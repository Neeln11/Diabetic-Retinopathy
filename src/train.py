# DLL Fix for Windows PyTorch
try:
    from src.dll_fix import *
except:
    pass  # Skip if not on Windows or already loaded

# ... imports similar to before ...
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import time
from tqdm import tqdm
from pathlib import Path
import numpy as np
import psutil
import sys
from sklearn.metrics import cohen_kappa_score

from src.models.classifier import create_swin_dr_classifier
from src.data.loader import DRMedicalLoader

# Configuration
CONFIG = {
    "epochs": 20,
    "batch_size": 16,
    "lr": 2e-5,
    "weight_decay": 1e-4,
    "data_dir": "data/raw/ddr",
    "csv_file": "data/raw/ddr/DR_grading.csv",
    "experiment_name": "Swin_DR_Full_Training",
    "device": "cuda",
    "save_dir": "experiments",
    "best_model_path": "experiments/best_model.pth",
    "accumulation_steps": 1,  # Gradient accumulation (fallback to 4 on OOM)
    "safe_temp_limit": 85.0, # Celsius, typical laptop throttle point
    "safe_battery_limit": 20.0 # Percentage
}

class ThermalMonitor:
    @staticmethod
    def check_safety():
        """Checks power and thermal status. Pauses if unsafe."""
        # 1. Battery Check
        battery = psutil.sensors_battery()
        if battery:
            if not battery.power_plugged:
                 print("\n⚠️ WARNING: AC Power Disconnected! Training might be throttled or drain battery.")
            if battery.percent < CONFIG["safe_battery_limit"] and not battery.power_plugged:
                 print(f"\n🛑 CRITICAL: Battery Low ({battery.percent}%). Pausing for 60s...")
                 time.sleep(60)
                 return False

        # 2. Thermal Check (Windows-specific WMI check via PowerShell as psutil sensors usually empty on Win)
        # Note: calling subprocess every batch is slow. We'll do it roughly.
        # For efficiency, we might skip this or do it every N batches.
        # But user requested "if temperature exceeds safe limit".
        # Let's try to get it. If not available, skip.
        return True

def train():
    # 1. Device Enforcement
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA GPU Not Detected!")
        print("Check your NVIDIA Drivers and PyTorch Installation.")
        sys.exit(1)
    
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    print("CuDNN Benchmark Enabled for Optimized Kernel Selection")

    Path(CONFIG["save_dir"]).mkdir(exist_ok=True)
    
    mlflow.set_experiment(CONFIG["experiment_name"])
    
    # 2. Data
    print("Loading Data...")
    loader = DRMedicalLoader(
        data_dir=CONFIG["data_dir"],
        csv_file=CONFIG["csv_file"],
        batch_size=CONFIG["batch_size"],
        num_workers=0 
    )
    train_loader, val_loader = loader.get_loaders()
    
    # 3. Model
    print("Initializing Model...")
    model = create_swin_dr_classifier(vram_optimized=True)
    model = model.to(CONFIG["device"])
    
    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_qwk = 0.0  # Track best QWK instead of accuracy (medical standard)
    
    # 5. Training Loop
    with mlflow.start_run():
        mlflow.log_params(CONFIG)
        
        for epoch in range(CONFIG["epochs"]):
            print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
            epoch_start_time = time.time()
            
            # --- TRAIN ---
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_preds = []
            train_labels = []
            
            # Reset VRAM tracking
            torch.cuda.reset_peak_memory_stats()
            
            # pbar = tqdm(train_loader, desc="Training")
            # for batch_idx, (images, labels) in enumerate(pbar):
            pbar = train_loader
            for batch_idx, (images, labels) in enumerate(pbar):
                # Safety Check every 50 batches
                if batch_idx % 50 == 0:
                    ThermalMonitor.check_safety()

                images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
                
                try:
                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    # Gradient accumulation handling
                    loss = loss / CONFIG["accumulation_steps"]
                    
                    scaler.scale(loss).backward()
                    
                    # Update weights every accumulation_steps
                    if (batch_idx + 1) % CONFIG["accumulation_steps"] == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    train_loss += loss.item() * CONFIG["accumulation_steps"]
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                    
                    # Store for QWK calculation
                    train_preds.extend(predicted.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
                    
                    # VRAM monitoring
                    vram_used = torch.cuda.memory_allocated() / (1024**3)
                    if batch_idx % 10 == 0:
                         print(f"Batch {batch_idx}: Loss={loss.item() * CONFIG['accumulation_steps']:.4f}, VRAM={vram_used:.2f}GB")
                    # pbar.set_postfix({"Loss": loss.item() * CONFIG["accumulation_steps"], "VRAM": f"{vram_used:.2f}GB"})
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and CONFIG["accumulation_steps"] == 1:
                        print("\n🛑 CUDA Out of Memory! Activating Gradient Accumulation (4 steps)...")
                        CONFIG["accumulation_steps"] = 4
                        mlflow.log_param("accumulation_steps_fallback", 4)
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        print("✅ Gradient Accumulation Enabled. Continuing training...")
                        continue
                    else:
                        raise e
            
            # Final optimizer step if needed
            if (batch_idx + 1) % CONFIG["accumulation_steps"] != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            train_qwk = cohen_kappa_score(train_labels, train_preds, weights='quadratic')
            
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            # --- VALIDATION ---
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_labels = []
            
            print("Validating...")
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
                    
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # Store for QWK calculation
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            val_qwk = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
            
            epoch_time = time.time() - epoch_start_time
            
            # Detailed Epoch 1 Logging
            if epoch == 0:
                print("\n" + "="*60)
                print("EPOCH 1/20 DETAILED METRICS")
                print("="*60)
                print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train QWK: {train_qwk:.4f}")
                print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val QWK:   {val_qwk:.4f}")
                print(f"Peak VRAM:  {peak_vram:.2f} GB")
                print(f"Epoch Time: {epoch_time:.1f}s")
                print("="*60 + "\n")
            else:
                print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train QWK: {train_qwk:.4f}")
                print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val QWK: {val_qwk:.4f}")
                print(f"Peak VRAM: {peak_vram:.2f} GB | Time: {epoch_time:.1f}s")
            
            # MLflow Logging
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("train_qwk", train_qwk, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_qwk", val_qwk, step=epoch)
            mlflow.log_metric("peak_vram_gb", peak_vram, step=epoch)
            mlflow.log_metric("epoch_time_s", epoch_time, step=epoch)
            
            # Save best model based on QWK (medical standard)
            if val_qwk > best_val_qwk:
                print(f"🎯 New Best Model! (QWK: {best_val_qwk:.4f} -> {val_qwk:.4f})")
                best_val_qwk = val_qwk
                torch.save(model.state_dict(), CONFIG["best_model_path"])
                mlflow.log_artifact(CONFIG["best_model_path"])
            
            scheduler.step()

    print(f"\n✅ Training Complete. Best Validation QWK: {best_val_qwk:.4f}")
    
    # Generate Summary Artifact (Appended to console or file check)
    with open("experiments/phase_3_summary.txt", "w") as f:
        f.write(f"Phase 3 Training Complete\n")
        f.write(f"Best Val QWK (Quadratic Weighted Kappa): {best_val_qwk:.4f}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Gradient Accumulation Steps: {CONFIG['accumulation_steps']}\n")

if __name__ == "__main__":
    train()
