
import argparse
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord, RandRotate90d, RandFlipd
from torchvision.models import efficientnet_b0, densenet121, EfficientNet_B0_Weights, DenseNet121_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

# Add project root to path
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.preprocessing.transforms import BenGrahamPreprocessingD
from src.utils.loss import OrdinalRegressionLoss
from src.data.balanced_loader import BalancedDRDataset # [NEW]
from sklearn.model_selection import StratifiedKFold # [NEW]

# Configuration
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10 # Back to full training
LEARNING_RATE = 1e-4
NUM_CLASSES = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Check if loader is using BalancedDataset and shuffle it
    if hasattr(loader.dataset, 'on_epoch_end'):
        loader.dataset.on_epoch_end()
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / total, correct / total

def validate(model, loader, criterion):
    # ... (Same as before) ...
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # [NEW] Confusion Matrix & Per-Class Recall
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3,4])
    
    # Calculate Recall per class (Sensitivity)
    # Recall = TP / (TP + FN) -> Diagonal / Sum(Rows)
    class_recall = cm.diagonal() / cm.sum(axis=1)
    
    print("\nEpoch Validation Report:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Stage 3 (Severe) Recall: {class_recall[3]:.4f}")
    print("-" * 30)
    
    # [NEW] VRAM Cleanup
    torch.cuda.empty_cache()
    
    return running_loss / len(loader.dataset), accuracy, qwk, cm

def main():
    parser = argparse.ArgumentParser(description="Train Ensemble Auxiliary Models (Balanced)")
    parser.add_argument("--model", type=str, required=True, choices=['efficientnet', 'densenet'], help="Model architecture")
    args = parser.parse_args()
    
    print(f"Starting Balanced Training for {args.model.upper()}...")
    
    # 1. Load Data (Unified Master) - Look in Archive now as requested by user previously
    master_csv = Path("archive_v1_prototype/unified_master.csv")
    if not master_csv.exists():
         # Fallback to old path if archive move failed or wasn't done yet (safety check)
         master_csv = Path("src/data/unified_master.csv")
         
    if not master_csv.exists():
        raise FileNotFoundError("unified_master.csv not found!")
        
    df = pd.read_csv(master_csv)
    df['image'] = df['image_path']
    df['label'] = df['dr_grade']
    
    # 2. Stratified K-Fold (Using Fold 0 for this run as standard val)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # We only take the first fold for this training session
    train_idx, val_idx = next(skf.split(df['image'], df['label']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"Train Size: {len(train_df)} | Val Size: {len(val_df)}")
    
    # 3. Datasets & Loaders
    # Wrap Training Data in BalancedLoader
    train_ds = BalancedDRDataset(
        df=train_df, 
        target_count=8200, # Per class! Total epoch size = 41,000 images
        mode='train'
    )
    
    # Validation is standard (representative of real world)
    # Re-use get_transforms from previous code logic or inline it? 
    # Let's use the BalancedLoader in 'val' mode which just applies standard transforms
    val_ds = BalancedDRDataset(
        df=val_df,
        mode='val'
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. Initialize Model
    print(f"Initializing {args.model} with ImageNet weights...")
    if args.model == 'efficientnet':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
        save_name = "efficientnet_b0_balanced.pth"
    elif args.model == 'densenet':
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(1024, NUM_CLASSES)
        save_name = "densenet121_balanced.pth"
        
    model = model.to(DEVICE)
    
    # 5. Loss Function (Weighted Class Matrix + Ordinal)
    # Weights: Stage 3 (Severe) = 2.0, Stage 0 = 0.5 (Downweight healthy majority), Others = 1.0 or 2.0
    # User Spec: Stage 3=2.0, Stage 0=0.5. Let's keep Stage 1 & 2 high too.
    # We need to modify OrdinalRegressionLoss or apply weights manually.
    # The current OrdinalRegressionLoss supports 'alpha' but not per-class weights easily in __init__.
    # It constructs weights dynamically in forward(). 
    # Let's UPDATE the logic in loss.py or subclass it?
    # Better: Update loss.py to accept a 'class_weights' dict or tensor.
    # For now, let's stick to the requested spec by modifying the weights tensor generation inside loss.py (via a new arg?)
    # Or just hardcode the "Stage 3 = 2.0" requirement in the existing logic if we edit loss.py.
    
    # 6. MLFlow Setup
    import mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Phase8_EfficientNet_Expert")
    
    with mlflow.start_run(run_name=f"{args.model}_balanced_v1"):
        mlflow.log_params({
            "model": args.model,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": "OrdinalRegressionLoss_1:2:3:4:5"
        })
        
        # [NEW] VRAM Safety Check (Batch 1 Profiling)
        print("\nRunning 1-Batch VRAM Safety Check...")
        torch.cuda.reset_peak_memory_stats()
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        _ = model(dummy_input)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"   Peak VRAM: {peak_mem:.4f} GB")
        
        if peak_mem > 4.0:
            print("WARNING: VRAM usage high!")
        else:
            print("VRAM Check Passed (< 4GB)")
            
        mlflow.log_metric("vram_check_gb", peak_mem)
        torch.cuda.empty_cache()

        criterion = OrdinalRegressionLoss(num_classes=5, alpha=1.0) 
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        best_qwk = -1.0
        
        print(f"\nTraining on {len(train_df)} samples, Validating on {len(val_df)} samples")
        print(f"Using Weighted Ordinal Loss (Prioritizing Stage 1+)")
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            
            # [NEW] VRAM Safety Check
            torch.cuda.empty_cache()
            
            val_loss, val_acc, val_qwk, val_cm = validate(model, val_loader, criterion)
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | QWK: {val_qwk:.4f}")
            
            # Log Metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_qwk": val_qwk,
                "epoch": epoch + 1
            }, step=epoch)
            
            if val_qwk > best_qwk:
                print(f"New Best Model! (QWK {best_qwk:.4f} -> {val_qwk:.4f})")
                best_qwk = val_qwk
                
                # Save artifact
                save_path = Path(f"src/models/{save_name}")
                torch.save(model.state_dict(), save_path)
                mlflow.log_artifact(str(save_path))
                
        print(f"\nTraining Complete. Best QWK: {best_qwk:.4f}")
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
