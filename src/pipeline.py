"""
Training Pipeline for Swin-DR Classifier
Implements pilot run logic with Mixed Precision and specific hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import os
from datetime import datetime

# Import project modules
from src.models.classifier import create_swin_dr_classifier
from src.preprocessing.quality_gate import QualityGate
from src.preprocessing.transforms import BenGrahamPreprocessingD
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord

# Configuration
CONFIG = {
    "img_size": 224,
    "batch_size": 16,  # Optimized for 4GB VRAM with FP16
    "lr": 2e-5,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "epochs": 1,  # Just a pilot run
    "data_dir": Path("data/raw/ddr"),
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"Running on device: {CONFIG['device']}")

class DDRPilotDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, limit=100):
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.quality_gate = QualityGate()
        self.limit = limit
        
        # Filter mostly existing files first to avoid slow checks later
        self.valid_samples = []
        print(f"Scanning first {limit} images for pilot run...")
        
        processed_count = 0
        rejection_count = 0
        
        for idx, row in self.df.iterrows():
            if processed_count >= limit:
                break
                
            img_name = row['id_code']
            # Handle nested directory structure
            possible_paths = [
                self.data_dir / "DR_grading" / img_name,
                self.data_dir / "DR_grading" / "DR_grading" / img_name
            ]
            
            img_path = None
            for p in possible_paths:
                # Check directly or with .jpg extension
                if p.exists():
                    img_path = p
                    break
                
                p_jpg = p.with_name(p.name + ".jpg")
                if p_jpg.exists():
                    img_path = p_jpg
                    break
            
            if img_path and img_path.exists():
                # Apply Quality Gate
                is_valid, metrics = self.quality_gate.is_gradable(str(img_path))
                
                if is_valid:
                    self.valid_samples.append({
                        "image": str(img_path),
                        "label": row['diagnosis']
                    })
                else:
                    rejection_count += 1
                    
                processed_count += 1
        
        print(f"Pilot Dataset: {len(self.valid_samples)} valid images, {rejection_count} rejected.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Load and transform
        data = {"image": sample["image"]}
        if self.transform:
            data = self.transform(data)
            
        return data["image"], torch.tensor(sample["label"], dtype=torch.long)

def run_pilot():
    # 1. Prepare Data
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(224, 224)),
        # ScaleIntensityd removed here - BenGraham needs 0-255
        BenGrahamPreprocessingD(keys=["image"], sigma=10),
        ScaleIntensityd(keys=["image"]), # Scale to [0, 1] after BenGraham
        ToTensord(keys=["image"])
    ])
    
    csv_path = CONFIG["data_dir"] / "DR_grading.csv"
    dataset = DDRPilotDataset(
        CONFIG["data_dir"], 
        csv_path, 
        transform=transforms, 
        limit=100
    )
    
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    # 2. Prepare Model
    model = create_swin_dr_classifier(vram_optimized=True)
    model = model.to(CONFIG["device"])
    
    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Warmup + Cosine Scheduler logic would go here, simplified for pilot
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # 4. Training Loop
    print("\nStarting Pilot Training...")
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(tqdm(loader)):
        images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
        
        optimizer.zero_grad()
        
        # Mixed Precision Forward
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        # Scaling Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(loader)
    duration = time.time() - start_time
    
    print(f"\nPilot Complete in {duration:.2f}s")
    print(f"Average Loss: {avg_loss:.4f}")
    
    # scheduler step
    scheduler.step()
    
    return {
        "duration": duration,
        "avg_loss": avg_loss,
        "valid_samples": len(dataset),
        "total_scanned": dataset.limit
    }

if __name__ == "__main__":
    run_pilot()
