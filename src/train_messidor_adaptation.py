
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized, 
    ScaleIntensityd, ToTensord
)
from src.preprocessing.transforms import BenGrahamPreprocessingD
from src.models.classifier import create_swin_dr_classifier
from src.utils.loss import OrdinalRegressionLoss
import time
from tqdm import tqdm

# Configuration
CONFIG = {
    "csv_file": "data/processed/messidor_few_shot_train.csv",
    "model_path": "src/models/production_v1.pth",
    "output_path": "src/models/messidor_adapted_v2.pth",
    "epochs": 20,
    "batch_size": 8,
    "lr": 5e-7, # Tiny learning rate as requested
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "img_size": 224
}

class AdaptationDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image']
        label = int(row['label'])
        
        data = {"image": img_path}
        if self.transform:
            try:
                data = self.transform(data)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # Return a zero tensor if failed (shouldn't happen on validated csv)
                data["image"] = torch.zeros((3, CONFIG['img_size'], CONFIG['img_size']))
                
        return data["image"], torch.tensor(label, dtype=torch.long)

def freeze_early_layers(model):
    """
    Freeze all layers except the last Swin Block (Stage 4) and the Head.
    MONAI SwinTransformer structure: layers1, layers2, layers3, layers4
    """
    print("❄️ Freezing early layers...")
    
    # Freeze entire backbone first
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # Unfreeze the last stage (Stage 4) of the Swin Encoder
    # In MONAI SwinTransformer, these are named layers1...layers4
    last_block = model.backbone.swinViT.layers4
    for param in last_block.parameters():
        param.requires_grad = True

    # Ensure LayerNorms at the end of network are trainable if they exist outside layers4
    # MONAI SwinViT doesn't expose a final norm attribute easily at top level usually, 
    # but check if there is a 'norm' attribute in swinViT
    if hasattr(model.backbone.swinViT, 'norm'):
        for param in model.backbone.swinViT.norm.parameters():
            param.requires_grad = True
        
    # Unfreeze Classifier and Global Pool
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Verify Trainable Parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable Params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({(trainable/total)*100:.1f}%)")

def load_production_model():
    print(f"Loading production model from {CONFIG['model_path']}...")
    model = create_swin_dr_classifier(
        img_size=CONFIG['img_size'],
        num_classes=5,
        pretrained=False
    )
    
    checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
    
    # Handle state dict format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(CONFIG['device'])
    return model

def train():
    print("="*60)
    print("MESSIDOR-2 DOMAIN ADAPTATION")
    print(f"Few-Shot Fine-Tuning (n={len(pd.read_csv(CONFIG['csv_file']))})")
    print(f"Learning Rate: {CONFIG['lr']}")
    print("="*60)
    
    # 1. Dataset & Loader
    # Enforcing Ben Graham Preprocessing regardless of filename
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(CONFIG['img_size'], CONFIG['img_size'])),
        BenGrahamPreprocessingD(keys=["image"], sigma=10), # Forcing this!
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"])
    ])
    
    dataset = AdaptationDataset(CONFIG['csv_file'], transform=transforms)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    
    # 2. Model
    model = load_production_model()
    freeze_early_layers(model)
    model.train()
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=CONFIG['lr'],
        weight_decay=1e-4
    )
    
    # Using Ordinal Loss
    criterion = OrdinalRegressionLoss(num_classes=5, alpha=1.0)
    
    # 4. Training Loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(loader)
        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Acc={acc:.2%} ({correct}/{total})")
        
    total_time = time.time() - start_time
    print(f"\n✅ Training Complete in {total_time:.1f}s")
    
    # Save adapted model
    torch.save(model.state_dict(), CONFIG['output_path'])
    print(f"💾 Saved adapted model to {CONFIG['output_path']}")

if __name__ == "__main__":
    train()
