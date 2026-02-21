import argparse
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from monai.data import DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.loss import OrdinalRegressionLoss
from src.data.balanced_loader import BalancedDRDataset

# Configuration
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3,4])
    
    print("\nEpoch Validation Report:")
    print(f"Confusion Matrix:\n{cm}")
    print("-" * 30)
    
    return running_loss / len(loader.dataset), accuracy, qwk, cm

def main():
    print("Starting Phase 10: DenseNet-121 Expert Training (Grade 1 Specialist)...")
    
    master_csv = Path("archive_v1_prototype/unified_master.csv")
    if not master_csv.exists():
         master_csv = Path("src/data/unified_master.csv")
         
    df = pd.read_csv(master_csv)
    df['image'] = df['image_path']
    df['label'] = df['dr_grade']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(df['image'], df['label']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"Train Size: {len(train_df)} | Val Size: {len(val_df)}")
    
    train_ds = BalancedDRDataset(df=train_df, target_count=8200, mode='train')
    val_ds = BalancedDRDataset(df=val_df, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("Initializing DenseNet-121 with ImageNet weights...")
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(1024, 5)
    model = model.to(DEVICE)
    
    # Custom Loss: Specifically boost Grade 1 weight to 2.5
    class_weights = [1.0, 2.5, 3.0, 4.0, 5.0]
    criterion = OrdinalRegressionLoss(num_classes=5, alpha=1.0, class_weights=class_weights) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_qwk = -1.0
    save_path = Path("production/weights/densenet_balanced.pth")
    
    # Dynamic Augmentation Tracking
    current_zoom_min = 0.9
    current_zoom_max = 1.1
    
    import mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Phase10_DenseNet_Expert")
    
    with mlflow.start_run(run_name="densenet_g1_expert_v1"):
        mlflow.log_params({
            "model": "densenet121",
            "epochs": NUM_EPOCHS,
            "loss_weights": str(class_weights)
        })
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            torch.cuda.empty_cache()
            
            val_loss, val_acc, val_qwk, val_cm = validate(model, val_loader, criterion)
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | QWK: {val_qwk:.4f}")
            
            mlflow.log_metrics({"val_qwk": val_qwk, "epoch": epoch + 1}, step=epoch)
            
            if val_qwk > best_qwk:
                print(f"New Best Model! (QWK {best_qwk:.4f} -> {val_qwk:.4f})")
                best_qwk = val_qwk
                torch.save(model.state_dict(), save_path)
                
            # Dynamic Zoom Logic check every 2 epochs
            if (epoch + 1) % 2 == 0:
                # Calculate Grade 1 miss rate specifically as Grade 0
                true_g1 = val_cm[1].sum()
                g1_pred_0 = val_cm[1][0]
                if true_g1 > 0:
                    miss_rate = g1_pred_0 / true_g1
                    print(f"Diagnostics: Grade 1 -> Grade 0 Miss Rate: {miss_rate:.2%}")
                    
                    if miss_rate > 0.50: # If >50% of Grade 1 are called Normal
                        current_zoom_min = min(1.0, current_zoom_min + 0.05)
                        current_zoom_max = min(1.3, current_zoom_max + 0.1)
                        print(f"WARNING: High Grade 1 miss rate. Increasing Zoom Augmentation to {current_zoom_minx}-{current_zoom_max}x")
                        train_ds.set_dynamic_zoom(current_zoom_min, current_zoom_max)

        print(f"\nTraining Complete. Best QWK: {best_qwk:.4f}")

if __name__ == "__main__":
    main()
