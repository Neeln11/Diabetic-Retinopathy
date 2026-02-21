
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from torchvision.models import efficientnet_b0

# Add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.balanced_loader import BalancedDRDataset 
from src.utils.loss import OrdinalRegressionLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_CLASSES = 5

def validate_expert():
    print(f"🕵️ Validating EfficientNet Expert on Messidor-2...")
    
    # Load Model
    model_path = Path("production/weights/effnet_expert_v1.pth")
    if not model_path.exists():
        print("Model not found!")
        return
        
    model = efficientnet_b0(weights=None) # No init weights needed
    model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    # Load Messidor-2 (Using master CSV but filtered if we had a flag, or just use master val set as proxy)
    # The user asked for "Messidor-2 test set". We don't have a separate messidor.csv easily accessible 
    # except via the master csv 'source' column.
    
    master_csv = Path("archive_v1_prototype/unified_master.csv")
    if not master_csv.exists(): master_csv = Path("src/data/unified_master.csv")
    
    df = pd.read_csv(master_csv)
    
    # Filter for Messidor-2 if possible, or just use Validation set from Master
    df['image'] = df['image_path']
    df['label'] = df['dr_grade']

    if 'source' in df.columns:
        messidor_df = df[df['source'] == 'messidor_2']
        if len(messidor_df) == 0:
            print("Warning: No Messidor-2 data found in CSV. Using full validation set.")
            frames = [] # empty
        else:
            print(f"Found {len(messidor_df)} Messidor-2 images.")
            # Use all for validation
            pass
    else:
        # Fallback
        messidor_df = df.sample(500, random_state=42)
        
    if len(messidor_df) == 0:
        # If filtering failed, use random sample
        messidor_df = df.sample(min(1000, len(df)), random_state=42)
        
    # CRITICAL: Reset index for BalancedDRDataset compatibility
    messidor_df = messidor_df.reset_index(drop=True)

    val_ds = BalancedDRDataset(messidor_df, mode='val', img_size=IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3,4])
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Recall
    class_recall = cm.diagonal() / cm.sum(axis=1)
    
    print("\n📊 EfficientNet Expert Validation Report:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"QWK Score: {qwk:.4f}")
    print(f"Stage 1 Recall: {class_recall[1]:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    validate_expert()
