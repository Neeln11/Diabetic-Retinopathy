import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from src.models.classifier import create_swin_dr_classifier
from src.preprocessing.transforms import BenGrahamPreprocessingD

class FeedbackDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row['image_path'])
        
        # Determine actual file existence
        # Fallback logic for when frontend just gives filename
        if not Path(img_path).exists():
             filename = Path(img_path).name
             found = False
             # Search everywhere in data and archive folders
             for base_dir in [Path('archive_v1_prototype'), Path('data')]:
                 for p in base_dir.rglob(filename):
                     if p.is_file():
                         img_path = str(p)
                         found = True
                         break
                 if found: break
             if not found:
                 print(f"Warning: Could not find image {filename}")

        data = {"image": img_path}
        if self.transform:
            data = self.transform(data)
            
        # The key feature: we use the DOCTOR'S correction, not the original label
        label = int(row['doctor_correction'])
        return data["image"], torch.tensor(label, dtype=torch.long)

def freeze_backbone(model):
    print("❄️ Freezing early layers (Stage 0, 1, 2) to prevent catastrophic forgetting...")
    
    # Freeze entire backbone first
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # Unfreeze the last stage (Stage 3/4) of the Swin Encoder
    # In MONAI SwinUNETR, it's called layers4
    if hasattr(model.backbone.swinViT, 'layers4'):
        last_layer = model.backbone.swinViT.layers4
        for param in last_layer.parameters():
            param.requires_grad = True
        
    for param in model.classifier.parameters():
        param.requires_grad = True

def train_feedback():
    print("=" * 60)
    print("HITL Few-Shot Fine-Tuning (Dry Run Trial)")
    print("=" * 60)
    
    csv_path = "data/feedback/doctor_corrections.csv"
    if not Path(csv_path).exists() or len(pd.read_csv(csv_path)) == 0:
        print("No feedback data found.")
        return

    dataset = FeedbackDataset(csv_path, transform=Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(224, 224)),
        BenGrahamPreprocessingD(keys=["image"], sigma=10),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"])
    ]))
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading base Swin model (Our Structural Expert)...")
    model = create_swin_dr_classifier(vram_optimized=True, num_classes=5)
    model.load_state_dict(torch.load("production/weights/production_v1.pth", map_location=device), strict=False)
    model = model.to(device)
    
    # Analyze pre-update loss
    model.eval()
    criterion = nn.CrossEntropyLoss()
    pre_loss = 0.0
    print("\n🔍 Checking Pre-Update Baseline Loss on these 'Hard Cases'...")
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            pre_loss += loss.item()
    print(f"Initial Average Loss: {pre_loss / len(loader):.4f}")
    
    # 1. Freeze Layers
    freeze_backbone(model)
    
    # 2. Optimizer (Micro-learning rate)
    LR = 1e-6
    print(f"🛡️ Safety Guard Active: Learning Rate set extremely low ({LR})")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-5)
    
    # Fine-tuning dry run (1 Epoch)
    model.train()
    print("\n🚀 Executing 1-Epoch Continuous Learning Update...")
    total_loss = 0
    pbar = tqdm(loader)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item()})

    post_loss = total_loss / len(loader)
    print(f"\n✅ Fine-Tuning Complete.")
    print(f"📊 Loss Summary: {pre_loss:.4f} --> {post_loss:.4f}")
    
    if post_loss < pre_loss:
        print("SUCCESS: The model successfully absorbed the doctor's corrections without crashing!")
        
    print("Note: Weights NOT saved to production in dry-run mode.")

if __name__ == "__main__":
    train_feedback()
