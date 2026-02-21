
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.classifier import create_swin_dr_classifier
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord
from src.preprocessing.transforms import BenGrahamPreprocessingD
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score

# Configuration
CONFIG = {
    "data_path": "data/processed/messidor_few_shot_train.csv",
    "original_model_path": "src/models/production_v1.pth",
    "adapted_model_path": "src/models/messidor_adapted_v2.pth",
    "plot_path": "hitl_progress_chart.png",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "img_size": 224,
    "batch_size": 32
}

class FeedbackDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image']
        label = int(row['label'])
        
        data = {"image": img_path}
        if self.transform:
            data = self.transform(data)
            
        return data['image'], label

def get_transforms():
    # Enforce Ben Graham as per Phase 3
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(CONFIG['img_size'], CONFIG['img_size'])),
        BenGrahamPreprocessingD(keys=["image"], sigma=10),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"])
    ])

def load_model(path, device):
    print(f"Loading model from {path}...")
    model = create_swin_dr_classifier(vram_optimized=True)
    try:
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model {path}: {e}")
        return None

def get_predictions(model, loader, device):
    preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # No temperature scaling needed for raw grade prediction comparison here
            # But adapted model v2 should serve logits.
            probs = torch.softmax(outputs, dim=1)
            batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
            preds.extend(batch_preds)
    return np.array(preds)

def analyze_bias(y_true, y_pred):
    """
    Detects if specific grades have high disagreement.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    
    # Calculate per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1) # Recall
    
    bias_report = {}
    for i, acc in enumerate(class_acc):
        if np.isnan(acc): acc = 0.0
        bias_report[i] = acc
        
    return bias_report, cm

def stabilization_check(disagreement_history, window=50):
    """
    Checks if disagreement rate is dropping over a rolling window.
    Takes a list of 1s (disagreement) and 0s (agreement).
    """
    if len(disagreement_history) < window:
        return "Insufficient Data", 0.0
        
    recent = disagreement_history[-window:]
    rate = sum(recent) / window
    
    # Simple heuristic: If rate < 0.2 (80% agreement), mostly stable?
    # Or comparing to previous window?
    # Let's just return the rate for now.
    
    return "Stabilizing" if rate < 0.2 else "Volatile", rate

def main():
    print("="*60)
    print("HITL FEEDBACK ANALYTICS & IMPACT REPORT")
    print("="*60)
    
    # 1. Load Data (Ground Truth / Doctor Corrections)
    df = pd.read_csv(CONFIG['data_path'])
    print(f"Loaded {len(df)} feedback cases from {CONFIG['data_path']}")
    
    ds = FeedbackDataset(df, transform=get_transforms())
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    y_true = df['label'].values
    
    # 2. Get Predictions
    # Original Model (Pre-Adaptation) - representing "Original AI Pred"
    model_v1 = load_model(CONFIG['original_model_path'], CONFIG['device'])
    y_pred_v1 = get_predictions(model_v1, loader, CONFIG['device'])
    
    # Adapted Model (Post-Feedback Training) - representing "Current Status"
    model_v2 = load_model(CONFIG['adapted_model_path'], CONFIG['device'])
    y_pred_v2 = get_predictions(model_v2, loader, CONFIG['device'])
    
    # 3. Disagreement Rate
    disagreement_v1 = np.mean(y_pred_v1 != y_true)
    disagreement_v2 = np.mean(y_pred_v2 != y_true)
    
    print(f"\nDisagreement Rate (Doctor vs AI):")
    print(f"  Original AI: {disagreement_v1:.2%} (Acc: {1-disagreement_v1:.2%})")
    print(f"  Adapted AI:  {disagreement_v2:.2%} (Acc: {1-disagreement_v2:.2%})")
    print(f"  Improvement: {disagreement_v1 - disagreement_v2:.2%} reduction in disagreement")
    
    # 4. Bias Detection
    print("\nBias Detection (Recall per Grade):")
    bias_rep_v1, _ = analyze_bias(y_true, y_pred_v1)
    bias_rep_v2, _ = analyze_bias(y_true, y_pred_v2)
    
    print("  Grade | Original | Adapted | Change")
    print("  ------+----------+---------+-------")
    for g in range(5):
        print(f"  {g}     | {bias_rep_v1[g]:6.1%} | {bias_rep_v2[g]:6.1%} | {bias_rep_v2[g]-bias_rep_v1[g]:+6.1%}")
        
    # 5. Critical Misses Analysis (Stage 2+ predicted as <2)
    # How many Referable cases were missed?
    referable_mask = y_true >= 2
    
    misses_v1 = np.sum((y_pred_v1 < 2) & referable_mask)
    misses_v2 = np.sum((y_pred_v2 < 2) & referable_mask)
    total_referable = np.sum(referable_mask)
    
    print(f"\nCritical Misses (Sensitivity Check):")
    print(f"  Total Referable Cases: {total_referable}")
    print(f"  Missed by Original:    {misses_v1} ({misses_v1/total_referable:.1%})")
    print(f"  Missed by Adapted:     {misses_v2} ({misses_v2/total_referable:.1%})")
    
    # 6. Stabilization check
    # Simulate a stream of feedback 
    # We'll use the Adapted model's agreement vector
    agreement_stream = (y_pred_v2 != y_true).astype(int) # 1 for disagreement
    status, rate = stabilization_check(agreement_stream, window=50)
    print(f"\nStabilization Status (Last 50 cases): {status} (Disagreement Rate: {rate:.2%})")
    
    # 7. Visualization
    plt.figure(figsize=(10, 6))
    
    # Bar chart of Misses
    models = ['Original AI', 'Adapted AI']
    miss_counts = [misses_v1, misses_v2]
    
    bars = plt.bar(models, miss_counts, color=['#ff9999', '#66b3ff'])
    plt.title('Critical Misses Reduction (HITL Feedback Impact)')
    plt.ylabel('Number of Missed Referable Cases')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}\n({height/total_referable:.1%})',
                 ha='center', va='bottom')
                 
    plt.savefig(CONFIG['plot_path'])
    print(f"\n✅ HITL Progress Chart saved to {CONFIG['plot_path']}")

if __name__ == "__main__":
    main()
