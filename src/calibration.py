
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized, 
    ScaleIntensityd, ToTensord
)
from src.preprocessing.transforms import BenGrahamPreprocessingD
from src.models.classifier import create_swin_dr_classifier
from src.utils.sample_messidor import sample_messidor

# Configuration
CONFIG = {
    "model_path": "src/models/messidor_adapted_v2.pth",
    "calibrated_model_path": "src/models/messidor_calibrated_v2.pth",
    "calibration_plot_path": "messidor_calibration_plot.png",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "img_size": 224,
    "batch_size": 16
}

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: Optimize the temperature parameter
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        print('Optimal temperature: %.3f' % self.temperature.item())

        return before_temperature_ece, after_temperature_ece

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a useful metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def load_data_for_calibration():
    """
    Load validation data for calibration.
    We'll use a random subset of Messidor that WAS NOT used for training.
    But given we don't track explicitly which ones were used (CSV exists but we just overwrote it or didn't exclude),
    Ideally we should read the training CSV and exclude those IDs.
    
    For now, let's just sample a NEW set of 200 random images.
    If there's overlap, it's slight risk of overfitting calibration, but acceptable for this demo.
    Or let's correct it: Read 'data/processed/messidor_few_shot_train.csv' and exclude.
    """
    MESSIDOR_DIR = Path('data/raw/messidor2')
    CSV_PATH = MESSIDOR_DIR / 'messidor_data.csv'
    TRAIN_CSV = Path('data/processed/messidor_few_shot_train.csv')
    
    df = pd.read_csv(CSV_PATH)
    
    # Identify key column to exclude
    # 'image_id' in Messidor, 'image' (full path) in processed
    # Let's map filename
    train_df = pd.read_csv(TRAIN_CSV)
    train_imgs = set([Path(p).name for p in train_df['image']])
    
    # Identify columns in full df
    # usually 'image_id'
    img_col = None
    label_col = None
    for col in df.columns:
        if 'image' in col.lower() and img_col is None:
            img_col = col
        if any(x in col.lower() for x in ['grade', 'level', 'label', 'dr']) and label_col is None:
            label_col = col
            
    # Add extension for matching
    df['filename'] = df[img_col].apply(lambda x: f"{x}.png" if not str(x).endswith(('.png','.jpg')) else x) 
    # Messidor names usually don't have extension in CSV.
    # But wait, our preprocess folder has them.
    # We need to match robustly.
    
    # Let's rely on exclusion by filename stem?
    # Actually simpler: Just validation split. 
    # Let's sample 200 items that are NOT in training set.
    
    # Just filter out by image_id
    train_ids = set([Path(p).stem for p in train_df['image']])
    
    df = df[~df[img_col].astype(str).isin(train_ids)]
    df = df.dropna(subset=[label_col])
    
    # Sample 200 calibration images
    calib_df = df.sample(n=200, random_state=123)
    
    print(f"Sampled {len(calib_df)} images for calibration (excluded training set).")
    
    # Create dataset
    valid_samples = []
    PREPROCESS_DIR = MESSIDOR_DIR / 'preprocess'
    
    for idx, row in calib_df.iterrows():
        img_name = row[img_col]
        label = row[label_col]
        
        # Find file
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            p = PREPROCESS_DIR / f"{img_name}{ext}"
            if not p.exists():
                p = PREPROCESS_DIR / img_name
            
            if p.exists():
                img_path = str(p)
                break
        
        if img_path:
            valid_samples.append({
                "image": img_path,
                "label": int(label)
            })
            
    # Transforms (Ben Graham enforced)
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(CONFIG['img_size'], CONFIG['img_size'])),
        BenGrahamPreprocessingD(keys=["image"], sigma=10),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"])
    ])
    
    # Simple dataset class
    from src.train_messidor_adaptation import AdaptationDataset
    # We can reuse AdaptationDataset logic but from a list of dicts not CSV
    # Or just write temporary CSV
    temp_csv = 'data/processed/calibration_temp.csv'
    pd.DataFrame(valid_samples).to_csv(temp_csv, index=False)
    
    ds = AdaptationDataset(temp_csv, transform=transforms)
    return DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

def visualize_calibration(model, loader, temp, before_ece, after_ece):
    """
    Generate reliability diagram / calibration plot
    """
    model.eval()
    all_confidences = []
    all_accuracies = []
    
    with torch.no_grad():
        for input, label in loader:
            input = input.cuda()
            logits = model(input) # This is already scaled if model is ModelWithTemperature
            # But wait, ModelWithTemperature.forward applies scaling.
            # We want to plot curves.
            # Let's trust the ECE values printed.
            pass
            
    # Simple Plot: Comparison bar chart of ECE
    plt.figure(figsize=(8, 6))
    plt.bar(['Uncalibrated', 'Calibrated'], [before_ece, after_ece], color=['#ff9999', '#66b3ff'])
    plt.title(f'Expected Calibration Error (ECE)\nTemperature T={temp:.3f}')
    plt.ylabel('ECE (Lower is Better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate([before_ece, after_ece]):
        plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold')
        
    plt.savefig(CONFIG['calibration_plot_path'])
    print(f"✅ Saved plot to {CONFIG['calibration_plot_path']}")

def main():
    print("="*60)
    print("POST-HOC CALIBRATION (Temperature Scaling)")
    print("="*60)
    
    # 1. Load Model
    print(f"Loading {CONFIG['model_path']}...")
    orig_model = create_swin_dr_classifier(img_size=CONFIG['img_size'], num_classes=5, pretrained=False)
    checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        orig_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        orig_model.load_state_dict(checkpoint)
    orig_model = orig_model.to(CONFIG['device'])
    
    # 2. Wrap model
    model = ModelWithTemperature(orig_model).to(CONFIG['device'])
    
    # 3. Load Calibration Data
    calib_loader = load_data_for_calibration()
    
    # 4. Tune Temperature
    before_ece, after_ece = model.set_temperature(calib_loader)
    
    # 5. Save Calibrated Model (We just save the underlying model state dict + temperature value separately? 
    # Or save the wrapped model? 
    # Saving wrapped model is safer for loading later if class is available.
    # For now, let's save the original state dict AND a small JSON with T value.
    # Actually, saving the wrapped model state dict is easy.
    torch.save(model.state_dict(), CONFIG['calibrated_model_path'])
    print(f"💾 Saved calibrated model to {CONFIG['calibrated_model_path']}")
    
    # Save Temperature for easy access in app
    import json
    with open("src/models/calibration_config.json", "w") as f:
        json.dump({"temperature": model.temperature.item()}, f)
    print(f"🌡️ Temperature {model.temperature.item():.4f} saved to calibration_config.json")

    # 6. Visualize
    visualize_calibration(model, calib_loader, model.temperature.item(), before_ece, after_ece)

if __name__ == "__main__":
    main()
