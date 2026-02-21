
import os
import sys
from pathlib import Path

# Force UTF-8 for Windows Console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import pandas as pd
from src.models.ensemble_factory import MetaPredictor
from src.preprocessing.transforms import BenGrahamPreprocessingD
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord

def get_transforms():
    """Matches app.py transforms"""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(224, 224)),
        BenGrahamPreprocessingD(keys=["image"], sigma=10),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"])
    ])

def main():
    print(">>> Starting Ensemble Validation: Swin vs EfficientNet vs DenseNet")
    
    # 1. Initialize Ensemble (Sequential Mode)
    predictor = MetaPredictor(swin_path="src/models/messidor_adapted_v2.pth")
    
    # 2. Select Test Cases
    try:
        df = pd.read_csv("data/processed/messidor_few_shot_train.csv")
        stage1_cases = df[df['label'] == 1].head(10)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"\n[INFO] Validating on {len(stage1_cases)} Stage 1 (Mild NPDR) cases...")
    
    results = []
    transforms = get_transforms()
    
    for idx, row in stage1_cases.iterrows():
        img_path = row['image']
        true_label = row['label']
        
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue
            
        data = {"image": img_path}
        data = transforms(data)
        img_tensor = data["image"].unsqueeze(0)
        
        print(f"  Processing {os.path.basename(img_path)}...", end="\r")
        
        probs, breakdown = predictor.predict_sequential(img_tensor)
        
        results.append({
            "Image": os.path.basename(img_path),
            "True_Label": true_label,
            "Swin_Pred": breakdown['Swin'],
            "EffNet_Pred": breakdown['EfficientNet'],
            "DenseNet_Pred": breakdown['DenseNet'],
            "Ensemble_Pred": breakdown['Ensemble'],
            "Corrected_By_Ensemble": (breakdown['Swin'] != true_label) and (breakdown['Ensemble'] == true_label)
        })
        
    print("\n\n=== Validation Results ===")
    print(f"{'Image':<30} | {'True':<4} | {'Swin':<4} | {'Eff':<4} | {'Dense':<5} | {'Ensmbl':<6} | {'Fixed?'}")
    print("-" * 85)
    
    fixed_count = 0
    total = len(results)
    
    for r in results:
        fixed_mark = "[FIXED]" if r['Corrected_By_Ensemble'] else ""
        if r['Corrected_By_Ensemble']: fixed_count += 1
        
        print(f"{r['Image']:<30} | {r['True_Label']:<4} | {r['Swin_Pred']:<4} | {r['EffNet_Pred']:<4} | {r['DenseNet_Pred']:<5} | {r['Ensemble_Pred']:<6} | {fixed_mark}")

    print("-" * 85)
    print(f"[RESULT] Outcome: {fixed_count}/{total} cases corrected by the Ensemble.")
    
    with open("docs/Ensemble_Validation_Report.md", "w") as f:
        f.write("# Ensemble Validation Report\n\n")
        f.write("Evaluation of 3-Model Ensemble on known challenging 'Stage 1' cases.\n\n")
        f.write("| Image | True Label | Swin (0.4) | EffNet (0.3) | DenseNet (0.3) | Ensemble | Outcome |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for r in results:
            outcome = "**CORRECTED**" if r['Corrected_By_Ensemble'] else ("Correct" if r['Ensemble_Pred'] == r['True_Label'] else "Missed")
            f.write(f"| {r['Image']} | {r['True_Label']} | {r['Swin_Pred']} | {r['EffNet_Pred']} | {r['DenseNet_Pred']} | {r['Ensemble_Pred']} | {outcome} |\n")
            
        f.write(f"\n**Summary**: The Ensemble corrected {fixed_count} out of {total} complex cases.\n")

if __name__ == "__main__":
    main()
