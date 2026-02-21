
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import os
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.classifier import create_swin_dr_classifier
from src.explainability.heatmaps import ExplainabilityEngine
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord
from src.preprocessing.transforms import BenGrahamPreprocessingD
from src.utils.risk_calculator import RiskCalculator

# Configuration
CONFIG = {
    "model_path": "src/models/messidor_adapted_v2.pth",
    "calibration_config": "src/models/calibration_config.json",
    "data_path": "data/processed/messidor_few_shot_train.csv", # Use some from here
    "output_path": "docs/Final_Detection_Gallery.png",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_images": 5
}

def load_model():
    print(f"Loading model from {CONFIG['model_path']}...")
    model = create_swin_dr_classifier(vram_optimized=True)
    try:
        checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(CONFIG['device'])
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    # Load Temperature
    temp = 1.0
    if Path(CONFIG['calibration_config']).exists():
        import json
        with open(CONFIG['calibration_config'], 'r') as f:
            temp = json.load(f).get('temperature', 1.0)
            
    return model, temp

def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(224, 224)),
        BenGrahamPreprocessingD(keys=["image"], sigma=10),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"])
    ])

def main():
    print("Generating Final Gallery...")
    
    # Load Model
    model, temperature = load_model()
    if not model: return
    
    explainer = ExplainabilityEngine(model)
    transforms = get_transforms()
    
    # Load Data (Sample 5 arbitrary images)
    # We can use the few-shot csv
    import pandas as pd
    df = pd.read_csv(CONFIG['data_path'])
    # Try to pick one of each grade if possible, or just random
    # Stratified sample if possible
    
    samples = []
    for g in range(5):
        grade_df = df[df['label'] == g]
        if len(grade_df) > 0:
            samples.append(grade_df.iloc[0])
            
    # Fill remaining if needed
    if len(samples) < 5:
        remaining = 5 - len(samples)
        extras = df.sample(n=remaining)
        for _, row in extras.iterrows():
            samples.append(row)
            
    # Process
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 4 * len(samples)))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    DR_LEVELS = {0: "No DR", 1: "Mild", 2: "Mod", 3: "Severe", 4: "PDR"}
    
    for idx, row in enumerate(samples):
        img_path = row['image']
        true_label = int(row['label'])
        
        # 1. Original
        orig_img = Image.open(img_path).resize((224, 224))
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f"Original (True: {DR_LEVELS[true_label]})")
        axes[idx, 0].axis('off')
        
        # 2. Heatmap
        data = {"image": img_path}
        data = transforms(data)
        input_tensor = data["image"].unsqueeze(0).to(CONFIG['device'])
        
        with torch.no_grad():
            logits = model(input_tensor)
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()
            prob_ref = probs[0][2:].sum().item()
            
        heatmap = explainer.generate_heatmap(input_tensor, target_class=pred)
        orig_cv = np.array(orig_img.convert('RGB')) # Already resized
        overlay = ExplainabilityEngine.overlay_heatmap(orig_cv, heatmap, alpha=0.5)
        
        axes[idx, 1].imshow(overlay)
        axes[idx, 1].set_title(f"AI: {DR_LEVELS[pred]} ({conf:.1%})\nRef Prob: {prob_ref:.1%}")
        axes[idx, 1].axis('off')
        
        # 3. Risk Score Card (Text visualization)
        # Dummy data for calc
        risk = RiskCalculator.calculate_risk(pred, hba1c=7.5, duration_years=10)
        
        text = f"RISK ASSESSMENT\n"
        text += f"---------------\n"
        text += f"Grade: {pred} ({DR_LEVELS[pred]})\n"
        text += f"Risk Score: {risk['score']}\n"
        text += f"Level: {risk['level']}\n\n"
        text += f"Rec:\n{risk['recommendation']}"
        
        axes[idx, 2].text(0.1, 0.5, text, fontsize=12, va='center', fontfamily='monospace')
        axes[idx, 2].axis('off')
        
        # Add safety flag border if ambiguous
        if 0.4 <= prob_ref <= 0.6:
            # Add yellow border to middle plot?
            for spine in axes[idx, 1].spines.values():
                spine.set_edgecolor('orange')
                spine.set_linewidth(5)
                
    # Save
    out_dir = Path(CONFIG['output_path']).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFIG['output_path'], bbox_inches='tight', dpi=150)
    print(f"✅ Gallery saved to {CONFIG['output_path']}")

if __name__ == "__main__":
    main()
