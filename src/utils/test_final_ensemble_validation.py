import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, recall_score, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.ensemble_factory import MetaPredictor 

def calculate_ece(probs_list, labels_list, n_bins=10):
    probs = np.vstack(probs_list)
    labels = np.array(labels_list)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(in_bin):
            prop_in_bin = in_bin.mean()
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def test_integration():
    print("Integration Test: Swin + EffNet + DenseNet (Final Clinical Validation)\n")
    
    # Check for DenseNet weights
    densenet_path = Path("production/weights/densenet_balanced.pth")
    if not densenet_path.exists():
        print(f"⚠️ DenseNet checkpoint not found at {densenet_path}")
        densenet_path = Path("production/weights/densenet121_aptos_finetuned.pth") # Fallback
    else:
        print(f"✅ Found DenseNet Balanced Checkpoint: {densenet_path}")
        
    # Config for integration
    config = {
        'swin_path': "production/weights/production_v1.pth", 
        'effnet_path': "production/weights/efficientnet_b0_balanced.pth",
        'densenet_path': str(densenet_path)
    }
    
    predictor = MetaPredictor(
        swin_path=config['swin_path'],
        effnet_path=config['effnet_path'],
        densenet_path=config['densenet_path']
    )
    
    master_csv = Path("archive_v1_prototype/unified_master.csv")
    if not master_csv.exists(): 
        master_csv = Path("src/data/unified_master.csv")
    df = pd.read_csv(master_csv)
    
    print("Sampling 50 Grade 0 and 50 Grade 1 cases (Focusing on Messidor-2)...")
    grade0_df = df[df['dr_grade'] == 0].sample(min(50, len(df[df['dr_grade'] == 0])), random_state=42)
    
    # Try to get Messidor
    messidor_grade1 = df[(df['dr_grade'] == 1) & (df['image_path'].str.contains('messidor', case=False, na=False))]
    if len(messidor_grade1) > 0:
        grade1_df = messidor_grade1.sample(min(50, len(messidor_grade1)), random_state=42)
    else:
        grade1_df = df[df['dr_grade'] == 1].sample(min(50, len(df[df['dr_grade'] == 1])), random_state=42)
        
    test_df = pd.concat([grade0_df, grade1_df]).sample(frac=1).reset_index(drop=True)
    
    print(f"🧪 Testing on {len(test_df)} Cases...")
    
    results = []
    swin_probs_all = []
    ens_probs_all = []
    y_true = []
    
    # Track metrics specifically for Grade 1
    grade_1_total = 0
    grade_1_swin_correct = 0
    grade_1_ens_correct = 0
    
    import tqdm
    for _, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = row['image_path']
        true_label = row['dr_grade']
        
        try:
            breakdown = predictor.predict(img_path, return_breakdown=True)
            
            # Swin
            swin_pred = breakdown['Swin']
            swin_probs = breakdown['Swin_Probs'].numpy()
            
            # Ensemble
            ens_pred = breakdown['Ensemble']
            ens_probs = breakdown['Ensemble_Probs'].numpy()
            is_ambiguous = breakdown['Ambiguous_Flag']
            
            swin_probs_all.append(swin_probs)
            ens_probs_all.append(ens_probs)
            y_true.append(true_label)
            
            if true_label == 1:
                grade_1_total += 1
                if swin_pred == 1:
                    grade_1_swin_correct += 1
                if ens_pred == 1:
                    grade_1_ens_correct += 1
            
            results.append({
                'Image': Path(img_path).name,
                'True Label': true_label,
                'Swin Pred': swin_pred,
                'Ensemble Pred': ens_pred,
                'Ambiguous': is_ambiguous
            })
        except Exception as e:
            print(f"Error predicting {img_path}: {e}")
            import traceback
            traceback.print_exc()
            
    # Calculate Metrics
    y_true_np = np.array(y_true)
    swin_preds = np.argmax(np.vstack(swin_probs_all), axis=1)
    ens_preds = np.argmax(np.vstack(ens_probs_all), axis=1)
    
    try:
        swin_qwk = cohen_kappa_score(y_true_np, swin_preds, weights='quadratic')
        ens_qwk = cohen_kappa_score(y_true_np, ens_preds, weights='quadratic')
    except:
        swin_qwk = 0.0
        ens_qwk = 0.0
    
    swin_recall_g1 = grade_1_swin_correct / max(1, grade_1_total)
    ens_recall_g1 = grade_1_ens_correct / max(1, grade_1_total)
    
    swin_ece = calculate_ece(swin_probs_all, y_true_np)
    ens_ece = calculate_ece(ens_probs_all, y_true_np)
    
    res_df = pd.DataFrame(results)
    ambig_count = res_df['Ambiguous'].sum()
    
    # Save Artifact
    with open("docs/Phase_11_Final_Validation.md", "w") as f:
        f.write("# Phase 11: Final Clinical Validation of the 3-Model Council\n\n")
        f.write("## 1. Metrics Comparison (Grade 0 vs 1 Hard Cases)\n\n")
        f.write("| Metric | Single Swin | 3-Model Ensemble (Calibrated) |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **QWK (on this subset)** | {swin_qwk:.4f} | {ens_qwk:.4f} |\n")
        f.write(f"| **Grade 1 Recall (Sensitivity)**| {swin_recall_g1:.2%} | {ens_recall_g1:.2%} |\n")
        f.write(f"| **Expected Calibration Error (ECE)**| {swin_ece:.4f} | {ens_ece:.4f} |\n\n")
        
        f.write("## 2. Grey-Zone Logic ('Ambiguous')\n")
        f.write("Cases flagged as Ambiguous (40-60% confidence):\n")
        f.write(f"- Total Flagged: {ambig_count} out of {len(res_df)}\n")
        f.write("- These cases would be routed to human doctors for manual review under clinical protocols.\n\n")
        
        f.write("## 3. Detailed Predictions Sample\n\n")
        f.write("```text\n")
        f.write(res_df.head(20).to_string(index=False))
        f.write("\n```\n")

    print("\nPhase 11 Validation Complete. Report saved to docs/Phase_11_Final_Validation.md")

if __name__ == "__main__":
    test_integration()
