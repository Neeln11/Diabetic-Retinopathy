
import os
import pandas as pd
from pathlib import Path

# Configuration
ROOT_DIR = Path(".")
IGNORE_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.idea', '.vscode', 'wandb', 'mlruns'}
IGNORE_EXTS = {'.pyc'}

# Knowledge Base of File Roles (Manual Mapping based on project history)
FILE_ROLES = {
    "src/app.py": ("Main Streamlit Dashboard Application", "Essential", "Keep"),
    "src/models/classifier.py": ("Swin-Tiny Model Definition (Production)", "Essential", "Keep"),
    "src/models/ensemble_factory.py": ("Ensemble Logic (Swin+EffNet+DenseNet)", "Essential", "Keep"),
    "src/models/vit_classifier.py": ("ViT-Tiny Definition (Benchmark Candidate)", "Legacy", "Delete (Outcome: Rejected)"),
    "src/data/balanced_loader.py": ("Balanced Dataset Wrapper (8.2k/class)", "Essential", "Keep"),
    "src/utils/loss.py": ("Ordinal Regression Loss Definition", "Essential", "Keep"),
    "src/utils/validate_expert.py": ("Validation Script for Expert Models", "Essential", "Keep"),
    "src/utils/test_ensemble_hard_cases.py": ("Integration Test for Ensemble", "Essential", "Keep"),
    "src/utils/count_data_distribution.py": ("Data Analysis Utility", "Useful", "Keep"),
    "src/benchmark_transformers.py": ("Benchmark Script (Swin vs ViT)", "Legacy", "Delete (Done)"),
    "src/train_ensemble_aux.py": ("Training Script for Aux Models", "Essential", "Keep"),
    "production/weights/production_v1.pth": ("Swin-Tiny Production Weights", "Essential", "Keep"),
    "production/weights/effnet_expert_v1.pth": ("EfficientNet Expert Weights", "Essential", "Keep"),
    "requirements.txt": ("Project Dependencies", "Essential", "Keep"),
    "README.md": ("Project Documentation", "Essential", "Keep"),
    "check_cuda.py": ("Environment Check Script", "Legacy", "Delete"),
    "inspect_swin.py": ("Model Inspection Script", "Legacy", "Delete"),
}

def analyze_file(filepath):
    rel_path = filepath.as_posix()
    
    # Check manual mapping first
    if rel_path in FILE_ROLES:
        return FILE_ROLES[rel_path]
    
    name = filepath.name
    
    # Heuristics
    if "production/" in rel_path:
        return ("Production Asset", "Essential", "Keep")
    if "archive_v1_prototype/" in rel_path:
        return ("Archived v1 Prototype Data", "Archive", "Keep")
    if "experiments/" in rel_path:
        return ("Experiment Log/Artifact", "Temporary", "Delete")
    if "docs/" in rel_path:
        return ("Project Documentation", "Essential", "Keep")
    if name.startswith("test_") or name.startswith("verify_"):
        return ("Verification Script", "Legacy", "Delete")
    if name.endswith(".csv") and "data/" in rel_path:
        return ("Dataset File", "Data", "Keep")
    
    return ("Unclassified File", "Unknown", "Review")

def scan_workspace():
    data = []
    
    for root, dirs, files in os.walk(ROOT_DIR):
        # Filter directories inplace
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in IGNORE_EXTS: continue
            
            rel_path = file_path.relative_to(ROOT_DIR).as_posix()
            
            description, status, action = analyze_file(file_path.relative_to(ROOT_DIR))
            
            data.append({
                "File Path": rel_path,
                "Description": description,
                "Status": status,
                "Action": action
            })
            
    df = pd.DataFrame(data)
    
    # Sort by Action (Delete first to highlight)
    df.sort_values(by=['Action', 'File Path'], inplace=True)
    
    # Generate Markdown Table
    md_table = df.to_markdown(index=False)
    
    # Save Artifact
    output_path = "docs/Project_File_Manifest.md"
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("# Project File Manifest & Cleanup Recommendation\n\n")
        f.write(md_table)
        
    print(f"Manifest generated at {output_path}")
    print(df[['File Path', 'Action']].to_markdown(index=False))

if __name__ == "__main__":
    scan_workspace()
