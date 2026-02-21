
import os
import ast
import pandas as pd
from pathlib import Path
import sys

# Configuration
ROOT_DIR = Path(".")
IGNORE_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.idea', '.vscode', 'wandb', 'mlruns', 'lightning_logs'}
IGNORE_EXTS = {'.pyc', '.gitkeep'}

def get_py_summary(filepath):
    """Extract class and function names from a python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
            
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        summary = []
        if classes:
            summary.append(f"Classes: {', '.join(classes)}")
        if functions:
            summary.append(f"Funcs: {', '.join(functions)}")
            
        return " | ".join(summary) if summary else "Script execution logic"
    except Exception as e:
        return "Using external libraries / Parsing Error"

def classify_file(filepath):
    path_str = filepath.as_posix()
    name = filepath.name
    
    # 1. Essential (Production) - Expanded
    if path_str in [
        "src/app.py", 
        "src/models/classifier.py", 
        "src/models/ensemble_factory.py",
        "src/data/balanced_loader.py",
        "src/models/segmentor.py",
        "src/utils/loss.py",
        "src/train_ensemble_aux.py",
        "src/utils/test_ensemble_hard_cases.py",
        "requirements.txt",
        "README.md",
        "src/calibration.py",
        "src/pipeline.py",
        "src/data/loader.py"
    ] or path_str.startswith("production/"):
        return "Essential (Production)", "Keep"

    if "explainability" in path_str or "preprocessing" in path_str:
        return "Essential (Feature)", "Keep"

    # 2. Key Documentation
    if path_str.startswith("docs/") and name.endswith(".md"):
        return "Documentation", "Keep"
        
    # 3. Data
    if "data/" in path_str:
        if name.endswith(".csv"):
            return "Dataset Index", "Keep"
        if "archive_v1_prototype" in path_str:
            return "Archived Data", "Archive"
        if "feedback" in path_str:
            return "HITL Data", "Keep"
        return "Raw Data Asset", "Keep"

    # 4. Developmental / Legacy
    if name == "vit_classifier.py":
        return "Legacy Model (ViT)", "Archive"
    if name.startswith("verify_") or name.startswith("check_") or name.startswith("inspect_"):
        return "Development Script", "Legacy/Delete"
    if "benchmark" in name.lower():
         return "Benchmark Script", "Legacy/Archive"
    if "audit_workspace.py" in name or "deep_audit.py" in name:
        return "Audit Tool", "Utility"
    
    # 5. Training & Utils
    if name.startswith("train_"):
        return "Training Script", "Keep"
    if name.startswith("validate_") or name.startswith("test_") or path_str.startswith("tests/"):
        return "Validation/Test Script", "Keep"
    if name.startswith("make_") or name.startswith("generate_"):
         return "Data Prep Script", "Keep"

    # 6. Artifacts
    if "experiments" in path_str:
        return "Experiment Artifact", "Delete (if not best)"
    if name.endswith(".log") or name.endswith(".tmp") or name.endswith(".db"):
        return "Temporary/DB File", "Review"
    if name.endswith(".bat"):
        return "Shell Script", "Keep"
        
    return "Unclassified", "Review"

def generate_manifest():
    print(f"🔍 Starting Deep Audit of {ROOT_DIR.absolute()}...")
    data = []
    
    for root, dirs, files in os.walk(ROOT_DIR):
        # Filter directories inplace
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in IGNORE_EXTS: continue
            
            rel_path = file_path.relative_to(ROOT_DIR)
            path_str = rel_path.as_posix()
            
            # Content Summary
            content_info = ""
            if file_path.suffix == ".py":
                content_info = get_py_summary(file_path)
            elif file_path.suffix in [".csv", ".json", ".pth"]:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                content_info = f"Data File ({size_mb:.2f} MB)"
            elif file_path.suffix in [".png", ".jpg", ".jpeg", ".tiff"]:
                # SKIP INDIVIDUAL IMAGES
                continue 
            else:
                content_info = "Text/Config File"
                
            # Classification
            status, action = classify_file(rel_path)
            
            data.append({
                "File Path": path_str,
                "Key Functions/Contents": content_info,
                "Status": status,
                "Recommendation": action
            })
            
    # Add summary rows for image directories (Hardcoded or dynamic? Let's just skip them as "Raw Data Asset" covers the directory usually)
    # The user asked for "Recursive directory listing", but listing 3000 images in a table is not useful.
    # I will add a manual entry for the image folders if traversing them.
    # actually, let's just skip adding them to 'data' list.
    
    # Check for specific huge directories and add a summary row
    image_dirs = ["data/raw/aptos2019/test_images", "data/raw/aptos2019/train_images", "data/raw/messidor2"]
    for img_dir_str in image_dirs:
        dpath = ROOT_DIR / img_dir_str
        if dpath.exists():
            count = len(list(dpath.glob("*")))
            data.append({
                "File Path": f"{img_dir_str}/ [Content: {count} images]",
                "Key Functions/Contents": "Raw Image Dataset",
                "Status": "Raw Data Asset",
                "Recommendation": "Keep"
            })
            
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort: Essential first, then by path
    df['SortKey'] = df['Status'].apply(lambda x: 0 if "Essential" in x else (1 if "Documentation" in x else 2))
    df.sort_values(by=['SortKey', 'File Path'], inplace=True)
    df.drop(columns=['SortKey'], inplace=True)
    
    # Custom Markdown Table Generation (No tabulate dependency)
    def to_md(dataframe):
        headers = dataframe.columns.tolist()
        # header row
        md = "| " + " | ".join(headers) + " |\n"
        # separator row
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        # data rows
        for _, row in dataframe.iterrows():
            md += "| " + " | ".join(str(x) for x in row) + " |\n"
        return md

    md_table = to_md(df)
    
    # Output
    output_path = "docs/Comprehensive_Project_Manifest.md"
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("# 📂 Comprehensive Project Manifest\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        f.write(md_table)
        
    print(f"\n✅ Manifest generated at: {output_path}")
    
    # Print preview of 'Recommendation: Delete'
    print("\n🗑️  Files Recommended for Deletion/Archival:")
    print(to_md(df[df['Recommendation'].str.contains("Delete|Archive")][['File Path', 'Status', 'Recommendation']]))

if __name__ == "__main__":
    generate_manifest()
