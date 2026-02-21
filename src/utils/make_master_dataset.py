
import pandas as pd
import os
from pathlib import Path

# Config
DATA_ROOT = Path("data/raw")
APTOS_DIR = DATA_ROOT / "aptos2019"
DDR_DIR = DATA_ROOT / "ddr"
IDRID_DIR = DATA_ROOT / "idrid"

OUTPUT_CSV = Path("src/data/unified_master.csv")

def process_aptos():
    print("Processing APTOS 2019...")
    df = pd.read_csv(APTOS_DIR / "train.csv")
    valid_rows = []
    
    for _, row in df.iterrows():
        img_name = row['id_code']
        label = row['diagnosis']
        img_path = APTOS_DIR / "train_images" / f"{img_name}.png"
        
        if img_path.exists():
            valid_rows.append({
                'image_path': str(img_path),
                'dr_grade': int(label),
                'source': 'aptos'
            })
            
    return pd.DataFrame(valid_rows)

def process_ddr():
    print("Processing DDR...")
    df = pd.read_csv(DDR_DIR / "DR_grading.csv")
    valid_rows = []
    
    # DDR layout found: data/raw/ddr/DR_grading/DR_grading/
    img_dir = DDR_DIR / "DR_grading" / "DR_grading"
    
    count = 0
    for _, row in df.iterrows():
        img_filename = row['id_code'] # e.g., 20170413102628830.jpg or 007-0004-000.jpg
        label = row['diagnosis']
        
        img_path = img_dir / img_filename
                
        if img_path.exists():
             valid_rows.append({
                'image_path': str(img_path),
                'dr_grade': int(label),
                'source': 'ddr'
            })
             count += 1
    
    print(f"Found {count} DDR images.")
    return pd.DataFrame(valid_rows)

def process_idrid():
    print("Processing IDRiD...")
    df = pd.read_csv(IDRID_DIR / "idrid_labels.csv")
    valid_rows = []
    
    # IDRiD layout found: data/raw/idrid/Imagenes/Imagenes/
    img_dir = IDRID_DIR / "Imagenes" / "Imagenes"
    
    count = 0
    for _, row in df.iterrows():
        img_name = row['id_code'] # IDRiD_001
        label = row['diagnosis']
        
        # IDRiD images often end in .jpg
        img_path = img_dir / f"{img_name}.jpg"
        
        if img_path.exists():
            valid_rows.append({
                'image_path': str(img_path),
                'dr_grade': int(label),
                'source': 'idrid'
            })
            count += 1
            
    print(f"Found {count} IDRiD images.")
    return pd.DataFrame(valid_rows)

def main():
    dfs = []
    
    # APTOS
    try:
        dfs.append(process_aptos())
    except Exception as e:
        print(f"Failed to process APTOS: {e}")

    # DDR
    try:
        dfs.append(process_ddr())
    except Exception as e:
        print(f"Failed to process DDR: {e}")
        
    # IDRiD
    try:
        dfs.append(process_idrid())
    except Exception as e:
        print(f"Failed to process IDRiD: {e}")
        
    if not dfs:
        print("No data found!")
        return

    master_df = pd.concat(dfs, ignore_index=True)
    
    # Shuffle
    master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    master_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Created unified master dataset: {OUTPUT_CSV}")
    print(f"Total Images: {len(master_df)}")
    print("\nBreakdown by Source:")
    print(master_df['source'].value_counts())
    print("\nBreakdown by Grade:")
    print(master_df['dr_grade'].value_counts().sort_index())
    print("\nFirst 5 rows:")
    print(master_df.head())

if __name__ == "__main__":
    main()
