
import pandas as pd
import os

def print_distribution(name, df, label_col):
    print(f"\n--- {name} ---")
    print(f"Total Images: {len(df)}")
    counts = df[label_col].value_counts().sort_index()
    percentages = df[label_col].value_counts(normalize=True).sort_index() * 100
    
    print(f"{'Grade':<6} | {'Count':<8} | {'%':<6}")
    print("-" * 26)
    for grade, count in counts.items():
        try:
            grade_int = int(grade)
            print(f"{grade_int:<6} | {count:<8} | {percentages[grade]:.1f}%")
        except:
            pass
            
    # Check for missing classes
    for i in range(5):
        if i not in counts.index:
            print(f"{i:<6} | {'0':<8} | 0.0%")

def main():
    # Paths
    ddr_path = r"data/raw/ddr/DR_grading.csv"
    messidor_adapt_path = r"data/processed/messidor_few_shot_train.csv"
    messidor_val_path = r"data/raw/messidor2/messidor_data.csv"
    
    # 1. DDR Training
    if os.path.exists(ddr_path):
        df_ddr = pd.read_csv(ddr_path)
        print_distribution("DDR Training Set (Original)", df_ddr, 'diagnosis')
    else:
        print(f"Warning: {ddr_path} not found.")

    # 2. Messidor Adaptation (Few-Shot)
    if os.path.exists(messidor_adapt_path):
        df_adapt = pd.read_csv(messidor_adapt_path)
        print_distribution("Messidor-2 Adaptation Set (Few-Shot)", df_adapt, 'label')
    else:
        print(f"Warning: {messidor_adapt_path} not found.")

    # 3. Messidor Validation
    if os.path.exists(messidor_val_path):
        df_val = pd.read_csv(messidor_val_path)
        # Filter for gradable if needed, usually adjudicated_gradable == 1
        if 'adjudicated_gradable' in df_val.columns:
            df_val = df_val[df_val['adjudicated_gradable'] == 1]
            print("(Filtered for 'adjudicated_gradable' == 1)")
            
        print_distribution("Messidor-2 Validation Set (Full)", df_val, 'adjudicated_dr_grade')
    else:
        print(f"Warning: {messidor_val_path} not found.")

if __name__ == "__main__":
    main()
