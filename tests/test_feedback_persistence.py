
import csv
import os
from pathlib import Path
from datetime import datetime

def simulate_feedback_submission():
    print("🧪 Simulating 'Submit Clinical Review' button click...")
    
    # Mock Data
    image_path = "data/raw/ddr/DR_grading/007-1234-000.jpg"
    prediction = 0
    correct_grade = 2
    confidence = 0.8850
    clinical_notes = "Simulated Logic Test: Microaneurysms present."
    
    feedback_csv = Path("data/feedback/doctor_corrections.csv")
    feedback_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Save feedback (Same logic as src/app.py)
    with open(feedback_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            image_path,
            prediction,
            correct_grade,
            f"{confidence:.4f}",
            clinical_notes
        ])
    
    print("✅ Feedback saved to CSV.")
    
    # Verify
    print("\n🔍 Verifying CSV Content:")
    with open(feedback_csv, 'r', encoding='utf-8') as f:
        rows = list(csv.reader(f))
        for row in rows[-5:]: # Show last 5
            print(row)
            
    # Assert
    last_row = rows[-1]
    assert last_row[1] == image_path
    assert int(last_row[3]) == correct_grade
    print("\n✅ Assertion Passed: Data persisted correctly.")

if __name__ == "__main__":
    simulate_feedback_submission()
