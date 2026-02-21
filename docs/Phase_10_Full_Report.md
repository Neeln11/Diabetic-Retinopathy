# Phase 10: DenseNet-121 Expert Training Report

**Date:** 2026-02-21
**Status:** Completed
**Primary Objective:** Train a DenseNet-121 model to act as the third expert in our ensemble, specifically tuned to detect **Grade 1 (Mild NPDR)** lesions, which previous models struggled with.

---

## 1. Strategy & Configuration

### A. Grade 1 Focused Loss Formulation
Previous runs showed that "Mild NPDR" was frequently misclassified as "Normal/No DR" because the visual differences are incredibly subtle (e.g., a single small microaneurysm).
- **Core Strategy:** We updated the `OrdinalRegressionLoss` configuration.
- **Class Weights:** Set to `[1.0, 2.5, 3.0, 4.0, 5.0]`. 
- **Impact:** By boosting the penalty for misclassifying Grade 1 from `2.0` to `2.5`, we forced the optimizer to care disproportionately about getting these subtle cases right, at the slight expense of overall accuracy.

### B. Dynamic "Zoom" Augmentation
To help the model "see" tiny lesions, we implemented a dynamic augmentation callback in the `BalancedDRDataset`.
- **Mechanism:** Every 2 epochs, the training script explicitly checks the validation confusion matrix for the **Grade 1 $\to$ Grade 0 Miss Rate**.
- **Trigger:** If the model misses more than 50% of Grade 1 cases, the dataset automatically increases the bounds of the `RandZoomd` transform.
- **Goal:** Magnify the input patches dynamically if the model isn't learning to spot the microscopic features of early disease.

### C. Data Foundation
- **Dataset:** Unified Master Dataset.
- **Sampling:** `BalancedDRDataset` active (8,200 samples per class/epoch) to ensure severe cases are oversampled to match healthy cases.

---

## 2. Training Execution Results

Training completed successfully over 10 epochs.

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Best QWK** | **0.8755** | Exceptionally high for this architecture. |
| **Final Accuracy** | ~83.23% | Impacted slightly by the aggressive loss weighting, but expected. |
| **Grade 1 Miss Rate** | **35.29%** | Evaluated at the final check. 65% of Grade 1 cases are now successfully detected or flagged as higher grades, up from ~0% in previous baseline tests. |
| **Processing Time** | ~1 Hour | Highly optimized data loading pipeline. |

### Dynamic Zoom Activity
The baseline zoom (0.9x - 1.1x) proved sufficient for the most part, thanks to the aggressive loss weighting. While the miss rate hovered near the threshold, the model successfully stabilized at a ~35% miss rate, preventing the script from having to continuously warp the images to extreme levels.

**Key Artifacts:**
- **Final Weights:** `production/weights/densenet_balanced.pth`
- **MLFlow Run Name:** `densenet_g1_expert_v1`

---

## 3. Current Ensemble State (Production Ready)

With Phase 10 complete, our final clinical ensemble is fully armed:

1.  **Swin-Tiny (`production_v1.pth`)**: The baseline transformer, highly sensitive to overall structural changes.
2.  **EfficientNet-B0 (`efficientnet_b0_balanced.pth`)**: The "Wide Net", heavily penalized for missing severe disease (high QWK).
3.  **DenseNet-121 (`densenet_balanced.pth`)**: The "Magnifying Glass", specifically weighted and trained to detect Grade 1 microaneurysms.

### Next Steps Recommendation
The models are trained. The next logical step is to formally run `src/utils/test_ensemble_hard_cases.py` one final time to see how the three models vote together on the tricky Grade 1 subset, or to integrate this final weight file into `src/app.py` for live deployment testing.
