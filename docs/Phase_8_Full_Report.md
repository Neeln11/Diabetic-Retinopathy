# Phase 8: Balanced Ensemble Execution Report

**Date:** 2026-02-20
**Status:** Completed
**Primary Objective:** Train an expert EfficientNet-B0 model using a class-balanced strategy to improve sensitivity on rare DR grades (Stages 1, 3, 4).

---

## 1. Strategy & Configuration

### A. Data Balancing
To address the severe class imbalance in the DDR/APTOS datasets (where Grade 0 dominates), we implemented a `BalancedDRDataset` wrapper.
- **Method:** Oversampling minority classes to achieve a uniform distribution.
- **Target:** **8,200 images per class** per epoch.
- **Total Epoch Size:** 41,000 images.
- **Impact:** Ensures the model sees an equal number of Severe/Proliferative cases as No DR cases, preventing bias towards the majority class.

### B. Loss Function: Weighted Ordinal Regression
We replaced standard Cross-Entropy with a custom `OrdinalRegressionLoss` to respect the progressive nature of Diabetic Retinopathy (0 -> 1 -> 2 -> 3 -> 4).
- **Class Weights:** **1 : 2 : 3 : 4 : 5** (Grade 0 : Grade 1 : ... : Grade 4)
- **Rationale:** Misses on higher grades (e.g., predicting Grade 2 for a Grade 4 patient) are penalized 5x more heavily than misses on Grade 0. This forces the model to prioritize sensitivity for severe disease.

### C. Model Architecture
- **Backbone:** EfficientNet-B0 (Pretrained on ImageNet)
- **Head:** Linear Layer (1280 -> 5)
- **Optimizer:** AdamW (LR: 1e-4, Weight Decay: 1e-4)

---

## 2. Training Execution Results

The training was conducted over 10 epochs.

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Best QWK** | **0.8726** | Achieved at Epoch 7 |
| **Final QWK** | 0.8655 | Epoch 10 (Slight overfitting or stochastic variance) |
| **VRAM Usage** | 0.13 GB | Extremely efficient (Batch Size 16) |
| **Training Time** | ~10 Hours | 10 full epochs on Balanced Dataset |

**Key Artifacts:**
- **Weights:** `production/weights/efficientnet_b0_balanced.pth`
- **MLFlow Run ID:** `652342233580417082b47e23f5cd2815`

---

## 3. Ensemble Integration Test

We integrated the new EfficientNet-B0 into the 3-model ensemble (Swin + EfficientNet + DenseNet).

### Verification Steps
1.  **Pipeline Functionality:** ✅ Success. The `MetaPredictor` correctly instantiates all three models and performs VRAM-safe sequential inference.
2.  **Edge Case Testing:**
    - **Target:** 5 randomly selected **Grade 1 (Mild NPDR)** cases, which are notoriously difficult to distinguish from Grade 0.
    - **Result:** **0% Accuracy** (0/5 Correct).
    - **Analysis:**
        - **Swin-Tiny:** Predicted Grade 0 (No DR) for most cases.
        - **EfficientNet-B0 (New):** Predicted Grade 2 (Moderate) or Grade 0.
        - **DenseNet-121:** Running on fallback APTOS weights (likely biased).
    - **Conclusion:** While the model excels at distinguishing severe disease (high QWK), Grade 1 remains a significant challenge. The ensemble tends to either under-call (0) or over-call (2) these subtle cases.

---

## 4. Next Steps (Phase 10)

1.  **Fine-Tune DenseNet:** Complete the training of the 3rd ensemble member (`densenet121_balanced.pth`) to replace the fallback weights.
2.  **Calibration:** Apply temperature scaling to the ensemble output to better calibrate probabilities for Grade 1.
3.  **Threshold Tuning:** Adjust the decision thresholds for Grade 1 specifically (e.g., lower the probability threshold required to predict Grade 1) to boost sensitivity at the cost of specificity.
