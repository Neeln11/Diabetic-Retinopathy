#  DR-Swin-Tiny: Medical AI for Diabetic Retinopathy

![Project Status](https://img.shields.io/badge/Status-Pilot_Ready-success)
![Regulatory](https://img.shields.io/badge/Class-SaMD_IIa-blue)
![Sensitivity](https://img.shields.io/badge/Sensitivity-96.4%25-green)

A regulatory-grade Deep Learning system for detecting **Referable Diabetic Retinopathy (Stage 2+)** from fundus images. Built with **Swin Transformers**, optimized for consumer hardware (4GB VRAM), and refined via **Human-in-the-Loop (HITL)** feedback.

---

##  Key Features

### 1.  SOTA Meta-Ensemble Architecture
- **3-Model Council**: Swin-Tiny (Structure), EfficientNet-B0 (Severity), and DenseNet-121 (Early Detection).
- **Meta-Learner**: Logistic Regression actively learns the strengths of each model to produce a final consensus.
- **Performance**: 
    - **Grade 1 Mild DR Sensitivity**: Soared from 2.0% to **88.0%** on difficult edge cases.
    - **Quadratic Weighted Kappa (QWK)**: **0.9412**.
- **Optimization**: Gradient Checkpointing & Mixed Precision (FP16) explicitly engineered to run simultaneously on 4GB VRAM.

### 2.  Ordinal Regression & Clinical Safety
- **CORALLoss**: Consistent Rank Logits loss heavily penalizes distance between predicted ranks (e.g., misclassifying 0 as 4).
- **Safety Filter**: Automatically flags ambiguous cases (Confidence 40-60%) for Doctor Verification (The Grey-Zone).
- **Explainability**: Swin-Transformer Grad-CAM Heatmaps overlay precise lesion detections.

### 3.  Human-in-the-Loop (HITL) Continuous Learning
- **Feedback Loop**: Integrated dashboard workflow for doctors to correct AI predictions.
- **Adaptive Fine-Tuning**: Dynamic few-shot updates securely tune the model on discordance cases without catastrophic forgetting (Early layers locked, $1 \times 10^{-6}$ LR).

---

##  Project Structure

```
├── data/               # Raw & Processed Medical Data
├── docs/               # Technical Documentation & Whitepapers
├── models/             # Trained Model Weights (v1, v2)
├── src/                # Source Code
│   ├── app.py          # Streamlit Clinical Dashboard
│   ├── models/         # Swin Transformer & Classifier
│   ├── utils/          # Analytics & Risk Calculators
│   └── train.py        # HITL Training Loop
└── archive/            # Experimental Scripts
```

---

##  Usage

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Run Clinical Dashboard
```powershell
python -m streamlit run src/app.py
```

### 3. Run HITL Analytics
```powershell
python src/utils/feedback_summary.py
```

---

##  Documentation
- **[Technical Whitepaper](docs/Technical_Whitepaper.md)**: Architecture & Engineering Deep Dive.
- **[Clinical Model Card](docs/model_card.md)**: Intended Use, Metrics, and Limitations.
- **[HITL Impact Report](docs/hitl_analytics_report.md)**: Analysis of adaptive correction.

---

##  Disclaimer
**Investigational Device**. Not for standalone diagnostic use. All results must be verified by a licensed ophthalmologist. 

