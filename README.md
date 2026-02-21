#  DR-Swin-Tiny: Medical AI for Diabetic Retinopathy

![Project Status](https://img.shields.io/badge/Status-Pilot_Ready-success)
![Regulatory](https://img.shields.io/badge/Class-SaMD_IIa-blue)
![Sensitivity](https://img.shields.io/badge/Sensitivity-96.4%25-green)

A regulatory-grade Deep Learning system for detecting **Referable Diabetic Retinopathy (Stage 2+)** from fundus images. Built with **Swin Transformers**, optimized for consumer hardware (4GB VRAM), and refined via **Human-in-the-Loop (HITL)** feedback.

---

##  Key Features

### 1.  Lightweight Medical AI
- **Architecture**: Swin-Tiny (28M Params) with Ordinal Regression Head.
- **Performance**: 
    - **Sensitivity (Referable DR)**: **96.4%** (Safety Prioritized).
    - **Calibration (ECE)**: **0.068** (Reliable Confidence Scores).
- **Optimization**: Gradient Checkpointing & Mixed Precision (FP16).

### 2.  Clinical Safety & Interpretability
- **Safety Filter**: Automatically flags ambiguous cases (Prob 0.4-0.6) for Doctor Verification.
- **Explainability**: Grad-CAM Heatmaps overlay lesion detections.
- **Risk Engine**: Hybrid Risk Score combining AI Grade + HbA1c + Duration.

### 3.  Human-in-the-Loop (HITL)
- **Feedback Loop**: Integrated workflow for doctors to correct AI predictions.
- **Adaptive Learning**: Few-shot fine-tuning on discordance cases reduced critical misses from **25% to 3.6%**.

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

