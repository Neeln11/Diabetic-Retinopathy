# Updated Streamlit UI Layout: Council Consensus & Safety Alerts

The `src/app.py` Streamlit dashboard has been updated to fully integrate the 3-Model Clinical Ensemble.

## 1. 🤖 AI Council Consensus Section
A new dedicated section appears just below the original fundus image after processing. It displays a "Board of Experts" voting table using Streamlit metrics:

| Swin-Tiny (Structure) | EfficientNet (Severity) | DenseNet (Early DR) | 🏆 Final Decision |
| :--- | :--- | :--- | :--- |
| Grade X (Weight: 0.4) | Grade Y (Weight: 0.3) | Grade Z (Weight: 0.3) | **Grade F** |

- This provides immediate transparency into how the three individual expert models voted before the weighted average is calculated.

## 2. ⚠️ Safety Alert Integration
If the `Ensemble_Confidence` falls into the calibrated "Grey-Zone" (between 40% and 60%), a prominent alert is triggered:

> ⚠️ **AMBIGUOUS CASE: Manual Clinical Review Required**
> 
> The Ensemble Confidence is within the Grey-Zone (XX.X%). This image may contain atypical features or be positioned between typical DR grades.

## 3. Explainability
The Grad-CAM Heatmap generation continues to use the **Swin-Tiny** model. This is critical because Swin-Tiny acts as our "Structural Expert," making its heatmaps the most reliable for pointing out geometric lesions (microaneurysms, hemorrhages) to the clinician. The backend `src/models/ensemble_factory.py` manages loading and unloading models to ensure Swin is available for this explainability step without OOM errors.
