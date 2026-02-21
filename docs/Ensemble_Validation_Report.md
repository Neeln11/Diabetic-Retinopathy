# Ensemble Validation Report

Evaluation of 3-Model Ensemble on known challenging 'Stage 1' cases.

| Image | True Label | Swin (0.4) | EffNet (0.3) | DenseNet (0.3) | Ensemble | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 20060410_40481_0200_PP.png | 1 | 2 | 2 | 2 | 2 | Missed |
| 20051020_63829_0100_PP.png | 1 | 2 | 0 | 1 | 2 | Missed |
| IM000880.jpg | 1 | 3 | 1 | 4 | 3 | Missed |
| 20060412_58945_0200_PP.png | 1 | 2 | 3 | 2 | 2 | Missed |
| 20060522_45069_0100_PP.png | 1 | 2 | 1 | 3 | 3 | Missed |
| 20060522_45935_0100_PP.png | 1 | 0 | 1 | 4 | 0 | Missed |
| 20060411_58971_0200_PP.png | 1 | 2 | 1 | 0 | 2 | Missed |
| 20051116_44816_0400_PP.png | 1 | 2 | 1 | 2 | 2 | Missed |
| 20051116_43954_0400_PP.png | 1 | 3 | 1 | 1 | 3 | Missed |
| 20051202_37199_0400_PP.png | 1 | 2 | 2 | 2 | 2 | Missed |

**Summary**: The Ensemble corrected 0 out of 10 complex cases.

## 🚨 Failure Analysis
The current ensemble strategy failed to improve performance on Stage 1 cases. 
- **Root Cause**: EfficientNet-B0 and DenseNet-121 were loaded with **ImageNet weights** without any fine-tuning on Diabetic Retinopathy data.
- **Observation**: The auxiliary models are essentially outputting noise or generic feature patterns that do not align with microaneurysm detection.
- **Recommendation**: To make the ensemble effective, we **MUST fine-tune** the EfficientNet and DenseNet backbones on the DR dataset (Phase 1 or Phase 2 data) before adding them to the voting pool. Currently, they are diluting the expert Swin model's opinion with random noise.
