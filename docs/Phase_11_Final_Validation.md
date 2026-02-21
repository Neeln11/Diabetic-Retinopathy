# Phase 11: Final Clinical Validation of the 3-Model Council

## 1. Metrics Comparison (Grade 0 vs 1 Hard Cases)

| Metric | Single Swin | 3-Model Ensemble (Calibrated) |
| :--- | :--- | :--- |
| **QWK (on this subset)** | 0.3904 | 0.4167 |
| **Grade 1 Recall (Sensitivity)**| 2.00% | 8.00% |
| **Expected Calibration Error (ECE)**| 0.1585 | 0.1485 |

## 2. Grey-Zone Logic ('Ambiguous')
Cases flagged as Ambiguous (40-60% confidence):
- Total Flagged: 40 out of 100
- These cases would be routed to human doctors for manual review under clinical protocols.

## 3. Detailed Predictions Sample

```text
                Image  True Label  Swin Pred  Ensemble Pred  Ambiguous
20170411115553873.jpg           0          0              0      False
     007-2549-100.jpg           1          2              2       True
     007-3237-100.jpg           1          2              0      False
     007-2785-100.jpg           1          2              0       True
     007-2704-100.jpg           1          2              0       True
        IDRiD_282.jpg           1          0              0      False
     007-2265-100.jpg           0          2              0       True
20170408222110964.jpg           0          0              0      False
     007-0221-000.jpg           0          0              0      False
20170429223942395.jpg           0          0              0      False
20170324103347095.jpg           0          0              0      False
     007-2324-100.jpg           1          2              0       True
20170429210854778.jpg           0          0              0      False
     c96f743915b5.png           1          2              2       True
20170427182001923.jpg           0          0              0      False
     007-2880-100.jpg           1          2              0       True
     007-7959-601.jpg           0          0              0       True
     007-3023-100.jpg           1          2              1      False
20170221141931314.jpg           0          0              0       True
     007-2633-100.jpg           1          2              2       True
```
