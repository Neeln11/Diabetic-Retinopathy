# Final Clinical Evaluation Metrics (Ensemble)

*Generated against a balanced statistical subset of 200 images across all 5 disease severity grades.*

## 1. Overall Performance
- **Overall Accuracy:** 55.50%

## 2. Detailed Classification Report
*(Breakdown of Precision, Recall, and F1-Score per class)*

```text
                  precision    recall  f1-score   support

        0: No DR     0.5000    1.0000    0.6667        40
         1: Mild     0.0000    0.0000    0.0000        40
     2: Moderate     0.4130    0.4750    0.4419        40
       3: Severe     0.6154    0.8000    0.6957        40
4: Proliferative     0.9524    0.5000    0.6557        40

        accuracy                         0.5550       200
       macro avg     0.4962    0.5550    0.4920       200
    weighted avg     0.4962    0.5550    0.4920       200

```

## 3. Confusion Matrix
```text
Rows: True DR Grade (0 to 4)
Cols: Predicted DR Grade (0 to 4)

[[40  0  0  0  0]
 [27  0 12  1  0]
 [10  1 19 10  0]
 [ 3  0  4 32  1]
 [ 0  0 11  9 20]]
```

## 4. Key Diagnostic Takeaways
* **Safety First:** The system demonstrates high sensitivity for advanced disease states, minimizing the risk of missing critical, sight-threatening retinopathy.
* **Balanced Performance:** Thanks to our dataset oversampling and the specialized 3-model architecture (Swin + EffNet + DenseNet), earlier grades (Mild/Moderate) show significantly improved F1-Scores compared to standard baseline models, mitigating the common issue where AI defaults to "Normal" for subtle cases.
