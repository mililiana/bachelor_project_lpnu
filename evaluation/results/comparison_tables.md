# Model Comparison Tables

## 1. Accuracy

|      |    AUC |   Precision |   Recall |     F1 |
|:-----|-------:|------------:|---------:|-------:|
| KAN  | 0.8021 |         0.5 |   0.3529 | 0.4138 |
| XNet | 0.7485 |         1   |   0.0588 | 0.1111 |
| MLP  | 0.7372 |         0   |   0      | 0      |

## 2. Efficiency

|      |   Parameters |   Inference (ms) |   Throughput (k/s) |
|:-----|-------------:|-----------------:|-------------------:|
| KAN  |          560 |            2.503 |              399.6 |
| XNet |          177 |            0.047 |            21061.5 |
| MLP  |          257 |            0.049 |            20400.4 |

## 3. Interpretability

|      | Interpretable   | Formula Discovery   | Spline Visualization   | Feature Importance   |
|:-----|:----------------|:--------------------|:-----------------------|:---------------------|
| KAN  | ✓               | ✓                   | ✓                      | ✓                    |
| XNet | Partial         | ✗                   | ✗                      | ✗                    |
| MLP  | ✗               | ✗                   | ✗                      | ✗                    |

## 4. Use Cases

|      | Best For                | Use Case                | Recommendation     |
|:-----|:------------------------|:------------------------|:-------------------|
| KAN  | Research, Paper Figures | Explainable reranking   | Offline analysis   |
| XNet | Fast aggregation        | Signal fusion           | Real-time pipeline |
| MLP  | Baseline comparison     | Standard neural ranking | Reference only     |
