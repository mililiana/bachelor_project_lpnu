# Baseline vs Approaches — Full Comparison

## Ranking Quality Metrics (Gold Standard)

| Approach | Category | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | MRR | MAP |
|----------|----------|--------|--------|--------|---------|-----|-----|
| BM25 (baseline) | Retrieval Baseline | 0.7594 | 0.8562 | 0.8886 | 0.8960 | 0.8257 | 0.8052 |
| Semantic (baseline) | Retrieval Baseline | 0.7509 | 0.8473 | 0.8877 | 0.8967 | 0.8186 | 0.8076 |
| RRF (baseline) | Retrieval Baseline | 0.7538 | 0.8673 | 0.8924 | 0.9012 | 0.8339 | 0.8173 |
| Logistic Regression **🏆** | Simple Baseline | 0.7861 | 0.8774 | 0.9045 | 0.9108 | 0.8416 | 0.8256 |
| KAN | Learned Reranker | 0.7890 | 0.8696 | 0.9010 | 0.9114 | 0.8515 | 0.8190 |
| XNet | Learned Reranker | 0.7284 | 0.8585 | 0.8837 | 0.8947 | 0.8225 | 0.8022 |
| MLP | Learned Reranker | 0.7723 | 0.8630 | 0.8953 | 0.9039 | 0.8297 | 0.8083 |

## Classification Metrics

| Approach | AUC | Precision | Recall | F1 |
|----------|-----|-----------|--------|-----|
| BM25 (baseline) | 0.6626 | 0.6454 | 0.6873 | 0.5978 |
| Semantic (baseline) | 0.6681 | 0.5428 | 0.7558 | 0.5911 |
| RRF (baseline) | 0.6843 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.7394 | 0.6999 | 0.5647 | 0.6189 |
| KAN | 0.7374 | 0.6941 | 0.6174 | 0.6447 |
| XNet | 0.7127 | 0.6605 | 0.5627 | 0.6012 |
| MLP | 0.7228 | 0.6843 | 0.5577 | 0.6057 |

## Efficiency & Interpretability

| Approach | Params | Inference (ms) | Interpretable | Explainability |
|----------|--------|----------------|---------------|----------------|
| BM25 (baseline) | 0 | — | ❌ | Black box |
| Semantic (baseline) | 0 | — | ❌ | Black box |
| RRF (baseline) | 0 | — | ❌ | Black box |
| Logistic Regression | 7 | — | ✅ Coefficients | Linear weights |
| KAN | 560 | 0.189 | ✅ Full | Formula + Feature Importance |
| XNet | 177 | 0.025 | ✅ Partial | Cauchy parameters |
| MLP | 257 | 0.024 | ❌ | Black box |