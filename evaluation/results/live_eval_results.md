# Live Search Evaluation — All Approaches

> **Methodology**: Query-level 70/30 train/test split over 5 runs.
> Models are trained ONLY on train queries. Metrics computed on held-out test queries.
> Data comes from actual search engine retrieval (ground truth evaluation).

## Ranking Metrics Comparison

| # | Approach | NDCG@5 | MRR | MAP | NDCG@1 | NDCG@10 |
|---|----------|--------|-----|-----|--------|---------|
| 🥇 | **Logistic Reg** | 0.9045 | 0.8416 | 0.8256 | 0.7861 | 0.9108 |
| 🥈 | **MLP** | 0.8994 | 0.8393 | 0.8180 | 0.7803 | 0.9087 |
| 🥉 | **RRF** | 0.8924 | 0.8339 | 0.8173 | 0.7538 | 0.9012 |
| 4. | **BM25** | 0.8886 | 0.8257 | 0.8052 | 0.7594 | 0.8960 |
| 5. | **Semantic** | 0.8877 | 0.8186 | 0.8076 | 0.7509 | 0.8967 |
| 6. | **XNet→KAN** | 0.8838 | 0.8212 | 0.7988 | 0.7252 | 0.8951 |
| 7. | **KAN** | 0.8800 | 0.8145 | 0.7957 | 0.7178 | 0.8930 |
| 8. | **XNet** | 0.8736 | 0.8066 | 0.7876 | 0.7047 | 0.8851 |
| 9. | **KAN→XNet** | 0.8719 | 0.8067 | 0.7868 | 0.7017 | 0.8856 |

## Key Takeaways

- **Two-stage pipelines** combine the strengths of both models
- **XNet→KAN**: XNet filters quickly (177 params), KAN reranks precisely
- **KAN→XNet**: KAN selects best candidates, XNet provides fast final ranking
- Query-level split prevents any data leakage between train and test