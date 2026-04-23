
import pandas as pd
import numpy as np
import torch
import json
import os
import sys
from sklearn.model_selection import KFold
from typing import List, Dict

# Add src to path
sys.path.append(os.getcwd())

from src.research.kan_model import KAN
from src.research.xnet_model import XNet
from src.research.ranking_metrics import compute_ranking_metrics, aggregate_metrics
import torch.nn as nn

# Constants
DATA_PATH = "evaluation/training_data_llm_labeled.csv" if os.path.exists("evaluation/training_data_llm_labeled.csv") else "evaluation/training_data_balanced.csv"
GROUND_TRUTH_PATH = "evaluation/ground_truth/ground_truth_augmented_clean.json"
DOCS_PATH = "data/chunked_documents_128.json"
SEED = 42
FEATURE_COLUMNS = ["semantic_score", "bm25_score", "title_overlap", "category_match", "chunk_position", "doc_length"]
RANKING_KS = [1, 3, 5, 10]

def train_model(model, X, y, epochs=100, lr=0.01):
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        if out.dim() == 1: out = out.unsqueeze(-1)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples, {df['query_id'].nunique()} unique queries.")
    
    print(f"Loading ground truth from {GROUND_TRUTH_PATH}...")
    with open(GROUND_TRUTH_PATH, "r") as f:
        raw_data = json.load(f)
        
    # Expand ground truth to match training data generation logic
    ground_truth = []
    for item in raw_data:
        if "original_question" in item:
            ground_truth.append({
                "question": item["original_question"],
                "ground_truth_answer": item["original_answer"],
                "doc_id": item.get("source_doc_id", "")
            })
            if "generated_question" in item and item["generated_question"]:
                ground_truth.append({
                    "question": item["generated_question"],
                    "ground_truth_answer": item["generated_answer"],
                    "doc_id": item.get("source_doc_id", "")
                })
        else:
            ground_truth.append(item)
            
    print(f"Expanded ground truth to {len(ground_truth)} items (Original + Augmented).")
        
    print(f"Loading documents from {DOCS_PATH}...")
    with open(DOCS_PATH, "r") as f:
        documents = json.load(f)
        
    return df, ground_truth, documents

def eval_fold_raw(model, df, qids, mtype):
    res = []
    for qid in qids:
        qdata = df[df["query_id"] == qid].copy()
        if len(qdata) < 2: continue
        # Use 'label' (binary) which now contains high-quality LLM judgments
        rel = qdata["label"].values
        
        # Skip queries with no relevant documents (according to LLM)
        if rel.sum() == 0:
            continue
        
        if mtype == "bm25": sc = qdata["bm25_score"].values
        elif mtype == "semantic": sc = qdata["semantic_score"].values
        elif mtype == "simplesum":
             b = qdata["bm25_score"]
             bn = (b - b.min()) / (b.max() - b.min() + 1e-9)
             sc = bn + qdata["semantic_score"]
        else:
             Xq = torch.FloatTensor(qdata[FEATURE_COLUMNS].values.astype(np.float32))
             with torch.no_grad(): sc = model(Xq).squeeze().numpy()
        
        idx = np.argsort(-sc)
        ord_rel = rel[idx].tolist()
        
        metrics = compute_ranking_metrics(ord_rel, RANKING_KS)
        # Add basic info for detailed analysis
        metrics["query_id"] = qid
        
        # Store for failure analysis later
        # Find the rank of the first relevant doc
        rank = -1
        for r_idx, r_val in enumerate(ord_rel):
            if r_val > 0: # Assuming rel > 0 is relevant
                rank = r_idx + 1
                break
        metrics["rank"] = rank if rank != -1 else 100 # punish not finding
        
        res.append(metrics)
    return res

# Tuned KAN: Tuning showed simple model is better (grid=5)
# EnsembleModel removed as it didn't improve performance

def run_evaluation():
    df_full, ground_truth, documents = load_data()
    query_ids_full = df_full["query_id"].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_results_flat = []

    print("\nStarting 5-Fold Cross-Validation...")
    fold = 0
    for train_idx, test_idx in kf.split(query_ids_full):
        fold += 1
        print(f"  Processing Fold {fold}/5...")
        
        train_qids = query_ids_full[train_idx]
        test_qids = query_ids_full[test_idx]
        
        train_df = df_full[df_full["query_id"].isin(train_qids)]
        test_df = df_full[df_full["query_id"].isin(test_qids)]
        
        X_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
        y_train = train_df["label"].values.astype(np.float32)

        # Train Models
        print(f"    Training KAN (Fold {fold})...")
        kan_cv = KAN(layers_hidden=[6, 8, 1])
        kan_cv = train_model(kan_cv, X_train, y_train, epochs=100, lr=0.01) 
        
        print(f"    Training XNet (Fold {fold})...")
        xnet_cv = XNet(input_dim=6)
        xnet_cv = train_model(xnet_cv, X_train, y_train, epochs=100, lr=0.01)
        
        # Evaluate
        # Note: We use "Model" (Cap) to match notebook logic, but failure analysis needs to check correct key
        cv_results_flat.extend([{"Model": "BM25", **m} for m in eval_fold_raw(None, test_df, test_qids, "bm25")])
        cv_results_flat.extend([{"Model": "Semantic", **m} for m in eval_fold_raw(None, test_df, test_qids, "semantic")])
        cv_results_flat.extend([{"Model": "BM25+Semantic", **m} for m in eval_fold_raw(None, test_df, test_qids, "simplesum")])
        cv_results_flat.extend([{"Model": "XNet", **m} for m in eval_fold_raw(xnet_cv, test_df, test_qids, "pytorch")])
        cv_results_flat.extend([{"Model": "KAN", **m} for m in eval_fold_raw(kan_cv, test_df, test_qids, "pytorch")])
        
    print("\n✅ Cross-Validation Complete!")
    
    # --- Aggregation ---
    print("\n=== Aggregate Results ===")
    df_results = pd.DataFrame(cv_results_flat)
    agg_results = df_results.groupby("Model")[["ndcg@5", "ndcg@10", "mrr", "map", "err@5", "err@10"]].mean().sort_values("ndcg@5", ascending=False)
    print(agg_results)
    
    # --- Detailed Failure Analysis ---
    print("\n=== Detailed Failure Analysis (Top 5 KAN Failures) ===")
    
    # 1. Collect all KAN results with Rank > 1
    failures = [res for res in cv_results_flat if res["Model"] == "KAN" and res["rank"] > 1]
    
    # 2. Sort by Rank (descending) to find worst failures
    worst_failures = sorted(failures, key=lambda x: x["rank"], reverse=True)[:5]
    
    if not worst_failures:
        print("🎉 No failures found! Perfect ranking.")
    else:
        for i, failure in enumerate(worst_failures):
            qid = failure["query_id"]
            rank = failure["rank"]
            
            # Get Query Text
            try:
                query_idx = int(qid)
                if 0 <= query_idx < len(ground_truth):
                    q_text = ground_truth[query_idx].get("question", "Unknown")
                    gt_answer = ground_truth[query_idx].get("ground_truth_answer", "Unknown")
                else:
                    q_text = "Index out of range"
                    gt_answer = "N/A"
            except ValueError:
                q_text = f"Invalid ID {qid}"
                gt_answer = "N/A"
                
            print(f"\n[{i+1}] Query ID: {qid} | Rank: {rank}")
            print(f"    Question: \"{q_text}\"")
            print(f"    GT Answer: \"{gt_answer[:100]}...\"")
            
            # TODO: In a real script, we would need to look up the top-1 doc vs the relevant doc
            # to see WHY it failed. But we don't have the scores here easily without re-running inference.
            # For now, just identifying the query is a huge step.

if __name__ == "__main__":
    run_evaluation()
