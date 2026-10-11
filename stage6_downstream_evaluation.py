"""
Stage 6 — Downstream Application & Representation Evaluation (Fixed)
===================================================================================
1. Downstream Classification Benchmark: probes (RF, SVM, GB, MLP) are now
   TRAINED on the train split and EVALUATED on the test split — the
   previous version ran cross-validation on the WHOLE dataset, which is a
   softer problem than a real held-out evaluation, especially since the
   encoder itself was trained using these same cluster labels.
2. Cross-Modal Similarity Retrieval Benchmark: now computed with TEST-split
   patches as queries, searched only against OTHER test-split patches —
   previously this searched across the entire dataset the encoder was
   trained on, which inflates retrieval accuracy for a SupCon-trained
   encoder almost by construction (SupCon's loss literally optimizes
   same-cluster cosine similarity on the training set).
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import warnings
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

EMBEDDINGS_CSV = "./patch_embeddings_stage5.csv"
OUTPUT_REPORT = "./stage6_evaluation_results.txt"
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------

def load_data():
    if not os.path.exists(EMBEDDINGS_CSV):
        raise FileNotFoundError(f"❌ Could not find {EMBEDDINGS_CSV}. Ensure Stage 5 ran successfully.")

    df = pd.read_csv(EMBEDDINGS_CSV)

    if "split" not in df.columns:
        raise ValueError(
            "❌ No `split` column in patch_embeddings_stage5.csv. "
            "Run utils/data_split.py + regenerate Stage 5 output first — "
            "evaluating without a held-out split invalidates these metrics."
        )

    feature_cols = [c for c in df.columns if c.startswith("fused_emb_")]
    if not feature_cols:
        raise ValueError("❌ No fused_emb_* columns found in embeddings CSV.")

    label_col = "cluster"  # written directly by the corrected Stage 5 script

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, label_col].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, label_col].values

    return X_train, y_train, X_test, y_test, feature_cols


# ----------------------------------------------------------------------
# TASK 1: DOWNSTREAM CLASSIFICATION BENCHMARK (train/test, not CV-on-all)
# ----------------------------------------------------------------------

def run_classification_benchmark(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print(" TASK 1: DOWNSTREAM LAND-COVER CLASSIFICATION PROBING")
    print(" (trained on TRAIN split, evaluated on held-out TEST split)")
    print("=" * 60)

    classifiers = {
        "Linear SVM (C=1.0)": SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE),
        "RBF SVM (C=1.0)": SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE),
        "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
        "MLP Probe (64->32)": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=RANDOM_STATE),
    }

    results = []
    raw_metrics = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds) * 100
        f1 = f1_score(y_test, preds, average="macro", zero_division=0) * 100
        prec = precision_score(y_test, preds, average="macro", zero_division=0) * 100
        rec = recall_score(y_test, preds, average="macro", zero_division=0) * 100

        results.append([name, f"{acc:.2f}%", f"{f1:.2f}%", f"{prec:.2f}%", f"{rec:.2f}%"])
        raw_metrics[name] = {"accuracy": acc, "macro_f1": f1, "macro_precision": prec, "macro_recall": rec}

    headers = ["Classifier", "Test Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]
    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))

    # Honest caveat print — small test sets make single-split numbers noisy.
    print(f"\nNote: test split has {len(y_test)} patches. With this few samples, "
          f"treat these numbers as indicative, not precise — consider repeating "
          f"with several random seeds for the split and reporting mean ± std "
          f"if you want a more defensible thesis claim.")

    return results, headers, raw_metrics


# ----------------------------------------------------------------------
# TASK 2: LATENT RETRIEVAL BENCHMARK — TEST-SPLIT ONLY, BOTH SIDES
# ----------------------------------------------------------------------

def run_retrieval_benchmark(X_test, y_test):
    print("\n" + "=" * 60)
    print(" TASK 2: LATENT EMBEDDING RETRIEVAL & SEMANTIC CONSISTENCY")
    print(" (query and gallery both restricted to held-out TEST split)")
    print("=" * 60)

    n_samples = len(y_test)
    if n_samples < 3:
        print(f"⚠️ Only {n_samples} test patches — too few for a meaningful retrieval benchmark. Skipping.")
        return [], {}

    sim_matrix = cosine_similarity(X_test, X_test)
    np.fill_diagonal(sim_matrix, -np.inf)

    top1_correct = top3_correct = top5_correct = 0
    mrr_total = 0.0

    for i in range(n_samples):
        query_label = y_test[i]
        ranked_indices = np.argsort(sim_matrix[i])[::-1]

        top1_label = y_test[ranked_indices[0]]
        top3_labels = y_test[ranked_indices[:3]]
        top5_labels = y_test[ranked_indices[:5]]

        if query_label == top1_label:
            top1_correct += 1
        if query_label in top3_labels:
            top3_correct += 1
        if query_label in top5_labels:
            top5_correct += 1

        matches = np.where(y_test[ranked_indices] == query_label)[0]
        if len(matches) > 0:
            mrr_total += 1.0 / (matches[0] + 1)

    top1_acc = (top1_correct / n_samples) * 100
    top3_acc = (top3_correct / n_samples) * 100
    top5_acc = (top5_correct / n_samples) * 100
    mrr = mrr_total / n_samples

    retrieval_results = [
        ["Top-1 Retrieval Accuracy", f"{top1_acc:.2f}%"],
        ["Top-3 Retrieval Accuracy", f"{top3_acc:.2f}%"],
        ["Top-5 Retrieval Accuracy", f"{top5_acc:.2f}%"],
        ["Mean Reciprocal Rank (MRR)", f"{mrr:.4f}"],
    ]
    print(tabulate(retrieval_results, headers=["Retrieval Metric", "Score"], tablefmt="fancy_grid"))
    print(f"\nNote: retrieval computed over {n_samples} held-out patches — a small "
          f"gallery size inflates Top-3/Top-5 accuracy somewhat by chance alone "
          f"(fewer distractors to rank ahead of the correct match). State this "
          f"gallery size explicitly wherever you report these numbers.")
    raw_metrics = {"top1": top1_acc, "top3": top3_acc, "top5": top5_acc, "mrr": mrr}
    return retrieval_results, raw_metrics


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print("🚀 Initializing Stage 6 Downstream Evaluation (train/test corrected)...")
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    print(f"Train: {X_train.shape[0]} patches | Test: {X_test.shape[0]} patches | "
          f"Feature dim: {X_train.shape[1]}")

    clf_results, clf_headers, clf_raw = run_classification_benchmark(X_train, y_train, X_test, y_test)
    retrieval_results, retrieval_raw = run_retrieval_benchmark(X_test, y_test)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("=======================================================\n")
        f.write("   STAGE 6 DOWNSTREAM EVALUATION REPORT (train/test split)\n")
        f.write("=======================================================\n\n")
        f.write(f"Train patches: {X_train.shape[0]} | Test patches: {X_test.shape[0]}\n\n")
        f.write("1. DOWNSTREAM CLASSIFICATION BENCHMARK (test-set performance)\n")
        f.write(tabulate(clf_results, headers=clf_headers, tablefmt="grid"))
        f.write("\n\n2. LATENT EMBEDDING RETRIEVAL METRICS (test-set only)\n")
        if retrieval_results:
            f.write(tabulate(retrieval_results, headers=["Metric", "Value"], tablefmt="grid"))
        f.write("\n\nStage 6 Evaluation Complete!\n")

    # Machine-readable summary — consumed by run_multi_seed_evaluation.py to
    # aggregate mean/std across several random splits.
    summary = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "classification": clf_raw,
        "retrieval": retrieval_raw,
    }
    with open("./stage6_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Report saved to: {OUTPUT_REPORT}")
    print("Stage 6 completed successfully!")


if __name__ == "__main__":
    main()