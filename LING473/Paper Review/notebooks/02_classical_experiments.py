"""02 — Classical Classifier Experiments

Replicates Bayram et al. (2022) Tables 2-3: classical classifiers across
all feature types with 10-fold Monte Carlo CV. Adds bootstrap CIs and
permutation tests that the original paper omits (Gap 2).

Experiment matrix:
  Classifiers: Logistic, MLP, SVM (new), Random Forest (new)
  Features:    unigram, bigram, n-gram, stopwords, network
  Total:       4 classifiers × 5 feature types = 20 experiments

Designed to run on CPU (no GPU required). Expected runtime ~15 minutes
on a modern machine with the SDCNL dataset (~1,450 texts).

Saves results to data/results/classical_results.json for downstream
analysis in notebook 04.

Usage:
    python notebooks/02_classical_experiments.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports from src/
# ---------------------------------------------------------------------------

from src.features.text_features import TextFeatureExtractor
from src.features.network_features import LexicalNetworkFeatureExtractor
from src.models.classifiers import (
    LogisticSoftmax,
    MLP,
    SVMBaseline,
    RandomForestBaseline,
)
from src.evaluation.metrics import (
    monte_carlo_cv,
    bootstrap_ci,
    paired_permutation_test,
    per_class_report,
    ExperimentResult,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FOLDS = 10
TEST_SIZE_PER_CLASS = 25
RANDOM_STATE = 42
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 10000

CLASSIFIER_CONFIGS = [
    ("Logistic", LogisticSoftmax),
    ("MLP", lambda: MLP(hidden_layer_sizes=(100,), max_iter=300)),
    ("SVM", SVMBaseline),
    ("RF", RandomForestBaseline),
]

TEXT_FEATURE_CONFIGS = [
    ("unigram", {"feature_type": "unigram", "min_df": 5}),
    ("bigram", {"feature_type": "bigram", "min_df": 5}),
    ("ngram", {"feature_type": "ngram", "min_df": 5}),
    ("stopwords", {"feature_type": "stopwords"}),
]

NETWORK_CONFIG = {
    "window_size": 5,
    "min_word_freq": 5,
    "min_edge_freq": 3,
}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def result_to_dict(result: ExperimentResult) -> dict:
    """Convert ExperimentResult to JSON-serializable dict."""
    fold_data = []
    for fr in result.fold_results:
        fold_data.append({
            "fold_id": fr.fold_id,
            "auc": fr.auc,
            "y_true": fr.y_true.tolist(),
            "y_prob": fr.y_prob.tolist(),
            "y_pred": fr.y_pred.tolist(),
        })
    return {
        "classifier_name": result.classifier_name,
        "feature_type": result.feature_type,
        "mean_auc": result.mean_auc,
        "std_auc": result.std_auc,
        "fold_results": fold_data,
    }


def save_results(
    results: list[ExperimentResult],
    filepath: Path,
) -> None:
    """Save experiment results to JSON."""
    data = [result_to_dict(r) for r in results]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filepath}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_classical_experiments() -> list[ExperimentResult]:
    """Run all classical classifier × feature type experiments."""
    # Load data
    combined = pd.read_parquet(DATA_DIR / "train_val_combined.parquet")
    texts = combined["text"].tolist()
    labels = combined["label"].values.astype(np.int64)
    print(f"Data loaded: {len(texts)} texts, "
          f"{labels.sum()} suicidal, {(1-labels).sum()} control")

    all_results: list[ExperimentResult] = []

    # ----- Text feature experiments -----
    for feat_name, feat_kwargs in TEXT_FEATURE_CONFIGS:
        extractor = TextFeatureExtractor(**feat_kwargs)
        X = extractor.fit_transform(texts)
        print(f"\n{'='*60}")
        print(f"Feature: {feat_name} ({X.shape[1]} dimensions)")
        print(f"{'='*60}")

        for clf_name, clf_factory in CLASSIFIER_CONFIGS:
            start = time.time()
            folds = monte_carlo_cv(
                X, labels, clf_factory,
                n_folds=N_FOLDS,
                test_size_per_class=TEST_SIZE_PER_CLASS,
                random_state=RANDOM_STATE,
            )
            elapsed = time.time() - start

            result = ExperimentResult(clf_name, feat_name, folds)
            all_results.append(result)

            # Bootstrap CI
            all_y = np.concatenate([f.y_true for f in folds])
            all_p = np.concatenate([f.y_prob for f in folds])
            lo, mid, hi = bootstrap_ci(
                all_y, all_p, n_bootstrap=N_BOOTSTRAP
            )

            print(f"  {clf_name:10s}: AUC={result.mean_auc:.3f} "
                  f"(±{result.std_auc:.3f})  "
                  f"95% CI=[{lo:.3f}, {hi:.3f}]  "
                  f"({elapsed:.1f}s)")

    # ----- Network feature experiments -----
    print(f"\n{'='*60}")
    print(f"Feature: network")
    print(f"{'='*60}")

    pos_texts = [t for t, l in zip(texts, labels) if l == 1]
    neg_texts = [t for t, l in zip(texts, labels) if l == 0]

    net_ext = LexicalNetworkFeatureExtractor(**NETWORK_CONFIG)
    X_net = net_ext.fit_transform(pos_texts, neg_texts, texts)
    print(f"  Network features: {X_net.shape[1]} dimensions")
    print(f"  Positive graph: {net_ext.graph_positive.number_of_nodes()} "
          f"nodes, {net_ext.graph_positive.number_of_edges()} edges")
    print(f"  Negative graph: {net_ext.graph_negative.number_of_nodes()} "
          f"nodes, {net_ext.graph_negative.number_of_edges()} edges")

    for clf_name, clf_factory in CLASSIFIER_CONFIGS:
        start = time.time()
        folds = monte_carlo_cv(
            X_net, labels, clf_factory,
            n_folds=N_FOLDS,
            test_size_per_class=TEST_SIZE_PER_CLASS,
            random_state=RANDOM_STATE,
        )
        elapsed = time.time() - start

        result = ExperimentResult(clf_name, "network", folds)
        all_results.append(result)

        all_y = np.concatenate([f.y_true for f in folds])
        all_p = np.concatenate([f.y_prob for f in folds])
        lo, mid, hi = bootstrap_ci(
            all_y, all_p, n_bootstrap=N_BOOTSTRAP
        )

        print(f"  {clf_name:10s}: AUC={result.mean_auc:.3f} "
              f"(±{result.std_auc:.3f})  "
              f"95% CI=[{lo:.3f}, {hi:.3f}]  "
              f"({elapsed:.1f}s)")

    return all_results


def run_significance_tests(
    results: list[ExperimentResult],
) -> dict[str, float]:
    """Run pairwise permutation tests between classifiers.

    Groups results by feature type and runs all pairwise tests.
    Returns dict mapping 'A_vs_B_feat' -> p-value.
    """
    from itertools import combinations

    pairwise = {}

    # Group by feature type
    by_feat: dict[str, list[ExperimentResult]] = {}
    for r in results:
        by_feat.setdefault(r.feature_type, []).append(r)

    print(f"\n{'='*60}")
    print("Pairwise permutation tests (p-values)")
    print(f"{'='*60}")

    for feat, feat_results in by_feat.items():
        print(f"\n  Feature: {feat}")
        for a, b in combinations(feat_results, 2):
            key = f"{a.classifier_name}_vs_{b.classifier_name}_{feat}"
            p = paired_permutation_test(
                a, b, n_permutations=N_PERMUTATIONS
            )
            pairwise[key] = p
            sig = "*" if p < 0.05 else " "
            print(f"    {a.classifier_name:10s} vs {b.classifier_name:10s}: "
                  f"p={p:.3f} {sig}")

    return pairwise


def print_summary_table(results: list[ExperimentResult]) -> None:
    """Print a formatted results table."""
    print(f"\n{'='*60}")
    print("Summary Table: Mean AUC (±std)")
    print(f"{'='*60}")

    # Build table
    classifiers = list(dict.fromkeys(
        r.classifier_name for r in results
    ))
    features = list(dict.fromkeys(
        r.feature_type for r in results
    ))

    # Header
    header = f"{'Classifier':12s}"
    for feat in features:
        header += f" | {feat:>12s}"
    print(header)
    print("-" * len(header))

    # Rows
    for clf in classifiers:
        row = f"{clf:12s}"
        for feat in features:
            matching = [
                r for r in results
                if r.classifier_name == clf and r.feature_type == feat
            ]
            if matching:
                r = matching[0]
                row += f" | {r.mean_auc:5.3f}±{r.std_auc:.3f}"
            else:
                row += f" | {'---':>12s}"
        print(row)


def print_top_features(texts: list[str]) -> None:
    """Print top discriminative features from logistic regression."""
    combined = pd.read_parquet(DATA_DIR / "train_val_combined.parquet")
    labels = combined["label"].values.astype(np.int64)

    ext = TextFeatureExtractor(feature_type="unigram", min_df=5)
    X = ext.fit_transform(texts)

    clf = LogisticSoftmax()
    clf.fit(X, labels)

    print(f"\n{'='*60}")
    print("Top 20 discriminative features (logistic, unigrams)")
    print(f"{'='*60}")

    top = clf.top_features(ext.feature_names, n=20)
    for rank, (fname, coef, direction) in enumerate(top, 1):
        print(f"  {rank:2d}. {fname:20s} coef={coef:+.4f}  ({direction})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Bayram et al. (2022) Replication: Classical Experiments")
    print(f"Monte Carlo CV: {N_FOLDS} folds, "
          f"{TEST_SIZE_PER_CLASS} test samples per class")
    print()

    # Run experiments
    results = run_classical_experiments()

    # Summary table
    print_summary_table(results)

    # Significance tests
    pairwise = run_significance_tests(results)

    # Top features
    combined = pd.read_parquet(DATA_DIR / "train_val_combined.parquet")
    print_top_features(combined["text"].tolist())

    # Save results
    save_results(results, RESULTS_DIR / "classical_results.json")

    # Save pairwise test results
    with open(RESULTS_DIR / "classical_pairwise.json", "w") as f:
        json.dump(pairwise, f, indent=2)
    print(f"Pairwise tests saved to {RESULTS_DIR / 'classical_pairwise.json'}")

    print("\nDone.")
