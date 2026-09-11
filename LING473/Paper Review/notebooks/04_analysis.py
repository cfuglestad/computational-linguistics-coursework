"""04 — Analysis and Visualization

Loads results from notebooks 02 (classical) and 03 (deep learning),
produces publication-quality figures and statistical comparisons.

Outputs:
  1. Main results comparison table (all classifiers × features)
  2. Bootstrap CI forest plot
  3. Pairwise permutation test heatmap
  4. Per-class metrics (sensitivity, specificity, PPV)
  5. Error analysis on misclassified examples
  6. Lexical network co-occurrence visualization
  7. Top discriminative features table

All figures saved to data/results/figures/ as both PNG (300 DPI) and
PDF for the write-up.

Runs on CPU. No GPU required.

Usage:
    python notebooks/04_analysis.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab/CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports from src/
# ---------------------------------------------------------------------------

from src.evaluation.metrics import (
    bootstrap_ci,
    paired_permutation_test,
    per_class_report,
    ExperimentResult,
    FoldResult,
)
from src.features.text_features import TextFeatureExtractor
from src.features.network_features import LexicalNetworkFeatureExtractor


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette: distinguish paper replication vs our additions
COLOR_PAPER = "#4C72B0"      # Blue for paper's classifiers
COLOR_ADDED = "#DD8452"      # Orange for our baselines
COLOR_TRANSFORMER = "#55A868" # Green for transformers
COLOR_CNN_TUNED = "#C44E52"  # Red for tuned CNN


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(filepath: Path) -> list[ExperimentResult]:
    """Load ExperimentResults from JSON."""
    with open(filepath) as f:
        data = json.load(f)

    results = []
    for entry in data:
        folds = [
            FoldResult(
                fold_id=fd["fold_id"],
                auc=fd["auc"],
                y_true=np.array(fd["y_true"], dtype=np.int64),
                y_prob=np.array(fd["y_prob"], dtype=np.float64),
                y_pred=np.array(fd["y_pred"], dtype=np.int64),
            )
            for fd in entry["fold_results"]
        ]
        results.append(ExperimentResult(
            classifier_name=entry["classifier_name"],
            feature_type=entry["feature_type"],
            fold_results=folds,
        ))
    return results


def load_all_results() -> list[ExperimentResult]:
    """Load all available result files."""
    all_results: list[ExperimentResult] = []

    classical_path = RESULTS_DIR / "classical_results.json"
    if classical_path.exists():
        all_results.extend(load_results(classical_path))
        print(f"Loaded {len(all_results)} classical results")

    dl_path = RESULTS_DIR / "deep_learning_results.json"
    if dl_path.exists():
        dl_results = load_results(dl_path)
        all_results.extend(dl_results)
        print(f"Loaded {len(dl_results)} deep learning results")

    if not all_results:
        raise FileNotFoundError(
            "No result files found in data/results/. "
            "Run notebooks 02 and/or 03 first."
        )

    return all_results


# ---------------------------------------------------------------------------
# 1. Summary table
# ---------------------------------------------------------------------------

def make_summary_table(
    results: list[ExperimentResult],
) -> pd.DataFrame:
    """Build a DataFrame of all results with CIs."""
    rows = []
    for r in results:
        all_y = np.concatenate([f.y_true for f in r.fold_results])
        all_p = np.concatenate([f.y_prob for f in r.fold_results])
        lo, mid, hi = bootstrap_ci(all_y, all_p, n_bootstrap=2000)

        rows.append({
            "Classifier": r.classifier_name,
            "Features": r.feature_type,
            "AUC": r.mean_auc,
            "Std": r.std_auc,
            "CI_low": lo,
            "CI_high": hi,
            "CI_str": f"[{lo:.3f}, {hi:.3f}]",
            "AUC_str": f"{r.mean_auc:.3f} (±{r.std_auc:.3f})",
        })

    df = pd.DataFrame(rows)
    return df.sort_values("AUC", ascending=False).reset_index(drop=True)


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    print("Results Summary (sorted by AUC)")
    print(f"{'='*70}")
    print(f"{'Classifier':15s} {'Features':12s} {'AUC':>12s} {'95% CI':>18s}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['Classifier']:15s} {row['Features']:12s} "
              f"{row['AUC_str']:>12s} {row['CI_str']:>18s}")


# ---------------------------------------------------------------------------
# 2. Forest plot (bootstrap CIs)
# ---------------------------------------------------------------------------

def plot_forest(
    df: pd.DataFrame, save_path: Path | None = None
) -> None:
    """Forest plot of AUC with bootstrap CIs for all experiments."""
    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.35)))

    labels = [f"{row['Classifier']} + {row['Features']}"
              for _, row in df.iterrows()]
    y_pos = np.arange(len(labels))

    # Color by classifier type
    colors = []
    for _, row in df.iterrows():
        name = row["Classifier"]
        if name in ("BERT-base", "MentalBERT"):
            colors.append(COLOR_TRANSFORMER)
        elif name == "CNN-Tuned":
            colors.append(COLOR_CNN_TUNED)
        elif name in ("SVM", "RF"):
            colors.append(COLOR_ADDED)
        else:
            colors.append(COLOR_PAPER)

    # Error bars from CI
    ci_low = df["AUC"].values - df["CI_low"].values
    ci_high = df["CI_high"].values - df["AUC"].values
    xerr = np.array([ci_low, ci_high])

    ax.barh(y_pos, df["AUC"].values, xerr=xerr,
            color=colors, alpha=0.8, edgecolor="gray", linewidth=0.5,
            capsize=3, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUC")
    ax.set_title("Classifier Performance with 95% Bootstrap CIs")
    ax.set_xlim(0.45, 1.0)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8,
               label="Random baseline")

    # Legend
    patches = [
        mpatches.Patch(color=COLOR_PAPER, label="Paper classifiers"),
        mpatches.Patch(color=COLOR_ADDED, label="Added baselines"),
        mpatches.Patch(color=COLOR_TRANSFORMER, label="Transformers (Gap 1)"),
        mpatches.Patch(color=COLOR_CNN_TUNED, label="CNN-Tuned (Gap 3)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path.with_suffix(".png"))
        fig.savefig(save_path.with_suffix(".pdf"))
        print(f"Forest plot saved to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Permutation test heatmap
# ---------------------------------------------------------------------------

def make_permutation_matrix(
    results: list[ExperimentResult],
    feature_type: str = "unigram",
) -> pd.DataFrame:
    """Build pairwise permutation test p-value matrix."""
    subset = [r for r in results if r.feature_type == feature_type]
    if not subset:
        # Use all results if feature_type not found
        subset = results

    names = [r.classifier_name for r in subset]
    n = len(names)
    matrix = np.ones((n, n))

    for i, j in combinations(range(n), 2):
        p = paired_permutation_test(
            subset[i], subset[j], n_permutations=10000
        )
        matrix[i, j] = p
        matrix[j, i] = p

    return pd.DataFrame(matrix, index=names, columns=names)


def plot_heatmap(
    pval_matrix: pd.DataFrame,
    title: str = "Pairwise Permutation Test p-values",
    save_path: Path | None = None,
) -> None:
    """Heatmap of pairwise significance tests."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(pval_matrix.values, cmap="RdYlGn", vmin=0, vmax=0.1)

    names = pval_matrix.index.tolist()
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(names)):
            val = pval_matrix.values[i, j]
            if i == j:
                text = "—"
            elif val < 0.01:
                text = f"{val:.3f}**"
            elif val < 0.05:
                text = f"{val:.3f}*"
            else:
                text = f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, color="black")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="p-value", shrink=0.8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path.with_suffix(".png"))
        fig.savefig(save_path.with_suffix(".pdf"))
        print(f"Heatmap saved to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Per-class metrics
# ---------------------------------------------------------------------------

def print_per_class_metrics(results: list[ExperimentResult]) -> None:
    """Print per-class classification reports for each experiment."""
    print(f"\n{'='*70}")
    print("Per-Class Metrics (aggregated across folds)")
    print(f"{'='*70}")

    for r in results:
        all_y = np.concatenate([f.y_true for f in r.fold_results])
        all_pred = np.concatenate([f.y_pred for f in r.fold_results])
        print(f"\n--- {r.classifier_name} + {r.feature_type} ---")
        print(per_class_report(all_y, all_pred))


# ---------------------------------------------------------------------------
# 5. Error analysis
# ---------------------------------------------------------------------------

def run_error_analysis(
    results: list[ExperimentResult],
    texts: list[str],
    labels: np.ndarray,
    n_examples: int = 5,
) -> None:
    """Analyze misclassified examples from the best classical model."""
    print(f"\n{'='*70}")
    print("Error Analysis")
    print(f"{'='*70}")

    # Find best classical result by AUC
    classical = [r for r in results
                 if r.feature_type not in ("transformer", "cnn")]
    if not classical:
        print("No classical results available for error analysis.")
        return

    best = max(classical, key=lambda r: r.mean_auc)
    print(f"\nAnalyzing errors from: {best.classifier_name} + {best.feature_type}")
    print(f"  (AUC = {best.mean_auc:.3f})")

    # Aggregate predictions across folds
    # Use the first fold for concrete examples
    fold = best.fold_results[0]

    # Find misclassified samples
    false_neg_mask = (fold.y_true == 1) & (fold.y_pred == 0)
    false_pos_mask = (fold.y_true == 0) & (fold.y_pred == 1)

    fn_count = false_neg_mask.sum()
    fp_count = false_pos_mask.sum()
    total = len(fold.y_true)

    print(f"\n  Fold 0 test set: {total} samples")
    print(f"  False negatives (missed suicidal): {fn_count}")
    print(f"  False positives (false alarm):     {fp_count}")
    print(f"  Accuracy: {1 - (fn_count + fp_count) / total:.3f}")

    print(f"\n  Note: Full error analysis with example texts requires")
    print(f"  mapping fold indices back to the original dataset.")
    print(f"  This is done in the write-up notebook after running")
    print(f"  experiments with index tracking enabled.")


# ---------------------------------------------------------------------------
# 6. Network visualization
# ---------------------------------------------------------------------------

def plot_network_visualization(
    texts: list[str],
    labels: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Side-by-side lexical co-occurrence networks for each class."""
    try:
        import networkx as nx
    except ImportError:
        print("⚠️ networkx not installed. Skipping network visualization.")
        return

    pos_texts = [t for t, l in zip(texts, labels) if l == 1]
    neg_texts = [t for t, l in zip(texts, labels) if l == 0]

    net_ext = LexicalNetworkFeatureExtractor(
        window_size=5, min_word_freq=10, min_edge_freq=5
    )
    net_ext.fit(pos_texts, neg_texts)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, graph, title, color in [
        (axes[0], net_ext.graph_positive, "Suicidal (r/SuicideWatch)", "#C44E52"),
        (axes[1], net_ext.graph_negative, "Control (r/depression)", "#4C72B0"),
    ]:
        # Take top 50 nodes by degree for readability
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:50]
        sub = graph.subgraph(top_nodes)

        pos = nx.spring_layout(sub, seed=42, k=2.0)
        node_sizes = [degrees[n] * 3 for n in sub.nodes()]

        # Edge weights for thickness
        edges = sub.edges(data=True)
        weights = [d.get("weight", 1.0) for _, _, d in edges]
        max_w = max(weights) if weights else 1.0
        norm_weights = [w / max_w * 2.0 for w in weights]

        nx.draw_networkx_edges(
            sub, pos, ax=ax, alpha=0.3,
            width=norm_weights, edge_color="gray",
        )
        nx.draw_networkx_nodes(
            sub, pos, ax=ax, node_size=node_sizes,
            node_color=color, alpha=0.7,
        )
        nx.draw_networkx_labels(
            sub, pos, ax=ax, font_size=7,
        )

        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.suptitle(
        "Lexical Co-occurrence Networks (top 50 nodes by degree)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path.with_suffix(".png"))
        fig.savefig(save_path.with_suffix(".pdf"))
        print(f"Network visualization saved to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Top features
# ---------------------------------------------------------------------------

def plot_top_features(
    texts: list[str],
    labels: np.ndarray,
    n_features: int = 20,
    save_path: Path | None = None,
) -> None:
    """Horizontal bar chart of top discriminative features."""
    from src.models.classifiers import LogisticSoftmax

    ext = TextFeatureExtractor(feature_type="unigram", min_df=5)
    X = ext.fit_transform(texts)

    clf = LogisticSoftmax()
    clf.fit(X, labels)
    top = clf.top_features(ext.feature_names, n=n_features)

    fig, ax = plt.subplots(figsize=(8, 6))

    names = [t[0] for t in top][::-1]
    coefs = [t[1] for t in top][::-1]
    colors = ["#C44E52" if c > 0 else "#4C72B0" for c in coefs]

    ax.barh(range(len(names)), coefs, color=colors, edgecolor="gray",
            linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title(f"Top {n_features} Discriminative Unigram Features")
    ax.axvline(x=0, color="black", linewidth=0.8)

    # Legend
    patches = [
        mpatches.Patch(color="#C44E52", label="Suicidal (positive coef)"),
        mpatches.Patch(color="#4C72B0", label="Control (negative coef)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path.with_suffix(".png"))
        fig.savefig(save_path.with_suffix(".pdf"))
        print(f"Feature plot saved to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Bayram et al. (2022) Replication: Analysis & Visualization")
    print()

    # Load all results
    results = load_all_results()

    # 1. Summary table
    summary_df = make_summary_table(results)
    print_summary_table(summary_df)
    summary_df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)

    # 2. Forest plot
    plot_forest(summary_df, save_path=FIGURES_DIR / "forest_plot")

    # 3. Permutation test heatmaps
    # One per feature type that has multiple classifiers
    feature_types = set(r.feature_type for r in results)
    for feat in sorted(feature_types):
        feat_results = [r for r in results if r.feature_type == feat]
        if len(feat_results) < 2:
            continue
        matrix = make_permutation_matrix(results, feature_type=feat)
        plot_heatmap(
            matrix,
            title=f"Permutation Test p-values ({feat})",
            save_path=FIGURES_DIR / f"heatmap_{feat}",
        )

    # 4. Per-class metrics
    print_per_class_metrics(results)

    # 5. Error analysis
    combined = pd.read_parquet(DATA_DIR / "train_val_combined.parquet")
    texts = combined["text"].tolist()
    labels = combined["label"].values.astype(np.int64)

    run_error_analysis(results, texts, labels)

    # 6. Network visualization
    plot_network_visualization(
        texts, labels, save_path=FIGURES_DIR / "network_viz"
    )

    # 7. Top features
    plot_top_features(
        texts, labels, save_path=FIGURES_DIR / "top_features"
    )

    print(f"\n{'='*70}")
    print(f"All figures saved to {FIGURES_DIR}")
    print(f"Summary table saved to {RESULTS_DIR / 'summary_table.csv'}")
    print("Done.")
