"""03 — Deep Learning Experiments (GPU required)

Runs transformer and CNN experiments that address two of the three gaps
identified in the review:

  Gap 1: No transformer baselines (BERT-base, MentalBERT)
  Gap 3: CNN uses default hyperparameters (Optuna search)

Designed for Google Colab with a T4 GPU (free tier).
Expected runtime:
  - Transformers: ~50 min (2 models × 10 folds × ~2.5 min/fold)
  - CNN default: ~20 min (10 folds × ~2 min/fold)
  - CNN Optuna: ~3-4 hours (50 trials × ~4 min/trial)

Saves results to data/results/deep_learning_results.json for downstream
analysis in notebook 04.

Colab setup:
    !pip install torch transformers optuna scikit-learn pandas numpy
    # Clone the repo or upload the src/ directory
    import sys; sys.path.insert(0, '/content/Paper Review')

Usage:
    python notebooks/03_deep_learning_experiments.py
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

from src.models.transformer_classifier import (
    TransformerClassifier,
    bert_base_factory,
    mentalbert_factory,
)
from src.models.cnn_classifier import (
    CNNClassifier,
    cnn_default_factory,
    cnn_tuned_factory,
    make_optuna_objective,
)
from src.evaluation.metrics import (
    monte_carlo_cv,
    bootstrap_ci,
    ExperimentResult,
    FoldResult,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FOLDS = 10
TEST_SIZE_PER_CLASS = 25
RANDOM_STATE = 42
N_BOOTSTRAP = 2000

# Optuna search budget
OPTUNA_N_TRIALS = 50
OPTUNA_N_FOLDS = 5  # Fewer folds during search to save time

# Transformer hyperparameters
TRANSFORMER_EPOCHS = 5
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_LR = 2e-5


# ---------------------------------------------------------------------------
# Serialization (same format as notebook 02)
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
    print(f"Results saved to {filepath}")


# ---------------------------------------------------------------------------
# Transformer experiments
# ---------------------------------------------------------------------------

def run_transformer_experiments(
    texts: list[str], labels: np.ndarray
) -> list[ExperimentResult]:
    """Fine-tune BERT-base and MentalBERT with Monte Carlo CV."""
    import torch
    print(f"\nDevice: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if not torch.cuda.is_available():
        print("⚠️ WARNING: No GPU detected. Training will be very slow.")

    # Convert to numpy object array for indexing compatibility
    X_text = np.array(texts, dtype=object)
    results: list[ExperimentResult] = []

    configs = [
        ("BERT-base", "bert-base-uncased"),
        ("MentalBERT", "mental/mental-bert-base-uncased"),
    ]

    for name, model_name in configs:
        print(f"\n{'='*60}")
        print(f"Transformer: {name} ({model_name})")
        print(f"{'='*60}")

        def factory(
            _mn=model_name,
        ) -> TransformerClassifier:
            return TransformerClassifier(
                model_name=_mn,
                epochs=TRANSFORMER_EPOCHS,
                batch_size=TRANSFORMER_BATCH_SIZE,
                learning_rate=TRANSFORMER_LR,
                patience=2,
                random_state=RANDOM_STATE,
            )

        start = time.time()
        folds = monte_carlo_cv(
            X_text, labels, factory,
            n_folds=N_FOLDS,
            test_size_per_class=TEST_SIZE_PER_CLASS,
            random_state=RANDOM_STATE,
        )
        elapsed = time.time() - start

        result = ExperimentResult(name, "transformer", folds)
        results.append(result)

        all_y = np.concatenate([f.y_true for f in folds])
        all_p = np.concatenate([f.y_prob for f in folds])
        lo, mid, hi = bootstrap_ci(all_y, all_p, n_bootstrap=N_BOOTSTRAP)

        print(f"  AUC={result.mean_auc:.3f} (±{result.std_auc:.3f})  "
              f"95% CI=[{lo:.3f}, {hi:.3f}]  ({elapsed/60:.1f} min)")

    return results


# ---------------------------------------------------------------------------
# CNN experiments
# ---------------------------------------------------------------------------

def run_cnn_default_experiment(
    texts: list[str], labels: np.ndarray
) -> ExperimentResult:
    """CNN with paper-default hyperparameters."""
    print(f"\n{'='*60}")
    print("CNN: Default hyperparameters (Kim 2014)")
    print(f"{'='*60}")

    X_text = np.array(texts, dtype=object)

    start = time.time()
    folds = monte_carlo_cv(
        X_text, labels, cnn_default_factory,
        n_folds=N_FOLDS,
        test_size_per_class=TEST_SIZE_PER_CLASS,
        random_state=RANDOM_STATE,
    )
    elapsed = time.time() - start

    result = ExperimentResult("CNN-Default", "cnn", folds)

    all_y = np.concatenate([f.y_true for f in folds])
    all_p = np.concatenate([f.y_prob for f in folds])
    lo, mid, hi = bootstrap_ci(all_y, all_p, n_bootstrap=N_BOOTSTRAP)

    print(f"  AUC={result.mean_auc:.3f} (±{result.std_auc:.3f})  "
          f"95% CI=[{lo:.3f}, {hi:.3f}]  ({elapsed/60:.1f} min)")

    return result


def run_cnn_optuna_experiment(
    texts: list[str], labels: np.ndarray
) -> tuple[ExperimentResult, dict]:
    """CNN with Optuna hyperparameter search, then evaluate best config."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n{'='*60}")
    print(f"CNN: Optuna hyperparameter search ({OPTUNA_N_TRIALS} trials)")
    print(f"{'='*60}")

    X_text = np.array(texts, dtype=object)

    # Phase 1: Search
    objective = make_optuna_objective(
        X_text, labels,
        n_folds=OPTUNA_N_FOLDS,
        test_size_per_class=TEST_SIZE_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    study = optuna.create_study(direction="maximize")
    start = time.time()
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS)
    search_time = time.time() - start

    best = study.best_params
    print(f"  Search completed in {search_time/60:.1f} min")
    print(f"  Best trial AUC: {study.best_value:.3f}")
    print(f"  Best params: {best}")

    # Phase 2: Evaluate best config with full 10-fold CV
    print(f"\n  Re-evaluating best config with {N_FOLDS}-fold CV...")

    # Reconstruct filter_sizes from individual flags
    filter_sizes = []
    for fs in [2, 3, 4, 5]:
        if best.get(f"use_filter_{fs}", False):
            filter_sizes.append(fs)
    if not filter_sizes:
        filter_sizes = [3]

    def best_factory() -> CNNClassifier:
        return CNNClassifier(
            embedding_dim=best.get("embedding_dim", 300),
            n_filters=best.get("n_filters", 128),
            filter_sizes=tuple(filter_sizes),
            dropout=best.get("dropout", 0.5),
            learning_rate=best.get("learning_rate", 1e-3),
            batch_size=best.get("batch_size", 64),
            epochs=20,
            patience=3,
            random_state=RANDOM_STATE,
        )

    start = time.time()
    folds = monte_carlo_cv(
        X_text, labels, best_factory,
        n_folds=N_FOLDS,
        test_size_per_class=TEST_SIZE_PER_CLASS,
        random_state=RANDOM_STATE,
    )
    elapsed = time.time() - start

    result = ExperimentResult("CNN-Tuned", "cnn", folds)

    all_y = np.concatenate([f.y_true for f in folds])
    all_p = np.concatenate([f.y_prob for f in folds])
    lo, mid, hi = bootstrap_ci(all_y, all_p, n_bootstrap=N_BOOTSTRAP)

    print(f"  AUC={result.mean_auc:.3f} (±{result.std_auc:.3f})  "
          f"95% CI=[{lo:.3f}, {hi:.3f}]  ({elapsed/60:.1f} min)")

    return result, best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Bayram et al. (2022) Extension: Deep Learning Experiments")
    print(f"Monte Carlo CV: {N_FOLDS} folds, "
          f"{TEST_SIZE_PER_CLASS} test samples per class")
    print()

    # Load data
    combined = pd.read_parquet(DATA_DIR / "train_val_combined.parquet")
    texts = combined["text"].tolist()
    labels = combined["label"].values.astype(np.int64)
    print(f"Data: {len(texts)} texts, "
          f"{labels.sum()} suicidal, {(1-labels).sum()} control")

    all_results: list[ExperimentResult] = []

    # Transformers (Gap 1)
    transformer_results = run_transformer_experiments(texts, labels)
    all_results.extend(transformer_results)

    # CNN default (replication)
    cnn_default_result = run_cnn_default_experiment(texts, labels)
    all_results.append(cnn_default_result)

    # CNN Optuna (Gap 3)
    cnn_tuned_result, best_params = run_cnn_optuna_experiment(
        texts, labels
    )
    all_results.append(cnn_tuned_result)

    # Save all results
    save_results(
        all_results,
        RESULTS_DIR / "deep_learning_results.json",
    )

    # Save Optuna best params separately
    with open(RESULTS_DIR / "optuna_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Optuna params saved to {RESULTS_DIR / 'optuna_best_params.json'}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r.classifier_name:15s}: AUC={r.mean_auc:.3f} "
              f"(±{r.std_auc:.3f})")

    print("\nDone.")
