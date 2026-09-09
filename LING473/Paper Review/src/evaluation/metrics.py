"""Evaluation metrics and statistical tests for the replication study.

Replicates the paper's Monte Carlo 10-fold CV evaluation, then adds
the statistical rigor the original paper lacks:
  - Bootstrap confidence intervals on AUC
  - Paired permutation tests between classifiers
  - Per-class metrics (sensitivity, specificity, PPV)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold


@dataclass
class FoldResult:
    """Results from a single CV fold."""

    fold_id: int
    auc: float
    y_true: NDArray[np.int64]
    y_prob: NDArray[np.float64]
    y_pred: NDArray[np.int64]


@dataclass
class ExperimentResult:
    """Aggregated results across all CV folds."""

    classifier_name: str
    feature_type: str
    fold_results: list[FoldResult]

    @property
    def auc_scores(self) -> NDArray[np.float64]:
        return np.array([f.auc for f in self.fold_results])

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.auc_scores))

    @property
    def std_auc(self) -> float:
        return float(np.std(self.auc_scores))

    def __str__(self) -> str:
        return (
            f"{self.classifier_name} + {self.feature_type}: "
            f"{self.mean_auc:.1f} (+/-{self.std_auc:.1f})"
        )


def monte_carlo_cv(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    classifier_factory: Callable[[], Any],
    n_folds: int = 10,
    test_size_per_class: int = 25,
    random_state: int = 42,
) -> list[FoldResult]:
    """Monte Carlo cross-validation matching Bayram et al. (2022).

    At each fold:
      - Hold out test_size_per_class samples from each class as test set
      - Use remaining samples for training
      - Report AUC on the held-out test set

    This matches the paper's description: "25 suicidal and 25 control
    subjects are randomly assigned as the test set" at each fold.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,) with values in {0, 1}
    classifier_factory : callable returning a fresh classifier instance
    n_folds : int
        Number of random CV folds (paper uses 10).
    test_size_per_class : int
        Number of samples per class in the test set (paper uses 25).
    random_state : int
        Base seed for reproducibility.
    """
    rng = np.random.RandomState(random_state)
    results: list[FoldResult] = []

    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    for fold in range(n_folds):
        # Random balanced test set
        test_pos = rng.choice(pos_indices, size=test_size_per_class, replace=False)
        test_neg = rng.choice(neg_indices, size=test_size_per_class, replace=False)
        test_idx = np.concatenate([test_pos, test_neg])

        train_idx = np.array([i for i in range(len(y)) if i not in set(test_idx)])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train and evaluate
        clf = classifier_factory()
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        auc = roc_auc_score(y_test, y_prob)

        results.append(FoldResult(
            fold_id=fold,
            auc=auc,
            y_true=y_test,
            y_prob=y_prob,
            y_pred=y_pred,
        ))

    return results


def bootstrap_ci(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for AUC.

    Returns (lower, point_estimate, upper).

    This addresses the paper's missing confidence intervals. With only
    50 test samples, the CI width reveals how uncertain the AUC estimate
    actually is.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    aucs: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    aucs_arr = np.array(aucs)
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(aucs_arr, 100 * alpha))
    upper = float(np.percentile(aucs_arr, 100 * (1 - alpha)))
    point = float(np.mean(aucs_arr))

    return lower, point, upper


def paired_permutation_test(
    result_a: ExperimentResult,
    result_b: ExperimentResult,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> float:
    """Paired permutation test between two classifiers' AUC scores.

    Tests the null hypothesis that the two classifiers have equal
    expected AUC. Returns a two-sided p-value.

    This directly addresses the paper's missing significance tests.
    When MLP gets 95.1% and logistic gets 92.0%, this test tells us
    whether the 3.1% gap is real or noise from the small test set.
    """
    rng = np.random.RandomState(random_state)

    scores_a = result_a.auc_scores
    scores_b = result_b.auc_scores

    assert len(scores_a) == len(scores_b), (
        "Both results must have the same number of folds."
    )

    observed_diff = np.abs(np.mean(scores_a) - np.mean(scores_b))
    diffs = scores_a - scores_b
    n_folds = len(diffs)

    count = 0
    for _ in range(n_permutations):
        # Randomly flip signs (equivalent to swapping assignment)
        signs = rng.choice([-1, 1], size=n_folds)
        perm_diff = np.abs(np.mean(diffs * signs))
        if perm_diff >= observed_diff:
            count += 1

    return count / n_permutations


def per_class_report(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    target_names: list[str] | None = None,
) -> str:
    """Per-class classification report (sensitivity, specificity, PPV).

    The paper reports only AUC. Per-class metrics matter more in a
    clinical screening context where false positive and false negative
    costs are asymmetric.
    """
    if target_names is None:
        target_names = ["control", "suicidal"]
    return classification_report(
        y_true, y_pred, target_names=target_names, digits=3
    )
