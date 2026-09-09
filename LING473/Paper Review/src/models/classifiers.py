"""Classifier wrappers replicating Bayram et al. (2022) and adding baselines.

Four classifiers from the original paper:
  1. Logistic softmax (no hidden layers) -- equivalent to logistic regression
  2. MLP (1 hidden layer, 1000 neurons, tanh activation, dropout 0.98)
  3. CNN (Kim 2014 architecture, pretrained word2vec, default hyperparameters)
  4. CNN with tuned hyperparameters (our addition)

Additional baselines (our contribution):
  5. SVM with RBF kernel (common clinical NLP baseline)
  6. Random Forest (common clinical NLP baseline)

Transformer baselines (BERT, MentalBERT) live in a separate module because
they use a different feature pipeline (tokenizer, not bag-of-words).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class Classifier(Protocol):
    """Protocol for all classifiers in the experiment."""

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> Any: ...
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]: ...


@dataclass
class LogisticSoftmax:
    """Logistic softmax classifier (no hidden layers).

    Bayram et al. implement this as a TensorFlow neural network with zero
    hidden layers. Functionally equivalent to multinomial logistic regression.
    We use scikit-learn's LogisticRegression with saga solver for consistency.

    This is the paper's interpretable baseline. Feature weights from the
    trained model directly indicate discriminative words (Table 4).
    """

    C: float = 1.0
    max_iter: int = 2000
    random_state: int = 42

    def __post_init__(self) -> None:
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> LogisticSoftmax:
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    @property
    def feature_importances(self) -> NDArray[np.float64]:
        """Absolute coefficient values for feature ranking (Table 4 replication)."""
        return np.abs(self._model.coef_[0])

    def top_features(
        self, feature_names: list[str], n: int = 20
    ) -> list[tuple[str, float, str]]:
        """Return top-n discriminative features with their class direction.

        Returns list of (feature_name, coefficient, class_label) tuples
        sorted by absolute coefficient magnitude.
        """
        coefs = self._model.coef_[0]
        indices = np.argsort(np.abs(coefs))[::-1][:n]
        results = []
        for idx in indices:
            name = feature_names[idx]
            coef = coefs[idx]
            direction = "suicidal" if coef > 0 else "control"
            results.append((name, float(coef), direction))
        return results


@dataclass
class MLP:
    """Multilayer perceptron matching Bayram et al. (2022) architecture.

    Paper specifies: 1 hidden layer, 1000 neurons, tanh activation,
    dropout rate 0.98 (which means 98% of neurons are KEPT, i.e., only
    2% dropout). This is unusual but matches their TensorFlow description.

    scikit-learn's MLPClassifier doesn't support dropout directly, so we
    use alpha (L2 regularization) as a proxy. For the exact architecture,
    use the PyTorch implementation in the notebooks.
    """

    hidden_layer_sizes: tuple[int, ...] = (1000,)
    activation: str = "tanh"
    alpha: float = 0.01  # L2 regularization (proxy for low dropout)
    max_iter: int = 500
    random_state: int = 42

    def __post_init__(self) -> None:
        self._scaler = StandardScaler()
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> MLP:
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)


@dataclass
class SVMBaseline:
    """SVM with RBF kernel (external baseline missing from original paper).

    Pestian et al. (2017) use SVM for the same task. Including it addresses
    the review's critique that the paper lacks external baselines.
    """

    C: float = 1.0
    kernel: str = "rbf"
    probability: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        self._scaler = StandardScaler()
        self._model = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=self.probability,
            random_state=self.random_state,
        )

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> SVMBaseline:
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)


@dataclass
class RandomForestBaseline:
    """Random Forest baseline (common in clinical text classification)."""

    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 42

    def __post_init__(self) -> None:
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> RandomForestBaseline:
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._model.predict_proba(X)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        return self._model.predict(X)


# Registry for experiment loops
CLASSIFIER_REGISTRY: dict[str, type] = {
    "logistic": LogisticSoftmax,
    "mlp": MLP,
    "svm": SVMBaseline,
    "random_forest": RandomForestBaseline,
}
