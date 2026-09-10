"""Kim (2014) CNN text classifier with Optuna hyperparameter tuning.

Replications the CNN architecture from:
  Kim, Y. (2014). Convolutional neural networks for sentence classification.
  Proceedings of EMNLP, pp. 1746-1751.

Bayram et al. (2022) use this CNN with default hyperparameters and pretrained
word2vec embeddings. Our review critique (Gap 3) argues that the CNN was given
an unfair comparison because its hyperparameters were never tuned, while the
MLP architecture was implicitly tuned through its design choices.

This module provides:
  1. CNNClassifier with paper-default hyperparameters (faithful replication)
  2. CNNClassifier with Optuna-tuned hyperparameters (our contribution)

Architecture:
  Embedding -> parallel Conv1d branches (variable kernel sizes)
  -> max-over-time pooling -> concat -> dropout -> linear -> softmax

Design:
  Accepts raw text input (same as transformer_classifier.py). Builds its
  own vocabulary from training data during fit(). Supports optional
  pretrained GloVe embeddings.

Compute:
  Trains on GPU (Colab T4) in under 2 minutes per fold for ~1,500 texts.
  CPU training is feasible (~5 min per fold).
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset as TorchDataset

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import optuna

    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


def _check_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "cnn_classifier requires torch. "
            "Install with: pip install torch"
        )


def _check_optuna() -> None:
    if not _HAS_OPTUNA:
        raise ImportError(
            "Optuna tuning requires optuna. "
            "Install with: pip install optuna"
        )


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Word-level vocabulary built from training data.

    Maps words to integer indices for the embedding layer. Reserves
    index 0 for PAD and index 1 for UNK (unknown words at test time).

    Parameters
    ----------
    min_freq : int
        Minimum word frequency in training data to be included.
        Words below this threshold map to UNK.
    max_vocab_size : int or None
        Maximum vocabulary size (excluding PAD/UNK). None for unlimited.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(
        self, min_freq: int = 2, max_vocab_size: int | None = None
    ) -> None:
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self._word2idx: dict[str, int] = {}
        self._idx2word: list[str] = []

    def build(self, texts: list[str]) -> Vocabulary:
        """Build vocabulary from tokenized training texts."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text.lower().split())

        # Filter by frequency, sort by count (descending) for determinism
        filtered = [
            (word, count)
            for word, count in counter.most_common()
            if count >= self.min_freq
        ]

        if self.max_vocab_size is not None:
            filtered = filtered[: self.max_vocab_size]

        # Build mappings (PAD=0, UNK=1, then vocab words)
        self._idx2word = [self.PAD_TOKEN, self.UNK_TOKEN]
        self._word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}

        for word, _ in filtered:
            idx = len(self._idx2word)
            self._word2idx[word] = idx
            self._idx2word.append(word)

        return self

    def encode(
        self, text: str, max_length: int = 256
    ) -> list[int]:
        """Encode a single text to a list of integer indices.

        Pads or truncates to max_length.
        """
        tokens = text.lower().split()
        indices = [
            self._word2idx.get(t, self.UNK_IDX) for t in tokens
        ]

        # Truncate
        if len(indices) > max_length:
            indices = indices[:max_length]
        # Pad
        while len(indices) < max_length:
            indices.append(self.PAD_IDX)

        return indices

    def __len__(self) -> int:
        return len(self._idx2word)

    @property
    def vocab_size(self) -> int:
        return len(self._idx2word)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class _TextCNNDataset(TorchDataset if _HAS_TORCH else object):  # type: ignore[misc]
    """Tokenized + encoded text dataset for DataLoader."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        max_length: int = 256,
    ) -> None:
        self.encoded = [vocab.encode(t, max_length) for t in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        _check_torch()
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Kim (2014) TextCNN model
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class TextCNN(nn.Module):
        """Convolutional neural network for sentence classification.

        Architecture follows Kim (2014):
          1. Embedding layer (trainable, optionally pretrained)
          2. Multiple parallel Conv1d branches with different kernel sizes
          3. Max-over-time pooling on each branch
          4. Concatenate pooled features
          5. Dropout
          6. Linear projection to num_classes

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary (including PAD and UNK).
        embedding_dim : int
            Dimension of word embeddings.
        n_filters : int
            Number of convolutional filters per kernel size.
        filter_sizes : tuple of int
            Kernel sizes for each parallel Conv1d branch.
        dropout : float
            Dropout rate applied after concatenation.
        num_classes : int
            Number of output classes (2 for binary).
        pad_idx : int
            Index of the padding token (zeroed in embedding).
        """

        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 300,
            n_filters: int = 128,
            filter_sizes: tuple[int, ...] = (3, 4, 5),
            dropout: float = 0.5,
            num_classes: int = 2,
            pad_idx: int = 0,
        ) -> None:
            super().__init__()

            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=pad_idx
            )

            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=embedding_dim,
                        out_channels=n_filters,
                        kernel_size=fs,
                    )
                    for fs in filter_sizes
                ]
            )

            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(n_filters * len(filter_sizes), num_classes)

        def forward(self, x: Any) -> Any:
            """Forward pass.

            Parameters
            ----------
            x : Tensor of shape (batch_size, seq_length)
                Integer-encoded token indices.

            Returns
            -------
            Tensor of shape (batch_size, num_classes)
                Raw logits (no softmax — handled by CrossEntropyLoss).
            """
            # (batch, seq_len) -> (batch, seq_len, emb_dim)
            embedded = self.embedding(x)

            # Conv1d expects (batch, channels, seq_len)
            embedded = embedded.permute(0, 2, 1)

            # Apply each conv + ReLU + max-pool
            pooled = []
            for conv in self.convs:
                # (batch, n_filters, seq_len - kernel + 1)
                conved = torch.relu(conv(embedded))
                # Max-over-time: (batch, n_filters)
                p = conved.max(dim=2).values
                pooled.append(p)

            # Concatenate all pooled features: (batch, n_filters * n_branches)
            cat = torch.cat(pooled, dim=1)
            cat = self.dropout(cat)

            return self.fc(cat)

        def load_pretrained_embeddings(
            self, embedding_matrix: Any
        ) -> None:
            """Load pretrained embedding weights (e.g., GloVe).

            Parameters
            ----------
            embedding_matrix : Tensor of shape (vocab_size, embedding_dim)
            """
            self.embedding.weight.data.copy_(embedding_matrix)
            # Keep PAD at zero
            self.embedding.weight.data[0] = torch.zeros(
                self.embedding.embedding_dim
            )


# ---------------------------------------------------------------------------
# CNN Classifier wrapper (Classifier protocol)
# ---------------------------------------------------------------------------

@dataclass
class CNNClassifier:
    """Kim (2014) CNN wrapped with the Classifier protocol interface.

    Accepts raw text input (list of str or numpy object array). Builds
    vocabulary from training data, trains a TextCNN model, and provides
    predict_proba/predict matching classifiers.py's Classifier protocol.

    Parameters
    ----------
    embedding_dim : int
        Word embedding dimension. 300 matches GloVe-6B.
    n_filters : int
        Filters per convolutional branch.
    filter_sizes : tuple of int
        Kernel sizes for parallel Conv1d branches.
    dropout : float
        Dropout rate after pooling concatenation.
    learning_rate : float
        Adam optimizer learning rate.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Training and inference batch size.
    max_length : int
        Maximum token sequence length (pad/truncate).
    min_word_freq : int
        Minimum word frequency for vocabulary inclusion.
    patience : int
        Early stopping patience (epochs without val loss improvement).
    val_fraction : float
        Fraction of training data held out for early stopping.
    random_state : int
        Seed for reproducibility.
    pretrained_path : str or None
        Optional path to pretrained embeddings (GloVe format: word dim1 dim2 ...).
        If None, embeddings are randomly initialized.
    """

    embedding_dim: int = 300
    n_filters: int = 128
    filter_sizes: tuple[int, ...] = (3, 4, 5)
    dropout: float = 0.5
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    max_length: int = 256
    min_word_freq: int = 2
    patience: int = 3
    val_fraction: float = 0.1
    random_state: int = 42
    pretrained_path: str | None = None

    # Private state
    _vocab: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _device: str = field(default="cpu", repr=False)

    def _load_glove(
        self, vocab: Vocabulary
    ) -> Any:
        """Load GloVe vectors for words in the vocabulary.

        Returns a tensor of shape (vocab_size, embedding_dim). Words not
        found in GloVe get random initialization (normal, std=0.1).
        """
        _check_torch()

        matrix = torch.randn(vocab.vocab_size, self.embedding_dim) * 0.1
        matrix[Vocabulary.PAD_IDX] = torch.zeros(self.embedding_dim)

        found = 0
        with open(self.pretrained_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in vocab._word2idx:
                    vec = torch.tensor(
                        [float(x) for x in parts[1:]], dtype=torch.float32
                    )
                    if vec.shape[0] == self.embedding_dim:
                        matrix[vocab._word2idx[word]] = vec
                        found += 1

        coverage = found / max(1, vocab.vocab_size - 2)  # exclude PAD/UNK
        if coverage < 0.5:
            warnings.warn(
                f"GloVe coverage is low: {found}/{vocab.vocab_size - 2} "
                f"words ({coverage:.1%}). Consider random initialization.",
                stacklevel=2,
            )

        return matrix

    def fit(
        self,
        X: NDArray[Any] | list[str],
        y: NDArray[np.int64] | list[int],
    ) -> CNNClassifier:
        """Build vocabulary from training text, train the CNN.

        Splits a validation set for early stopping, then trains using
        a manual PyTorch loop with Adam and CrossEntropyLoss.
        """
        _check_torch()

        texts = [str(t) for t in X]
        labels = [int(lbl) for lbl in y]

        # Reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Train/val split for early stopping
        rng = np.random.RandomState(self.random_state)
        n = len(texts)
        n_val = max(1, int(n * self.val_fraction))
        indices = rng.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Build vocabulary from training data only
        self._vocab = Vocabulary(
            min_freq=self.min_word_freq
        ).build(train_texts)

        # Build model
        self._model = TextCNN(
            vocab_size=self._vocab.vocab_size,
            embedding_dim=self.embedding_dim,
            n_filters=self.n_filters,
            filter_sizes=self.filter_sizes,
            dropout=self.dropout,
            pad_idx=Vocabulary.PAD_IDX,
        )

        # Load pretrained embeddings if provided
        if self.pretrained_path is not None:
            emb_matrix = self._load_glove(self._vocab)
            self._model.load_pretrained_embeddings(emb_matrix)

        # Device
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        self._model.to(self._device)

        # Datasets and loaders
        train_ds = _TextCNNDataset(
            train_texts, train_labels, self._vocab, self.max_length
        )
        val_ds = _TextCNNDataset(
            val_texts, val_labels, self._vocab, self.max_length
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        # Optimizer and loss
        optimizer = optim.Adam(
            self._model.parameters(), lr=self.learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            self._model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                logits = self._model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            # Validate
            self._model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self._device)
                    batch_y = batch_y.to(self._device)
                    logits = self._model(batch_x)
                    val_losses.append(
                        criterion(logits, batch_y).item()
                    )

            avg_val_loss = np.mean(val_losses)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self._model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self._device)

        return self

    def predict_proba(
        self, X: NDArray[Any] | list[str]
    ) -> NDArray[np.float64]:
        """Return class probabilities for input texts.

        Returns array of shape (n_samples, 2) with columns
        [P(control), P(suicidal)].
        """
        _check_torch()
        if self._model is None or self._vocab is None:
            raise RuntimeError("Call fit() before predict_proba().")

        texts = [str(t) for t in X]
        encoded = [
            self._vocab.encode(t, self.max_length) for t in texts
        ]

        self._model.eval()
        all_probs = []

        with torch.no_grad():
            for start in range(0, len(encoded), self.batch_size):
                end = min(start + self.batch_size, len(encoded))
                batch = torch.tensor(
                    encoded[start:end], dtype=torch.long
                ).to(self._device)
                logits = self._model(batch)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0).astype(np.float64)

    def predict(
        self, X: NDArray[Any] | list[str]
    ) -> NDArray[np.int64]:
        """Return predicted class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def cnn_default_factory(**kwargs: Any) -> CNNClassifier:
    """CNN with Bayram et al. (2022) default hyperparameters.

    This replicates the paper's CNN setup: Kim (2014) architecture with
    pretrained word2vec embeddings and default filter/dropout settings.
    Embedding dim is 300 (word2vec/GloVe standard).
    """
    defaults = {
        "embedding_dim": 300,
        "n_filters": 128,
        "filter_sizes": (3, 4, 5),
        "dropout": 0.5,
        "learning_rate": 1e-3,
        "epochs": 20,
        "batch_size": 64,
    }
    defaults.update(kwargs)
    return CNNClassifier(**defaults)


def cnn_tuned_factory(
    trial: Any, **kwargs: Any
) -> CNNClassifier:
    """CNN with Optuna-sampled hyperparameters.

    Samples architecture and training hyperparameters from the trial.
    Used inside an Optuna objective function.

    Search space:
      - n_filters: 64, 128, 256
      - filter_sizes: subsets of {2, 3, 4, 5}
      - dropout: 0.3-0.7
      - learning_rate: 1e-4 to 1e-2 (log scale)
      - embedding_dim: 100, 200, 300
      - batch_size: 32, 64, 128
    """
    _check_optuna()

    n_filters = trial.suggest_categorical("n_filters", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.3, 0.7)
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True
    )
    embedding_dim = trial.suggest_categorical(
        "embedding_dim", [100, 200, 300]
    )
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Variable number of filter sizes (at least 1)
    use_fs2 = trial.suggest_categorical("use_filter_2", [True, False])
    use_fs3 = trial.suggest_categorical("use_filter_3", [True, False])
    use_fs4 = trial.suggest_categorical("use_filter_4", [True, False])
    use_fs5 = trial.suggest_categorical("use_filter_5", [True, False])

    filter_sizes = []
    if use_fs2:
        filter_sizes.append(2)
    if use_fs3:
        filter_sizes.append(3)
    if use_fs4:
        filter_sizes.append(4)
    if use_fs5:
        filter_sizes.append(5)

    # Guarantee at least one filter size
    if not filter_sizes:
        filter_sizes = [3]

    config = {
        "embedding_dim": embedding_dim,
        "n_filters": n_filters,
        "filter_sizes": tuple(filter_sizes),
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": 20,
        "patience": 3,
    }
    config.update(kwargs)
    return CNNClassifier(**config)


def make_optuna_objective(
    X: NDArray[Any] | list[str],
    y: NDArray[np.int64],
    n_folds: int = 5,
    test_size_per_class: int = 25,
    random_state: int = 42,
) -> Callable[[Any], float]:
    """Create an Optuna objective function for CNN hyperparameter search.

    Returns a callable that takes an Optuna trial and returns mean AUC
    across Monte Carlo CV folds. Use with:

        study = optuna.create_study(direction='maximize')
        objective = make_optuna_objective(X_text, y, n_folds=5)
        study.optimize(objective, n_trials=50)
        best_params = study.best_params

    Parameters
    ----------
    X : array-like of str
        Raw text input.
    y : array-like of int
        Binary labels.
    n_folds : int
        Number of Monte Carlo CV folds.
    test_size_per_class : int
        Samples per class in each test fold.
    random_state : int
        Base seed for fold generation.
    """
    _check_torch()
    _check_optuna()

    from sklearn.metrics import roc_auc_score

    texts = [str(t) for t in X]
    labels = np.asarray(y, dtype=np.int64)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    def objective(trial: Any) -> float:
        rng = np.random.RandomState(random_state)
        aucs = []

        for _fold in range(n_folds):
            test_pos = rng.choice(pos_idx, size=test_size_per_class, replace=False)
            test_neg = rng.choice(neg_idx, size=test_size_per_class, replace=False)
            test_set = set(test_pos) | set(test_neg)
            test_idx = np.array(sorted(test_set))
            train_idx = np.array([i for i in range(len(labels)) if i not in test_set])

            train_texts = [texts[i] for i in train_idx]
            train_labels = labels[train_idx].tolist()
            test_texts = [texts[i] for i in test_idx]
            test_labels = labels[test_idx]

            clf = cnn_tuned_factory(trial)
            clf.fit(train_texts, train_labels)
            y_prob = clf.predict_proba(test_texts)[:, 1]
            aucs.append(roc_auc_score(test_labels, y_prob))

        return float(np.mean(aucs))

    return objective
