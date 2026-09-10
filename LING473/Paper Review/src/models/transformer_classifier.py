"""Transformer-based classifiers for suicidal ideation detection.

Adds the BERT and MentalBERT baselines that Bayram et al. (2022) omit.
These are the core "missing baseline" contribution from the review critique:
by 2022, pretrained language models were well established in clinical NLP,
and omitting them weakens the paper's claim of a comprehensive comparison.

Supported models:
  - bert-base-uncased (general-domain BERT)
  - mental/mental-bert-base-uncased (MentalBERT, pretrained on Reddit
    mental health text including r/SuicideWatch)

Design:
  The TransformerClassifier wraps HuggingFace's Trainer API and exposes
  fit / predict_proba / predict to match the Classifier protocol in
  classifiers.py. Input is raw text (numpy object array or list of str),
  not pre-extracted features. The Monte Carlo CV pipeline passes text
  arrays indexed by sample, which numpy object arrays support.

Compute:
  Fine-tuning runs on GPU (Colab T4 free tier). The dataset is small
  enough (~1,500 texts) that each fold trains in under 5 minutes on a T4.
  CPU training works but is slow (~30 min per fold).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


def _check_deps() -> None:
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformer_classifier requires torch and transformers. "
            "Install with: pip install torch transformers"
        )


# ---------------------------------------------------------------------------
# Dataset wrapper for HuggingFace Trainer
# ---------------------------------------------------------------------------

class _TextDataset(TorchDataset if _HAS_TRANSFORMERS else object):  # type: ignore[misc]
    """Tokenized text dataset for Trainer."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int = 256,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Transformer classifier
# ---------------------------------------------------------------------------

@dataclass
class TransformerClassifier:
    """Fine-tuned transformer for binary text classification.

    Wraps HuggingFace Trainer with an interface matching the Classifier
    protocol (fit, predict_proba, predict). Accepts raw text as input.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Use 'bert-base-uncased' or
        'mental/mental-bert-base-uncased'.
    max_length : int
        Maximum token sequence length. 256 covers >95% of SDCNL posts
        without excessive padding.
    epochs : int
        Maximum training epochs. Early stopping may halt sooner.
    batch_size : int
        Training and evaluation batch size. 16 fits comfortably on a
        Colab T4 (16GB VRAM) with max_length=256.
    learning_rate : float
        AdamW learning rate. 2e-5 is standard for BERT fine-tuning.
    weight_decay : float
        L2 regularization coefficient.
    warmup_ratio : float
        Fraction of total steps used for linear warmup.
    patience : int
        Early stopping patience (number of evaluations without improvement).
    val_fraction : float
        Fraction of training data held out for early stopping evaluation.
    random_state : int
        Seed for reproducibility.
    output_dir : str
        Directory for Trainer checkpoints. Use a temp dir for CV folds.
    """

    model_name: str = "bert-base-uncased"
    max_length: int = 256
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 2
    val_fraction: float = 0.1
    random_state: int = 42
    output_dir: str = "/tmp/transformer_checkpoints"

    # Private state
    _tokenizer: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _trainer: Any = field(default=None, repr=False)
    _device: str = field(default="cpu", repr=False)

    def fit(
        self,
        X: NDArray[Any] | list[str],
        y: NDArray[np.int64] | list[int],
    ) -> TransformerClassifier:
        """Fine-tune the transformer on text and labels.

        Splits a validation set from X for early stopping, then trains
        using HuggingFace Trainer.
        """
        _check_deps()

        texts = [str(t) for t in X]
        labels = [int(lbl) for lbl in y]

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

        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )

        # Detect device
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        # Build datasets
        train_dataset = _TextDataset(
            train_texts, train_labels, self._tokenizer, self.max_length
        )
        val_dataset = _TextDataset(
            val_texts, val_labels, self._tokenizer, self.max_length
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            seed=self.random_state,
            logging_steps=50,
            report_to="none",  # No wandb/mlflow during CV
            disable_tqdm=True,
        )

        # Trainer with early stopping
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._trainer.train()

        return self

    def predict_proba(
        self, X: NDArray[Any] | list[str]
    ) -> NDArray[np.float64]:
        """Return class probabilities for input texts.

        Returns array of shape (n_samples, 2) with columns
        [P(control), P(suicidal)].
        """
        _check_deps()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call fit() before predict_proba().")

        texts = [str(t) for t in X]

        # Tokenize
        encodings = self._tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device
        self._model.eval()
        device = next(self._model.parameters()).device
        encodings = {k: v.to(device) for k, v in encodings.items()}

        # Inference in batches to avoid OOM
        all_probs = []
        n = len(texts)
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch = {k: v[start:end] for k, v in encodings.items()}
                outputs = self._model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0).astype(np.float64)

    def predict(
        self, X: NDArray[Any] | list[str]
    ) -> NDArray[np.int64]:
        """Return predicted class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Convenience factories for the experiment pipeline
# ---------------------------------------------------------------------------

def bert_base_factory(**kwargs: Any) -> TransformerClassifier:
    """Create a BERT-base classifier with default hyperparameters."""
    defaults = {"model_name": "bert-base-uncased"}
    defaults.update(kwargs)
    return TransformerClassifier(**defaults)


def mentalbert_factory(**kwargs: Any) -> TransformerClassifier:
    """Create a MentalBERT classifier with default hyperparameters."""
    defaults = {"model_name": "mental/mental-bert-base-uncased"}
    defaults.update(kwargs)
    return TransformerClassifier(**defaults)
