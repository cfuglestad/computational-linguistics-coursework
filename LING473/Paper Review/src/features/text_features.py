"""Standard text feature extractors replicating Bayram et al. (2022).

Extracts unigram, bigram, combined n-gram, and stopwords-only feature
representations from text. All extractors return sparse matrices compatible
with scikit-learn classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


FeatureType = Literal["unigram", "bigram", "ngram", "stopwords"]

# NLTK English stopwords list (frozen here to avoid runtime download).
# Matches the set used by Bayram et al. via NLTK 3.x.
ENGLISH_STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
})


@dataclass
class TextFeatureExtractor:
    """Extract bag-of-words features from text documents.

    Supports four feature types matching Bayram et al. (2022) Table 2:
      - unigram: individual word frequencies
      - bigram: adjacent word-pair frequencies
      - ngram: unigrams + bigrams combined
      - stopwords: only stopword unigrams (function words)

    Parameters
    ----------
    feature_type : FeatureType
        Which feature representation to extract.
    use_tfidf : bool
        If True, apply TF-IDF weighting. Bayram et al. use raw counts,
        but TF-IDF is available for ablation experiments.
    max_features : int | None
        Maximum vocabulary size. None keeps all features.
    min_df : int
        Minimum document frequency for a term to be included.
    """

    feature_type: FeatureType = "unigram"
    use_tfidf: bool = False
    max_features: int | None = None
    min_df: int = 2

    def __post_init__(self) -> None:
        self._vectorizer: CountVectorizer | None = None
        self._tfidf: TfidfTransformer | None = None

    @property
    def _ngram_range(self) -> tuple[int, int]:
        if self.feature_type == "unigram":
            return (1, 1)
        if self.feature_type == "bigram":
            return (2, 2)
        if self.feature_type == "ngram":
            return (1, 2)
        if self.feature_type == "stopwords":
            return (1, 1)
        raise ValueError(f"Unknown feature type: {self.feature_type}")

    def _build_stopword_vocabulary(self, texts: list[str]) -> list[str]:
        """Build vocabulary containing only stopwords."""
        vocab: set[str] = set()
        for text in texts:
            tokens = text.lower().split()
            vocab.update(t for t in tokens if t in ENGLISH_STOPWORDS)
        return sorted(vocab)

    def fit(self, texts: list[str]) -> TextFeatureExtractor:
        """Learn vocabulary from training texts."""
        if self.feature_type == "stopwords":
            vocabulary = self._build_stopword_vocabulary(texts)
            self._vectorizer = CountVectorizer(
                vocabulary=vocabulary,
                lowercase=True,
            )
        else:
            self._vectorizer = CountVectorizer(
                ngram_range=self._ngram_range,
                max_features=self.max_features,
                min_df=self.min_df,
                lowercase=True,
            )
        self._vectorizer.fit(texts)

        if self.use_tfidf:
            X = self._vectorizer.transform(texts)
            self._tfidf = TfidfTransformer()
            self._tfidf.fit(X)

        return self

    def transform(self, texts: list[str]) -> NDArray[np.float64]:
        """Transform texts into feature matrix.

        Returns a dense numpy array (not sparse) for compatibility with
        all downstream classifiers including the MLP.
        """
        if self._vectorizer is None:
            raise RuntimeError("Call fit() before transform().")
        X = self._vectorizer.transform(texts)
        if self.use_tfidf and self._tfidf is not None:
            X = self._tfidf.transform(X)
        return X.toarray().astype(np.float64)

    def fit_transform(self, texts: list[str]) -> NDArray[np.float64]:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)

    @property
    def feature_names(self) -> list[str]:
        """Return learned feature names."""
        if self._vectorizer is None:
            raise RuntimeError("Call fit() before accessing feature_names.")
        return list(self._vectorizer.get_feature_names_out())

    @property
    def n_features(self) -> int:
        """Number of features in the learned vocabulary."""
        return len(self.feature_names)
