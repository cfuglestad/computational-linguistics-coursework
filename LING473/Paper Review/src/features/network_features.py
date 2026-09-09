"""Lexical associative network features replicating Algorithm 1 of Bayram et al. (2022).

Builds class-specific word co-occurrence networks and computes edge-weight
differences as features. The intuition: word pairs that co-occur more strongly
in suicidal text than control text (or vice versa) carry discriminative signal
that raw frequency features miss.

Algorithm 1 summary (from the paper):
  1. For each class c in {suicidal, control}:
     a. Build a word co-occurrence graph G_c where nodes are words and edges
        connect words that appear within a sliding window in the same document.
     b. Weight each edge by the Pearson correlation coefficient between the
        two words' occurrence vectors across all documents of class c.
  2. Compute the shared vocabulary V = nodes(G_suicidal) ∩ nodes(G_control).
  3. For each word pair (w_i, w_j) where both appear in V:
     feature(w_i, w_j) = weight_suicidal(w_i, w_j) - weight_control(w_i, w_j)
     (Use 0 if the edge doesn't exist in one graph.)
  4. The feature vector for a new document is the subset of these edge-weight
     differences for word pairs that appear in the document.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from numpy.typing import NDArray


@dataclass
class LexicalNetworkFeatureExtractor:
    """Extract features from class-specific lexical co-occurrence networks.

    Parameters
    ----------
    window_size : int
        Sliding window size for co-occurrence counting. Bayram et al. use 5.
    min_word_freq : int
        Minimum corpus frequency for a word to be included as a node.
    min_edge_freq : int
        Minimum co-occurrence count for an edge to be retained.
    correlation_method : str
        Edge weighting method. 'pearson' matches the paper.
    """

    window_size: int = 5
    min_word_freq: int = 3
    min_edge_freq: int = 2
    correlation_method: str = "pearson"

    # Learned state
    _graph_positive: nx.Graph = field(default_factory=nx.Graph, repr=False)
    _graph_negative: nx.Graph = field(default_factory=nx.Graph, repr=False)
    _shared_edges: list[tuple[str, str]] = field(default_factory=list, repr=False)
    _edge_weight_diffs: dict[tuple[str, str], float] = field(
        default_factory=dict, repr=False
    )
    _fitted: bool = field(default=False, repr=False)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing.

        Bayram et al. use NLTK word_tokenize. We match the lowercasing
        behavior but keep tokenization simple for reproducibility.
        """
        return text.lower().split()

    def _build_cooccurrence_graph(
        self, texts: list[str]
    ) -> nx.Graph:
        """Build a weighted co-occurrence graph from a set of texts.

        Step 1a-b of Algorithm 1: count co-occurrences within sliding windows,
        then weight edges by correlation.
        """
        # Count word frequencies and co-occurrences
        word_freq: dict[str, int] = {}
        cooccurrence: dict[tuple[str, str], int] = {}
        doc_word_vectors: dict[str, list[int]] = {}  # word -> [0/1 per doc]

        for doc_idx, text in enumerate(texts):
            tokens = self._tokenize(text)
            doc_words: set[str] = set()

            # Word frequencies
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
                doc_words.add(token)

            # Sliding window co-occurrences
            for i, token_i in enumerate(tokens):
                window_end = min(i + self.window_size, len(tokens))
                for j in range(i + 1, window_end):
                    token_j = tokens[j]
                    if token_i != token_j:
                        pair = tuple(sorted([token_i, token_j]))
                        cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

            # Track document presence for correlation
            for word in doc_words:
                if word not in doc_word_vectors:
                    doc_word_vectors[word] = [0] * doc_idx
                doc_word_vectors[word].append(1)

            # Pad all vectors to current length
            for word in doc_word_vectors:
                if len(doc_word_vectors[word]) <= doc_idx:
                    doc_word_vectors[word].append(0)

        n_docs = len(texts)
        # Pad all vectors to final length
        for word in doc_word_vectors:
            while len(doc_word_vectors[word]) < n_docs:
                doc_word_vectors[word].append(0)

        # Filter by frequency
        valid_words = {
            w for w, freq in word_freq.items() if freq >= self.min_word_freq
        }

        # Build graph with correlation-weighted edges
        G = nx.Graph()
        G.add_nodes_from(valid_words)

        for (w1, w2), count in cooccurrence.items():
            if (
                w1 in valid_words
                and w2 in valid_words
                and count >= self.min_edge_freq
            ):
                # Compute Pearson correlation between document vectors
                vec1 = np.array(doc_word_vectors.get(w1, [0] * n_docs), dtype=np.float64)
                vec2 = np.array(doc_word_vectors.get(w2, [0] * n_docs), dtype=np.float64)

                std1 = np.std(vec1)
                std2 = np.std(vec2)

                if std1 > 0 and std2 > 0:
                    correlation = float(np.corrcoef(vec1, vec2)[0, 1])
                else:
                    correlation = 0.0

                if not np.isnan(correlation):
                    G.add_edge(w1, w2, weight=correlation)

        return G

    def fit(
        self,
        texts_positive: list[str],
        texts_negative: list[str],
    ) -> LexicalNetworkFeatureExtractor:
        """Build class-specific networks and compute edge-weight differences.

        Parameters
        ----------
        texts_positive : list[str]
            Texts from the positive class (suicidal).
        texts_negative : list[str]
            Texts from the negative class (control).
        """
        self._graph_positive = self._build_cooccurrence_graph(texts_positive)
        self._graph_negative = self._build_cooccurrence_graph(texts_negative)

        # Shared vocabulary (Algorithm 1, Step 2)
        shared_nodes = (
            set(self._graph_positive.nodes()) & set(self._graph_negative.nodes())
        )

        # Compute edge-weight differences (Algorithm 1, Step 3)
        # Collect all edges where both endpoints are in shared vocabulary
        pos_edges = {
            tuple(sorted([u, v])): d["weight"]
            for u, v, d in self._graph_positive.edges(data=True)
            if u in shared_nodes and v in shared_nodes
        }
        neg_edges = {
            tuple(sorted([u, v])): d["weight"]
            for u, v, d in self._graph_negative.edges(data=True)
            if u in shared_nodes and v in shared_nodes
        }

        all_edge_keys = set(pos_edges.keys()) | set(neg_edges.keys())

        self._edge_weight_diffs = {}
        for edge in all_edge_keys:
            w_pos = pos_edges.get(edge, 0.0)
            w_neg = neg_edges.get(edge, 0.0)
            self._edge_weight_diffs[edge] = w_pos - w_neg

        # Sort edges for consistent feature ordering
        self._shared_edges = sorted(self._edge_weight_diffs.keys())
        self._fitted = True

        return self

    def transform(self, texts: list[str]) -> NDArray[np.float64]:
        """Transform texts into network feature vectors.

        For each document, the feature vector contains the edge-weight
        difference for each shared edge whose both words appear in the
        document. Edges whose words are absent get value 0.

        Parameters
        ----------
        texts : list[str]
            Documents to transform.

        Returns
        -------
        NDArray of shape (n_documents, n_shared_edges)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        n_docs = len(texts)
        n_features = len(self._shared_edges)
        X = np.zeros((n_docs, n_features), dtype=np.float64)

        for doc_idx, text in enumerate(texts):
            doc_words = set(self._tokenize(text))
            for feat_idx, (w1, w2) in enumerate(self._shared_edges):
                if w1 in doc_words and w2 in doc_words:
                    X[doc_idx, feat_idx] = self._edge_weight_diffs[(w1, w2)]

        return X

    def fit_transform(
        self,
        texts_positive: list[str],
        texts_negative: list[str],
        all_texts: list[str] | None = None,
    ) -> NDArray[np.float64]:
        """Fit on class-separated texts, then transform all texts.

        Parameters
        ----------
        texts_positive : list[str]
            Positive class training texts.
        texts_negative : list[str]
            Negative class training texts.
        all_texts : list[str] | None
            Texts to transform. If None, concatenates positive + negative.
        """
        self.fit(texts_positive, texts_negative)
        if all_texts is None:
            all_texts = texts_positive + texts_negative
        return self.transform(all_texts)

    @property
    def feature_names(self) -> list[str]:
        """Return edge labels as feature names (e.g. 'word1--word2')."""
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing feature_names.")
        return [f"{w1}--{w2}" for w1, w2 in self._shared_edges]

    @property
    def n_features(self) -> int:
        """Number of network features."""
        return len(self._shared_edges)

    @property
    def graph_positive(self) -> nx.Graph:
        """The positive-class co-occurrence graph (for visualization)."""
        return self._graph_positive

    @property
    def graph_negative(self) -> nx.Graph:
        """The negative-class co-occurrence graph (for visualization)."""
        return self._graph_negative
