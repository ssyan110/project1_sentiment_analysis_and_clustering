"""Utility functions for text clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


def vectorize_texts(texts: list[str], *, n_components: int = 50, max_features: int = 4000) -> np.ndarray:
    """Convert a list of documents to a dense matrix using TF-IDF and SVD."""
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X)


def evaluate_models(X: np.ndarray, k_range: range = range(2, 7), random_state: int = 42) -> pd.DataFrame:
    """Run several clustering algorithms and return metric scores."""
    results = []
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(X_scaled)
        results.append(
            {
                "algorithm": "KMeans",
                "k": k,
                "silhouette": silhouette_score(X_scaled, labels),
                "calinski": calinski_harabasz_score(X_scaled, labels),
            }
        )

        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X_scaled)
        results.append(
            {
                "algorithm": "Agglomerative",
                "k": k,
                "silhouette": silhouette_score(X_scaled, labels),
                "calinski": calinski_harabasz_score(X_scaled, labels),
            }
        )

        gmm = GaussianMixture(n_components=k, random_state=random_state)
        labels = gmm.fit_predict(X_scaled)
        results.append(
            {
                "algorithm": "GaussianMixture",
                "k": k,
                "silhouette": silhouette_score(X_scaled, labels),
                "calinski": calinski_harabasz_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(results).sort_values("silhouette", ascending=False)


def cluster_texts(texts: list[str], k_range: range = range(2, 7)) -> pd.DataFrame:
    """High level helper: vectorize texts and evaluate clustering models."""
    X = vectorize_texts(texts)
    return evaluate_models(X, k_range=k_range)
