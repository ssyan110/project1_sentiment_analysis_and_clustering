# coding: utf-8
"""Standalone pipeline implementing the notebook workflow.

Steps implemented:
1. Load raw review data from Excel
2. Lexicon-based sentiment labelling
3. Train classical ML classifiers
4. Aggregate reviews per company and build LDA topic vectors
5. Compare clustering algorithms and save results
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def load_list(file: Path) -> list[str]:
    with open(file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def clean_text(text: str) -> str:
    return str(text).lower()


def count_matches(lex: list[str], text: str) -> tuple[int, list[str]]:
    hits = [w for w in lex if w in text]
    return len(hits), hits

def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    pos_words = load_list(DATA_DIR / "positive_VN.txt")
    neg_words = load_list(DATA_DIR / "negative_VN.txt")
    pos_emojis = load_list(DATA_DIR / "positive_emoji.txt")
    neg_emojis = load_list(DATA_DIR / "negative_emoji.txt")

    df = df.copy()
    df["clean_review"] = df["review"].fillna("").apply(clean_text)

    df[["positive_word_count", "positive_words"]] = df["clean_review"].apply(
        lambda s: pd.Series(count_matches(pos_words, s))
    )
    df[["negative_word_count", "negative_words"]] = df["clean_review"].apply(
        lambda s: pd.Series(count_matches(neg_words, s))
    )
    df[["positive_emoji_count", "positive_emoji_list"]] = df["review"].apply(
        lambda s: pd.Series(count_matches(pos_emojis, str(s)))
    )
    df[["negative_emoji_count", "negative_emoji_list"]] = df["review"].apply(
        lambda s: pd.Series(count_matches(neg_emojis, str(s)))
    )

    def lexicon_sent(row):
        pos = row.positive_word_count + row.positive_emoji_count
        neg = row.negative_word_count + row.negative_emoji_count
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    df["sentiment"] = df.apply(lexicon_sent, axis=1)
    df.to_csv(OUT_DIR / "clean_reviews.csv", index=False)
    return df

def build_features(df: pd.DataFrame):
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=5,
        sublinear_tf=True,
    )
    X_text = tfidf.fit_transform(df["clean_review"])
    extras = df[[
        "positive_word_count",
        "negative_word_count",
        "positive_emoji_count",
        "negative_emoji_count",
    ]].values
    from scipy.sparse import csr_matrix, hstack
    X = hstack([X_text, csr_matrix(extras)], format="csr")
    y = df["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}).values
    return X, y


def evaluate_models(X, y) -> pd.DataFrame:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(),
    }
    records = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        records.append({
            "model": name,
            "accuracy": accuracy_score(y_te, preds),
            "f1": f1_score(y_te, preds, average="macro"),
        })
    res_df = pd.DataFrame(records).sort_values("f1", ascending=False)
    res_df.to_csv(OUT_DIR / "model_results.csv", index=False)
    return res_df

def aggregate_by_company(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("id")
          .agg(
              review_cnt=("clean_review", "size"),
              pos_word_sum=("positive_word_count", "sum"),
              neg_word_sum=("negative_word_count", "sum"),
              pos_emoji_sum=("positive_emoji_count", "sum"),
              neg_emoji_sum=("negative_emoji_count", "sum"),
          )
    )
    docs = df.groupby("id")["clean_review"].apply(" ".join).rename("doc")
    company_df = stats.join(docs, how="left").reset_index()
    company_df["sent_score"] = (
        company_df.pos_word_sum + company_df.pos_emoji_sum
        - company_df.neg_word_sum - company_df.neg_emoji_sum
    )
    return company_df


def lda_topics(docs: list[str], n_topics: int = 3, top_n: int = 12):
    cv = CountVectorizer(ngram_range=(1, 2), min_df=7, max_df=0.95)
    X_cnt = cv.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_mat = lda.fit_transform(X_cnt)
    feat = cv.get_feature_names_out()
    topics = {
        f"topic_{t}": [feat[i] for i in comp.argsort()[-top_n:][::-1]]
        for t, comp in enumerate(lda.components_)
    }
    return topic_mat, topics, X_cnt, feat


def compare_clustering(topic_mat, k_range=range(2, 7)) -> pd.DataFrame:
    results = []
    for k in k_range:
        km_labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(topic_mat)
        results.append({
            "algorithm": "KMeans", "k": k,
            "silhouette": silhouette_score(topic_mat, km_labels),
        })
        agg_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(topic_mat)
        results.append({
            "algorithm": "Agglomerative", "k": k,
            "silhouette": silhouette_score(topic_mat, agg_labels),
        })
        gmm_labels = GaussianMixture(n_components=k, random_state=42).fit_predict(topic_mat)
        results.append({
            "algorithm": "GaussianMixture", "k": k,
            "silhouette": silhouette_score(topic_mat, gmm_labels),
        })
    return pd.DataFrame(results).sort_values("silhouette", ascending=False)


def final_clustering(company_df: pd.DataFrame, topic_mat, results: pd.DataFrame, X_cnt, feat, topics: dict[str, list[str]]):
    best = results.iloc[0]
    if best.algorithm == "KMeans":
        model = KMeans(n_clusters=int(best.k), n_init=10, random_state=42)
    elif best.algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=int(best.k), linkage="ward")
    else:
        model = GaussianMixture(n_components=int(best.k), random_state=42)
    labels = model.fit_predict(topic_mat)
    company_df["cluster_best"] = labels

    cluster_terms = {}
    for lab in sorted(set(labels)):
        mask = labels == lab
        counts = X_cnt[mask].sum(axis=0)
        idx = np.asarray(counts).ravel().argsort()[-10:][::-1]
        cluster_terms[f"cluster_{lab}"] = [feat[i] for i in idx]

    company_df.to_csv(OUT_DIR / "company_clusters_with_topics.csv", index=False)
    results.to_csv(OUT_DIR / "clustering_results.csv", index=False)
    with open(OUT_DIR / "lda_topics.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "cluster_terms.json", "w", encoding="utf-8") as f:
        json.dump(cluster_terms, f, ensure_ascii=False, indent=2)


def main():
    df = pd.read_excel(DATA_DIR / "Reviews.xlsx")
    df = preprocess_reviews(df)
    X, y = build_features(df)
    evaluate_models(X, y)

    company_df = aggregate_by_company(df)
    topic_mat, topics, X_cnt, feat = lda_topics(company_df["doc"].tolist())
    results = compare_clustering(topic_mat)
    final_clustering(company_df, topic_mat, results, X_cnt, feat, topics)


if __name__ == "__main__":
    main()
