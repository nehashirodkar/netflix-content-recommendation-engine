"""Content-based recommender using TF-IDF over movie titles + genres."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src import config


class ContentModel:
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None  # sparse (n_items, n_features)
        self.item_indices: np.ndarray | None = None  # item_idx values aligned to tfidf rows

    def fit(self, movies_df):
        """movies_df must have item_idx, title, genres columns."""
        movies_df = movies_df.sort_values("item_idx").reset_index(drop=True)
        text = (movies_df["title"].fillna("") + " " + movies_df["genres"].fillna("")).tolist()
        self.vectorizer = TfidfVectorizer(
            max_features=config.CONTENT_MAX_FEATURES,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(text)
        self.item_indices = movies_df["item_idx"].to_numpy(dtype=np.int64)
        return self

    def similar_items(self, item_idx: int, top_k: int = 10) -> list[tuple[int, float]]:
        if item_idx not in set(self.item_indices.tolist()):
            return []
        row = int(np.where(self.item_indices == item_idx)[0][0])
        sims = linear_kernel(self.tfidf_matrix[row], self.tfidf_matrix).flatten()
        sims[row] = -np.inf
        top = np.argpartition(-sims, top_k)[:top_k]
        top = top[np.argsort(-sims[top])]
        return [(int(self.item_indices[i]), float(sims[i])) for i in top]

    def recommend_for_user(self, user_history_item_idxs: list[int], top_k: int = 10) -> list[tuple[int, float]]:
        """Score all items by mean similarity to the user's history."""
        if not user_history_item_idxs:
            return self.popular_items(top_k)
        rows = []
        index_map = {int(v): i for i, v in enumerate(self.item_indices)}
        for item_idx in user_history_item_idxs:
            if item_idx in index_map:
                rows.append(index_map[item_idx])
        if not rows:
            return self.popular_items(top_k)
        user_profile = self.tfidf_matrix[rows].mean(axis=0)
        user_profile = np.asarray(user_profile)
        sims = (self.tfidf_matrix @ user_profile.T).A.flatten()
        seen = set(rows)
        for r in seen:
            sims[r] = -np.inf
        top = np.argpartition(-sims, top_k)[:top_k]
        top = top[np.argsort(-sims[top])]
        return [(int(self.item_indices[i]), float(sims[i])) for i in top]

    def popular_items(self, top_k: int = 10) -> list[tuple[int, float]]:
        """Cold-start fallback — return first top_k items (no popularity signal here, kept simple)."""
        return [(int(idx), 0.0) for idx in self.item_indices[:top_k]]

    def score_for_user(self, user_history_item_idxs: list[int]) -> np.ndarray:
        """Return scores aligned to self.item_indices for hybrid ensembling."""
        if not user_history_item_idxs:
            return np.zeros(len(self.item_indices), dtype=np.float32)
        index_map = {int(v): i for i, v in enumerate(self.item_indices)}
        rows = [index_map[i] for i in user_history_item_idxs if i in index_map]
        if not rows:
            return np.zeros(len(self.item_indices), dtype=np.float32)
        user_profile = self.tfidf_matrix[rows].mean(axis=0)
        sims = (self.tfidf_matrix @ np.asarray(user_profile).T).A.flatten()
        return sims.astype(np.float32)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "item_indices": self.item_indices,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path):
        data = joblib.load(path)
        m = cls()
        m.vectorizer = data["vectorizer"]
        m.tfidf_matrix = data["tfidf_matrix"]
        m.item_indices = data["item_indices"]
        return m
