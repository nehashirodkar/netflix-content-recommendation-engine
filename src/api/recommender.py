"""Inference layer: loads all three models + metadata, exposes recommendation methods."""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config
from src.models.content import ContentModel
from src.models.hybrid import hybrid_scores
from src.models.ncf import NCFModel
from src.models.svd import SVDModel


class Recommender:
    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
        self._svd: SVDModel | None = None
        self._ncf: NCFModel | None = None
        self._content: ContentModel | None = None
        self._movies: pd.DataFrame | None = None
        self._user_to_idx: dict[int, int] | None = None
        self._item_to_idx: dict[int, int] | None = None
        self._idx_to_item: dict[int, int] | None = None
        self._idx_to_title: dict[int, str] | None = None
        self._popular_items: list[int] | None = None
        self._metadata: dict | None = None

    def load(self):
        # Indices + movies
        with open(self.models_dir / "user_index.json") as f:
            self._user_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
        with open(self.models_dir / "item_index.json") as f:
            self._item_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
        self._idx_to_item = {v: k for k, v in self._item_to_idx.items()}
        self._movies = pd.read_parquet(self.models_dir / "movies.parquet")
        self._idx_to_title = dict(zip(self._movies["item_idx"], self._movies["title"]))

        # Models
        self._svd = SVDModel.load(self.models_dir / "svd.pt")
        self._ncf = NCFModel.load(self.models_dir / "ncf.pt")
        self._content = ContentModel.load(self.models_dir / "content.joblib")

        # Metadata
        meta_path = self.models_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}

        # Build a popularity prior for cold start (not stored, computed from movies count via item_idx 0..N)
        # As fallback we just use first N item indices; if movies.parquet has it, prefer that.
        self._popular_items = self._movies["item_idx"].head(50).astype(int).tolist()
        return self

    @property
    def n_users(self) -> int:
        return len(self._user_to_idx)

    @property
    def n_items(self) -> int:
        return len(self._item_to_idx)

    @property
    def metadata(self) -> dict:
        return self._metadata or {}

    def list_movies(self, query: str | None = None, limit: int = 20) -> list[dict]:
        df = self._movies
        if query:
            mask = df["title"].str.contains(query, case=False, na=False)
            df = df[mask]
        out = df.head(limit)[["movie_id", "item_idx", "title", "genres"]].to_dict(orient="records")
        return out

    def get_user_idx(self, user_id: int) -> int | None:
        return self._user_to_idx.get(int(user_id))

    def _format(self, ranked_idx: list[int], scores: np.ndarray | None = None) -> list[dict]:
        out = []
        for i, idx in enumerate(ranked_idx):
            out.append({
                "rank": i + 1,
                "item_idx": int(idx),
                "movie_id": int(self._idx_to_item.get(int(idx), -1)),
                "title": self._idx_to_title.get(int(idx), "Unknown"),
                "score": float(scores[idx]) if scores is not None else None,
            })
        return out

    def _topk(self, scores: np.ndarray, k: int, exclude: set[int] | None = None) -> list[int]:
        if exclude:
            scores = scores.copy()
            mask = np.fromiter(exclude, dtype=np.int64)
            mask = mask[mask < len(scores)]
            scores[mask] = -np.inf
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, k)[:k]
            order = top[np.argsort(-scores[top])]
        return [int(i) for i in order[:k]]

    def recommend(
        self,
        model: str = "hybrid",
        user_id: int | None = None,
        history_movie_ids: list[int] | None = None,
        k: int = 10,
    ) -> dict:
        """
        model: 'svd' | 'ncf' | 'content' | 'hybrid' | 'auto'
        Returns dict with 'model_used', 'recommendations', 'cold_start' flag.
        """
        history_idx = []
        if history_movie_ids:
            history_idx = [self._item_to_idx[m] for m in history_movie_ids if m in self._item_to_idx]

        user_idx = self.get_user_idx(user_id) if user_id is not None else None
        cold_start = user_idx is None

        # Cold-start fallback to content if user is unknown
        effective_model = model
        if effective_model == "auto":
            effective_model = "content" if cold_start else "hybrid"
        if cold_start and effective_model in ("svd", "ncf", "hybrid"):
            effective_model = "content"

        if effective_model == "content":
            seed_history = history_idx
            if not seed_history and not cold_start:
                # Use top movies from user's known history is unavailable here without train; just popular
                pass
            recs = self._content.recommend_for_user(seed_history, top_k=k)
            ranked = [r[0] for r in recs] or self._popular_items[:k]
            return {
                "model_used": "content",
                "cold_start": cold_start,
                "recommendations": [
                    {
                        "rank": i + 1,
                        "item_idx": int(idx),
                        "movie_id": int(self._idx_to_item.get(int(idx), -1)),
                        "title": self._idx_to_title.get(int(idx), "Unknown"),
                        "score": None,
                    }
                    for i, idx in enumerate(ranked[:k])
                ],
            }

        seen = set(history_idx)
        if effective_model == "svd":
            scores = self._svd.score_all_items(user_idx)
        elif effective_model == "ncf":
            scores = self._ncf.score_all_items(user_idx)
        elif effective_model == "hybrid":
            svd_s = self._svd.score_all_items(user_idx)
            ncf_s = self._ncf.score_all_items(user_idx)
            scores = hybrid_scores(svd_s, ncf_s)
        else:
            raise ValueError(f"Unknown model: {model}")

        top = self._topk(scores, k, exclude=seen)
        return {
            "model_used": effective_model,
            "cold_start": cold_start,
            "recommendations": self._format(top, scores),
        }

    def compare(self, user_id: int | None = None, history_movie_ids: list[int] | None = None, k: int = 10) -> dict:
        out = {}
        for m in ("svd", "ncf", "content", "hybrid"):
            try:
                out[m] = self.recommend(model=m, user_id=user_id, history_movie_ids=history_movie_ids, k=k)
            except Exception as e:
                out[m] = {"error": str(e)}
        return out


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    """Singleton — loads from MODELS_DIR; on Spaces, models are pulled to MODELS_DIR at startup."""
    rec = Recommender()
    rec.load()
    return rec
