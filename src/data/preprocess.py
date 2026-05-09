"""Build user/item indices, train/val/test split, persist as parquet."""
import json

import numpy as np
import pandas as pd

from src import config
from src.data.loader import load_movies, load_ratings


def build_indices(ratings: pd.DataFrame):
    unique_users = sorted(ratings["user_id"].unique())
    unique_items = sorted(ratings["movie_id"].unique())
    user_to_idx = {int(u): i for i, u in enumerate(unique_users)}
    item_to_idx = {int(m): i for i, m in enumerate(unique_items)}
    return user_to_idx, item_to_idx


def split_by_time(ratings: pd.DataFrame, val_frac: float, test_frac: float):
    """Time-based split: oldest = train, newest = test. Avoids leakage."""
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    n = len(ratings)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    train = ratings.iloc[:train_end].copy()
    val = ratings.iloc[train_end:val_end].copy()
    test = ratings.iloc[val_end:].copy()
    return train, val, test


def preprocess(save: bool = True):
    config.ensure_dirs()
    print("Loading ratings + movies")
    ratings = load_ratings()
    movies = load_movies()

    print(f"Raw ratings: {len(ratings):,}, users: {ratings['user_id'].nunique():,}, movies: {ratings['movie_id'].nunique():,}")

    user_to_idx, item_to_idx = build_indices(ratings)
    ratings["user_idx"] = ratings["user_id"].map(user_to_idx)
    ratings["item_idx"] = ratings["movie_id"].map(item_to_idx)

    movies = movies[movies["movie_id"].isin(item_to_idx)].copy()
    movies["item_idx"] = movies["movie_id"].map(item_to_idx)

    train, val, test = split_by_time(ratings, config.VAL_FRACTION, config.TEST_FRACTION)
    print(f"Split: train={len(train):,} val={len(val):,} test={len(test):,}")

    if save:
        ratings.to_parquet(config.RATINGS_PARQUET, index=False)
        movies.to_parquet(config.MOVIES_PARQUET, index=False)
        train.to_parquet(config.TRAIN_PARQUET, index=False)
        val.to_parquet(config.VAL_PARQUET, index=False)
        test.to_parquet(config.TEST_PARQUET, index=False)
        with open(config.USER_INDEX_PATH, "w") as f:
            json.dump(user_to_idx, f)
        with open(config.ITEM_INDEX_PATH, "w") as f:
            json.dump(item_to_idx, f)
        print(f"Saved to {config.PROCESSED_DIR}")

    return {
        "ratings": ratings,
        "movies": movies,
        "train": train,
        "val": val,
        "test": test,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
    }


def load_processed():
    train = pd.read_parquet(config.TRAIN_PARQUET)
    val = pd.read_parquet(config.VAL_PARQUET)
    test = pd.read_parquet(config.TEST_PARQUET)
    movies = pd.read_parquet(config.MOVIES_PARQUET)
    with open(config.USER_INDEX_PATH) as f:
        user_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(config.ITEM_INDEX_PATH) as f:
        item_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
    return {
        "train": train,
        "val": val,
        "test": test,
        "movies": movies,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
    }


if __name__ == "__main__":
    preprocess()
