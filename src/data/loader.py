"""Download and load MovieLens 1M into pandas DataFrames."""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src import config


def download_movielens(force: bool = False) -> Path:
    config.ensure_dirs()
    if config.MOVIELENS_DIR.exists() and not force:
        return config.MOVIELENS_DIR
    print(f"Downloading MovieLens 1M from {config.MOVIELENS_URL}")
    resp = requests.get(config.MOVIELENS_URL, stream=True, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(config.RAW_DIR)
    print(f"Extracted to {config.MOVIELENS_DIR}")
    return config.MOVIELENS_DIR


def load_ratings() -> pd.DataFrame:
    if not config.RATINGS_FILE.exists():
        download_movielens()
    df = pd.read_csv(
        config.RATINGS_FILE,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df


def load_movies() -> pd.DataFrame:
    if not config.MOVIES_FILE.exists():
        download_movielens()
    df = pd.read_csv(
        config.MOVIES_FILE,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    df["genres"] = df["genres"].fillna("").str.replace("|", " ", regex=False)
    return df


def load_users() -> pd.DataFrame:
    if not config.USERS_FILE.exists():
        download_movielens()
    df = pd.read_csv(
        config.USERS_FILE,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    return df
