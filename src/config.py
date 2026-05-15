"""Central configuration. Paths resolve relative to repo root, so it works locally and on Colab."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# MovieLens 1M
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_DIR = RAW_DIR / "ml-1m"
RATINGS_FILE = MOVIELENS_DIR / "ratings.dat"
MOVIES_FILE = MOVIELENS_DIR / "movies.dat"
USERS_FILE = MOVIELENS_DIR / "users.dat"

# Processed artifacts
RATINGS_PARQUET = PROCESSED_DIR / "ratings.parquet"
MOVIES_PARQUET = PROCESSED_DIR / "movies.parquet"
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
VAL_PARQUET = PROCESSED_DIR / "val.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
USER_INDEX_PATH = PROCESSED_DIR / "user_index.json"
ITEM_INDEX_PATH = PROCESSED_DIR / "item_index.json"

# Model artifacts
SVD_MODEL_PATH = MODELS_DIR / "svd.pt"
NCF_MODEL_PATH = MODELS_DIR / "ncf.pt"
CONTENT_MODEL_PATH = MODELS_DIR / "content.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Splits
RANDOM_SEED = 42
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

# Evaluation
TOP_K = 10
EVAL_USERS_SAMPLE = 1000  # sample users for ranking eval (full eval is too slow)

# SVD (PyTorch matrix factorization)
SVD_EMBEDDING_DIM = 50
SVD_LR = 5e-3
SVD_WEIGHT_DECAY = 1e-5
SVD_BATCH_SIZE = 4096
SVD_EPOCHS = 15

# NCF
NCF_EMBEDDING_DIM = 32
NCF_MLP_LAYERS = [64, 32, 16]
NCF_DROPOUT = 0.2
NCF_LR = 1e-3
NCF_WEIGHT_DECAY = 1e-5
NCF_BATCH_SIZE = 1024
NCF_EPOCHS = 5
NCF_NEG_SAMPLES = 4

# Content-based
CONTENT_MAX_FEATURES = 5000

# Hybrid
HYBRID_SVD_WEIGHT = 0.5
HYBRID_NCF_WEIGHT = 0.5

# W&B
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "netflix-recsys")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")  # None = personal account
WANDB_MODE = os.environ.get("WANDB_MODE", "online")  # "disabled" to skip

# Hugging Face Hub
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "NehaS98/netflix-recsys-models")
HF_TOKEN = os.environ.get("HF_TOKEN")

# API
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
API_BASE_URL = os.environ.get("API_BASE_URL", f"http://localhost:{API_PORT}")


def ensure_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
