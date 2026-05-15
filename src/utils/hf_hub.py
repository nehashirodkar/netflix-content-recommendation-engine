"""Push trained models to Hugging Face Hub and pull them back at serving time."""
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from src import config


def _api() -> HfApi:
    return HfApi(token=config.HF_TOKEN or os.environ.get("HF_TOKEN"))


def ensure_repo(repo_id: str | None = None) -> str:
    repo_id = repo_id or config.HF_MODEL_REPO
    api = _api()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    return repo_id


def push_models(repo_id: str | None = None, commit_message: str = "Update models"):
    repo_id = ensure_repo(repo_id)
    api = _api()
    # All files should live in MODELS_DIR by now (train_all copies indices/movies in)
    files = [
        config.SVD_MODEL_PATH,
        config.NCF_MODEL_PATH,
        config.CONTENT_MODEL_PATH,
        config.METADATA_PATH,
        config.MODELS_DIR / "user_index.json",
        config.MODELS_DIR / "item_index.json",
        config.MODELS_DIR / "movies.parquet",
    ]
    for f in files:
        if not f.exists():
            print(f"Skipping (not found): {f}")
            continue
        print(f"Uploading {f.name}")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
    print(f"Pushed to https://huggingface.co/{repo_id}")


def pull_models(repo_id: str | None = None, local_dir: Path | None = None) -> Path:
    repo_id = repo_id or config.HF_MODEL_REPO
    local_dir = Path(local_dir) if local_dir else config.MODELS_DIR
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        token=config.HF_TOKEN or os.environ.get("HF_TOKEN"),
    )
    print(f"Pulled {repo_id} -> {local_dir}")
    return local_dir
