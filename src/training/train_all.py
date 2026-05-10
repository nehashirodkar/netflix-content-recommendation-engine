"""End-to-end training: preprocess -> train SVD/NCF/content -> evaluate -> save -> (optional) push to HF Hub."""
import argparse
import json
import os
import time

import numpy as np
import torch

from src import config
from src.data.preprocess import load_processed, preprocess
from src.evaluation.metrics import evaluate_ranking, rmse
from src.models.content import ContentModel
from src.models.hybrid import hybrid_scores, normalize
from src.models.ncf import build_user_pos_items, train_ncf, NCFModel
from src.models.svd import SVDModel, train_svd


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_wandb(disabled: bool):
    if disabled or config.WANDB_MODE == "disabled":
        return None
    try:
        import wandb
        run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            mode=config.WANDB_MODE,
            config={
                "svd_embedding_dim": config.SVD_EMBEDDING_DIM,
                "svd_epochs": config.SVD_EPOCHS,
                "ncf_embedding_dim": config.NCF_EMBEDDING_DIM,
                "ncf_mlp_layers": config.NCF_MLP_LAYERS,
                "ncf_epochs": config.NCF_EPOCHS,
                "ncf_neg_samples": config.NCF_NEG_SAMPLES,
                "top_k": config.TOP_K,
            },
        )
        return run
    except Exception as e:
        print(f"W&B init failed, continuing without it: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preprocess", action="store_true", help="Reuse existing processed parquets")
    parser.add_argument("--skip-train", action="store_true", help="Load existing model checkpoints instead of training")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--push-hf", action="store_true", help="Push trained models to Hugging Face Hub")
    parser.add_argument("--eval-users", type=int, default=config.EVAL_USERS_SAMPLE)
    args = parser.parse_args()

    config.ensure_dirs()
    device = get_device()
    print(f"Device: {device}")

    if not args.skip_preprocess or not config.TRAIN_PARQUET.exists():
        preprocess()
    data = load_processed()
    train, val, test = data["train"], data["val"], data["test"]
    movies = data["movies"]
    user_to_idx = data["user_to_idx"]
    item_to_idx = data["item_to_idx"]
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    print(f"n_users={n_users:,} n_items={n_items:,}")

    wandb_run = init_wandb(args.no_wandb)

    if args.skip_train and config.SVD_MODEL_PATH.exists():
        print("\n=== Loading existing SVD checkpoint ===")
        svd_model = SVDModel.load(config.SVD_MODEL_PATH).to(device)
        svd_time = 0.0
    else:
        print("\n=== Training SVD ===")
        t0 = time.time()
        svd_model = train_svd(train, val, n_users, n_items, device=device, wandb_run=wandb_run)
        svd_time = time.time() - t0
        svd_model.save(config.SVD_MODEL_PATH)
        print(f"SVD saved to {config.SVD_MODEL_PATH} ({svd_time:.1f}s)")

    if args.skip_train and config.NCF_MODEL_PATH.exists():
        print("\n=== Loading existing NCF checkpoint ===")
        ncf_model = NCFModel.load(config.NCF_MODEL_PATH).to(device)
        ncf_time = 0.0
    else:
        print("\n=== Training NCF ===")
        t0 = time.time()
        ncf_model = train_ncf(train, val, n_users, n_items, device=device, wandb_run=wandb_run)
        ncf_time = time.time() - t0
        ncf_model.save(config.NCF_MODEL_PATH)
        print(f"NCF saved to {config.NCF_MODEL_PATH} ({ncf_time:.1f}s)")

    if args.skip_train and config.CONTENT_MODEL_PATH.exists():
        print("\n=== Loading existing Content checkpoint ===")
        content_model = ContentModel.load(config.CONTENT_MODEL_PATH)
        content_time = 0.0
    else:
        print("\n=== Fitting Content (TF-IDF) ===")
        t0 = time.time()
        content_model = ContentModel().fit(movies)
        content_time = time.time() - t0
        content_model.save(config.CONTENT_MODEL_PATH)
        print(f"Content saved to {config.CONTENT_MODEL_PATH} ({content_time:.1f}s)")

    # Evaluate — RMSE on test for SVD; ranking metrics for SVD/NCF/Content/Hybrid
    print("\n=== Evaluating ===")
    train_user_items = build_user_pos_items(train)
    user_history = {u: list(s) for u, s in train_user_items.items()}

    # RMSE (SVD on rating prediction)
    test_users = torch.tensor(test["user_idx"].to_numpy(dtype=np.int64), device=device)
    test_items = torch.tensor(test["item_idx"].to_numpy(dtype=np.int64), device=device)
    test_ratings = test["rating"].to_numpy(dtype=np.float32)
    svd_model.eval()
    with torch.no_grad():
        svd_test_pred = svd_model(test_users, test_items).cpu().numpy()
    svd_rmse = rmse(test_ratings, svd_test_pred)
    print(f"SVD test RMSE: {svd_rmse:.4f}")

    # Score functions for ranking eval
    def svd_score_fn(u: int) -> np.ndarray:
        return svd_model.score_all_items(u)

    def ncf_score_fn(u: int) -> np.ndarray:
        return ncf_model.score_all_items(u)

    # Align content scores to item_idx order
    item_idx_to_content_pos = {int(v): i for i, v in enumerate(content_model.item_indices)}
    content_pos = np.array([item_idx_to_content_pos.get(i, -1) for i in range(n_items)])

    def content_score_fn(u: int) -> np.ndarray:
        history = user_history.get(int(u), [])
        scores_aligned_to_content = content_model.score_for_user(history)
        out = np.zeros(n_items, dtype=np.float32)
        valid = content_pos >= 0
        out[valid] = scores_aligned_to_content[content_pos[valid]]
        return out

    def hybrid_score_fn(u: int) -> np.ndarray:
        return hybrid_scores(svd_score_fn(u), ncf_score_fn(u))

    k = config.TOP_K
    print(f"Ranking eval on {args.eval_users} sampled users, K={k}")
    svd_metrics = evaluate_ranking(svd_score_fn, test, train_user_items, n_items, k=k, user_sample=args.eval_users)
    ncf_metrics = evaluate_ranking(ncf_score_fn, test, train_user_items, n_items, k=k, user_sample=args.eval_users)
    content_metrics = evaluate_ranking(content_score_fn, test, train_user_items, n_items, k=k, user_sample=args.eval_users)
    hybrid_metrics = evaluate_ranking(hybrid_score_fn, test, train_user_items, n_items, k=k, user_sample=args.eval_users)

    print(f"SVD: {svd_metrics}")
    print(f"NCF: {ncf_metrics}")
    print(f"Content: {content_metrics}")
    print(f"Hybrid: {hybrid_metrics}")

    metadata = {
        "n_users": n_users,
        "n_items": n_items,
        "k": k,
        "svd_test_rmse": svd_rmse,
        "svd_train_seconds": svd_time,
        "ncf_train_seconds": ncf_time,
        "content_train_seconds": content_time,
        "metrics": {
            "svd": svd_metrics,
            "ncf": ncf_metrics,
            "content": content_metrics,
            "hybrid": hybrid_metrics,
        },
        "config": {
            "svd_embedding_dim": config.SVD_EMBEDDING_DIM,
            "svd_epochs": config.SVD_EPOCHS,
            "ncf_embedding_dim": config.NCF_EMBEDDING_DIM,
            "ncf_mlp_layers": config.NCF_MLP_LAYERS,
            "ncf_epochs": config.NCF_EPOCHS,
            "ncf_neg_samples": config.NCF_NEG_SAMPLES,
            "hybrid_svd_weight": config.HYBRID_SVD_WEIGHT,
            "hybrid_ncf_weight": config.HYBRID_NCF_WEIGHT,
        },
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {config.METADATA_PATH}")

    if wandb_run is not None:
        wandb_run.log({
            "test/svd_rmse": svd_rmse,
            **{f"test/svd_{k}": v for k, v in svd_metrics.items()},
            **{f"test/ncf_{k}": v for k, v in ncf_metrics.items()},
            **{f"test/content_{k}": v for k, v in content_metrics.items()},
            **{f"test/hybrid_{k}": v for k, v in hybrid_metrics.items()},
        })
        wandb_run.finish()

    if args.push_hf:
        from src.utils.hf_hub import push_models
        push_models(commit_message=f"Trained models — hybrid ndcg@{k}={hybrid_metrics[f'ndcg@{k}']:.4f}")


if __name__ == "__main__":
    main()
