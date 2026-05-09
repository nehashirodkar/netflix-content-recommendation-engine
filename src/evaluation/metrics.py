"""Ranking + rating metrics: Precision@K, Recall@K, NDCG@K, RMSE, MAE."""
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not recommended:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def dcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = dcg_at_k(recommended, relevant, k)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    score_fn,
    test_df,
    train_user_items: dict[int, set[int]],
    n_items: int,
    k: int = 10,
    user_sample: int | None = None,
    rating_threshold: float = 4.0,
    seed: int = 42,
) -> dict:
    """
    score_fn(user_idx) -> np.ndarray of length n_items (higher = better)
    Relevant items are test ratings >= rating_threshold.
    Items already in train are excluded from candidates.
    """
    rng = np.random.default_rng(seed)
    user_to_relevant: dict[int, set[int]] = {}
    for u, i, r in zip(
        test_df["user_idx"].to_numpy(),
        test_df["item_idx"].to_numpy(),
        test_df["rating"].to_numpy(),
    ):
        if r >= rating_threshold:
            user_to_relevant.setdefault(int(u), set()).add(int(i))

    eligible_users = [u for u, items in user_to_relevant.items() if len(items) > 0]
    if user_sample is not None and len(eligible_users) > user_sample:
        eligible_users = rng.choice(eligible_users, size=user_sample, replace=False).tolist()

    precisions, recalls, ndcgs = [], [], []
    for u in eligible_users:
        scores = score_fn(int(u))
        seen = train_user_items.get(int(u), set())
        if seen:
            scores = scores.copy()
            seen_arr = np.fromiter(seen, dtype=np.int64)
            seen_arr = seen_arr[seen_arr < n_items]
            scores[seen_arr] = -np.inf
        top_idx = np.argpartition(-scores, k)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])].tolist()
        relevant = user_to_relevant[int(u)]
        precisions.append(precision_at_k(top_idx, relevant, k))
        recalls.append(recall_at_k(top_idx, relevant, k))
        ndcgs.append(ndcg_at_k(top_idx, relevant, k))

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_evaluated": len(precisions),
    }
