"""Weighted ensemble of SVD + NCF (and optionally content) scores."""
import numpy as np

from src import config


def normalize(scores: np.ndarray) -> np.ndarray:
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-9:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def hybrid_scores(
    svd_scores: np.ndarray,
    ncf_scores: np.ndarray,
    svd_weight: float = config.HYBRID_SVD_WEIGHT,
    ncf_weight: float = config.HYBRID_NCF_WEIGHT,
) -> np.ndarray:
    s = normalize(svd_scores)
    n = normalize(ncf_scores)
    return svd_weight * s + ncf_weight * n
