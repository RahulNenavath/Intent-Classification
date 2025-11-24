import numpy as np
from typing import List

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / norm


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between two 2D arrays (n_a, d) and (n_b, d).
    Returns (n_a, n_b)
    """
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return np.matmul(a_norm, b_norm.T)

def k_center_greedy_select(
    embeddings: np.ndarray,
    k: int,
    query_embedding: np.ndarray | None = None,
) -> List[int]:
    """
    k-center greedy selection:
        Minimizes max distance to nearest selected point.

    Args:
        embeddings: (N, d)
        k: number to select
        query_embedding: optional (d,)
            If provided, the first representative will be the point
            closest to this query (e.g., intent-name + description embedding).
    """
    n = embeddings.shape[0]
    if n == 0:
        return []
    if k >= n:
        return list(range(n))

    # distances helper
    def euclid(a, b):
        return np.linalg.norm(a - b, axis=-1)

    # ---- Initialization step ----
    if query_embedding is not None:
        dists = euclid(embeddings, query_embedding[None, :])
        first = int(np.argmin(dists))
    else:
        first = 0

    selected = [first]
    # track distance of each pt to closest of selected
    min_dists = euclid(embeddings, embeddings[first:first+1, :])

    # ---- Greedy expansion ----
    for _ in range(1, k):
        next_idx = int(np.argmax(min_dists))     # farthest point
        selected.append(next_idx)

        new_dists = euclid(embeddings, embeddings[next_idx:next_idx+1, :])
        min_dists = np.minimum(min_dists, new_dists)

    return selected