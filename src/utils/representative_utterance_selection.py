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

def mmr_select(
    embeddings: np.ndarray,
    k: int,
    lambda_mult: float = 0.5,
) -> List[int]:
    """
    Maximal Marginal Relevance selection.

    Args:
        embeddings: (N, d) array of utterance embeddings
        k: number of representatives to select
        lambda_mult: trade-off between relevance & diversity

    Strategy:
        - Query = centroid of all embeddings
        - Relevance = cos_sim(embedding_i, query)
        - Diversity = max cos_sim(embedding_i, already_selected)

    Returns:
        List of selected indices into embeddings (length <= k)
    """
    n = embeddings.shape[0]
    if n == 0:
        return []
    if k >= n:
        return list(range(n))

    # query is centroid
    query = embeddings.mean(axis=0, keepdims=True)  # (1, d)
    sim_to_query = cosine_sim_matrix(embeddings, query).reshape(-1)  # (N,)

    selected: List[int] = []
    candidate_indices = list(range(n))

    # 1) pick the most relevant first
    first_idx = int(np.argmax(sim_to_query))
    selected.append(first_idx)
    candidate_indices.remove(first_idx)

    if k == 1:
        return selected

    # precompute pairwise cosine similarity between utterances
    pairwise_sim = cosine_sim_matrix(embeddings, embeddings)  # (N, N)

    while len(selected) < k and candidate_indices:
        # For each candidate, compute MMR score
        mmr_scores = []
        for idx in candidate_indices:
            # diversity term: max similarity to any already selected
            diversity = np.max(pairwise_sim[idx, selected])
            score = lambda_mult * sim_to_query[idx] - (1.0 - lambda_mult) * diversity
            mmr_scores.append((idx, score))

        # pick best candidate
        mmr_scores.sort(key=lambda x: x[1], reverse=True)
        best_idx = mmr_scores[0][0]
        selected.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected