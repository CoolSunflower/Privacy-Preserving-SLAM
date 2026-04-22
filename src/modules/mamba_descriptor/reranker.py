"""
MNN Reranking Module for Loop Closure Detection
Stage 2: Local Feature Reranking using Mutual Nearest Neighbors

This module provides reranking functionality for loop closure candidates
using MambaVision Stage 3 V-features and entropy-based keypoint selection.

Reference:
- EffoVPR (Tzachor et al., 2024) for MNN scoring
- Novel entropy-based keypoint selection for MambaVision (no CLS token)
"""

import numpy as np
from typing import List, Tuple

def select_keypoints_entropy(
    attention_matrix: np.ndarray,
    threshold_t1: float = 0.3,
    min_keypoints: int = 10,
) -> np.ndarray:
    """
    Select keypoints with low attention entropy (discriminative patches).
    Used for MambaVision which has no CLS token.

    Low entropy = focused attention = discriminative patch.
    High entropy = diffuse attention = generic patch.

    Args:
        attention_matrix: [num_patches, num_patches] attention weights (softmaxed)
        threshold_t1: normalized entropy threshold (select below this)
        min_keypoints: fallback minimum number of keypoints

    Returns:
        mask: [num_patches] boolean mask of selected keypoints
    """
    # Compute per-position entropy
    entropy = -(attention_matrix * np.log(attention_matrix + 1e-8)).sum(axis=-1)  # [N]

    # Normalize to [0, 1]
    e_min, e_max = entropy.min(), entropy.max()
    if e_max - e_min > 1e-8:
        entropy_norm = (entropy - e_min) / (e_max - e_min)
    else:
        entropy_norm = np.zeros_like(entropy)

    # Low entropy = discriminative
    mask = entropy_norm < threshold_t1
    if mask.sum() < min_keypoints:
        topk_idx = np.argsort(entropy_norm)[:min_keypoints]
        mask = np.zeros_like(entropy_norm, dtype=bool)
        mask[topk_idx] = True
    return mask


def compute_mnn_scores(
    query_features: np.ndarray,
    candidate_features: List[np.ndarray],
    threshold_t2: float = 0.05
) -> np.ndarray:
    """
    Compute Mutual Nearest Neighbor (MNN) count for each candidate.
    Implements Eq. 2 from EffoVPR (Tzachor et al., 2024).

    Args:
        query_features: [num_kp_q, feat_dim], L2-normalized
        candidate_features: list of [num_kp_ci, feat_dim], L2-normalized
        threshold_t2: minimum cosine similarity for MNN pair

    Returns:
        scores: [num_candidates] MNN counts per candidate
    """
    scores = np.zeros(len(candidate_features), dtype=np.float32)
    if len(query_features) == 0:
        return scores

    q_idxs = np.arange(len(query_features))

    for i, cand_feat in enumerate(candidate_features):
        if len(cand_feat) == 0:
            continue
        # Cosine similarity matrix (features are L2-normalized)
        sim = query_features @ cand_feat.T          # [num_kp_q, num_kp_c]
        nn_q2c = np.argmax(sim, axis=1)             # [num_kp_q]
        nn_c2q = np.argmax(sim, axis=0)             # [num_kp_c]
        # Vectorised MNN check: q->c->q cycle + similarity threshold
        is_mutual = (nn_c2q[nn_q2c] == q_idxs)
        above_thresh = (sim[q_idxs, nn_q2c] > threshold_t2)
        scores[i] = float(np.sum(is_mutual & above_thresh))

    return scores


def blend_with_global(
    mnn_scores: np.ndarray,
    global_scores: np.ndarray,
    global_weight: float = 0.5,
) -> np.ndarray:
    """
    Blend normalized MNN counts with normalized global descriptor similarity.

    When MNN produces no matches (all zeros) - which happens under extreme
    appearance change such as day/night or seasonal changes - falls back entirely
    to the global score, preserving stage-1 ranking. When MNN does produce
    matches, their normalized contribution is weighted against the global
    similarity according to `global_weight`.

    Args:
        mnn_scores:    [num_candidates] raw MNN counts (>= 0)
        global_scores: [num_candidates] cosine similarities from FAISS (stage-1)
        global_weight: weight in [0, 1] assigned to the global score component.
                       Higher = more trust in global retrieval, less in local MNN.

    Returns:
        combined: [num_candidates] blended scores, higher is better.
    """
    # Normalise global scores to [0, 1]
    g = global_scores.astype(np.float32)
    g_norm = (g - g.min()) / (g.max() - g.min() + 1e-8)

    mnn_max = float(mnn_scores.max())
    if mnn_max == 0.0:
        # No local matches at all -> trust global ranking entirely
        return g_norm

    mnn_norm = mnn_scores.astype(np.float32) / mnn_max
    return global_weight * g_norm + (1.0 - global_weight) * mnn_norm


def rerank_candidates(
    query_local: np.ndarray,
    candidate_locals: List[np.ndarray],
    global_scores: np.ndarray,
    threshold_t2: float = 0.05,
    global_weight: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full reranking pipeline: MNN scoring + global blending.

    Args:
        query_local: [num_kp_q, feat_dim] query local features
        candidate_locals: list of [num_kp_ci, feat_dim] candidate local features
        global_scores: [num_candidates] FAISS similarity scores
        threshold_t2: MNN similarity threshold
        global_weight: blend weight for global vs MNN scores

    Returns:
        blended_scores: [num_candidates] final reranked scores
        rerank_order: [num_candidates] indices sorted by blended score (descending)
    """
    mnn_scores = compute_mnn_scores(query_local, candidate_locals, threshold_t2)
    blended_scores = blend_with_global(mnn_scores, global_scores, global_weight)
    rerank_order = np.argsort(-blended_scores)  # Descending order
    return blended_scores, rerank_order
