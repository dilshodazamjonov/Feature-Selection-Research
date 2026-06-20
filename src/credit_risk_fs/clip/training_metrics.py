from __future__ import annotations

from typing import Any

import numpy as np
import torch

from credit_risk_fs.clip.loss import symmetric_masked_contrastive_loss


def retrieval_metrics(
    text_projection: torch.Tensor,
    statistical_projection: torch.Tensor,
    *,
    false_negative_mask: torch.Tensor,
    temperature: torch.Tensor,
    split: str,
) -> dict[str, float | str]:
    with torch.no_grad():
        loss = symmetric_masked_contrastive_loss(
            text_projection,
            statistical_projection,
            temperature=temperature,
            false_negative_mask=false_negative_mask,
        ).loss
        similarity = text_projection @ statistical_projection.T
        mask = false_negative_mask.to(device=similarity.device, dtype=torch.bool).clone()
        mask.fill_diagonal_(False)
        retrieval_scores = similarity.masked_fill(mask, torch.finfo(similarity.dtype).min)
        labels = torch.arange(similarity.shape[0], device=similarity.device)
        t2s = _direction_metrics(retrieval_scores, labels)
        s2t = _direction_metrics(retrieval_scores.T, labels)
        positives = similarity.diag()
        negative_values = similarity[~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device) & ~mask]
        negative_mean = negative_values.mean() if negative_values.numel() else torch.tensor(0.0, device=similarity.device)
        return {
            "split": split,
            "validation_contrastive_loss" if split == "validation" else "contrastive_loss": float(loss.item()),
            "positive_pair_cosine_mean": float(positives.mean().item()),
            "allowed_negative_cosine_mean": float(negative_mean.item()),
            "positive_minus_negative_margin": float((positives.mean() - negative_mean).item()),
            "text_to_statistical_recall_at_1": t2s["recall_at_1"],
            "text_to_statistical_recall_at_5": t2s["recall_at_5"],
            "text_to_statistical_recall_at_10": t2s["recall_at_10"],
            "statistical_to_text_recall_at_1": s2t["recall_at_1"],
            "statistical_to_text_recall_at_5": s2t["recall_at_5"],
            "statistical_to_text_recall_at_10": s2t["recall_at_10"],
            "text_to_statistical_mrr": t2s["mrr"],
            "statistical_to_text_mrr": s2t["mrr"],
            "mean_reciprocal_rank": float((t2s["mrr"] + s2t["mrr"]) / 2.0),
        }


def collapse_diagnostics(
    embeddings: torch.Tensor,
    *,
    thresholds: dict[str, float],
    label: str,
) -> dict[str, Any]:
    with torch.no_grad():
        values = embeddings.detach().cpu().numpy().astype(float)
    variance_mean = float(np.var(values, axis=0).mean())
    rounded_unique = int(np.unique(np.round(values, decimals=6), axis=0).shape[0])
    cosine = values @ values.T
    off_diag = cosine[~np.eye(cosine.shape[0], dtype=bool)]
    mean_pairwise = float(off_diag.mean()) if off_diag.size else 0.0
    std_pairwise = float(off_diag.std()) if off_diag.size else 0.0
    norm_error = float(np.abs(np.linalg.norm(values, axis=1) - 1.0).max())
    variance_min = float(thresholds.get("variance_min", 1e-6))
    unique_min_fraction = float(thresholds.get("unique_min_fraction", 0.25))
    pairwise_cosine_max = float(thresholds.get("mean_pairwise_cosine_max", 0.98))
    pairwise_std_min = float(thresholds.get("pairwise_cosine_std_min", 1e-4))
    norm_tolerance = float(thresholds.get("norm_tolerance", 1e-4))
    warnings = []
    if variance_mean < variance_min:
        warnings.append("near_zero_embedding_variance")
    if rounded_unique < max(2, int(values.shape[0] * unique_min_fraction)):
        warnings.append("too_many_identical_embeddings")
    if mean_pairwise > pairwise_cosine_max:
        warnings.append("mean_pairwise_cosine_too_high")
    if std_pairwise < pairwise_std_min:
        warnings.append("mean_pairwise_cosine_too_uniform")
    if norm_error > norm_tolerance:
        warnings.append("representation_norm_violation")
    return {
        "label": label,
        "embedding_count": int(values.shape[0]),
        "embedding_dimension": int(values.shape[1]) if values.ndim == 2 else 0,
        "variance_mean": variance_mean,
        "unique_embedding_count_rounded_6": rounded_unique,
        "mean_pairwise_cosine": mean_pairwise,
        "std_pairwise_cosine": std_pairwise,
        "max_norm_error": norm_error,
        "status": "pass" if not warnings else "warn",
        "warnings": warnings,
    }


def _direction_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    order = torch.argsort(scores, dim=1, descending=True)
    matches = order.eq(labels[:, None])
    ranks = matches.float().argmax(dim=1) + 1
    return {
        "recall_at_1": float((ranks <= 1).float().mean().item()),
        "recall_at_5": float((ranks <= min(5, scores.shape[1])).float().mean().item()),
        "recall_at_10": float((ranks <= min(10, scores.shape[1])).float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
    }
