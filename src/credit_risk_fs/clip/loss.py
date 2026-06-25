from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ContrastiveLossOutput:
    loss: torch.Tensor
    text_to_statistical_loss: torch.Tensor
    statistical_to_text_loss: torch.Tensor
    masked_negative_count: int


def symmetric_masked_contrastive_loss(
    text_projection: torch.Tensor,
    statistical_projection: torch.Tensor,
    *,
    temperature: torch.Tensor,
    false_negative_mask: torch.Tensor | None = None,
) -> ContrastiveLossOutput:
    if text_projection.shape != statistical_projection.shape:
        raise ValueError("text and statistical projections must have the same shape")
    if text_projection.ndim != 2:
        raise ValueError("projection tensors must be 2-dimensional")
    batch_size = int(text_projection.shape[0])
    if batch_size < 2:
        raise ValueError("contrastive loss requires at least two rows")
    logits = text_projection @ statistical_projection.T / temperature.clamp_min(1e-8)
    masked_negative_count = 0
    if false_negative_mask is not None:
        if false_negative_mask.shape != (batch_size, batch_size):
            raise ValueError("false_negative_mask shape must match batch")
        mask = false_negative_mask.to(device=logits.device, dtype=torch.bool).clone()
        if not torch.equal(mask, mask.T):
            raise ValueError("false_negative_mask must be symmetric")
        mask.fill_diagonal_(False)
        valid_negatives = (~mask).sum(dim=1) - 1
        if torch.any(valid_negatives < 1):
            raise ValueError("false_negative_mask leaves a row with zero valid negatives")
        masked_negative_count = int(mask.sum().item())
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    labels = torch.arange(batch_size, device=logits.device)
    text_to_stat = F.cross_entropy(logits, labels)
    stat_to_text = F.cross_entropy(logits.T, labels)
    loss = (text_to_stat + stat_to_text) / 2.0
    return ContrastiveLossOutput(
        loss=loss,
        text_to_statistical_loss=text_to_stat,
        statistical_to_text_loss=stat_to_text,
        masked_negative_count=masked_negative_count,
    )
