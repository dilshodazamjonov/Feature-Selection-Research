from __future__ import annotations

import torch

from credit_risk_fs.clip.loss import symmetric_masked_contrastive_loss


def test_symmetric_loss_uses_both_directions_and_masks_false_negatives():
    text = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]]), dim=1)
    stat = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]]), dim=1)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0, 1] = True
    mask[1, 0] = True

    unmasked = symmetric_masked_contrastive_loss(text, stat, temperature=torch.tensor(0.07))
    masked = symmetric_masked_contrastive_loss(text, stat, temperature=torch.tensor(0.07), false_negative_mask=mask)

    assert unmasked.text_to_statistical_loss.item() > 0
    assert unmasked.statistical_to_text_loss.item() > 0
    assert masked.masked_negative_count == 2
    assert masked.loss.item() < unmasked.loss.item()


def test_loss_rejects_asymmetric_and_zero_valid_negative_masks():
    text = torch.eye(3)
    stat = torch.eye(3)
    asymmetric = torch.zeros(3, 3, dtype=torch.bool)
    asymmetric[0, 1] = True
    try:
        symmetric_masked_contrastive_loss(text, stat, temperature=torch.tensor(0.07), false_negative_mask=asymmetric)
    except ValueError as error:
        assert "symmetric" in str(error)
    else:
        raise AssertionError("asymmetric mask was accepted")

    complete = ~torch.eye(3, dtype=torch.bool)
    try:
        symmetric_masked_contrastive_loss(text, stat, temperature=torch.tensor(0.07), false_negative_mask=complete)
    except ValueError as error:
        assert "zero valid negatives" in str(error)
    else:
        raise AssertionError("zero-valid-negative mask was accepted")
