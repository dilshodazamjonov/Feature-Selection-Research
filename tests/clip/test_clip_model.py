from __future__ import annotations

import torch

from credit_risk_fs.clip.model import ClipModelConfig, SemanticStatisticalContrastiveEncoder, count_trainable_parameters


def test_clip_model_outputs_normalized_vectors_and_expected_dimensions():
    config = ClipModelConfig(text_input_dim=384, statistical_input_dim=1, shared_embedding_dim=32)
    model = SemanticStatisticalContrastiveEncoder(config)
    text = torch.randn(5, 384)
    stat = torch.randn(5, 1)

    text_out, stat_out = model(text, stat)

    assert text_out.shape == (5, 32)
    assert stat_out.shape == (5, 32)
    assert torch.allclose(text_out.norm(dim=1), torch.ones(5), atol=1e-5)
    assert torch.allclose(stat_out.norm(dim=1), torch.ones(5), atol=1e-5)
    assert count_trainable_parameters(model) == 27296
    assert not any("sentence" in name.lower() or "transformer" in name.lower() for name, _ in model.named_parameters())
