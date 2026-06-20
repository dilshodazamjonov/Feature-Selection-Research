from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ClipModelConfig:
    text_input_dim: int
    statistical_input_dim: int
    text_hidden_dim: int = 64
    statistical_hidden_dim: int = 16
    shared_embedding_dim: int = 32
    dropout: float = 0.05
    activation: str = "gelu"
    initial_temperature: float = 0.07
    trainable_temperature: bool = False
    min_temperature: float = 0.02
    max_temperature: float = 0.5


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float, activation: str) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("projection dimensions must be positive")
        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            act = nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(values), p=2, dim=-1)


class SemanticStatisticalContrastiveEncoder(nn.Module):
    """CLIP-style semantic-statistical feature encoder over fixed input views."""

    def __init__(self, config: ClipModelConfig) -> None:
        super().__init__()
        self.config = config
        self.text_projection = ProjectionHead(
            config.text_input_dim,
            config.text_hidden_dim,
            config.shared_embedding_dim,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.statistical_projection = ProjectionHead(
            config.statistical_input_dim,
            config.statistical_hidden_dim,
            config.shared_embedding_dim,
            dropout=config.dropout,
            activation=config.activation,
        )
        initial = torch.tensor(float(config.initial_temperature)).log()
        if config.trainable_temperature:
            self.log_temperature = nn.Parameter(initial)
        else:
            self.register_buffer("log_temperature", initial)

    def forward(self, text: torch.Tensor, statistical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_text(text), self.encode_statistical(statistical)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.text_projection(text)

    def encode_statistical(self, statistical: torch.Tensor) -> torch.Tensor:
        return self.statistical_projection(statistical)

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(
            min=float(self.config.min_temperature),
            max=float(self.config.max_temperature),
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
