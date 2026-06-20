from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TextEncoderProtocol(Protocol):
    model_name: str
    revision: str
    embedding_dimension: int

    def encode(self, texts: list[str], *, batch_size: int, normalize_embeddings: bool) -> np.ndarray:
        ...


@dataclass
class FrozenSentenceTransformerEncoder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "main"
    local_model_path: str | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it or provide an environment with the configured pretrained model."
            ) from exc

        model_ref = self.local_model_path or self.model_name
        kwargs = {"device": self.device} if self.device else {}
        self.model = SentenceTransformer(model_ref, revision=self.revision, **kwargs)
        if hasattr(self.model, "eval"):
            self.model.eval()
        for parameter in getattr(self.model, "parameters", lambda: [])():
            parameter.requires_grad = False
        dimension_getter = getattr(self.model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = self.model.get_sentence_embedding_dimension
        self.embedding_dimension = int(dimension_getter())

    def encode(self, texts: list[str], *, batch_size: int, normalize_embeddings: bool) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


@dataclass
class MockFrozenTextEncoder:
    model_name: str = "mock-frozen-text-encoder"
    revision: str = "test"
    embedding_dimension: int = 8
    loaded: bool = False

    def encode(self, texts: list[str], *, batch_size: int, normalize_embeddings: bool) -> np.ndarray:
        rows = []
        for text in texts:
            seed = sum(ord(char) for char in text)
            values = np.array([(seed + idx * 17) % 101 for idx in range(self.embedding_dimension)], dtype=np.float32)
            if normalize_embeddings:
                norm = float(np.linalg.norm(values))
                if norm:
                    values = values / norm
            rows.append(values)
        return np.vstack(rows).astype(np.float32)


def resolve_device(device_policy: str) -> str | None:
    if device_policy == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_policy in {"cpu", "cuda"}:
        return device_policy
    return None
