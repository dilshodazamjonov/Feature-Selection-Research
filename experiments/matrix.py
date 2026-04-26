from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


MODELS = ["lr", "catboost"]
STAT_SELECTORS = ["mrmr", "boruta", "pca"]
HYBRID_SELECTORS = ["mrmr", "boruta"]

LLM_SELECTOR = "llm"
EXPERIMENT_TYPES = {"statistical", "llm", "hybrid"}


@dataclass(frozen=True, slots=True)
class MatrixRunSpec:
    """One atomic experiment in the full research matrix."""

    model: str
    selector: str
    experiment_type: str
    experiment_name: str
    selector_name: str
    output_bucket: str

    @property
    def run_label(self) -> str:
        return f"{self.model}_{self.experiment_name}"


def iter_matrix() -> Iterator[MatrixRunSpec]:
    """Yield the full model/selector matrix in a stable, explicit order."""
    for model in MODELS:
        for selector in STAT_SELECTORS:
            yield MatrixRunSpec(
                model=model,
                selector=selector,
                experiment_type="statistical",
                experiment_name=selector,
                selector_name=selector,
                output_bucket="statistical",
            )

        yield MatrixRunSpec(
            model=model,
            selector=LLM_SELECTOR,
            experiment_type="llm",
            experiment_name=LLM_SELECTOR,
            selector_name=LLM_SELECTOR,
            output_bucket="llm",
        )

        for selector in HYBRID_SELECTORS:
            yield MatrixRunSpec(
                model=model,
                selector=selector,
                experiment_type="hybrid",
                experiment_name=f"llm_then_{selector}",
                selector_name=f"llm_then_{selector}",
                output_bucket=f"hybrid_{selector}",
            )


def validate_matrix() -> None:
    """Fail fast if the explicit matrix constants drift into invalid values."""
    if sorted(set(MODELS)) != sorted(MODELS):
        raise ValueError("MODELS contains duplicates.")
    if sorted(set(STAT_SELECTORS)) != sorted(STAT_SELECTORS):
        raise ValueError("STAT_SELECTORS contains duplicates.")
    if sorted(set(HYBRID_SELECTORS)) != sorted(HYBRID_SELECTORS):
        raise ValueError("HYBRID_SELECTORS contains duplicates.")
    if not set(HYBRID_SELECTORS).issubset(set(STAT_SELECTORS)):
        raise ValueError("Every hybrid downstream selector must also be statistical.")

