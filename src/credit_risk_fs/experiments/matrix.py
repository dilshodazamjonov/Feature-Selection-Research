from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


MODELS = ["lr", "catboost"]
STAT_SELECTORS = ["mrmr", "boruta", "pca", "domain_rule_baseline"]
HYBRID_VARIANTS = [
    ("mrmr", "llm_then_mrmr", "llm_then_mrmr", "hybrid_mrmr"),
    ("boruta", "llm_then_boruta", "llm_then_boruta", "hybrid_boruta"),
    (
        "stable_core_llm_fill",
        "stable_core_llm_fill",
        "stable_core_llm_fill",
        "hybrid_stable_core_llm_fill",
    ),
]

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

        for selector, experiment_name, selector_name, output_bucket in HYBRID_VARIANTS:
            yield MatrixRunSpec(
                model=model,
                selector=selector,
                experiment_type="hybrid",
                experiment_name=experiment_name,
                selector_name=selector_name,
                output_bucket=output_bucket,
            )


def validate_matrix() -> None:
    """Fail fast if the explicit matrix constants drift into invalid values."""
    if sorted(set(MODELS)) != sorted(MODELS):
        raise ValueError("MODELS contains duplicates.")
    if sorted(set(STAT_SELECTORS)) != sorted(STAT_SELECTORS):
        raise ValueError("STAT_SELECTORS contains duplicates.")

    hybrid_selector_names = [selector for selector, *_ in HYBRID_VARIANTS]
    if sorted(set(hybrid_selector_names)) != sorted(hybrid_selector_names):
        raise ValueError("HYBRID_VARIANTS contains duplicate selector ids.")

    if not {"mrmr", "boruta"}.issubset(set(STAT_SELECTORS)):
        raise ValueError("Statistical baselines must include mrmr and boruta.")
