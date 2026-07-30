"""Runtime-only workload composition for the frozen full-baseline matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from credit_risk_fs.experiments.atomic_io import sha256_file
from credit_risk_fs.selectors.lightweight.registry import get_method_descriptor


RUNTIME_POLICY_SCHEMA_VERSION = "full_baseline_runtime_policy_v1"
DEFAULT_RUNTIME_POLICY_PATH = Path(
    "configs/execution/full_baseline_runtime_v1.yaml"
)


@dataclass(frozen=True, slots=True)
class FullBaselineRuntimePolicy:
    schema_version: str
    profile_name: str
    cost_order: tuple[str, ...]
    dataset_components: dict[str, dict[str, Any]]
    final_model_components: dict[str, dict[str, Any]]
    source_path: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class WorkloadClassification:
    selector_cost_class: str
    selector_wall_clock_limit_seconds: float
    final_model_cost_class: str
    final_model_wall_clock_limit_seconds: float
    dataset_cost_class: str
    dataset_wall_clock_limit_seconds: float
    effective_cost_class: str
    effective_wall_clock_limit_seconds: float
    composition_rule: str
    policy_profile: str
    policy_path: str
    policy_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _component(
    values: Mapping[str, Any],
    *,
    name: str,
    cost_order: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError(f"runtime component {name!r} must be a mapping")
    cost = str(values.get("cost_class", "")).strip()
    if cost not in cost_order:
        raise ValueError(f"runtime component {name!r} has unknown cost class")
    try:
        timeout = float(values["wall_clock_limit_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"runtime component {name!r} timeout must be numeric") from exc
    if timeout <= 0:
        raise ValueError(f"runtime component {name!r} timeout must be positive")
    return {
        "cost_class": cost,
        "wall_clock_limit_seconds": timeout,
    }


def load_full_baseline_runtime_policy(
    repository_root: str | Path,
    path: str | Path = DEFAULT_RUNTIME_POLICY_PATH,
) -> FullBaselineRuntimePolicy:
    root = Path(repository_root).resolve()
    supplied = Path(path)
    resolved = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise FileNotFoundError(f"full-baseline runtime policy is missing: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("full-baseline runtime policy must be a mapping")
    if payload.get("schema_version") != RUNTIME_POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported full-baseline runtime policy schema")
    profile = str(payload.get("profile_name", "")).strip()
    if not profile:
        raise ValueError("full-baseline runtime policy profile is empty")
    order = tuple(map(str, payload.get("cost_order", ())))
    if order != ("light", "heavy"):
        raise ValueError("runtime cost order must be exactly [light, heavy]")
    raw_datasets = payload.get("dataset_components", {})
    raw_models = payload.get("final_model_components", {})
    if not isinstance(raw_datasets, Mapping) or not isinstance(raw_models, Mapping):
        raise ValueError("runtime component tables must be mappings")
    datasets = {
        str(name): _component(values, name=f"dataset:{name}", cost_order=order)
        for name, values in raw_datasets.items()
    }
    models = {
        str(name): _component(values, name=f"final_model:{name}", cost_order=order)
        for name, values in raw_models.items()
    }
    return FullBaselineRuntimePolicy(
        schema_version=RUNTIME_POLICY_SCHEMA_VERSION,
        profile_name=profile,
        cost_order=order,
        dataset_components=datasets,
        final_model_components=models,
        source_path=resolved.relative_to(root).as_posix(),
        source_sha256=sha256_file(resolved),
    )


def classify_full_baseline_workload(
    cell: Any,
    policy: FullBaselineRuntimePolicy,
) -> WorkloadClassification:
    descriptor = get_method_descriptor(str(cell.method_id))
    selector_cost = str(descriptor.cost_class)
    if selector_cost not in policy.cost_order:
        raise ValueError(f"selector has unknown cost class: {selector_cost}")
    try:
        dataset = policy.dataset_components[str(cell.dataset)]
        final_model = policy.final_model_components[str(cell.model)]
    except KeyError as exc:
        raise ValueError(f"runtime policy has no component for {exc.args[0]!r}") from exc
    selector_timeout = float(cell.wall_clock_limit_seconds)
    if selector_timeout <= 0:
        raise ValueError("selector wall-clock limit must be positive")
    component_costs = (
        selector_cost,
        str(final_model["cost_class"]),
        str(dataset["cost_class"]),
    )
    effective_cost = max(
        component_costs, key=lambda value: policy.cost_order.index(value)
    )
    effective_timeout = max(
        selector_timeout,
        float(final_model["wall_clock_limit_seconds"]),
        float(dataset["wall_clock_limit_seconds"]),
    )
    return WorkloadClassification(
        selector_cost_class=selector_cost,
        selector_wall_clock_limit_seconds=selector_timeout,
        final_model_cost_class=str(final_model["cost_class"]),
        final_model_wall_clock_limit_seconds=float(
            final_model["wall_clock_limit_seconds"]
        ),
        dataset_cost_class=str(dataset["cost_class"]),
        dataset_wall_clock_limit_seconds=float(dataset["wall_clock_limit_seconds"]),
        effective_cost_class=effective_cost,
        effective_wall_clock_limit_seconds=effective_timeout,
        composition_rule="maximum_cost_and_timeout_across_selector_final_model_dataset",
        policy_profile=policy.profile_name,
        policy_path=Path(policy.source_path).as_posix(),
        policy_sha256=policy.source_sha256,
    )


__all__ = [
    "DEFAULT_RUNTIME_POLICY_PATH",
    "FullBaselineRuntimePolicy",
    "RUNTIME_POLICY_SCHEMA_VERSION",
    "WorkloadClassification",
    "classify_full_baseline_workload",
    "load_full_baseline_runtime_policy",
]
