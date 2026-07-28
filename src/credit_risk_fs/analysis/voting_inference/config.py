"""Frozen-input authentication and analysis configuration loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from credit_risk_fs.experiments.result_paths import reject_historical_write
from credit_risk_fs.utils.hashing import sha256_file

ANALYSIS_SCHEMA_VERSION = "cross_dataset_voting_inference_analysis_v1"
DEFAULT_CONFIG_PATH = "configs/analysis/cross_dataset_voting_inference_v1.yaml"


class AuthenticationError(RuntimeError):
    """Raised when a required frozen input cannot be authenticated."""


@dataclass(frozen=True)
class AnalysisConfig:
    """Resolved Prompt 6 analysis configuration."""

    repository_root: Path
    config_path: Path
    config_sha256: str
    payload: Mapping[str, Any]
    frozen_input_hashes: Mapping[str, str] = field(default_factory=dict)

    # -- resolved locations -------------------------------------------------
    @property
    def package_root(self) -> Path:
        return self._resolved("package_root")

    @property
    def audit_root(self) -> Path:
        return self._resolved("audit_root")

    @property
    def run_root(self) -> Path:
        return self._resolved("run_root")

    @property
    def figures_root(self) -> Path:
        return self.package_root / str(self.payload["paths"]["figures_subdirectory"])

    def _resolved(self, key: str) -> Path:
        relative = str(self.payload["paths"][key])
        return reject_historical_write((self.repository_root / relative).resolve())

    # -- frequently used sections ------------------------------------------
    @property
    def expected(self) -> Mapping[str, Any]:
        return self.payload["expected_structure"]

    @property
    def metric_definitions(self) -> Mapping[str, Any]:
        return self.payload["metric_definitions"]

    @property
    def inference(self) -> Mapping[str, Any]:
        return self.payload["inference"]

    @property
    def preservation(self) -> Mapping[str, Any]:
        return self.payload["preservation"]

    @property
    def tolerance(self) -> float:
        return float(self.payload["independent_recalculation"]["absolute_tolerance"])

    def dataset_universe_size(self, dataset: str) -> int:
        sizes = self.metric_definitions["kuncheva"]["universe_size"]
        if dataset not in sizes:
            raise AuthenticationError(f"no authenticated Kuncheva universe for {dataset!r}")
        return int(sizes[dataset])

    def final_budget(self, model: str) -> int:
        budgets = self.expected["final_feature_budgets"]
        if model not in budgets:
            raise AuthenticationError(f"no authenticated final feature budget for {model!r}")
        return int(budgets[model])


def load_analysis_config(
    repository_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> AnalysisConfig:
    """Load the Prompt 6 analysis configuration without authenticating inputs."""

    root = Path(repository_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = (root / path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AuthenticationError(f"analysis configuration is not a mapping: {path}")
    if payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise AuthenticationError(
            f"unexpected analysis schema_version: {payload.get('schema_version')!r}"
        )
    return AnalysisConfig(
        repository_root=root,
        config_path=path,
        config_sha256=sha256_file(path),
        payload=payload,
    )


def authenticate_frozen_inputs(config: AnalysisConfig) -> dict[str, Any]:
    """Hash every declared frozen input and compare against expectations.

    A declared ``expected_sha256`` of ``None`` records the observed hash without
    asserting it, which keeps run-produced registries auditable without
    pretending they were frozen before execution.
    """

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for name, entry in config.payload["frozen_inputs"].items():
        relative = str(entry["path"])
        path = (config.repository_root / relative).resolve()
        present = path.is_file()
        observed = sha256_file(path) if present else None
        expected = entry.get("expected_sha256")
        if not present:
            status = "missing"
            failures.append(f"{name}: missing {relative}")
        elif expected is None:
            status = "recorded_not_pinned"
        elif str(expected) == observed:
            status = "hash_match"
        else:
            status = "hash_mismatch"
            failures.append(
                f"{name}: expected {expected} observed {observed} for {relative}"
            )
        records.append(
            {
                "input_name": name,
                "path": relative,
                "present": present,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "size_bytes": path.stat().st_size if present else None,
                "status": status,
            }
        )
    try:
        config_label = str(
            config.config_path.relative_to(config.repository_root)
        ).replace("\\", "/")
    except ValueError:
        config_label = str(config.config_path).replace("\\", "/")
    return {
        "schema_version": "prompt_06_frozen_input_authentication_v1",
        "analysis_config_path": config_label,
        "analysis_config_sha256": config.config_sha256,
        "inputs": records,
        "failures": failures,
        "status": "PASS" if not failures else "BLOCKED",
    }


def read_json(path: str | Path) -> Any:
    """Read one JSON document from an immutable input path."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "AnalysisConfig",
    "AuthenticationError",
    "authenticate_frozen_inputs",
    "load_analysis_config",
    "read_json",
]
