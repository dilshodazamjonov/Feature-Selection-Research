from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from credit_risk_fs.clip.datasets import default_training_evidence_paths
from credit_risk_fs.clip.leakage_policy import ALLOWED_DATASETS, LEGACY_DATASETS


@dataclass(frozen=True)
class ClipReadinessManifest:
    datasets: tuple[str, ...]
    legacy_datasets: tuple[str, ...]
    training_evidence: dict[str, Path]
    cross_dataset_summary: Path
    min_allowed_rows: dict[str, int]


def default_manifest() -> ClipReadinessManifest:
    return ClipReadinessManifest(
        datasets=ALLOWED_DATASETS,
        legacy_datasets=LEGACY_DATASETS,
        training_evidence=default_training_evidence_paths(),
        cross_dataset_summary=Path(
            "results/cross_dataset_v2/analysis/clip_readiness/dev_only_clip_training_evidence_summary.csv"
        ),
        min_allowed_rows={"homecredit": 100, "lendingclub_v2": 100},
    )


def load_readiness_manifest(path: Path = Path("configs/clip/readiness.yaml")) -> ClipReadinessManifest:
    if not path.exists():
        return default_manifest()

    data = _read_simple_yaml(path)
    default = default_manifest()
    datasets = tuple(data.get("datasets", default.datasets))
    legacy_datasets = tuple(data.get("legacy_datasets", default.legacy_datasets))
    raw_evidence = data.get("training_evidence", {})
    training_evidence = {
        dataset: Path(raw_evidence.get(dataset, default.training_evidence.get(dataset, Path())))
        for dataset in datasets
    }
    min_allowed_rows = {
        dataset: int(data.get("min_allowed_rows", {}).get(dataset, default.min_allowed_rows.get(dataset, 1)))
        for dataset in datasets
    }
    return ClipReadinessManifest(
        datasets=datasets,
        legacy_datasets=legacy_datasets,
        training_evidence=training_evidence,
        cross_dataset_summary=Path(data.get("cross_dataset_summary", default.cross_dataset_summary)),
        min_allowed_rows=min_allowed_rows,
    )


def _read_simple_yaml(path: Path) -> dict[str, object]:
    """Parse the small readiness manifest without adding a YAML dependency."""
    result: dict[str, object] = {}
    current_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_key = line[:-1].strip()
            result[current_key] = []
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip('"')
            current_key = None
            continue
        if current_key is None:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            value = stripped[2:].strip().strip('"')
            target = result.setdefault(current_key, [])
            if isinstance(target, list):
                target.append(value)
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            target = result.setdefault(current_key, {})
            if not isinstance(target, dict):
                target = {}
                result[current_key] = target
            target[key.strip()] = value.strip().strip('"')

    return result
