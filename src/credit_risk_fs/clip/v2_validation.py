from __future__ import annotations

from pathlib import Path
from typing import Any

from credit_risk_fs.clip.statistical_schema_v2 import DESCRIPTOR_COLUMNS_V2
from credit_risk_fs.clip.versioning import CLIP_V2, CLIP_V2_STATISTICAL_VIEW, assert_version_output_root


def validate_clip_v2_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("experiment_version") != CLIP_V2:
        errors.append("experiment_version must be clip_v2")
    if config.get("statistical_view_version") != CLIP_V2_STATISTICAL_VIEW:
        errors.append("statistical_view_version must be compact_target_free_v2")
    if int(config.get("statistical_input_dimension", -1)) != len(DESCRIPTOR_COLUMNS_V2):
        errors.append("statistical_input_dimension must be 13")
    try:
        assert_version_output_root(experiment_version=CLIP_V2, output_root=config.get("output_root", ""))
    except Exception as exc:
        errors.append(str(exc))
    reference = Path(str(config.get("v1_reference_manifest", "")))
    if not reference.exists():
        errors.append(f"v1_reference_manifest does not exist: {reference}")
    return errors


def validate_no_v1_output_paths(paths: list[str]) -> None:
    violations = [path for path in paths if Path(path).as_posix().startswith("results/clip/")]
    if violations:
        raise ValueError(f"CLIP-v2 paths must not target CLIP-v1 outputs: {violations}")
