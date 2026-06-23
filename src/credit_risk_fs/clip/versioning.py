from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CLIP_V1 = "clip_v1"
CLIP_V2 = "clip_v2"
CLIP_V1_STATISTICAL_VIEW = "missingness_only"
CLIP_V2_STATISTICAL_VIEW = "compact_target_free_v2"


@dataclass(frozen=True)
class ClipVersionPaths:
    experiment_version: str
    output_root: Path
    config_root: Path
    report_prefix: str


V1_PATHS = ClipVersionPaths(CLIP_V1, Path("results/clip"), Path("configs/clip"), "final_clip")
V2_PATHS = ClipVersionPaths(CLIP_V2, Path("results/clip_v2"), Path("configs/clip_v2"), "clip_v2")


def assert_version_output_root(*, experiment_version: str, output_root: str | Path) -> None:
    root = Path(output_root).as_posix().rstrip("/")
    if experiment_version == CLIP_V1 and root != V1_PATHS.output_root.as_posix():
        raise ValueError(f"CLIP-v1 output root must remain {V1_PATHS.output_root}")
    if experiment_version == CLIP_V2 and root != V2_PATHS.output_root.as_posix():
        raise ValueError(f"CLIP-v2 output root must be isolated under {V2_PATHS.output_root}")
    if experiment_version == CLIP_V2 and root == V1_PATHS.output_root.as_posix():
        raise ValueError("CLIP-v2 must not write to CLIP-v1 results/clip")


def versioned_cache_namespace(*, experiment_version: str, statistical_view_version: str) -> str:
    if not experiment_version or not statistical_view_version:
        raise ValueError("cache namespace requires explicit experiment and statistical-view versions")
    return f"{experiment_version}:{statistical_view_version}"
