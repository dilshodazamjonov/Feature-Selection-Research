import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.experiments.result_paths import (  # noqa: E402
    CLIP_COMPLETE_PROFILE,
    configured_legacy_results_root,
    resolve_legacy_artifact,
    validate_legacy_evidence_profile,
)


@pytest.fixture
def legacy_artifact_path():
    """Resolve CLIP evidence only from a declared, hash-valid complete profile."""

    root = configured_legacy_results_root()
    if root is None:
        pytest.skip(
            f"required_external_evidence_unavailable: {CLIP_COMPLETE_PROFILE} "
            "(CREDIT_RISK_LEGACY_RESULTS_ROOT is not configured)"
        )
    profile = validate_legacy_evidence_profile(root)
    if profile is None:
        pytest.skip(
            f"required_external_evidence_unavailable: {CLIP_COMPLETE_PROFILE}"
        )

    def resolve(path: str | Path, *, required: bool = True) -> Path:
        return resolve_legacy_artifact(path, required=required)

    return resolve


@pytest.fixture
def legacy_config_paths(legacy_artifact_path):
    """Rebuild a frozen config with historical read paths under the explicit root."""

    def remap(config, *, output_dir: str | Path | None = None):
        if not is_dataclass(config):
            raise TypeError("legacy_config_paths requires a dataclass configuration")
        values = {}
        for field in fields(config):
            value = getattr(config, field.name)
            if isinstance(value, Path):
                normalized = str(value).replace("\\", "/")
                if normalized == "results" or normalized.startswith("results/"):
                    value = legacy_artifact_path(value, required=value.suffix != "")
            values[field.name] = value
        if output_dir is not None:
            values["output_dir"] = Path(output_dir)
        return config.__class__(**values)

    return remap
