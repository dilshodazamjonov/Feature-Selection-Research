from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_fs.clip_final_comparison.constants import OUTPUT_ROOT


def assert_isolated_output_path(path: str | Path) -> Path:
    output = Path(path)
    try:
        output.resolve().relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"final-comparison output must stay under {OUTPUT_ROOT.as_posix()}: {output}") from exc
    return output


def ensure_layout() -> None:
    for directory in [
        "manifests",
        "candidate_pool/plans",
        "candidate_pool/screening_scores",
        "candidate_pool/selected_pools",
        "candidate_pool/runs",
        "candidate_pool/predictions",
        "candidate_pool/aggregates",
        "seed_robustness",
        "ablations",
        "temporal_validation",
        "uncertainty",
        "final_analysis/tables",
        "final_analysis/plots",
        "audit",
    ]:
        assert_isolated_output_path(OUTPUT_ROOT / directory).mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: str | Path, content: str) -> Path:
    output = assert_isolated_output_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(output)
    return output


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    output = assert_isolated_output_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    tmp.replace(output)
    return output


def read_json_if_exists(path: str | Path, default: Any) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    return json.loads(file_path.read_text(encoding="utf-8-sig"))
