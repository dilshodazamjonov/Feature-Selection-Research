from __future__ import annotations

from pathlib import Path

from credit_risk_fs.clip.leakage_policy import ALLOWED_DATASETS


def default_training_evidence_paths(results_root: Path = Path("results")) -> dict[str, Path]:
    return {
        dataset: results_root / dataset / "analysis" / "clip_readiness" / "dev_only_clip_training_evidence.csv"
        for dataset in ALLOWED_DATASETS
    }

