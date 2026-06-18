from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk_fs.clip.leakage_policy import is_forbidden_training_column
from credit_risk_fs.clip.training_manifest import load_readiness_manifest
from scripts.validate_clip_readiness import validate_manifest


def _write_training_evidence(path: Path, dataset: str, include_forbidden_column: bool = False) -> None:
    rows = [
        {
            "dataset": dataset,
            "feature": "income_ratio",
            "clip_training_split": "DEV_ONLY",
            "clip_training_text": "feature=income_ratio | description=Income ratio",
            "description": "Income ratio",
            "semantic_group": "income_capacity",
            "allowed_for_clip_training": True,
            "clip_training_exclusion_reason": "",
            "leakage_review_status": "safe",
            "leakage_review_action": "include",
            "leakage_rule": "DEV-only metadata and DEV statistics only.",
            "prohibited_training_fields": "psi;oot;target",
            "evaluation_only_fields": "psi_dev_oot_if_available",
        },
        {
            "dataset": dataset,
            "feature": "PC1",
            "clip_training_split": "DEV_ONLY",
            "clip_training_text": "feature=PC1",
            "description": "",
            "semantic_group": "other",
            "allowed_for_clip_training": False,
            "clip_training_exclusion_reason": "missing_description",
            "leakage_review_status": "safe",
            "leakage_review_action": "include",
            "leakage_rule": "DEV-only metadata and DEV statistics only.",
            "prohibited_training_fields": "psi;oot;target",
            "evaluation_only_fields": "psi_dev_oot_if_available",
        },
    ]
    frame = pd.DataFrame(rows)
    if include_forbidden_column:
        frame["psi_dev_oot_if_available"] = [0.1, 0.2]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_summary(path: Path, evidence_paths: dict[str, Path]) -> None:
    rows = []
    for dataset, evidence_path in evidence_paths.items():
        frame = pd.read_csv(evidence_path)
        allowed = frame["allowed_for_clip_training"].astype(bool)
        rows.append(
            {
                "dataset": dataset,
                "total_rows": len(frame),
                "allowed_for_clip_training": int(allowed.sum()),
                "blocked_for_clip_training": int((~allowed).sum()),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_clip_readiness_manifest_uses_v2_not_old_lendingclub():
    manifest = load_readiness_manifest(Path("configs/clip/readiness.yaml"))

    assert manifest.datasets == ("homecredit", "lendingclub_v2")
    assert "lendingclub" not in manifest.datasets


def test_forbidden_column_policy_does_not_flag_bootstrap():
    assert is_forbidden_training_column("psi_dev_oot_if_available")
    assert is_forbidden_training_column("mean_oot_if_available")
    assert not is_forbidden_training_column("bootstrap_selection_frequency_if_available")


def test_validate_clip_readiness_accepts_dev_only_training_tables(tmp_path):
    config = tmp_path / "readiness.yaml"
    hc = tmp_path / "homecredit.csv"
    lc = tmp_path / "lendingclub_v2.csv"
    summary = tmp_path / "summary.csv"
    _write_training_evidence(hc, "homecredit")
    _write_training_evidence(lc, "lendingclub_v2")
    _write_summary(summary, {"homecredit": hc, "lendingclub_v2": lc})
    config.write_text(
        "\n".join(
            [
                "datasets:",
                "  - homecredit",
                "  - lendingclub_v2",
                "legacy_datasets:",
                "  - lendingclub",
                "training_evidence:",
                f"  homecredit: {hc}",
                f"  lendingclub_v2: {lc}",
                f"cross_dataset_summary: {summary}",
                "min_allowed_rows:",
                "  homecredit: 1",
                "  lendingclub_v2: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = load_readiness_manifest(config)

    assert validate_manifest(manifest) == []


def test_validate_clip_readiness_rejects_oot_psi_training_columns(tmp_path):
    config = tmp_path / "readiness.yaml"
    hc = tmp_path / "homecredit.csv"
    lc = tmp_path / "lendingclub_v2.csv"
    summary = tmp_path / "summary.csv"
    _write_training_evidence(hc, "homecredit", include_forbidden_column=True)
    _write_training_evidence(lc, "lendingclub_v2")
    _write_summary(summary, {"homecredit": hc, "lendingclub_v2": lc})
    config.write_text(
        "\n".join(
            [
                "datasets:",
                "  - homecredit",
                "  - lendingclub_v2",
                "legacy_datasets:",
                "  - lendingclub",
                "training_evidence:",
                f"  homecredit: {hc}",
                f"  lendingclub_v2: {lc}",
                f"cross_dataset_summary: {summary}",
                "min_allowed_rows:",
                "  homecredit: 1",
                "  lendingclub_v2: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    errors = validate_manifest(load_readiness_manifest(config))

    assert any("forbidden training columns" in error for error in errors)
