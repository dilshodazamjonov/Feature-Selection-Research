from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_risk_fs.clip.evidence_loader import ClipEvidenceError, load_clip_evidence
from credit_risk_fs.clip.schemas import ClipDatasetRole


REAL_HOME = "results/homecredit/analysis/clip_readiness/dev_only_clip_training_evidence.csv"
REAL_LC_V2 = "results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv"


def test_dev_only_evidence_files_load(legacy_artifact_path):
    home = load_clip_evidence(
        dataset="homecredit",
        role=ClipDatasetRole.TRAIN,
        source_path=legacy_artifact_path(REAL_HOME),
        statistical_fields=["missing_rate_dev", "iv_score_if_available"],
    )
    lc = load_clip_evidence(
        dataset="lendingclub_v2",
        role=ClipDatasetRole.EXTERNAL_VALIDATION,
        source_path=legacy_artifact_path(REAL_LC_V2),
        statistical_fields=["missing_rate_dev", "iv_score_if_available"],
    )

    assert len(home.allowed) == 436
    assert len(home.blocked) == 171
    assert len(lc.allowed) == 576
    assert len(lc.blocked) == 220
    assert home.allowed["clip_training_exclusion_reason"].fillna("").eq("").all()


def test_feature_level_evidence_for_clip_is_rejected():
    with pytest.raises(ClipEvidenceError, match="feature_level_evidence_for_clip"):
        load_clip_evidence(
            dataset="homecredit",
            role=ClipDatasetRole.TRAIN,
            source_path="results/homecredit/analysis/clip_readiness/feature_level_evidence_for_clip.csv",
            statistical_fields=["missing_rate_dev"],
        )


def test_legacy_lendingclub_path_is_rejected():
    with pytest.raises(ClipEvidenceError, match="legacy_lendingclub_results"):
        load_clip_evidence(
            dataset="lendingclub",
            role=ClipDatasetRole.EXTERNAL_VALIDATION,
            source_path="results/lendingclub/analysis/clip_readiness/dev_only_clip_training_evidence.csv",
            statistical_fields=["missing_rate_dev"],
        )


@pytest.mark.parametrize("field", ["psi_dev_oot_if_available", "mean_oot_if_available", "TARGET", "member_id"])
def test_forbidden_fields_cannot_be_statistical_inputs(field: str, legacy_artifact_path):
    with pytest.raises(ClipEvidenceError, match="forbidden statistical input field"):
        load_clip_evidence(
            dataset="homecredit",
            role=ClipDatasetRole.TRAIN,
            source_path=legacy_artifact_path(REAL_HOME),
            statistical_fields=[field],
        )


def test_unsafe_rows_cannot_enter_allowed_set(tmp_path, legacy_artifact_path):
    path = tmp_path / "results" / "homecredit" / "analysis" / "clip_readiness" / "dev_only_clip_training_evidence.csv"
    frame = pd.read_csv(legacy_artifact_path(REAL_HOME)).head(2).copy()
    frame["dataset"] = "homecredit"
    frame["allowed_for_clip_training"] = True
    frame["clip_training_exclusion_reason"] = ""
    frame["leakage_review_status"] = ["safe", "needs_manual_review"]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

    with pytest.raises(ClipEvidenceError, match="unsafe rows included"):
        load_clip_evidence(
            dataset="homecredit",
            role=ClipDatasetRole.TRAIN,
            source_path=path,
            statistical_fields=["missing_rate_dev"],
        )


def test_blocked_rows_preserve_block_reasons(legacy_artifact_path):
    home = load_clip_evidence(
        dataset="homecredit",
        role=ClipDatasetRole.TRAIN,
        source_path=legacy_artifact_path(REAL_HOME),
        statistical_fields=["missing_rate_dev"],
    )

    assert home.blocked["clip_training_exclusion_reason"].fillna("").str.len().gt(0).all()
