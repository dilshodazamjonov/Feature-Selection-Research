"""Prompt 6 alignment-gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk_fs.analysis.voting_inference.alignment import (
    PredictionFrame,
    align_predictions,
    canonical_identity,
    dev_oot_disjoint_audit,
    load_prediction_frame,
)
from credit_risk_fs.analysis.voting_inference.config import AuthenticationError
from credit_risk_fs.analysis.voting_inference.inventory import RunRecord

PREDICTION_HEADER = (
    "stable_row_id,target,prediction_probability,predicted_class,fold_id,split,"
    "row_position_or_order_key,dataset,run_id,method,model,seed,coverage_type,"
    "research_eligible,comparison_eligible,probability_orientation"
)


def _frame(
    identities: list[str],
    targets: list[int],
    scores: list[float],
    *,
    run_id: str = "run",
    split: str = "OOT",
) -> PredictionFrame:
    return PredictionFrame(
        run_id=run_id,
        dataset="homecredit",
        split=split,
        path=Path("memory"),
        frame=pd.DataFrame(
            {
                "stable_row_id": canonical_identity(identities),
                "target": pd.Series(targets, dtype="int8"),
                "score": pd.Series(scores, dtype=float),
                "fold_id": pd.Series(["final"] * len(identities), dtype="string"),
            }
        ),
        metadata={"identity_target_sha256": "abc"},
    )


def test_identical_ids_in_different_row_orders_align() -> None:
    reference = _frame(["10", "11", "12"], [0, 1, 0], [0.1, 0.9, 0.4], run_id="ref")
    comparator = _frame(["12", "10", "11"], [0, 0, 1], [0.5, 0.2, 0.8], run_id="cmp")
    aligned, audit = align_predictions(reference, comparator)
    assert audit["decision"] == "aligned"
    assert audit["aligned_row_count"] == 3
    assert audit["target_mismatch_count"] == 0
    assert list(aligned["stable_row_id"]) == ["10", "11", "12"]
    assert list(aligned["score_reference"]) == [0.1, 0.9, 0.4]
    assert list(aligned["score_comparator"]) == [0.2, 0.8, 0.5]


def test_missing_identity_blocks_alignment() -> None:
    reference = _frame(["10", "11", "12"], [0, 1, 0], [0.1, 0.9, 0.4], run_id="ref")
    comparator = _frame(["10", "11"], [0, 1], [0.2, 0.8], run_id="cmp")
    aligned, audit = align_predictions(reference, comparator)
    assert audit["decision"] == "BLOCKED"
    assert audit["missing_comparator_ids"] == 1
    assert aligned.empty


def test_duplicate_identity_blocks_alignment() -> None:
    reference = _frame(["10", "10", "12"], [0, 1, 0], [0.1, 0.9, 0.4], run_id="ref")
    comparator = _frame(["10", "11", "12"], [0, 1, 0], [0.2, 0.8, 0.5], run_id="cmp")
    _, audit = align_predictions(reference, comparator)
    assert audit["decision"] == "BLOCKED"
    assert audit["duplicate_id_count"] >= 1


def test_target_mismatch_blocks_alignment() -> None:
    reference = _frame(["10", "11"], [0, 1], [0.1, 0.9], run_id="ref")
    comparator = _frame(["10", "11"], [1, 1], [0.2, 0.8], run_id="cmp")
    _, audit = align_predictions(reference, comparator)
    assert audit["decision"] == "BLOCKED"
    assert audit["target_mismatch_count"] == 1


def test_score_outside_unit_interval_blocks_alignment() -> None:
    reference = _frame(["10", "11"], [0, 1], [0.1, 0.9], run_id="ref")
    comparator = _frame(["10", "11"], [0, 1], [0.2, 1.4], run_id="cmp")
    _, audit = align_predictions(reference, comparator)
    assert audit["decision"] == "BLOCKED"
    assert audit["comparator_score_range_violations"] == 1


def test_dev_oot_overlap_is_reported() -> None:
    dev = _frame(["1", "2"], [0, 1], [0.2, 0.7], run_id="r", split="DEV_OOF")
    oot = _frame(["2", "3"], [1, 0], [0.3, 0.6], run_id="r", split="OOT")
    audit = dev_oot_disjoint_audit(dev, oot)
    assert audit["dev_oot_identity_overlap"] == 1
    assert audit["decision"] == "BLOCKED"


def test_missing_identity_value_is_rejected() -> None:
    with pytest.raises(AuthenticationError):
        canonical_identity(["1", None])


def _write_run(directory: Path, *, orientation: str, run_id: str = "r") -> RunRecord:
    results = directory / "results"
    results.mkdir(parents=True, exist_ok=True)
    rows = [
        f"1,0,0.10,0,final,OOT,1,homecredit,{run_id},rank_voting_v1,lr,42,"
        f"locked_complete_oot,True,True,{orientation}",
        f"2,1,0.90,1,final,OOT,2,homecredit,{run_id},rank_voting_v1,lr,42,"
        f"locked_complete_oot,True,True,{orientation}",
    ]
    (results / "oot_predictions.csv").write_text(
        PREDICTION_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    (results / "oot_prediction_metadata.json").write_text(
        json.dumps({"coverage_type": "locked_complete_oot"}), encoding="utf-8"
    )
    return RunRecord(
        run_id=run_id,
        dataset="homecredit",
        model="lr",
        configuration="reference",
        candidate_pool_budget=None,
        arm="reference",
        designation="reference",
        comparison_family="homecredit_lr",
        directory=directory,
        manifest={},
        config={},
    )


def test_loader_accepts_the_frozen_positive_class_orientation(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "ok", orientation="class_1_higher_default_risk")
    loaded = load_prediction_frame(run, split="OOT")
    assert loaded.row_count == 2
    assert loaded.positive_count == 1
    assert loaded.split == "OOT"


def test_loader_rejects_a_reversed_positive_class_orientation(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "reversed", orientation="class_0_higher_default_risk")
    with pytest.raises(AuthenticationError, match="probability orientation"):
        load_prediction_frame(run, split="OOT")
