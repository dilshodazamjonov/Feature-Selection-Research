from __future__ import annotations

import hashlib
import json
from pathlib import Path

import credit_risk_fs.experiments.selector_combinations as runner


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/protocols/selector_combinations_v1/oot_scope_freeze.json"


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _load_authenticated_freeze() -> dict:
    payload = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    observed = payload.pop("artifact_authentication_sha256")
    assert observed == _canonical_sha(payload)
    payload["artifact_authentication_sha256"] = observed
    return payload


def test_plan_only_freezes_exact_oot_matrix_without_loading_data_or_starting_workers(
    monkeypatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan-only validation invoked an execution-only path")

    monkeypatch.setattr(runner, "_load_fold", forbidden)
    monkeypatch.setattr(runner, "_run_supervised", forbidden)
    plan = runner.render_plan(runner.load_combination_plan(ROOT))

    assert plan["raw_dataset_paths_resolved"] is False
    assert plan["workers_started"] == 0
    assert plan["oot_selection_count"] == 18
    assert plan["oot_evaluation_count"] == 24


def test_safe_plan_and_scope_freeze_have_identical_ordered_oot_identities() -> None:
    freeze = _load_authenticated_freeze()
    plan = runner.render_plan(runner.load_combination_plan(ROOT))

    assert freeze["ordered_oot_selection_ids"] == [
        item["selection_id"] for item in plan["oot_selections"]
    ]
    assert freeze["ordered_oot_evaluation_cell_ids"] == [
        item["cell_id"] for item in plan["oot_evaluations"]
    ]
    assert freeze["configuration_sha256"] == plan["configuration_sha256"]
    assert freeze["protocol_lock_sha256"] == plan["protocol_lock_sha256"]


def test_scope_freeze_precedes_dev_outcome_review_and_retains_every_iv_pool() -> None:
    freeze = _load_authenticated_freeze()
    declarations = freeze["freeze_declarations"]

    assert declarations["combination_dev_performance_opened_before_scope_freeze"] is False
    assert declarations["combination_oot_accessed"] is False
    assert declarations["all_frozen_iv_pools_retained"] == [100, 200, 300]
    assert declarations["configuration_selected_from_dev_performance"] is False
    assert declarations["configuration_removed_from_dev_performance"] is False
    assert declarations["configuration_tuned_from_dev_performance"] is False
    assert declarations["configuration_reordered_from_dev_performance"] is False


def test_natural_support_cases_remain_26_of_40_and_never_pad() -> None:
    freeze = _load_authenticated_freeze()
    labels = freeze["required_natural_support_labels"]

    assert len(labels) == 2
    assert {item["method_id"] for item in labels} == {
        "boruta_then_mrmr_mutual_information",
        "boruta_then_rfe_catboost",
    }
    assert all(item["dataset"] == "homecredit" for item in labels)
    assert all(item["model"] == "catboost" for item in labels)
    assert all(item["requested_k"] == 40 for item in labels)
    assert all(item["authenticated_reference_realized_support"] == 26 for item in labels)
    assert all(item["label"] == "natural_support_26_of_requested_40" for item in labels)
    assert "never pad" in freeze["no_padding_rule"].lower()
