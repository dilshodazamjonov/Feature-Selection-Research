"""Shared-contract tests that every lightweight selector must satisfy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.selectors.base import get_selected_features
from credit_risk_fs.selectors.lightweight import (
    CONTRACT_VERSION,
    LONG_FRAME_COLUMNS,
    ControlledSelectorFailure,
    FullCandidateFeaturesSelector,
    InformationValueSelector,
    L1LogisticSelector,
    MutualInformationMRMRSelector,
    RandomKSelector,
    SelectionResult,
    SelectorContractError,
)
from credit_risk_fs.selectors.lightweight.contract import (
    ordered_name_hash,
    rank_by_score,
    training_identity_hash,
)

SUPERVISED = (InformationValueSelector, MutualInformationMRMRSelector, L1LogisticSelector)
UNSUPERVISED = (RandomKSelector, FullCandidateFeaturesSelector)
ALL_SELECTORS = SUPERVISED + UNSUPERVISED

EXCLUDED = ("SK_ID_CURR", "TARGET", "split", "time_index")


@pytest.fixture()
def fixture() -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(3)
    n = 400
    latent = generator.normal(size=n)
    target = pd.Series((latent + generator.normal(scale=0.6, size=n) > 0).astype(int))
    frame = pd.DataFrame(
        {
            "alpha_signal": latent,
            "beta_copy": latent + generator.normal(scale=0.05, size=n),
            "gamma_noise": generator.normal(size=n),
            "delta_constant": np.ones(n),
            "epsilon_holes": np.where(
                generator.random(n) < 0.25, np.nan, generator.normal(size=n)
            ),
        }
    )
    return frame, target


def _fit(selector_cls, frame, target, **kwargs):
    selector = selector_cls(**kwargs)
    return selector.fit(frame, target if selector.supervised else None)


@pytest.mark.parametrize("selector_cls", ALL_SELECTORS)
def test_result_satisfies_the_universal_invariants(selector_cls, fixture) -> None:
    frame, target = fixture
    selector = _fit(selector_cls, frame, target, k=3)
    result = selector.result

    assert result.contract_version == CONTRACT_VERSION
    assert result.method_id == selector_cls.method_id
    assert result.implementation_id == selector_cls.implementation_id
    assert result.display_label == selector_cls.display_label
    assert result.fit_scope == "dev_fold_training_only"

    selected = list(result.selected_features)
    assert len(selected) == len(set(selected))
    assert set(selected).issubset(set(frame.columns))
    assert result.actual_selected_count == len(selected)
    assert result.candidate_universe == tuple(frame.columns)
    assert result.candidate_universe_sha256 == ordered_name_hash(frame.columns)

    # The legacy accessor still works, so the fold runner needs no change.
    assert get_selected_features(selector) == selected
    assert list(selector.transform(frame).columns) == selected


@pytest.mark.parametrize("selector_cls", ALL_SELECTORS)
def test_ranks_are_one_based_dense_and_deterministic(selector_cls, fixture) -> None:
    frame, target = fixture
    first = _fit(selector_cls, frame, target, k=3)
    second = _fit(selector_cls, frame, target, k=3)

    assert first.result.selected_features == second.result.selected_features
    assert first.result.ranking == second.result.ranking

    long_frame = first.result.to_long_frame()
    assert list(long_frame.columns) == list(LONG_FRAME_COLUMNS)
    assert list(long_frame["rank"]) == list(range(1, len(long_frame) + 1))
    assert long_frame["feature"].is_unique
    assert long_frame["tie_rule"].nunique() == 1


@pytest.mark.parametrize("selector_cls", ALL_SELECTORS)
def test_excluded_columns_can_never_be_selected(selector_cls, fixture) -> None:
    frame, target = fixture
    contaminated = frame.copy()
    contaminated["TARGET"] = target.to_numpy()
    contaminated["SK_ID_CURR"] = np.arange(len(frame))

    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(selector_cls, contaminated, target, k=3, excluded_columns=EXCLUDED)
    assert error.value.stage == "candidate_universe_validation"
    assert "TARGET" in error.value.cause


@pytest.mark.parametrize("selector_cls", ALL_SELECTORS)
def test_duplicate_feature_names_are_rejected(selector_cls, fixture) -> None:
    frame, target = fixture
    duplicated = pd.concat([frame, frame.iloc[:, :1]], axis=1)
    with pytest.raises(ValueError, match="unique"):
        _fit(selector_cls, duplicated, target, k=2)


@pytest.mark.parametrize("selector_cls", ALL_SELECTORS)
def test_empty_universe_produces_a_controlled_outcome(selector_cls, fixture) -> None:
    _, target = fixture
    empty = pd.DataFrame(index=range(len(target)))
    selector = _fit(selector_cls, empty, target, k=3)
    result = selector.result
    assert result.selected_features == ()
    assert result.budget_status == "empty_universe"
    assert result.warnings
    assert selector.selected_features_ == []


@pytest.mark.parametrize("selector_cls", SUPERVISED)
def test_missing_target_fails_explicitly_for_supervised_methods(
    selector_cls, fixture
) -> None:
    frame, _ = fixture
    with pytest.raises(ControlledSelectorFailure) as error:
        selector_cls(k=2).fit(frame, None)
    assert error.value.stage == "target_validation"


@pytest.mark.parametrize("selector_cls", SUPERVISED)
def test_single_class_target_fails_explicitly(selector_cls, fixture) -> None:
    frame, target = fixture
    with pytest.raises(ControlledSelectorFailure) as error:
        selector_cls(k=2).fit(frame, pd.Series(np.zeros(len(target), dtype=int)))
    assert "single class" in error.value.cause


@pytest.mark.parametrize("selector_cls", SUPERVISED)
def test_non_binary_target_fails_explicitly(selector_cls, fixture) -> None:
    frame, target = fixture
    labels = pd.Series(np.arange(len(target)) % 3)
    with pytest.raises(ControlledSelectorFailure) as error:
        selector_cls(k=2).fit(frame, labels)
    assert "0/1" in error.value.cause


@pytest.mark.parametrize("selector_cls", (InformationValueSelector, MutualInformationMRMRSelector, RandomKSelector))
def test_budget_equal_to_universe_is_satisfied(selector_cls, fixture) -> None:
    frame, target = fixture
    selector = _fit(selector_cls, frame, target, k=frame.shape[1])
    assert selector.result.budget_status == "satisfied"
    assert selector.result.actual_selected_count == frame.shape[1]


@pytest.mark.parametrize("selector_cls", (InformationValueSelector, MutualInformationMRMRSelector, RandomKSelector))
def test_budget_larger_than_universe_is_clipped_and_recorded(selector_cls, fixture) -> None:
    frame, target = fixture
    selector = _fit(selector_cls, frame, target, k=frame.shape[1] + 25)
    result = selector.result
    assert result.budget_status == "clipped_to_universe"
    assert result.requested_budget == frame.shape[1] + 25
    # Never padded or duplicated to reach the request.
    assert result.actual_selected_count == frame.shape[1]
    assert len(set(result.selected_features)) == frame.shape[1]
    assert any("exceeds the eligible candidate universe" in item for item in result.warnings)


@pytest.mark.parametrize("selector_cls", ALL_SELECTORS)
def test_serialization_round_trip_is_exact(selector_cls, fixture) -> None:
    frame, target = fixture
    original = _fit(selector_cls, frame, target, k=3).result
    restored = SelectionResult.from_json(original.to_json())

    assert restored.method_id == original.method_id
    assert restored.implementation_id == original.implementation_id
    assert restored.selection_mode == original.selection_mode
    assert restored.selected_features == original.selected_features
    assert restored.ranking == original.ranking
    assert restored.candidate_universe == original.candidate_universe
    assert restored.candidate_universe_sha256 == original.candidate_universe_sha256
    assert restored.requested_budget == original.requested_budget
    assert restored.budget_status == original.budget_status
    assert restored.natural_selected == original.natural_selected
    assert restored.seed == original.seed
    assert restored.training_identity_sha256 == original.training_identity_sha256
    if original.raw_scores is None:
        assert restored.raw_scores is None
    else:
        for name, value in original.raw_scores.items():
            assert restored.raw_scores[name] == pytest.approx(value, abs=0.0)


def test_reload_rejects_an_unknown_contract_version(fixture) -> None:
    frame, target = fixture
    payload = InformationValueSelector(k=2).fit(frame, target).result.to_dict()
    payload["contract_version"] = "something_else_v9"
    with pytest.raises(SelectorContractError, match="contract version"):
        SelectionResult.from_dict(payload)


@pytest.mark.parametrize("selector_cls", SUPERVISED)
def test_training_identity_hash_detects_a_different_row_set(selector_cls, fixture) -> None:
    """The fit-boundary hash is what makes a leaked row detectable."""

    frame, target = fixture
    half = len(frame) // 2
    inside = _fit(selector_cls, frame.iloc[:half], target.iloc[:half], k=2)
    leaked = _fit(selector_cls, frame, target, k=2)
    assert inside.result.training_identity_sha256 != leaked.result.training_identity_sha256
    assert inside.result.training_row_count == half


def test_training_identity_hash_ignores_nothing_relevant() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    left = pd.Series([0, 1, 0])
    right = pd.Series([0, 1, 1])
    assert training_identity_hash(frame, left) != training_identity_hash(frame, right)
    assert training_identity_hash(frame, None) != training_identity_hash(frame, left)


def test_rank_by_score_is_independent_of_mapping_order() -> None:
    order = ["a", "b", "c"]
    forward = rank_by_score({"a": 1.0, "b": 1.0, "c": 2.0}, candidate_order=order)
    backward = rank_by_score({"c": 2.0, "b": 1.0, "a": 1.0}, candidate_order=order)
    assert forward == backward == ["c", "a", "b"]


def test_contract_rejects_a_selection_outside_the_universe() -> None:
    with pytest.raises(SelectorContractError, match="outside the authenticated"):
        SelectionResult(
            method_id="iv_woe",
            display_label="x",
            implementation_id="y",
            selection_mode="matched_budget",
            supervised=True,
            selected_features=("ghost",),
            candidate_universe=("real",),
            requested_budget=1,
            budget_status="satisfied",
            score_orientation="higher_is_better",
            tie_rule="descending_score_then_ascending_feature_name",
        )


def test_contract_rejects_a_satisfied_budget_that_was_not_met() -> None:
    with pytest.raises(SelectorContractError, match="budget_status='satisfied'"):
        SelectionResult(
            method_id="iv_woe",
            display_label="x",
            implementation_id="y",
            selection_mode="matched_budget",
            supervised=True,
            selected_features=("a",),
            candidate_universe=("a", "b"),
            requested_budget=2,
            budget_status="satisfied",
            score_orientation="higher_is_better",
            tie_rule="descending_score_then_ascending_feature_name",
        )


def test_contract_rejects_a_selected_feature_that_carries_no_rank() -> None:
    with pytest.raises(SelectorContractError, match="carry no"):
        SelectionResult(
            method_id="iv_woe",
            display_label="x",
            implementation_id="y",
            selection_mode="matched_budget",
            supervised=True,
            selected_features=("b",),
            candidate_universe=("a", "b"),
            requested_budget=1,
            budget_status="satisfied",
            score_orientation="higher_is_better",
            tie_rule="descending_score_then_ascending_feature_name",
            ranking=("a",),
        )


def test_controlled_failure_reports_method_stage_cause_and_configuration() -> None:
    error = ControlledSelectorFailure(
        method_id="iv_woe",
        stage="binning",
        cause="deliberate",
        configuration={"n_bins": 10},
    )
    text = str(error)
    assert "iv_woe" in text
    assert "binning" in text
    assert "deliberate" in text
    assert "n_bins" in text
