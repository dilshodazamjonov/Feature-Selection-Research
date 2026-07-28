"""Tests for the random-k and full-candidate scientific controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.selectors.lightweight.contract import TIE_RULE_UNIVERSE_ORDER
from credit_risk_fs.selectors.lightweight.controls import (
    FullCandidateFeaturesSelector,
    RandomKSelector,
)


@pytest.fixture()
def frame() -> pd.DataFrame:
    generator = np.random.default_rng(2)
    return pd.DataFrame(
        {name: generator.normal(size=200) for name in ("alpha", "bravo", "charlie", "delta", "echo")}
    )


def test_random_k_matches_the_explicit_local_generator(frame) -> None:
    """Oracle: reproduce the draw from numpy directly, not from the class."""

    order = list(frame.columns)
    expected_priority = [
        order[index] for index in np.random.default_rng(1234).permutation(len(order))
    ]

    selector = RandomKSelector(k=3, random_state=1234).fit(frame)
    assert list(selector.result.ranking or ()) == expected_priority
    assert list(selector.result.selected_features) == expected_priority[:3]


def test_random_k_is_reproducible_and_seed_sensitive(frame) -> None:
    first = RandomKSelector(k=3, random_state=7).fit(frame)
    repeat = RandomKSelector(k=3, random_state=7).fit(frame)
    other = RandomKSelector(k=3, random_state=8).fit(frame)

    assert first.result.selected_features == repeat.result.selected_features
    assert first.result.selected_features != other.result.selected_features
    # A different seed must not change the method's identity.
    assert first.result.method_id == other.result.method_id == "random_k"
    assert first.result.implementation_id == other.result.implementation_id
    assert other.result.seed == 8


def test_random_k_does_not_touch_global_rng_state(frame) -> None:
    np.random.seed(99)
    before = np.random.get_state()[1][:8].copy()
    RandomKSelector(k=2, random_state=5).fit(frame)
    after = np.random.get_state()[1][:8]
    assert np.array_equal(before, after)


def test_random_k_never_inspects_the_target(frame) -> None:
    honest = pd.Series(np.arange(len(frame)) % 2)
    inverted = 1 - honest

    baseline = RandomKSelector(k=2, random_state=3).fit(frame, honest)
    flipped = RandomKSelector(k=2, random_state=3).fit(frame, inverted)
    unlabelled = RandomKSelector(k=2, random_state=3).fit(frame, None)

    assert baseline.result.selected_features == flipped.result.selected_features
    assert baseline.result.selected_features == unlabelled.result.selected_features
    assert baseline.result.supervised is False
    # The target never enters the fit-boundary hash either, because it was never
    # made available to the selector.
    assert baseline.result.training_identity_sha256 == flipped.result.training_identity_sha256


def test_random_k_records_the_universe_it_sampled_from(frame) -> None:
    selector = RandomKSelector(k=2, random_state=11).fit(frame)
    result = selector.result
    assert result.candidate_universe == tuple(frame.columns)
    assert result.candidate_universe_count == 5
    assert result.requested_budget == 2
    assert result.actual_selected_count == 2
    assert result.configuration["uses_global_rng_state"] is False
    assert result.configuration["inspects_target"] is False


def test_full_features_returns_exactly_the_eligible_universe(frame) -> None:
    selector = FullCandidateFeaturesSelector().fit(frame)
    result = selector.result

    assert list(result.selected_features) == list(frame.columns)
    assert result.selection_mode == "full_control"
    assert result.budget_status == "not_applicable"
    assert result.requested_budget is None
    assert result.actual_selected_count == frame.shape[1]
    assert result.tie_rule == TIE_RULE_UNIVERSE_ORDER
    assert result.score_orientation == "not_applicable"
    assert result.raw_scores is None


def test_full_features_preserves_the_authenticated_candidate_order(frame) -> None:
    reordered = frame[["echo", "alpha", "delta", "bravo", "charlie"]]
    selector = FullCandidateFeaturesSelector().fit(reordered)
    assert list(selector.result.selected_features) == list(reordered.columns)
    assert list(selector.result.ranking or ()) == list(reordered.columns)


def test_full_features_ignores_a_budget_explicitly(frame) -> None:
    selector = FullCandidateFeaturesSelector(k=2).fit(frame)
    result = selector.result

    assert result.actual_selected_count == frame.shape[1]
    assert result.requested_budget is None
    assert result.configuration["ignored_budget_request"] == 2
    assert any("ignores the fixed budget 2" in item for item in result.warnings)


def test_full_features_matches_the_historical_no_selection_route(frame) -> None:
    """The legacy 'none' route stays available and means the same thing."""

    from credit_risk_fs.selectors.registry import get_selector

    legacy_cls, legacy_kwargs = get_selector("none")
    assert legacy_cls is None
    assert legacy_kwargs == {}

    # With no selector object the pipeline uses every supplied column; the
    # explicit control must select exactly that set, in that order.
    explicit = FullCandidateFeaturesSelector().fit(frame)
    assert list(explicit.result.selected_features) == list(frame.columns)
    assert list(explicit.transform(frame).columns) == list(frame.columns)
    pd.testing.assert_frame_equal(explicit.transform(frame), frame)


def test_full_features_never_inspects_the_target(frame) -> None:
    target = pd.Series(np.arange(len(frame)) % 2)
    with_target = FullCandidateFeaturesSelector().fit(frame, target)
    without = FullCandidateFeaturesSelector().fit(frame)
    assert with_target.result.selected_features == without.result.selected_features
    assert with_target.result.supervised is False
    assert (
        with_target.result.training_identity_sha256
        == without.result.training_identity_sha256
    )
