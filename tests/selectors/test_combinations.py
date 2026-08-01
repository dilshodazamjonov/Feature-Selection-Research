from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.selectors.combinations import (
    BorutaThenCatBoostRFESelector,
    BorutaThenMutualInformationMRMRSelector,
    CombinationContractError,
    IVThenBorutaSelector,
    STATISTICAL_VOTERS,
    StatisticalNormalizedAverageRankSelector,
)
from credit_risk_fs.selectors.lightweight.contract import (
    SelectionResult,
    ordered_name_hash,
    training_identity_hash,
)


LOCK = "a" * 64
IMPLEMENTATIONS = {
    "iv_woe": "iv_woe_quantile_binned_v1",
    "lasso_l1_logistic": "lasso_l1_logistic_v1",
    "rfe_catboost": "rfe_catboost_fractional_step_v1",
    "boruta_random_forest": "boruta_random_forest_confirmed_tentative_v1",
    "catboost_shap": "catboost_native_shap_regular_mean_abs_train_sample_v1",
    "mrmr_mutual_information": "mrmr_mutual_information_discrete_plugin_v1",
}


def _frame(n_features: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        np.arange(20 * n_features, dtype=float).reshape(20, n_features),
        columns=[f"f{i:02d}" for i in range(n_features)],
        index=[f"row-{i}" for i in range(20)],
    )
    return X, pd.Series([0, 1] * 10)


class StubFactory:
    def __init__(self, method: str, *, support: int | None = None, score_mode: str = "descending"):
        self.method = method
        self.support = support
        self.score_mode = score_mode
        self.seen_columns: list[str] | None = None
        self.result: SelectionResult | None = None
        self.selected_features_: list[str] = []

    def __call__(self, **kwargs):
        method, support, score_mode, outer = self.method, self.support, self.score_mode, self

        class Fitted:
            def fit(self, X, y):
                outer.seen_columns = list(X.columns)
                k = kwargs.get("k")
                count = min(len(X.columns), support if support is not None else (k or len(X.columns)))
                selected = list(X.columns[:count])
                ranking = list(X.columns)
                scores = {name: float(len(X.columns) - index) for index, name in enumerate(X.columns)}
                heavy = None
                mode = "matched_budget"
                requested = k
                status = "satisfied"
                if method == "boruta_random_forest":
                    mode = "natural_confirmed"
                    requested = None
                    status = "not_applicable"
                    states = {
                        name: ("confirmed" if index < count else "rejected")
                        for index, name in enumerate(X.columns)
                    }
                    heavy = {"support_states": states, "engine_ranking": {name: index + 1 for index, name in enumerate(X.columns)}}
                if method == "lasso_l1_logistic" and score_mode == "zero_tie":
                    scores = {name: (1.0 if index == 0 else 0.0) for index, name in enumerate(X.columns)}
                self.selected_features_ = selected
                self.result = SelectionResult(
                    method_id=method,
                    display_label=method,
                    implementation_id=IMPLEMENTATIONS[method],
                    selection_mode=mode,
                    supervised=True,
                    selected_features=tuple(selected),
                    candidate_universe=tuple(X.columns),
                    requested_budget=requested,
                    budget_status=status,
                    score_orientation="rank_1_is_best" if method == "rfe_catboost" else "higher_is_better",
                    tie_rule="synthetic",
                    fit_scope=kwargs.get("fit_scope", "dev_fold_training_only"),
                    seed=kwargs.get("random_state", 42),
                    configuration=kwargs,
                    ranking=tuple(ranking),
                    raw_scores=scores,
                    natural_selected=tuple(selected),
                    training_row_count=len(X),
                    training_identity_sha256=training_identity_hash(X, y),
                    estimator_config_sha256=hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest(),
                    heavy_metadata=heavy,
                )
                outer.result = self.result
                outer.selected_features_ = selected
                return self

        return Fitted()


def test_iv_then_boruta_never_recovers_an_iv_excluded_feature() -> None:
    X, y = _frame(320)
    iv = StubFactory("iv_woe")
    boruta = StubFactory("boruta_random_forest", support=17)
    selector = IVThenBorutaSelector(
        iv_pool_budget=100, protocol_lock_sha256=LOCK, iv_factory=iv, boruta_factory=boruta
    ).fit(X, y)
    assert len(selector.result.intermediate_features) == 100
    assert len(selector.selected_features_) == 17
    assert boruta.seen_columns == list(X.columns[:100])
    assert set(selector.selected_features_) <= set(selector.result.intermediate_features)


@pytest.mark.parametrize(
    ("support", "state", "refiner_fits"),
    [(19, "infeasible_natural_support", False), (20, "no_refinement_possible", False), (25, "completed", True)],
)
def test_boruta_chain_support_semantics(support: int, state: str, refiner_fits: bool) -> None:
    X, y = _frame()
    boruta = StubFactory("boruta_random_forest", support=support)
    refiner = StubFactory("rfe_catboost")
    selector = BorutaThenCatBoostRFESelector(
        k=20, protocol_lock_sha256=LOCK, boruta_factory=boruta, refiner_factory=refiner
    ).fit(X, y)
    assert selector.result.feasibility_state == state
    assert (refiner.seen_columns is not None) is refiner_fits
    assert set(selector.selected_features_) <= set(X.columns[:support])
    assert len(selector.selected_features_) == min(support, 20)


def test_boruta_mrmr_uses_the_canonical_mi_configuration() -> None:
    X, y = _frame()
    refiner = StubFactory("mrmr_mutual_information")
    selector = BorutaThenMutualInformationMRMRSelector(
        k=20,
        protocol_lock_sha256=LOCK,
        boruta_factory=StubFactory("boruta_random_forest", support=30),
        refiner_factory=refiner,
    ).fit(X, y)
    assert selector.result.component_ids[-1] == "mrmr_mutual_information"
    assert selector.result.configuration["refiner_kwargs"] == {"n_bins": 10, "objective": "mid"}


def _component_results(X: pd.DataFrame, y: pd.Series) -> dict[str, SelectionResult]:
    output = {}
    for method in STATISTICAL_VOTERS:
        factory = StubFactory(method, support=20, score_mode="zero_tie" if method == "lasso_l1_logistic" else "descending")
        fitted = factory(random_state=42, fit_scope="dev_fold_training_only", k=20).fit(X, y)
        output[method] = fitted.result
    return output


def test_statistical_voter_is_exact_equal_weight_midrank_and_top_k() -> None:
    X, y = _frame()
    selector = StatisticalNormalizedAverageRankSelector(
        k=20, protocol_lock_sha256=LOCK, component_results=_component_results(X, y)
    ).fit(X, y)
    evidence = selector.voting_frame_.set_index("feature")
    assert selector.result.component_ids == STATISTICAL_VOTERS
    assert all(evidence[f"{method}__weight"].eq(0.2).all() for method in STATISTICAL_VOTERS)
    # LASSO's 49 zero coefficients are one valid tied block.
    zero_quality = evidence.loc[X.columns[1:], "lasso_l1_logistic__normalized_quality"]
    assert zero_quality.nunique() == 1
    assert len(selector.selected_features_) == 20
    assert selector.result.configuration["outcome_weighting"] is False


def test_statistical_voter_rejects_near_match_component_identity() -> None:
    X, y = _frame()
    components = _component_results(X, y)
    components = {**components, "mrmr_mutual_information": components.pop("catboost_shap")}
    with pytest.raises(CombinationContractError, match="exact ordered five-voter"):
        StatisticalNormalizedAverageRankSelector(
            k=20, protocol_lock_sha256=LOCK, component_results=components
        ).fit(X, y)


def test_statistical_voter_rejects_candidate_universe_mismatch() -> None:
    X, y = _frame()
    components = _component_results(X, y)
    bad = components["iv_woe"]
    components["iv_woe"] = SelectionResult(
        **{**bad.__dict__, "candidate_universe": tuple(reversed(bad.candidate_universe))}
    )
    with pytest.raises(CombinationContractError, match="candidate-universe"):
        StatisticalNormalizedAverageRankSelector(
            k=20, protocol_lock_sha256=LOCK, component_results=components
        ).fit(X, y)
