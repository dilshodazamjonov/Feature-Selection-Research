from __future__ import annotations

# Prompt 7 lightweight controls. Listed by name rather than imported eagerly so
# resolving an unrelated selector does not pull in scikit-learn estimators.
_LIGHTWEIGHT_IDS = frozenset(
    {
        "iv_woe",
        "mrmr_mutual_information",
        "lasso_l1_logistic",
        "random_k",
        "full_features",
        "none_explicit",
        # Prompt 8 heavy methods. They share the same contract and registry, so
        # they resolve through the same descriptor path; only their cost class and
        # estimator differ. The historical "rfe" and "boruta" routes above are
        # untouched and continue to resolve to the legacy implementations.
        "rfe_catboost",
        "boruta_random_forest",
        "catboost_shap",
    }
)


def get_selector(selector_name: str):
    """Return a canonical selector class and deterministic default kwargs."""

    name = selector_name.lower()

    if name == "boruta":
        from credit_risk_fs.selectors.boruta import BorutaSelector

        return BorutaSelector, {
            "max_iter": 15,
            "random_state": 42,
            "n_features": 40,
            "n_jobs": 1,
        }

    if name == "boruta_rfe":
        from credit_risk_fs.selectors.boruta_then_rfe import BorutaThenRFESelector

        return BorutaThenRFESelector, {
            "boruta_kwargs": {"max_iter": 15, "random_state": 42, "n_jobs": 1},
            "rfe_kwargs": {"n_features": 40, "step": 10, "random_state": 42, "thread_count": 1},
            "use_rfe": True,
            "n_features": 40,
        }

    if name in {
        "iv_then_boruta",
        "boruta_then_rfe_catboost",
        "boruta_then_mrmr_mutual_information",
        "statistical_normalized_average_rank",
    }:
        from credit_risk_fs.selectors.combinations import COMBINATION_CLASSES

        # Scientific parameters are intentionally not defaulted here: callers
        # must supply the committed protocol-lock hash and the applicable frozen
        # candidate/final budget.  That prevents an ad-hoc registry lookup from
        # creating an unauthenticated combination configuration.
        return COMBINATION_CLASSES[name], {}

    if name == "rfe":
        from credit_risk_fs.selectors.rfe import RFESelector

        return RFESelector, {
            "n_features": 40,
            "step": 10,
            "random_state": 42,
            "thread_count": 1,
        }

    # "mrmr" is a historical alias. It has always meant the random-forest
    # relevance / absolute-correlation redundancy selector, so it keeps resolving
    # there and every artifact written under that name stays truthful. Canonical
    # mutual-information mRMR is a separate ID ("mrmr_mutual_information") and is
    # deliberately unreachable from this alias.
    if name in {"mrmr", "legacy_rf_relevance_corr"}:
        from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector

        return RandomForestRelevanceMRMRSelector, {
            "k": 50,
            "method": "mrmr",
            "random_state": 42,
            "n_jobs": 1,
        }

    if name in _LIGHTWEIGHT_IDS:
        from credit_risk_fs.selectors.lightweight.registry import get_method_descriptor

        descriptor = get_method_descriptor(name)
        return descriptor.load(), dict(descriptor.default_kwargs)

    if name == "pca":
        from credit_risk_fs.selectors.pca import PCASelector

        return PCASelector, {"n_components": 0.95, "save_dir": None, "random_state": 42}

    if name == "llm":
        from credit_risk_fs.selectors.llm_screening import LLMSelector

        return LLMSelector, {
            "description_csv_path": None,
            "cache_dir": "artifacts/llm_cache",
            "model": "gpt-4.1-mini",
            "temperature": 0.0,
            "max_features": 100,
            "max_missing_rate": 0.95,
            "iv_filter_kwargs": {
                "min_iv": 0.01,
                "max_iv_for_leakage": 0.5,
                "encode": True,
                "n_jobs": 1,
                "verbose": False,
            },
        }

    if name in {"llm_then_stat", "llm_then_mrmr"}:
        from credit_risk_fs.selectors.llm_then_stat import LLMThenStatSelector
        from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector

        return LLMThenStatSelector, {
            "description_csv_path": None,
            "stat_selector_cls": RandomForestRelevanceMRMRSelector,
            "stat_selector_kwargs": {"k": 40, "method": "mrmr", "random_state": 42, "n_jobs": 1},
            "cache_dir": "artifacts/llm_cache",
        }

    if name == "llm_then_boruta":
        from credit_risk_fs.selectors.boruta import BorutaSelector
        from credit_risk_fs.selectors.llm_then_stat import LLMThenStatSelector

        return LLMThenStatSelector, {
            "description_csv_path": None,
            "stat_selector_cls": BorutaSelector,
            "stat_selector_kwargs": {
                "max_iter": 15,
                "random_state": 42,
                "n_features": 40,
                "n_jobs": 1,
            },
            "cache_dir": "artifacts/llm_cache",
        }

    if name == "domain_rule_baseline":
        from credit_risk_fs.selectors.domain_rule_baseline import DomainRuleBaselineSelector

        return DomainRuleBaselineSelector, {
            "description_csv_path": None,
            "feature_budget": 40,
        }

    if name == "stable_core_llm_fill":
        from credit_risk_fs.selectors.stable_core_llm_fill import StableCoreLLMFillSelector

        return StableCoreLLMFillSelector, {}

    if name in {"none", ""}:
        return None, {}

    raise ValueError(
        f"Unsupported selector: {selector_name}. "
        "Available: boruta, boruta_rfe, rfe, mrmr, legacy_rf_relevance_corr, pca, "
        "llm, llm_then_stat, llm_then_mrmr, llm_then_boruta, domain_rule_baseline, "
        "stable_core_llm_fill, iv_woe, mrmr_mutual_information, lasso_l1_logistic, "
        "random_k, full_features, rfe_catboost, boruta_random_forest, catboost_shap, "
        "iv_then_boruta, boruta_then_rfe_catboost, "
        "boruta_then_mrmr_mutual_information, statistical_normalized_average_rank, "
        "none"
    )
