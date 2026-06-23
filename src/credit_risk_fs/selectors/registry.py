from __future__ import annotations


def get_selector(selector_name: str):
    """Return selector class plus conservative default kwargs."""
    name = selector_name.lower()

    if name in ("boruta", "boruta_rfe"):
        from credit_risk_fs.selectors.boruta import BorutaRFESelector

        return BorutaRFESelector, {
            "boruta_kwargs": {"max_iter": 15, "random_state": 42},
            "rfe_kwargs": {"n_features": 40, "step": 10, "random_state": 42},
            "use_rfe": False,
            "n_features": 40,
        }

    if name == "mrmr":
        from credit_risk_fs.selectors.mrmr import MRMR

        return MRMR, {"k": 50, "method": "mrmr", "random_state": 42}

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
        from credit_risk_fs.selectors.mrmr import MRMR

        return LLMThenStatSelector, {
            "description_csv_path": None,
            "stat_selector_cls": MRMR,
            "stat_selector_kwargs": {"k": 40, "method": "mrmr", "random_state": 42},
            "cache_dir": "artifacts/llm_cache",
        }

    if name == "llm_then_boruta":
        from credit_risk_fs.selectors.boruta import BorutaRFESelector
        from credit_risk_fs.selectors.llm_then_stat import LLMThenStatSelector

        return LLMThenStatSelector, {
            "description_csv_path": None,
            "stat_selector_cls": BorutaRFESelector,
            "stat_selector_kwargs": {
                "boruta_kwargs": {"max_iter": 15, "random_state": 42},
                "rfe_kwargs": {"n_features": 40, "step": 10, "random_state": 42},
                "use_rfe": False,
                "n_features": 40,
            },
            "cache_dir": "artifacts/llm_cache",
        }

    if name == "clip":
        from credit_risk_fs.selectors.clip_screening import ClipScreeningSelector

        return ClipScreeningSelector, {
            "config_path": "configs/clip/selector.yaml",
            "dataset": "homecredit",
            "feature_budget": 40,
            "missing_feature_policy": "error",
        }

    if name == "clip_v2":
        from credit_risk_fs.selectors.clip_screening import ClipScreeningSelector

        return ClipScreeningSelector, {
            "config_path": "configs/clip_v2/selector.yaml",
            "dataset": "homecredit",
            "feature_budget": 40,
            "missing_feature_policy": "exclude_with_manifest",
            "selector_label": "clip_v2",
        }

    if name == "clip_then_mrmr":
        from credit_risk_fs.selectors.clip_then_mrmr import ClipThenMRMRSelector

        return ClipThenMRMRSelector, {
            "config_path": "configs/clip/selector.yaml",
            "dataset": "homecredit",
            "feature_budget": 40,
            "screening_pool_size": 100,
            "missing_feature_policy": "error",
            "random_state": 42,
        }

    if name == "clip_v2_then_mrmr":
        from credit_risk_fs.selectors.clip_then_mrmr import ClipThenMRMRSelector

        return ClipThenMRMRSelector, {
            "config_path": "configs/clip_v2/selector.yaml",
            "dataset": "homecredit",
            "feature_budget": 40,
            "screening_pool_size": 100,
            "missing_feature_policy": "exclude_with_manifest",
            "random_state": 42,
            "selector_label": "clip_v2_then_mrmr",
        }

    if name == "domain_rule_baseline":
        from credit_risk_fs.selectors.domain_rule_baseline import (
            DomainRuleBaselineSelector,
        )

        return DomainRuleBaselineSelector, {
            "description_csv_path": None,
            "feature_budget": 40,
        }

    if name == "stable_core_llm_fill":
        from credit_risk_fs.selectors.stable_core_llm_fill import (
            StableCoreLLMFillSelector,
        )

        return StableCoreLLMFillSelector, {}

    if name in {"none", ""}:
        return None, {}

    raise ValueError(
        f"Unsupported selector: {selector_name}. "
        "Available: boruta, boruta_rfe, mrmr, pca, llm, llm_then_stat, "
        "llm_then_mrmr, llm_then_boruta, clip, clip_then_mrmr, clip_v2, "
        "clip_v2_then_mrmr, domain_rule_baseline, "
        "stable_core_llm_fill, none"
    )
