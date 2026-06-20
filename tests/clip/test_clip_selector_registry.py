from __future__ import annotations

from credit_risk_fs.experiments.config import apply_feature_budget_to_selector_kwargs
from credit_risk_fs.selectors.clip_screening import ClipScreeningSelector
from credit_risk_fs.selectors.clip_then_mrmr import ClipThenMRMRSelector
from credit_risk_fs.selectors.llm_screening import LLMSelector
from credit_risk_fs.selectors.registry import get_selector


def test_clip_selectors_are_registered_without_changing_llm_selector():
    llm_cls, llm_kwargs = get_selector("llm")
    clip_cls, clip_kwargs = get_selector("clip")
    hybrid_cls, hybrid_kwargs = get_selector("clip_then_mrmr")

    assert llm_cls is LLMSelector
    assert llm_kwargs["model"] == "gpt-4.1-mini"
    assert clip_cls is ClipScreeningSelector
    assert hybrid_cls is ClipThenMRMRSelector
    assert clip_kwargs["config_path"] == "configs/clip/selector.yaml"
    assert hybrid_kwargs["screening_pool_size"] == 100


def test_clip_selector_budgets_are_model_specific():
    assert apply_feature_budget_to_selector_kwargs("clip", {}, 20)["feature_budget"] == 20
    assert apply_feature_budget_to_selector_kwargs("clip_then_mrmr", {}, 40)["feature_budget"] == 40
