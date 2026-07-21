from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import generate_homecredit_domain_rule_ranking as canonical
from scripts import generate_homecredit_llm_feature_ranking as legacy


def _metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_id": "2",
                "feature_name": "AMT_CREDIT",
                "source_table": "application_train",
                "semantic_group": "application_amounts",
            },
            {
                "feature_id": "1",
                "feature_name": "EXT_SOURCE_1",
                "source_table": "application_train",
                "semantic_group": "external_score",
            },
        ]
    )


def test_domain_rule_ranking_is_deterministic_and_accurately_named():
    first = canonical.build_ranking(_metadata_frame())
    second = canonical.build_ranking(_metadata_frame().sample(frac=1, random_state=9))

    pd.testing.assert_frame_equal(first, second)
    assert first["feature_name"].tolist() == ["EXT_SOURCE_1", "AMT_CREDIT"]
    assert "no llm" in (canonical.__doc__ or "").lower()
    assert "domain_rule" in canonical.OUTPUT_PATH.as_posix()
    assert "llm" not in canonical.OUTPUT_PATH.name.lower()


def test_legacy_script_delegates_with_visible_deprecation(monkeypatch):
    monkeypatch.setattr(legacy._canonical, "main", lambda: 17)

    with pytest.warns(FutureWarning, match="deterministic domain-rule ranker"):
        assert legacy.main() == 17
    assert legacy.build_ranking is canonical.build_ranking


def test_active_source_has_no_imports_from_pre_refactor_selector_owners():
    root = Path(__file__).resolve().parents[1]
    active_python = list((root / "src").rglob("*.py")) + list((root / "scripts").glob("*.py"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in active_python)

    assert "from credit_risk_fs.selectors.boruta import RFESelector" not in source
    assert "from credit_risk_fs.selectors.boruta import BorutaRFESelector" not in source
    assert "from credit_risk_fs.selectors.llm_then_stat import DomainRuleBaselineSelector" not in source
    assert "from credit_risk_fs.selectors.llm_then_stat import StableCoreLLMFillSelector" not in source
