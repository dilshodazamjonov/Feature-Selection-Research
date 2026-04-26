import json
from pathlib import Path

import pandas as pd

from feature_selection.llm_selector import LLMSelector


def _description(path: Path) -> Path:
    path.write_text(
        "row,description,table\n"
        "f1,Feature one,application_train\n"
        "f2,Feature two,application_train\n"
        "f3,Feature three,application_train\n",
        encoding="utf-8",
    )
    return path


def _selector(tmp_path, feature_budget=2):
    selector = LLMSelector(
        description_csv_path=str(_description(tmp_path / "descriptions.csv")),
        cache_dir=str(tmp_path / "cache"),
        max_features=3,
        ranking_budget=3,
        feature_budget=feature_budget,
        config_hash="cfg",
        iv_filter_kwargs={},
    )
    selector._call_llm = lambda prompt: {
        "selected_features": ["f1", "f2", "f3"],
        "feature_reasons": {"f1": "best"},
        "reasoning_summary": "ranked",
        "raw_response": json.dumps({"selected_features": ["f1", "f2", "f3"]}),
    }
    selector.set_ranking_context(
        scope="fold",
        fold_id=1,
        ranking_artifact_dir=tmp_path / "rankings",
    )
    return selector


def test_shared_llm_ranking_cache_reuses_same_fold_and_signature(tmp_path):
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [2.0, 3.0], "f3": [3.0, 4.0]})
    y = pd.Series([0, 1])

    first = _selector(tmp_path, feature_budget=2).fit(X, y)
    second = _selector(tmp_path, feature_budget=3).fit(X, y)

    assert first.llm_calls_made_ == 1
    assert second.llm_cache_hits_ == 1
    assert first.ranked_features_ == ["f1", "f2", "f3"]
    assert first.selected_features_ == ["f1", "f2"]
    assert second.selected_features_ == ["f1", "f2", "f3"]
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_shared_llm_ranking_cache_keeps_fold_and_final_dev_separate(tmp_path):
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [2.0, 3.0], "f3": [3.0, 4.0]})
    y = pd.Series([0, 1])

    fold_selector = _selector(tmp_path, feature_budget=2)
    fold_selector.set_ranking_context(scope="fold", fold_id=1)
    fold_selector.fit(X, y)

    final_selector = _selector(tmp_path, feature_budget=2)
    final_selector.set_ranking_context(scope="final_dev", fold_id=None)
    final_selector.fit(X, y)

    assert len(list((tmp_path / "cache").glob("*.json"))) == 2
    assert final_selector.llm_calls_made_ == 1


def test_shared_llm_ranking_expected_six_calls_for_five_folds_plus_final_dev(tmp_path):
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [2.0, 3.0], "f3": [3.0, 4.0]})
    y = pd.Series([0, 1])

    first_pass_calls = 0
    second_pass_cache_hits = 0
    for fold_id in range(1, 6):
        selector = _selector(tmp_path, feature_budget=2)
        selector.set_ranking_context(scope="fold", fold_id=fold_id)
        selector.fit(X, y)
        first_pass_calls += selector.llm_calls_made_

    final_selector = _selector(tmp_path, feature_budget=2)
    final_selector.set_ranking_context(scope="final_dev", fold_id=None)
    final_selector.fit(X, y)
    first_pass_calls += final_selector.llm_calls_made_

    for fold_id in range(1, 6):
        selector = _selector(tmp_path, feature_budget=3)
        selector.set_ranking_context(scope="fold", fold_id=fold_id)
        selector.fit(X, y)
        second_pass_cache_hits += selector.llm_cache_hits_

    final_selector = _selector(tmp_path, feature_budget=3)
    final_selector.set_ranking_context(scope="final_dev", fold_id=None)
    final_selector.fit(X, y)
    second_pass_cache_hits += final_selector.llm_cache_hits_

    assert first_pass_calls == 6
    assert second_pass_cache_hits == 6
    assert len(list((tmp_path / "cache").glob("*.json"))) == 6
