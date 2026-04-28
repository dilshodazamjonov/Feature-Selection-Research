import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from feature_selection.llm_selector import LLMSelector


def test_llm_selector_cache_is_keyed_by_training_signature():
    runtime_dir = Path("tests_runtime") / uuid4().hex
    runtime_dir.mkdir(parents=True, exist_ok=True)

    description_path = runtime_dir / "descriptions.csv"
    description_path.write_text(
        "row,description,table\nf1,Feature one,application_train\nf2,Feature two,application_train\n",
        encoding="utf-8",
    )

    cache_dir = runtime_dir / "cache"

    X_first = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.0, 0.0, 1.0]})
    y_first = pd.Series([0, 0, 1])
    selector_first = LLMSelector(
        description_csv_path=str(description_path),
        cache_dir=str(cache_dir),
        iv_filter_kwargs={},
        max_features=1,
    )
    selector_first._call_llm = lambda prompt: {
        "selected_features": ["f1"],
        "reasoning_summary": "pick f1",
        "selection_principles": ["stability"],
        "raw_response": json.dumps({"selected_features": ["f1"]}),
    }
    selector_first.fit(X_first, y_first)

    X_second = pd.DataFrame({"f1": [0.0, 0.0, 0.0], "f2": [1.0, 2.0, 3.0]})
    y_second = pd.Series([0, 1, 1])
    selector_second = LLMSelector(
        description_csv_path=str(description_path),
        cache_dir=str(cache_dir),
        iv_filter_kwargs={},
        max_features=1,
    )
    selector_second._call_llm = lambda prompt: {
        "selected_features": ["f2"],
        "reasoning_summary": "pick f2",
        "selection_principles": ["behavioral signal"],
        "raw_response": json.dumps({"selected_features": ["f2"]}),
    }
    selector_second.fit(X_second, y_second)

    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 2
    assert selector_first.selected_features == ["f1"]
    assert selector_second.selected_features == ["f2"]
    assert selector_first.selection_payload_["selection_principles"] == ["stability"]
    assert selector_second.selection_payload_["selection_principles"] == ["behavioral signal"]


def test_llm_selector_call_llm_parses_selection_principles():
    selector = LLMSelector(
        description_csv_path="unused.csv",
        cache_dir="unused_cache",
        iv_filter_kwargs={},
        max_features=2,
        ranking_budget=2,
    )

    class _DummyUsage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class _DummyMessage:
        content = json.dumps(
            {
                "selected_features": ["f1", "f2", "f1"],
                "reasoning_summary": "High-level reasoning.",
                "selection_principles": ["stability", "low missingness", "stability", ""],
            }
        )

    class _DummyChoice:
        message = _DummyMessage()

    class _DummyResponse:
        choices = [_DummyChoice()]
        usage = _DummyUsage()
        model = "mock-model"
        id = "mock-response"

    class _DummyCompletions:
        def create(self, **kwargs):
            return _DummyResponse()

    class _DummyChat:
        completions = _DummyCompletions()

    class _DummyClient:
        chat = _DummyChat()

    selector._client = _DummyClient()

    payload = selector._call_llm("prompt")

    assert payload["selected_features"] == ["f1", "f2"]
    assert payload["reasoning_summary"] == "High-level reasoning."
    assert payload["selection_principles"] == ["stability", "low missingness"]


def test_llm_shared_ranking_can_produce_top_100(tmp_path):
    description_path = tmp_path / "descriptions.csv"
    rows = ["row,description,table"]
    columns = {}
    for i in range(120):
        feature = f"f{i}"
        rows.append(f"{feature},Feature {i},application_train")
        columns[feature] = [float(i), float(i + 1), float(i + 2)]
    description_path.write_text("\n".join(rows), encoding="utf-8")

    selector = LLMSelector(
        description_csv_path=str(description_path),
        cache_dir=str(tmp_path / "cache"),
        max_features=100,
        ranking_budget=100,
        feature_budget=20,
        iv_filter_kwargs={},
    )
    selector._call_llm = lambda prompt: {
        "selected_features": [f"f{i}" for i in range(120)],
        "reasoning_summary": "broad ranking",
        "selection_principles": ["stability"],
        "raw_response": json.dumps({"selected_features": [f"f{i}" for i in range(120)]}),
    }

    selector.fit(pd.DataFrame(columns), pd.Series([0, 1, 0]))

    assert len(selector.ranked_features_) == 100
    assert selector.selected_features_ == [f"f{i}" for i in range(20)]


def test_cache_invalidates_when_prompt_version_changes(tmp_path):
    description_path = tmp_path / "descriptions.csv"
    description_path.write_text(
        "row,description,table\nf1,Feature one,application_train\n",
        encoding="utf-8",
    )
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 0])

    for prompt_version in ["stability_expert_v2", "stability_expert_v3"]:
        selector = LLMSelector(
            description_csv_path=str(description_path),
            cache_dir=str(tmp_path / "cache"),
            ranking_budget=1,
            feature_budget=1,
            prompt_version=prompt_version,
            iv_filter_kwargs={},
        )
        selector._call_llm = lambda prompt: {
            "selected_features": ["f1"],
            "reasoning_summary": "ranked",
            "selection_principles": ["stability"],
            "raw_response": json.dumps({"selected_features": ["f1"]}),
        }
        selector.fit(X, y)

    assert len(list((tmp_path / "cache").glob("*.json"))) == 2


def test_missing_invalid_llm_ranked_features_are_filtered_safely(tmp_path):
    description_path = tmp_path / "descriptions.csv"
    description_path.write_text(
        "row,description,table\nf1,Feature one,application_train\nf2,Feature two,application_train\nf3,Feature three,application_train\n",
        encoding="utf-8",
    )
    selector = LLMSelector(
        description_csv_path=str(description_path),
        cache_dir=str(tmp_path / "cache"),
        ranking_budget=3,
        feature_budget=3,
        iv_filter_kwargs={},
    )
    selector._call_llm = lambda prompt: {
        "selected_features": ["missing_a", "f2", "missing_b"],
        "reasoning_summary": "ranked",
        "selection_principles": ["stability"],
        "raw_response": json.dumps({"selected_features": ["missing_a", "f2", "missing_b"]}),
    }

    selector.fit(
        pd.DataFrame({"f1": [1.0, 2.0], "f2": [2.0, 3.0], "f3": [3.0, 4.0]}),
        pd.Series([0, 1]),
    )

    assert selector.ranked_features_ == ["f2", "f1", "f3"]
    assert selector.selected_features_ == ["f2", "f1", "f3"]
    assert selector.selection_payload_["filtered_invalid_features"] == ["missing_a", "missing_b"]


def test_lr_llm_only_truncates_broad_ranking_to_top20(tmp_path):
    description_path = tmp_path / "descriptions.csv"
    rows = ["row,description,table"]
    columns = {}
    for i in range(70):
        feature = f"f{i}"
        rows.append(f"{feature},Feature {i},application_train")
        columns[feature] = [float(i), float(i + 1), float(i + 2)]
    description_path.write_text("\n".join(rows), encoding="utf-8")

    selector = LLMSelector(
        description_csv_path=str(description_path),
        cache_dir=str(tmp_path / "cache"),
        ranking_budget=60,
        feature_budget=20,
        iv_filter_kwargs={},
    )
    selector._call_llm = lambda prompt: {
        "selected_features": [f"f{i}" for i in range(70)],
        "reasoning_summary": "ranked",
        "selection_principles": ["stability"],
        "raw_response": json.dumps({"selected_features": [f"f{i}" for i in range(70)]}),
    }

    selector.fit(pd.DataFrame(columns), pd.Series([0, 1, 0]))

    assert len(selector.ranked_features_) == 60
    assert selector.selected_features_ == [f"f{i}" for i in range(20)]


def test_catboost_llm_only_truncates_broad_ranking_to_top40(tmp_path):
    description_path = tmp_path / "descriptions.csv"
    rows = ["row,description,table"]
    columns = {}
    for i in range(110):
        feature = f"f{i}"
        rows.append(f"{feature},Feature {i},application_train")
        columns[feature] = [float(i), float(i + 1), float(i + 2)]
    description_path.write_text("\n".join(rows), encoding="utf-8")

    selector = LLMSelector(
        description_csv_path=str(description_path),
        cache_dir=str(tmp_path / "cache"),
        ranking_budget=100,
        feature_budget=40,
        iv_filter_kwargs={},
    )
    selector._call_llm = lambda prompt: {
        "selected_features": [f"f{i}" for i in range(110)],
        "reasoning_summary": "ranked",
        "selection_principles": ["stability"],
        "raw_response": json.dumps({"selected_features": [f"f{i}" for i in range(110)]}),
    }

    selector.fit(pd.DataFrame(columns), pd.Series([0, 1, 0]))

    assert len(selector.ranked_features_) == 100
    assert selector.selected_features_ == [f"f{i}" for i in range(40)]
