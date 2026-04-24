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
        "raw_response": json.dumps({"selected_features": ["f2"]}),
    }
    selector_second.fit(X_second, y_second)

    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 2
    assert selector_first.selected_features == ["f1"]
    assert selector_second.selected_features == ["f2"]
