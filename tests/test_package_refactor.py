from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk_fs.data.dataset_registry import get_dataset_config
from credit_risk_fs.evaluation.redundancy import correlation_redundancy_frame, redundancy_summary
from credit_risk_fs.evaluation.semantic_coverage import semantic_coverage_frame, semantic_coverage_summary
from credit_risk_fs.experiments.config import experiment_config_path, load_project_config
from credit_risk_fs.pipelines.common import ExperimentConfig, _resolve_selector, create_run_output_dir
from credit_risk_fs.preprocessing.leakage import apply_leakage_blacklist
from credit_risk_fs.preprocessing.temporal_split import split_dev_oot
from credit_risk_fs.selectors.registry import get_selector


def test_credit_risk_fs_import_surface_is_available():
    homecredit = get_dataset_config("homecredit")
    lendingclub = get_dataset_config("lendingclub")

    assert homecredit.name == "homecredit"
    assert lendingclub.name == "lendingclub"
    assert homecredit.mode == "homecredit_multitable"
    assert lendingclub.mode == "single_table"


def test_matrix_config_loading_works_for_both_datasets():
    homecredit = load_project_config(experiment_config_path("homecredit", "matrix"))
    lendingclub = load_project_config(experiment_config_path("lendingclub", "matrix"))

    assert homecredit["dataset_name"] == "homecredit"
    assert lendingclub["dataset_name"] == "lendingclub"
    assert homecredit["data_dir"].endswith("data/homecredit/raw")
    assert lendingclub["data_dir"].endswith("data/lendingclub/processed")


def test_temporal_split_keeps_future_rows_out_of_dev():
    df = pd.DataFrame(
        {
            "recent_decision": [-5, -4, -3, -2, -1, 0],
            "TARGET": [0, 1, 0, 1, 0, 1],
            "feature": [1, 2, 3, 4, 5, 6],
        }
    )

    dev, oot = split_dev_oot(
        df,
        time_col="recent_decision",
        target_col="TARGET",
        dev_start_day=-5,
        oot_start_day=-2,
        oot_end_day=0,
    )

    assert dev["recent_decision"].max() < oot["recent_decision"].min()


def test_leakage_blacklist_is_applied():
    df = pd.DataFrame(
        {
            "TARGET": [0, 1],
            "recent_decision": [-1, 0],
            "safe_feature": [1.0, 2.0],
        }
    )

    cleaned = apply_leakage_blacklist(df, ["TARGET", "recent_decision"])

    assert list(cleaned.columns) == ["safe_feature"]


def test_selector_registry_supports_core_research_selectors():
    for selector_name in ["mrmr", "llm", "llm_then_stat", "stable_core_llm_fill"]:
        selector_cls, selector_kwargs = get_selector(selector_name)
        assert selector_cls is not None
        assert isinstance(selector_kwargs, dict)


def test_hybrid_selectors_receive_description_path_and_cache_dir():
    config = ExperimentConfig(
        experiment_name="smoke",
        selector_name="llm_then_mrmr",
        dataset_name="lendingclub",
        description_path="data/lendingclub/metadata/columns_description.csv",
        feature_budget=20,
    )

    _, selector_kwargs = _resolve_selector(config)

    assert selector_kwargs["description_csv_path"] == "data/lendingclub/metadata/columns_description.csv"
    assert selector_kwargs["cache_dir"] == "artifacts/llm_cache"


def test_result_folder_helper_creates_run_directories(tmp_path):
    config = ExperimentConfig(
        experiment_name="smoke",
        selector_name="mrmr",
        dataset_name="homecredit",
        base_output_dir=str(tmp_path),
    )

    run_dir = create_run_output_dir(config.base_output_dir, f"{config.dataset_name}_{config.experiment_name}")

    assert run_dir.exists()
    assert Path(run_dir).parent == tmp_path


def test_semantic_coverage_and_redundancy_run_on_toy_feature_sets():
    coverage = semantic_coverage_frame(["EXT_SOURCE_1", "BURO_AMT_CREDIT_SUM_MEAN", "AMT_INCOME_TOTAL"])
    summary = semantic_coverage_summary(["EXT_SOURCE_1", "BURO_AMT_CREDIT_SUM_MEAN", "AMT_INCOME_TOTAL"])
    redundancy_frame = correlation_redundancy_frame(
        pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0],
                "f2": [0.0, 1.0, 2.0, 3.0],
                "f3": [3.0, 2.0, 1.0, 0.0],
            }
        ),
        threshold=0.95,
    )
    redundancy = redundancy_summary(
        pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0],
                "f2": [0.0, 1.0, 2.0, 3.0],
                "f3": [3.0, 2.0, 1.0, 0.0],
            }
        ),
        threshold=0.95,
    )

    assert not coverage.empty
    assert summary["semantic_group_count"] >= 1
    assert not redundancy_frame.empty
    assert redundancy["redundant_pair_count"] >= 1
