from __future__ import annotations

import pandas as pd
import pytest

from scripts import build_clip_final_report_figures as figures


def test_final_report_fixed_method_set_is_prespecified():
    assert figures.FIXED_METHODS == ["clip", "clip_then_mrmr", "mrmr", "llm", "llm_then_mrmr"]


def test_manifest_rejects_more_than_five_main_methods():
    with pytest.raises(ValueError, match="more than five methods"):
        figures._manifest_row(
            file="x.png",
            question="q",
            methods=["a", "b", "c", "d", "e", "f"],
            scope="scope",
            source_artifacts=[],
            source_columns=[],
            main_finding="finding",
            limitation="limitation",
            reason="reason",
        )


def test_semantic_plot_frame_uses_selected_feature_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    features_dir = run_dir / "features"
    features_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "feature_name": ["A_GROUP_MEAN", "A_GROUP_MAX", "B"],
            "semantic_group": ["income", "income", "credit"],
        }
    ).to_csv(features_dir / "final_selected_features.csv", index=False)
    comparison = pd.DataFrame(
        [
            {
                "dataset_name": "homecredit",
                "model": "lr",
                "selector": "clip",
                "result_origin": "clip_extension",
                "run_dir": str(run_dir),
                "output_folder": "",
            }
        ]
    )

    frame = figures.build_semantic_plot_frame(comparison)

    assert frame.loc[0, "semantic_group_count"] == 2
    assert frame.loc[0, "repeated_base_family_share"] == pytest.approx(2 / 3)
