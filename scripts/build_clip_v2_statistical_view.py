from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.statistical_preprocessor_v2 import RobustStatisticalPreprocessorV2  # noqa: E402
from credit_risk_fs.clip.statistical_schema_v2 import DESCRIPTOR_COLUMNS_V2  # noqa: E402
from credit_risk_fs.clip.statistical_view_v2 import build_statistical_view_frame  # noqa: E402
from credit_risk_fs.clip.v2_validation import validate_clip_v2_config, validate_no_v1_output_paths  # noqa: E402
from credit_risk_fs.experiments.config import _parse_simple_yaml, load_named_project_config  # noqa: E402
from credit_risk_fs.pipelines.common import ExperimentConfig, drop_excluded_feature_columns, prepare_modeling_data  # noqa: E402
from credit_risk_fs.utils.hashing import sha256_file, sha256_text  # noqa: E402
from credit_risk_fs.utils.io import read_json, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP-v2 DEV-only compact statistical view.")
    parser.add_argument("--config", default="configs/clip_v2/statistical_view.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print the build plan without loading datasets.")
    parser.add_argument("--status", action="store_true", help="Inspect expected output files without modifying them.")
    parser.add_argument("--execute", action="store_true", help="Actually compute and write CLIP-v2 statistical vectors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    errors = validate_clip_v2_config(config)
    validate_no_v1_output_paths([str(config.get("output_dir", "")), str(config.get("output_root", ""))])
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 1
    if args.status:
        print(json.dumps(_status_payload(config), indent=2, default=str))
        return 0
    if args.dry_run or not args.execute:
        payload = _plan_payload(config)
        payload["dry_run"] = True
        payload["requires_execute_for_full_build"] = True
        print(json.dumps(payload, indent=2, default=str))
        return 0
    return _execute(config)


def _plan_payload(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(config.get("output_dir", "results/clip_v2/statistical_view")))
    return {
        "status": "planned",
        "experiment_version": config["experiment_version"],
        "statistical_view_version": config["statistical_view_version"],
        "statistical_input_dimension": int(config["statistical_input_dimension"]),
        "descriptor_order": list(DESCRIPTOR_COLUMNS_V2),
        "output_dir": str(output_dir).replace("\\", "/"),
        "will_load_full_datasets": False,
        "will_train_model": False,
        "model_trained": False,
        "will_run_downstream_evaluation": False,
        "fit_boundary": "Home Credit training-split feature vectors only",
        "external_policy": "LendingClub v2 descriptors are transformed with the unchanged Home Credit-fitted scaler",
    }


def _status_payload(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(config.get("output_dir", "results/clip_v2/statistical_view")))
    expected = [
        "homecredit_statistical_vectors.parquet",
        "lendingclub_v2_statistical_vectors.parquet",
        "statistical_preprocessor.json",
        "statistical_feature_order.json",
        "statistical_anchor_manifest.json",
        "statistical_view_summary.json",
    ]
    return {
        "status": "complete" if all((output_dir / name).exists() for name in expected) else "incomplete",
        "output_dir": str(output_dir).replace("\\", "/"),
        "files": {name: (output_dir / name).exists() for name in expected},
        "model_trained": False,
        "downstream_evaluation_run": False,
    }


def _execute(config: dict[str, Any]) -> int:
    output_dir = Path(str(config.get("output_dir", "results/clip_v2/statistical_view")))
    manifest = read_json("results/clip/dry_run/training_manifest.json")
    training_features = pd.read_csv("results/clip/dry_run/training_features.csv")
    external_features = pd.read_csv("results/clip/dry_run/external_validation_features.csv")
    group_split = pd.read_csv("results/clip/text_baseline/homecredit_group_split.csv")
    anchors = pd.read_csv("results/clip/text_baseline/homecredit_anchor_features.csv")
    source_manifest_hash = sha256_file("results/clip/dry_run/training_manifest.json")

    home_prepared = _prepare_dev_frame("homecredit")
    lc_prepared = _prepare_dev_frame("lendingclub_v2")
    home_allowed = training_features["feature"].astype(str).tolist()
    lc_allowed = external_features["feature"].astype(str).tolist()
    home_data = home_prepared.reindex(columns=home_allowed)
    lc_data = lc_prepared.reindex(columns=lc_allowed)

    home_descriptors = build_statistical_view_frame(
        home_data,
        metadata=_metadata_type_frame(training_features),
        metadata_type_column="metadata_type",
    )
    lc_descriptors = build_statistical_view_frame(
        lc_data,
        metadata=_metadata_type_frame(external_features),
        metadata_type_column="metadata_type",
    )

    home_meta = _home_metadata(training_features, group_split, source_manifest_hash)
    lc_meta = _lc_metadata(external_features, source_manifest_hash)
    preprocessor = RobustStatisticalPreprocessorV2()
    train_features = set(group_split.loc[group_split["split"].astype(str).eq("train"), "feature_name"].astype(str))
    home_values = home_meta.merge(home_descriptors[["feature_name", *DESCRIPTOR_COLUMNS_V2]], on="feature_name", how="left")
    lc_values = lc_meta.merge(lc_descriptors[["feature_name", *DESCRIPTOR_COLUMNS_V2]], on="feature_name", how="left")
    fit_values = home_values[home_values["feature_name"].astype(str).isin(train_features)]
    preprocessor.fit(fit_values, dataset="homecredit", split="train")
    home_vectors = _build_vector_frame(metadata=home_meta, transformed=preprocessor.transform(home_values), preprocessor=preprocessor)
    lc_vectors = _build_vector_frame(metadata=lc_meta, transformed=preprocessor.transform(lc_values), preprocessor=preprocessor)

    output_dir.mkdir(parents=True, exist_ok=True)
    home_vectors.to_parquet(output_dir / "homecredit_statistical_vectors.parquet", index=False)
    lc_vectors.to_parquet(output_dir / "lendingclub_v2_statistical_vectors.parquet", index=False)
    write_json(output_dir / "statistical_preprocessor.json", preprocessor.to_state())
    write_json(
        output_dir / "statistical_feature_order.json",
        {
            "field_order": list(DESCRIPTOR_COLUMNS_V2),
            "input_field_hash": sha256_text(json.dumps(list(DESCRIPTOR_COLUMNS_V2))),
            "vector_dimension": len(DESCRIPTOR_COLUMNS_V2),
            "statistical_view_version": config["statistical_view_version"],
        },
    )
    anchor_manifest = _write_anchor(output_dir, home_vectors, anchors, preprocessor.preprocessor_hash_, source_manifest_hash)
    summary = {
        **_plan_payload(config),
        "dry_run": False,
        "homecredit_vectors": int(len(home_vectors)),
        "lendingclub_v2_vectors": int(len(lc_vectors)),
        "homecredit_train_fit_vectors": int(len(fit_values)),
        "preprocessor_hash": preprocessor.preprocessor_hash_,
        "anchor_hash": anchor_manifest["anchor_hash"],
        "source_manifest_hash": source_manifest_hash,
        "source_files": manifest.get("source_files", {}),
        "model_trained": False,
        "downstream_evaluation_run": False,
    }
    write_json(output_dir / "statistical_view_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _prepare_dev_frame(dataset: str) -> pd.DataFrame:
    project = load_named_project_config(dataset)
    config = ExperimentConfig(
        experiment_name="clip_v2_statistical_view",
        selector_name="mrmr",
        dataset_name=dataset,
        data_dir=str(project["data_dir"]),
        description_path=str(project["description_path"]),
        dev_start_day=int(project["dev_start_day"]),
        oot_start_day=int(project["oot_start_day"]),
        oot_end_day=int(project["oot_end_day"]),
        n_splits=int(project["n_splits"]),
        cv_gap_groups=int(project["cv_gap_groups"]),
        excluded_feature_columns=tuple(project.get("excluded_feature_columns", [])),
        preprocessor_kwargs=dict(project.get("preprocessor_kwargs", {})),
    )
    prepared = prepare_modeling_data(config)
    return drop_excluded_feature_columns(
        prepared.X_train,
        time_col=prepared.time_col,
        excluded_columns=config.excluded_feature_columns,
    )


def _metadata_type_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features[["feature", "clip_training_text"]].copy()
    frame = frame.rename(columns={"feature": "feature_name"})
    frame["metadata_type"] = frame["clip_training_text"].astype(str).str.extract(r"dtype=([^|]+)", expand=False).fillna("")
    return frame[["feature_name", "metadata_type"]]


def _home_metadata(features: pd.DataFrame, split: pd.DataFrame, source_manifest_hash: str) -> pd.DataFrame:
    home = features[["dataset", "feature", "semantic_group", "source_table"]].rename(
        columns={"feature": "feature_name", "source_table": "source_table_or_formula"}
    )
    split_cols = [
        "feature_name",
        "split",
        "group_key",
        "canonical_feature_family",
        "family_resolution_source",
        "family_resolution_rule",
        "family_member_count",
    ]
    home = home.merge(split[[col for col in split_cols if col in split.columns]], on="feature_name", how="left")
    home["source_manifest_hash"] = source_manifest_hash
    return home.sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)


def _lc_metadata(features: pd.DataFrame, source_manifest_hash: str) -> pd.DataFrame:
    lc = features[["dataset", "feature", "semantic_group", "source_table"]].rename(
        columns={"feature": "feature_name", "source_table": "source_table_or_formula"}
    )
    lc["split"] = "external_validation"
    lc["group_key"] = "external_validation:lendingclub_v2"
    lc["canonical_feature_family"] = lc["feature_name"].astype(str)
    lc["family_resolution_source"] = "external_validation_dataset"
    lc["family_resolution_rule"] = "external_validation_not_grouped"
    lc["family_member_count"] = 1
    lc["source_manifest_hash"] = source_manifest_hash
    return lc.sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)


def _build_vector_frame(*, metadata: pd.DataFrame, transformed: pd.DataFrame, preprocessor: RobustStatisticalPreprocessorV2) -> pd.DataFrame:
    vector_columns = [f"stat_{idx:04d}" for idx in range(len(DESCRIPTOR_COLUMNS_V2))]
    values = transformed[DESCRIPTOR_COLUMNS_V2].to_numpy(dtype=np.float32)
    vectors = pd.DataFrame(values, columns=vector_columns, index=metadata.index)
    base = metadata.copy()
    base["stable_row_index"] = range(len(base))
    base["stable_row_id"] = [sha256_text(f"{row.dataset}|{row.feature_name}|{row.split}") for row in base.itertuples(index=False)]
    base["input_field_hash"] = sha256_text(json.dumps(list(DESCRIPTOR_COLUMNS_V2)))
    base["preprocessor_hash"] = preprocessor.preprocessor_hash_
    base["statistical_vector"] = [json.dumps([float(value) for value in row]) for row in values]
    base["statistical_vector_hash"] = base["statistical_vector"].map(sha256_text)
    base["vector_dimension"] = len(DESCRIPTOR_COLUMNS_V2)
    return pd.concat([base, vectors], axis=1).sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)


def _write_anchor(
    output_dir: Path,
    home_vectors: pd.DataFrame,
    anchors: pd.DataFrame,
    preprocessor_hash: str,
    source_manifest_hash: str,
) -> dict[str, Any]:
    stat_cols = [f"stat_{idx:04d}" for idx in range(len(DESCRIPTOR_COLUMNS_V2))]
    train = home_vectors[home_vectors["split"].astype(str).eq("train")]
    names = sorted(set(anchors["feature_name"].astype(str)).intersection(set(train["feature_name"].astype(str))))
    anchor_frame = train[train["feature_name"].astype(str).isin(names)].copy()
    centroid = anchor_frame[stat_cols].to_numpy(dtype=float).mean(axis=0)
    anchor_hash = sha256_text(json.dumps([float(value) for value in centroid], sort_keys=False))
    anchor_frame.to_csv(output_dir / "homecredit_statistical_anchor_features.csv", index=False)
    payload = {
        "anchor_dataset": "homecredit",
        "external_validation_dataset": "lendingclub_v2",
        "anchor_policy": "Home Credit training split stable-core anchors only",
        "lendingclub_v2_anchor_policy": "uses unchanged Home Credit training-split statistical anchor centroid",
        "anchor_count": int(len(anchor_frame)),
        "anchor_hash": anchor_hash,
        "preprocessor_hash": preprocessor_hash,
        "source_manifest_hash": source_manifest_hash,
        "anchor_features": names,
    }
    write_json(output_dir / "statistical_anchor_manifest.json", payload)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
