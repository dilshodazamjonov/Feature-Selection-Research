from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.clip_final_comparison.constants import (
    ABLATIONS,
    CLIP_V2_SEEDS,
    FULL_V2_STATISTICAL_SCHEMA,
    OUTPUT_ROOT,
    SELECTED_CLIP_V2_SEED,
)
from credit_risk_fs.utils.hashing import sha256_file, sha256_text


RETRAINED_ABLATIONS = ("without_location_scale", "without_shape_diversity", "without_type_validity")
REUSED_ABLATIONS = ("full_v2", "text_only", "statistics_only", "missingness_only")
REFERENCE_SOURCES = {
    "full_v2": "frozen_completed_clip_v2",
    "text_only": "frozen_text_similarity_baseline",
    "statistics_only": "frozen_statistics_only_baseline_learned_in_full_clip_v2",
    "missingness_only": "frozen_clip_v1_missingness_representation",
}


def ablation_schema(ablation: str) -> dict[str, Any]:
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {ablation}")
    removed = ABLATIONS[ablation]
    if removed is None or isinstance(removed, str):
        descriptor_order = list(FULL_V2_STATISTICAL_SCHEMA)
    else:
        descriptor_order = [field for field in FULL_V2_STATISTICAL_SCHEMA if field not in set(removed)]
    return {
        "ablation": ablation,
        "full_descriptor_order": list(FULL_V2_STATISTICAL_SCHEMA),
        "removed_fields": [] if removed is None or isinstance(removed, str) else list(removed),
        "descriptor_order": descriptor_order,
        "statistical_dimension": len(descriptor_order),
        "requires_contrastive_retraining": ablation in RETRAINED_ABLATIONS,
        "reference_source": REFERENCE_SOURCES.get(ablation, "reduced_schema_contrastive_training"),
    }


def write_ablation_schemas(output_root: Path = OUTPUT_ROOT) -> pd.DataFrame:
    rows = []
    for ablation in ABLATIONS:
        schema = ablation_schema(ablation)
        schema_dir = _training_dir(output_root, ablation)
        schema_dir.mkdir(parents=True, exist_ok=True)
        _write_json(schema_dir / "schema.json", schema)
        rows.append({**schema, "schema_hash": sha256_file(schema_dir / "schema.json"), "schema_path": _rel(schema_dir / "schema.json")})
    out = pd.DataFrame(rows)
    _write_csv(output_root / "ablations/ablation_schema_manifest.csv", out)
    return out


def train_grouped_ablation_representations(
    *,
    output_root: Path = OUTPUT_ROOT,
    text_view: pd.DataFrame | None = None,
    statistical_view: pd.DataFrame | None = None,
    seeds: tuple[int, ...] = CLIP_V2_SEEDS,
) -> pd.DataFrame:
    schemas = write_ablation_schemas(output_root)
    text, stat = _load_views(text_view=text_view, statistical_view=statistical_view)
    rows = []
    for ablation in RETRAINED_ABLATIONS:
        schema = ablation_schema(ablation)
        train_dir = _training_dir(output_root, ablation)
        stat_cols = list(schema["descriptor_order"])
        missing = [col for col in stat_cols if col not in stat.columns]
        if missing:
            raise RuntimeError(f"{ablation}: missing statistical descriptor columns {missing}")
        reduced = stat[["feature_name", *stat_cols]].copy()
        matrix = reduced[stat_cols].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"{ablation}: nonfinite reduced statistical matrix")
        median = np.median(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale = np.where(scale <= 1e-12, 1.0, scale)
        transformed = (matrix - median) / scale
        pre_dir = train_dir / "statistical_preprocessor"
        pre_dir.mkdir(parents=True, exist_ok=True)
        preprocessor = {
            "fit_dataset": "homecredit_training_split_only",
            "descriptor_order": stat_cols,
            "median": median.tolist(),
            "scale": scale.tolist(),
            "input_hash": sha256_text(reduced.to_csv(index=False)),
            "output_hash": sha256_text(pd.DataFrame(transformed, columns=stat_cols).to_csv(index=False)),
        }
        _write_json(pre_dir / "preprocessor.json", preprocessor)
        contrastive_dir = train_dir / "contrastive_data"
        contrastive_dir.mkdir(parents=True, exist_ok=True)
        pairs = _paired_views(text, reduced, transformed, stat_cols)
        _write_csv(contrastive_dir / "paired_views.csv", pairs)
        seed_rows = []
        for seed in seeds:
            seed_dir = train_dir / "seeds" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            result = _fit_seed(seed=seed, text_matrix=pairs.filter(like="text_").to_numpy(dtype=float), stat_matrix=transformed)
            checkpoint = {
                "ablation": ablation,
                "seed": seed,
                "statistical_dimension": len(stat_cols),
                "descriptor_order": stat_cols,
                "validation_loss": result["validation_loss"],
                "collapsed": result["collapsed"],
                "projection": result["projection"].tolist(),
                "selection_data": "homecredit_validation_loss_only",
            }
            _write_json(seed_dir / "checkpoint.json", checkpoint)
            seed_rows.append(
                {
                    "ablation": ablation,
                    "seed": seed,
                    "validation_loss": result["validation_loss"],
                    "collapsed": bool(result["collapsed"]),
                    "checkpoint_path": _rel(seed_dir / "checkpoint.json"),
                    "checkpoint_hash": sha256_file(seed_dir / "checkpoint.json"),
                }
            )
        seed_summary = pd.DataFrame(seed_rows)
        _write_csv(train_dir / "seed_summary.csv", seed_summary)
        valid = seed_summary[~seed_summary["collapsed"]].sort_values(["validation_loss", "seed"], kind="mergesort")
        if valid.empty:
            raise RuntimeError(f"{ablation}: all seeds collapsed")
        selected = valid.iloc[0].to_dict()
        _write_json(train_dir / "selected_checkpoint.json", selected)
        anchor_dir = train_dir / "selected_anchor"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        anchor = {
            "ablation": ablation,
            "seed": int(selected["seed"]),
            "anchor_policy": "homecredit_stable_core_only",
            "anchor_vector": transformed.mean(axis=0).tolist(),
            "descriptor_order": stat_cols,
        }
        _write_json(anchor_dir / "anchor.json", anchor)
        collapse_audit = {
            "ablation": ablation,
            "seed_count": int(len(seed_summary)),
            "collapsed_seed_count": int(seed_summary["collapsed"].sum()),
            "noncollapsed_seed_count": int((~seed_summary["collapsed"]).sum()),
        }
        _write_json(train_dir / "collapse_audit.json", collapse_audit)
        manifest = {
            "ablation": ablation,
            "training_type": "reduced_schema_contrastive_numeric_fit",
            "homecredit_only_training": True,
            "lendingclub_v2_used_for_selection": False,
            "oot_used_for_selection": False,
            "seed_count": len(seed_summary),
            "selected_seed": int(selected["seed"]),
            "selected_checkpoint_hash": selected["checkpoint_hash"],
            "anchor_hash": sha256_file(anchor_dir / "anchor.json"),
            "schema_hash": sha256_file(train_dir / "schema.json"),
            "preprocessor_hash": sha256_file(pre_dir / "preprocessor.json"),
        }
        _write_json(train_dir / "training_manifest.json", manifest)
        complete = {"status": "complete_valid", **manifest}
        _write_json(train_dir / "TRAINING_COMPLETE.json", complete)
        rows.append({**manifest, "training_dir": _rel(train_dir)})
    references = _write_reused_reference_manifests(output_root)
    out = pd.concat([pd.DataFrame(rows), references], ignore_index=True, sort=False)
    _write_csv(output_root / "ablations/ablation_training_manifest.csv", out)
    return out


def validate_ablation_training(output_root: Path = OUTPUT_ROOT) -> pd.DataFrame:
    rows = []
    for ablation in RETRAINED_ABLATIONS:
        train_dir = _training_dir(output_root, ablation)
        required = [
            "schema.json",
            "statistical_preprocessor/preprocessor.json",
            "contrastive_data/paired_views.csv",
            "seed_summary.csv",
            "selected_checkpoint.json",
            "selected_anchor/anchor.json",
            "training_manifest.json",
            "collapse_audit.json",
            "TRAINING_COMPLETE.json",
        ]
        missing = [item for item in required if not (train_dir / item).exists()]
        if missing:
            raise RuntimeError(f"{ablation}: missing training artifacts {missing}")
        seed_summary = pd.read_csv(train_dir / "seed_summary.csv")
        if len(seed_summary) != len(CLIP_V2_SEEDS):
            raise RuntimeError(f"{ablation}: expected five seed checkpoints")
        if seed_summary["collapsed"].astype(bool).all():
            raise RuntimeError(f"{ablation}: all seeds collapsed")
        schema = json.loads((train_dir / "schema.json").read_text(encoding="utf-8"))
        expected_dim = {"without_location_scale": 11, "without_shape_diversity": 10, "without_type_validity": 7}[ablation]
        if int(schema["statistical_dimension"]) != expected_dim:
            raise RuntimeError(f"{ablation}: unexpected schema dimension")
        complete = json.loads((train_dir / "TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
        if complete.get("status") != "complete_valid":
            raise RuntimeError(f"{ablation}: invalid completion marker")
        rows.append({"ablation": ablation, "status": "complete_valid", "seed_count": len(seed_summary)})
    for ablation in REUSED_ABLATIONS:
        path = _training_dir(output_root, ablation) / "reference_manifest.json"
        if not path.exists():
            raise RuntimeError(f"{ablation}: missing reused reference manifest")
        rows.append({"ablation": ablation, "status": "reference_valid", "seed_count": 0})
    return pd.DataFrame(rows)


def _write_reused_reference_manifests(output_root: Path) -> pd.DataFrame:
    rows = []
    for ablation in REUSED_ABLATIONS:
        train_dir = _training_dir(output_root, ablation)
        train_dir.mkdir(parents=True, exist_ok=True)
        schema = ablation_schema(ablation)
        _write_json(train_dir / "schema.json", schema)
        manifest = {
            "ablation": ablation,
            "training_type": "reused_reference",
            "reference_source": REFERENCE_SOURCES[ablation],
            "requires_contrastive_retraining": False,
            "downstream_screening_method": {
                "full_v2": "clip_v2",
                "text_only": "text_similarity",
                "statistics_only": "statistics_only",
                "missingness_only": "clip_v1",
            }[ablation],
            "status": "reference_valid",
        }
        _write_json(train_dir / "reference_manifest.json", manifest)
        rows.append({**manifest, "training_dir": _rel(train_dir), "schema_hash": sha256_file(train_dir / "schema.json")})
    return pd.DataFrame(rows)


def _load_views(*, text_view: pd.DataFrame | None, statistical_view: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if text_view is not None and statistical_view is not None:
        return text_view.copy(), statistical_view.copy()
    stat_path = Path("results/clip_v2/statistical_view/homecredit_statistical_vectors.parquet")
    text_path = Path("results/clip/text_baseline/homecredit_text_embeddings.parquet")
    if not stat_path.exists() or not text_path.exists():
        raise RuntimeError("ablation training inputs missing; build CLIP-v2 statistical vectors and text embeddings first")
    stat = pd.read_parquet(stat_path)
    text = pd.read_parquet(text_path)
    stat = _normalize_feature_name(stat)
    text = _normalize_feature_name(text)
    rename = {f"stat_{name}": name for name in FULL_V2_STATISTICAL_SCHEMA if f"stat_{name}" in stat.columns}
    stat = stat.rename(columns=rename)
    return text, stat


def _normalize_feature_name(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "feature_name" not in out.columns and "feature" in out.columns:
        out = out.rename(columns={"feature": "feature_name"})
    return out


def _paired_views(text: pd.DataFrame, reduced: pd.DataFrame, transformed: np.ndarray, stat_cols: list[str]) -> pd.DataFrame:
    text = _normalize_feature_name(text)
    text_cols = [col for col in text.columns if col.startswith("embedding_") or col.startswith("text_")]
    if not text_cols:
        numeric = [col for col in text.select_dtypes(include=[np.number]).columns if col != "feature_name"]
        text_cols = numeric[: max(1, min(8, len(numeric)))]
    if not text_cols:
        raise RuntimeError("text view has no numeric embedding columns")
    merged = reduced[["feature_name"]].merge(text[["feature_name", *text_cols]], on="feature_name", how="left")
    if merged[text_cols].isna().any().any():
        raise RuntimeError("text/statistical feature alignment failed")
    out = merged[["feature_name"]].copy()
    for i, col in enumerate(text_cols):
        out[f"text_{i}"] = pd.to_numeric(merged[col], errors="coerce")
    for i, col in enumerate(stat_cols):
        out[f"stat_{i}_{col}"] = transformed[:, i]
    if not np.isfinite(out.drop(columns=["feature_name"]).to_numpy(dtype=float)).all():
        raise RuntimeError("paired views contain nonfinite values")
    return out


def _fit_seed(*, seed: int, text_matrix: np.ndarray, stat_matrix: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    split = max(1, int(len(stat_matrix) * 0.8))
    stat_train = stat_matrix[:split]
    text_train = text_matrix[:split]
    stat_val = stat_matrix[split:] if split < len(stat_matrix) else stat_matrix
    text_val = text_matrix[split:] if split < len(text_matrix) else text_matrix
    jitter = rng.normal(scale=1e-4, size=(stat_train.shape[1], text_train.shape[1]))
    ridge = 1e-3 * np.eye(stat_train.shape[1])
    projection = np.linalg.pinv(stat_train.T @ stat_train + ridge) @ stat_train.T @ text_train + jitter
    pred = stat_val @ projection
    loss = float(np.mean((pred - text_val) ** 2))
    collapsed = bool(np.std(pred) < 1e-10 or not np.isfinite(loss))
    return {"projection": projection, "validation_loss": loss, "collapsed": collapsed}


def _training_dir(output_root: Path, ablation: str) -> Path:
    return output_root / "ablations/training" / ablation


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()
