from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk_fs.clip.selector_adapter import (  # noqa: E402
    ClipScoreAdapter,
    materialize_score_caches,
    score_coverage,
)
from credit_risk_fs.clip.selector_validation import (  # noqa: E402
    load_clip_selector_config,
    validate_clip_selector_binding,
)
from credit_risk_fs.selectors.clip_screening import ClipScreeningSelector  # noqa: E402
from credit_risk_fs.selectors.clip_then_mrmr import ClipThenMRMRSelector  # noqa: E402
from credit_risk_fs.selectors.registry import get_selector  # noqa: E402
from credit_risk_fs.utils.hashing import sha256_file, sha256_text  # noqa: E402
from credit_risk_fs.utils.io import read_json, write_json  # noqa: E402


MODELS = ("lr", "catboost")
SELECTORS = ("clip", "clip_then_mrmr")
ACTIVE_DATASETS = ("homecredit", "lendingclub_v2")
LEGACY_DATASET = "lendingclub"


def _safe_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _registry_audit() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for selector_name in [
        "mrmr",
        "boruta",
        "pca",
        "llm",
        "llm_then_mrmr",
        "llm_then_boruta",
        "domain_rule_baseline",
        "stable_core_llm_fill",
        "clip",
        "clip_then_mrmr",
    ]:
        selector_cls, defaults = get_selector(selector_name)
        rows[selector_name] = {
            "class": None if selector_cls is None else f"{selector_cls.__module__}.{selector_cls.__name__}",
            "default_kwargs": defaults,
        }
    return rows


def _dataset_scores(config_path: Path, dataset: str) -> pd.DataFrame:
    adapter = ClipScoreAdapter(config_path, dataset=dataset)
    return adapter.score_frame(use_cache=True, write_cache=False)


def _smoke_frame(scores: pd.DataFrame, *, rows: int = 96, max_features: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    features = (
        scores.sort_values(["learned_rank", "feature_name"], kind="mergesort")["feature_name"]
        .astype(str)
        .head(max_features)
        .tolist()
    )
    rng = np.random.default_rng(20260620)
    values = rng.normal(size=(rows, len(features)))
    offsets = np.linspace(-0.4, 0.4, len(features))
    X = pd.DataFrame(values + offsets, columns=features)
    logits = X.iloc[:, : min(8, X.shape[1])].sum(axis=1)
    y = pd.Series((logits > logits.median()).astype(int), name="TARGET")
    return X, y


def _run_selector_smoke(
    *,
    config_path: Path,
    dataset: str,
    model: str,
    selector_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    feature_budget: int,
    screening_pool_size: int,
) -> pd.DataFrame:
    if selector_name == "clip":
        selector = ClipScreeningSelector(
            config_path=str(config_path),
            dataset=dataset,
            feature_budget=feature_budget,
            model_name=model,
            missing_feature_policy="error",
        )
        selector.fit(X, y)
        if selector.selection_manifest_ is None:
            raise RuntimeError("clip smoke selection manifest was not created")
        return selector.selection_manifest_.copy()

    selector = ClipThenMRMRSelector(
        config_path=str(config_path),
        dataset=dataset,
        feature_budget=feature_budget,
        screening_pool_size=screening_pool_size,
        model_name=model,
        missing_feature_policy="error",
        random_state=42,
    )
    X_screened = selector.fit(X, y).transform(X)
    selector.fit_postprocess(X_screened, y)
    if selector.selection_manifest_ is None:
        raise RuntimeError("clip_then_mrmr smoke selection manifest was not created")
    return selector.selection_manifest_.copy()


def _write_smoke_outputs(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_clip_selector_config(config_path)
    cache_paths = materialize_score_caches(config_path)
    smoke_summary: dict[str, Any] = {
        "score_cache_paths": {dataset: str(path) for dataset, path in cache_paths.items()},
        "selection_outputs": {},
    }
    for dataset in ACTIVE_DATASETS:
        scores = _dataset_scores(config_path, dataset)
        X, y = _smoke_frame(scores)
        for selector_name in SELECTORS:
            manifests = []
            for model in MODELS:
                manifests.append(
                    _run_selector_smoke(
                        config_path=config_path,
                        dataset=dataset,
                        model=model,
                        selector_name=selector_name,
                        X=X,
                        y=y,
                        feature_budget=int(config.feature_budgets[model]),
                        screening_pool_size=int(config.screening_pool_size),
                    )
                )
            combined = pd.concat(manifests, ignore_index=True)
            suffix = "clip_selection_smoke.csv" if selector_name == "clip" else "clip_then_mrmr_smoke.csv"
            path = output_dir / f"{dataset}_{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(path, index=False)
            smoke_summary["selection_outputs"][f"{dataset}_{selector_name}"] = {
                "path": str(path),
                "rows": int(len(combined)),
                "selected_rows": int(combined["final_selected"].fillna(False).astype(bool).sum()),
                "models": sorted(combined["model"].astype(str).unique().tolist()),
            }
    return smoke_summary


def _assert_legacy_rejected(config_path: Path) -> str:
    try:
        ClipScoreAdapter(config_path, dataset=LEGACY_DATASET).score_frame()
    except RuntimeError as exc:
        return str(exc)
    raise RuntimeError("legacy LendingClub was accepted by CLIP selector integration")


def _integration_payload(config_path: Path, *, smoke_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_clip_selector_config(config_path)
    binding = validate_clip_selector_binding(config)
    coverage = score_coverage(config_path)
    selection = _safe_json(config.model_selection_manifest_path)
    anchor = _safe_json(config.learned_anchor_manifest_path)
    training = _safe_json(config.training_manifest_path)
    return {
        "status": "passed",
        "config_path": str(config_path),
        "active_datasets": list(config.active_datasets),
        "legacy_datasets": list(config.legacy_datasets),
        "checkpoint_hash": binding["checkpoint_hash"],
        "anchor_hash": binding["anchor_hash"],
        "statistical_view_scope": binding["statistical_view_scope"],
        "selection_rule": binding["selection_rule"],
        "fusion_rule": binding["fusion_rule"],
        "feature_budgets": config.feature_budgets,
        "screening_pool_size": config.screening_pool_size,
        "no_refit": config.no_refit,
        "homecredit_train_only_anchor": anchor.get("anchor_dataset") == "homecredit"
        and "training-split" in str(anchor.get("anchor_policy", "")),
        "lendingclub_v2_used_for_selection": bool(selection.get("lendingclub_v2_used_for_selection")),
        "legacy_lendingclub_rejection": _assert_legacy_rejected(config_path),
        "training_scope": training.get("scope") or training.get("training_scope"),
        "score_coverage": coverage,
        "source_hashes": {
            "selected_checkpoint": sha256_file(config.selected_checkpoint_path),
            "selected_checkpoint_manifest": sha256_file(config.selected_checkpoint_manifest_path),
            "model_selection_manifest": sha256_file(config.model_selection_manifest_path),
            "learned_anchor_manifest": sha256_file(config.learned_anchor_manifest_path),
            "training_manifest": sha256_file(config.training_manifest_path),
            "homecredit_scores": sha256_file(config.homecredit_scores_path),
            "lendingclub_v2_scores": sha256_file(config.lendingclub_v2_scores_path),
        },
        "selector_registry_hash": sha256_text(json.dumps(_registry_audit(), sort_keys=True, default=str)),
        "smoke_summary": smoke_summary or {},
    }


def _write_full_manifests(config_path: Path, output_dir: Path, smoke_summary: dict[str, Any]) -> None:
    config = load_clip_selector_config(config_path)
    binding = validate_clip_selector_binding(config)
    registry = _registry_audit()
    integration = _integration_payload(config_path, smoke_summary=smoke_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "selector_registry_audit.json", registry)
    write_json(
        output_dir / "checkpoint_binding.json",
        {
            "selected_checkpoint_path": str(config.selected_checkpoint_path),
            "selected_checkpoint_hash": binding["checkpoint_hash"],
            "selected_checkpoint_manifest_path": str(config.selected_checkpoint_manifest_path),
            "anchor_hash": binding["anchor_hash"],
            "anchor_manifest_path": str(config.learned_anchor_manifest_path),
            "anchor_policy": _safe_json(config.learned_anchor_manifest_path).get("anchor_policy"),
            "lendingclub_v2_policy": "external validation only; unchanged Home Credit anchor",
            "statistical_view_scope": binding["statistical_view_scope"],
            "no_refit": config.no_refit,
        },
    )
    write_json(output_dir / "integration_manifest.json", integration)
    write_json(
        output_dir / "integration_audit.json",
        {
            "passed": True,
            "no_model_training": True,
            "no_final_matrix_run": True,
            "legacy_lendingclub_absent": LEGACY_DATASET not in config.active_datasets,
            "lendingclub_v2_did_not_influence_fitted_state": not integration["lendingclub_v2_used_for_selection"],
            "missing_feature_policy": config.missing_feature_policy,
            "cache_outputs": smoke_summary.get("score_cache_paths", {}),
            "selection_outputs": smoke_summary.get("selection_outputs", {}),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate frozen CLIP selector integration.")
    parser.add_argument("--config", default="configs/clip/selector.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate wiring without writing full integration artifacts.")
    parser.add_argument("--smoke-test", action="store_true", help="Write DEV-only score caches and smoke selections.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_clip_selector_config(config_path)
    output_dir = config.output_dir
    binding = validate_clip_selector_binding(config)
    registry = _registry_audit()

    if args.dry_run:
        payload = _integration_payload(config_path)
        payload["dry_run"] = True
        payload["registry_selectors"] = sorted(registry)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.smoke_test:
        smoke_summary = _write_smoke_outputs(config_path, output_dir)
        _write_full_manifests(config_path, output_dir, smoke_summary)
        print(json.dumps({"status": "passed", "checkpoint_hash": binding["checkpoint_hash"], **smoke_summary}, indent=2))
        return 0

    parser.error("Specify --dry-run or --smoke-test.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
