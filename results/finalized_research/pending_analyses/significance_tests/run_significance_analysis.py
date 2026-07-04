from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # The manual exhaustive calculation remains authoritative.
    stats = None


ANALYSIS_NAME = "Paired Fold Significance and Consistency Analysis"
ANALYSIS_VERSION = "1.0.0"
DATASETS = ("homecredit", "lendingclub_v2")
MODELS = ("lr", "catboost")
PIPELINES = ("mrmr", "stable_core_llm_fill", "llm", "llm_then_mrmr")
CANDIDATES = ("stable_core_llm_fill", "llm", "llm_then_mrmr")
PIPELINE_LABELS = {
    "mrmr": "mRMR",
    "stable_core_llm_fill": "Stable-core + LLM-fill",
    "llm": "pure LLM",
    "llm_then_mrmr": "LLM-then-mRMR",
}
DATASET_LABELS = {
    "homecredit": "Home Credit",
    "lendingclub_v2": "LendingClub v2",
}
MODEL_LABELS = {"lr": "Logistic Regression", "catboost": "CatBoost"}
RESULT_COLUMNS = [
    "comparison_id",
    "dataset",
    "dataset_version",
    "model",
    "pipeline_a",
    "pipeline_b",
    "planned_folds",
    "paired_folds",
    "nonzero_pairs",
    "mean_delta_auc",
    "median_delta_auc",
    "std_delta_auc",
    "min_delta_auc",
    "max_delta_auc",
    "pipeline_a_wins",
    "pipeline_b_wins",
    "ties",
    "wilcoxon_w_positive",
    "wilcoxon_w_negative",
    "exact_two_sided_p",
    "holm_adjusted_p",
    "rank_biserial_correlation",
    "raw_significant_0_05",
    "holm_significant_0_05",
    "direction",
    "consistency_rating",
    "comparison_status",
    "interpretation",
    "limitations",
]
MASTER_COLUMNS = [
    "dataset",
    "dataset_version",
    "model",
    "pipeline_id",
    "pipeline_label",
    "run_id",
    "fold_id",
    "fold_identity",
    "validation_start",
    "validation_end",
    "validation_rows",
    "validation_auc",
    "auc_source_column",
    "cv_results_path",
    "run_manifest_path",
    "split_manifest_path",
    "model_config_identity",
    "feature_budget",
    "source_status",
    "notes",
]
DETAIL_COLUMNS = [
    "comparison_id",
    "dataset",
    "model",
    "pipeline_a",
    "pipeline_b",
    "fold_id",
    "fold_identity",
    "auc_pipeline_a",
    "auc_pipeline_b",
    "delta_auc",
    "winner",
    "source_path_a",
    "source_path_b",
    "pairing_status",
    "notes",
]
LIMITATION_TEXT = (
    "Five folds provide very low inferential power; non-significance is not "
    "equivalence; overlapping CV training samples are not independent "
    "scientific replications; this fold-consistency test does not replace "
    "authenticated OOT evaluation."
)


@dataclass
class AuthenticatedRun:
    dataset: str
    model: str
    pipeline: str
    run_id: str
    dataset_version: str
    feature_budget: float
    cv_path: str
    manifest_path: str
    split_path: str
    auc_column: str
    model_config_identity: str
    preprocessing_identity: str
    split_contract_identity: str
    data_manifest_hash: str
    target_definition: str
    source_status: str
    oot_auc: float
    master_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=ANALYSIS_NAME)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Repository root (defaults to the root inferred from this script).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def relpath(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def normalized_number(value: Any) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Fold identity contains a non-finite number: {value}")
    if number.is_integer():
        return int(number)
    return float(f"{number:.15g}")


def require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_canonical_indexes(
    root: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]], list[str]]:
    base = root / "results" / "finalized_research"
    starting_files = [
        base / "README.md",
        base / "STATUS.md",
        base / "canonical_artifact_manifest.json",
        base / "canonical_artifact_inventory.csv",
    ]
    for path in starting_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required canonical starting file is missing: {path}")

    inventory_frame = pd.read_csv(starting_files[3], dtype=str, keep_default_na=False)
    require_columns(
        inventory_frame,
        {"path", "sha256", "status", "purpose"},
        relpath(root, starting_files[3]),
    )
    if inventory_frame["path"].duplicated().any():
        duplicates = inventory_frame.loc[
            inventory_frame["path"].duplicated(keep=False), "path"
        ].tolist()
        raise ValueError(f"Canonical inventory contains duplicate paths: {duplicates}")
    inventory = {
        str(row["path"]): {str(key): str(value) for key, value in row.items()}
        for row in inventory_frame.to_dict("records")
    }

    manifest = read_json(starting_files[2])
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Canonical artifact manifest has no artifact list.")
    manifest_index: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError("Malformed canonical artifact manifest entry.")
        path = str(item["path"])
        if path in manifest_index:
            raise ValueError(f"Canonical artifact manifest duplicates path: {path}")
        manifest_index[path] = item
    if int(manifest.get("artifact_count", -1)) != len(manifest_index):
        raise ValueError("Canonical artifact manifest count does not match its entries.")

    return inventory, manifest_index, [relpath(root, path) for path in starting_files]


def authenticate_source(
    root: Path,
    relative_path: str,
    inventory: dict[str, dict[str, str]],
    manifest_index: dict[str, dict[str, Any]],
) -> str:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Canonical source is missing: {relative_path}")
    if relative_path not in inventory:
        raise ValueError(f"Source is absent from canonical inventory: {relative_path}")
    if relative_path not in manifest_index:
        raise ValueError(f"Source is absent from canonical manifest: {relative_path}")
    actual = sha256_file(path)
    inventory_hash = str(inventory[relative_path].get("sha256", ""))
    manifest_hash = str(manifest_index[relative_path].get("sha256", ""))
    if actual != inventory_hash or actual != manifest_hash:
        raise ValueError(f"Canonical hash mismatch for {relative_path}")
    return actual


def identify_auc_column(
    cv: pd.DataFrame,
    run_manifest: dict[str, Any],
    source: str,
) -> str:
    plausible = [
        column
        for column in cv.columns
        if "auc" in column.lower()
        and not any(
            excluded in column.lower()
            for excluded in ("train", "oot", "mean", "std", "final")
        )
    ]
    numeric_fold = pd.to_numeric(cv.get("fold"), errors="coerce")
    fold_rows = cv[numeric_fold.between(1, 5, inclusive="both")].copy()
    expected_mean = run_manifest.get("summary", {}).get("cv_auc_mean")
    authenticated: list[str] = []
    for column in plausible:
        values = pd.to_numeric(fold_rows[column], errors="coerce")
        if values.notna().all() and len(values) == 5 and values.between(0, 1).all():
            if expected_mean is None or math.isclose(
                float(values.mean()),
                float(expected_mean),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                authenticated.append(column)
    if len(authenticated) != 1:
        raise ValueError(
            f"{source}: expected one authenticated validation AUC column; "
            f"found {authenticated} from plausible columns {plausible}"
        )
    auc_column = authenticated[0]
    if "gini" in fold_rows.columns:
        auc = pd.to_numeric(fold_rows[auc_column], errors="raise")
        gini = pd.to_numeric(fold_rows["gini"], errors="raise")
        if not np.allclose(gini.to_numpy(), 2.0 * auc.to_numpy() - 1.0, atol=1e-12):
            raise ValueError(f"{source}: AUC is inconsistent with the stored Gini values.")
    return auc_column


def preprocessing_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest["config"]
    return {
        "data_dir": str(config.get("data_dir", "")).replace("\\", "/"),
        "description_path": str(config.get("description_path", "")).replace(
            "\\", "/"
        ),
        "time_column": config.get("time_col", "recent_decision"),
        "excluded_feature_columns": sorted(config.get("excluded_feature_columns", [])),
        "preprocessor_kwargs": config.get("preprocessor_kwargs", {}),
    }


def model_contract(manifest: dict[str, Any], model: str) -> dict[str, Any]:
    config = manifest["config"]
    return {
        "model_family": model,
        "model_parameters": config.get("model_params", {}).get(model, {}),
        "random_seed": manifest.get("random_seed", config.get("random_seed")),
    }


def split_contract(manifest: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    config = manifest["config"]
    return {
        "time_column": split.get("time_column"),
        "configured_windows": split.get("configured_windows"),
        "dev_window": split.get("DEV_window"),
        "oot_window": split.get("OOT_window"),
        "dev_summary": split.get("dev"),
        "oot_summary": split.get("oot"),
        "source_row_count": split.get("source_row_count"),
        "n_splits": config.get("n_splits"),
        "cv_gap_groups": config.get("cv_gap_groups"),
    }


def load_authenticated_run(
    root: Path,
    registry_row: dict[str, str],
    dataset: str,
    model: str,
    pipeline: str,
    inventory: dict[str, dict[str, str]],
    manifest_index: dict[str, dict[str, Any]],
    source_hashes: dict[str, str],
) -> AuthenticatedRun:
    run_id = registry_row["run_id"]
    manifest_path = registry_row["manifest_path"].replace("\\", "/")
    run_dir = Path(manifest_path).parent
    cv_path = (run_dir / "results" / "cv_results.csv").as_posix()
    split_path = (run_dir / "data_split_manifest.json").as_posix()
    for source in (manifest_path, cv_path, split_path):
        source_hashes[source] = authenticate_source(
            root, source, inventory, manifest_index
        )

    manifest = read_json(root / manifest_path)
    split = read_json(root / split_path)
    cv = pd.read_csv(root / cv_path)
    require_columns(
        cv,
        {
            "fold",
            "val_size",
            "val_start_idx",
            "val_end_idx",
            "val_time_start",
            "val_time_end",
        },
        cv_path,
    )
    if manifest.get("status") != "completed":
        raise ValueError(f"Run is not completed: {run_id}")
    if manifest.get("run_id") != run_id:
        raise ValueError(f"Registry/run-manifest ID mismatch: {run_id}")
    if manifest.get("model") != model:
        raise ValueError(f"Registry/run-manifest model mismatch: {run_id}")
    if manifest.get("selector") != pipeline:
        raise ValueError(f"Registry/run-manifest selector mismatch: {run_id}")
    if manifest.get("config", {}).get("dataset_name") != dataset:
        raise ValueError(f"Registry/run-manifest dataset mismatch: {run_id}")
    if "TARGET" not in manifest["config"].get("excluded_feature_columns", []):
        raise ValueError(f"Target column contract is not explicit for {run_id}")

    auc_column = identify_auc_column(cv, manifest, cv_path)
    cv = cv.copy()
    cv["_fold_numeric"] = pd.to_numeric(cv["fold"], errors="coerce")
    folds = cv[cv["_fold_numeric"].between(1, 5, inclusive="both")].copy()
    folds["_fold_numeric"] = folds["_fold_numeric"].astype(int)
    if len(folds) != 5 or set(folds["_fold_numeric"]) != {1, 2, 3, 4, 5}:
        raise ValueError(f"{run_id} does not contain exactly five unique fold rows.")
    if folds["_fold_numeric"].duplicated().any():
        raise ValueError(f"{run_id} contains duplicate fold IDs.")
    auc_values = pd.to_numeric(folds[auc_column], errors="raise")
    if not np.isfinite(auc_values.to_numpy()).all() or not auc_values.between(
        0, 1
    ).all():
        raise ValueError(f"{run_id} contains invalid fold AUC values.")

    data_manifest_hash = registry_row["data_manifest_hash"]
    if not data_manifest_hash:
        raise ValueError(f"Registry data-manifest hash is missing for {run_id}")
    dataset_version = f"{dataset}@{data_manifest_hash[:12]}"
    target_definition = f"TARGET@{data_manifest_hash[:12]}"
    split_payload = split_contract(manifest, split)
    split_identity = canonical_json_hash(split_payload)
    preprocessing_identity = canonical_json_hash(preprocessing_contract(manifest))
    model_identity = canonical_json_hash(model_contract(manifest, model))
    feature_budget = float(registry_row["feature_budget"])
    if not math.isclose(
        feature_budget,
        float(manifest.get("feature_budget")),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Registry/run-manifest feature-budget mismatch: {run_id}")

    master_rows: list[dict[str, Any]] = []
    for row in folds.sort_values("_fold_numeric").to_dict("records"):
        fold_id = int(row["_fold_numeric"])
        fold_payload = {
            "dataset_version": dataset_version,
            "data_manifest_hash": data_manifest_hash,
            "target_definition": target_definition,
            "split_contract_identity": split_identity,
            "fold_id": fold_id,
            "validation_start_index": normalized_number(row["val_start_idx"]),
            "validation_end_index": normalized_number(row["val_end_idx"]),
            "validation_time_start": normalized_number(row["val_time_start"]),
            "validation_time_end": normalized_number(row["val_time_end"]),
            "validation_rows": normalized_number(row["val_size"]),
        }
        master_rows.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "dataset_version": dataset_version,
                "model": MODEL_LABELS[model],
                "pipeline_id": pipeline,
                "pipeline_label": PIPELINE_LABELS[pipeline],
                "run_id": run_id,
                "fold_id": fold_id,
                "fold_identity": canonical_json_hash(fold_payload),
                "validation_start": normalized_number(row["val_time_start"]),
                "validation_end": normalized_number(row["val_time_end"]),
                "validation_rows": normalized_number(row["val_size"]),
                "validation_auc": float(row[auc_column]),
                "auc_source_column": auc_column,
                "cv_results_path": cv_path,
                "run_manifest_path": manifest_path,
                "split_manifest_path": split_path,
                "model_config_identity": model_identity,
                "feature_budget": feature_budget,
                "source_status": (
                    f"active_registry:{registry_row['reuse_status']};"
                    f"canonical_inventory:{inventory[cv_path]['status']}"
                ),
                "notes": (
                    "Fold identity hashes dataset/data-manifest/target/split "
                    "contract plus validation indices, time bounds, and row count; "
                    "ROC AUC authenticated against manifest mean and Gini."
                ),
            }
        )

    oot_auc = float(manifest.get("summary", {}).get("oot_auc"))
    if not math.isfinite(oot_auc) or not 0 <= oot_auc <= 1:
        raise ValueError(f"Authenticated OOT AUC is unavailable for {run_id}")
    return AuthenticatedRun(
        dataset=dataset,
        model=model,
        pipeline=pipeline,
        run_id=run_id,
        dataset_version=dataset_version,
        feature_budget=feature_budget,
        cv_path=cv_path,
        manifest_path=manifest_path,
        split_path=split_path,
        auc_column=auc_column,
        model_config_identity=model_identity,
        preprocessing_identity=preprocessing_identity,
        split_contract_identity=split_identity,
        data_manifest_hash=data_manifest_hash,
        target_definition=target_definition,
        source_status=registry_row["reuse_status"],
        oot_auc=oot_auc,
        master_rows=master_rows,
    )


def resolve_runs(
    root: Path,
    inventory: dict[str, dict[str, str]],
    manifest_index: dict[str, dict[str, Any]],
    source_hashes: dict[str, str],
) -> dict[tuple[str, str, str], AuthenticatedRun]:
    registry_path = "results/research_summary/run_index.csv"
    registry = pd.read_csv(root / registry_path, dtype=str, keep_default_na=False)
    require_columns(
        registry,
        {
            "run_id",
            "dataset",
            "method",
            "model",
            "feature_budget",
            "data_manifest_hash",
            "manifest_path",
            "reuse_status",
        },
        registry_path,
    )
    source_hashes[registry_path] = sha256_file(root / registry_path)
    runs: dict[tuple[str, str, str], AuthenticatedRun] = {}
    for dataset, model, pipeline in itertools.product(DATASETS, MODELS, PIPELINES):
        matches = registry[
            registry["dataset"].eq(dataset)
            & registry["model"].eq(model)
            & registry["method"].eq(pipeline)
            & registry["reuse_status"].eq("reusable_existing")
        ]
        if len(matches) != 1:
            raise ValueError(
                "Canonical run resolution must yield exactly one active row for "
                f"{dataset}/{model}/{pipeline}; found {len(matches)}"
            )
        row = {key: str(value) for key, value in matches.iloc[0].to_dict().items()}
        forbidden = ("failed", "incomplete", "smoke", "dry_run", "clip_v2")
        joined = f"{row['run_id']} {row['manifest_path']}".lower()
        if any(token in joined for token in forbidden):
            raise ValueError(f"Forbidden run family resolved from registry: {joined}")
        runs[(dataset, model, pipeline)] = load_authenticated_run(
            root,
            row,
            dataset,
            model,
            pipeline,
            inventory,
            manifest_index,
            source_hashes,
        )
    return runs


def comparison_id(dataset: str, model: str, candidate: str) -> str:
    return f"{dataset}__{model}__{candidate}_vs_mrmr"


def exact_wilcoxon(
    differences: np.ndarray,
) -> tuple[float, float, float, float, int, bool, str]:
    zero_mask = np.isclose(differences, 0.0, rtol=0.0, atol=1e-15)
    nonzero = differences[~zero_mask]
    if len(nonzero) == 0:
        return 0.0, 0.0, 1.0, 0.0, 0, False, "all_zero"
    ranks = (
        pd.Series(np.abs(nonzero), dtype=float)
        .rank(method="average", ascending=True)
        .to_numpy(dtype=float)
    )
    w_positive = float(ranks[nonzero > 0].sum())
    w_negative = float(ranks[nonzero < 0].sum())
    rank_total = float(ranks.sum())
    observed = min(w_positive, w_negative)
    extreme = 0
    for signs in itertools.product((0, 1), repeat=len(ranks)):
        perm_positive = float(
            sum(rank for rank, positive in zip(ranks, signs) if positive)
        )
        perm_statistic = min(perm_positive, rank_total - perm_positive)
        if perm_statistic <= observed + 1e-12:
            extreme += 1
    exact_p = extreme / (2 ** len(ranks))
    rank_biserial = (w_positive - w_negative) / rank_total
    tied_absolute_ranks = len(np.unique(np.abs(nonzero))) < len(nonzero)
    scipy_note = "not_cross_checked"
    if stats is not None and not tied_absolute_ranks and not zero_mask.any():
        scipy_p = float(
            stats.wilcoxon(
                nonzero,
                zero_method="wilcox",
                alternative="two-sided",
                method="exact",
            ).pvalue
        )
        if not math.isclose(scipy_p, exact_p, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                f"Manual exact Wilcoxon p={exact_p} disagrees with SciPy p={scipy_p}"
            )
        scipy_note = f"scipy_exact_cross_check={scipy_p:.15g}"
    return (
        w_positive,
        w_negative,
        exact_p,
        rank_biserial,
        len(nonzero),
        tied_absolute_ranks,
        scipy_note,
    )


def consistency_rating(wins_a: int, wins_b: int, ties: int) -> str:
    strongest = max(wins_a, wins_b)
    if strongest == 5 and ties == 0:
        return "strong"
    if strongest == 4:
        return "moderate"
    if strongest == 3:
        return "weak"
    return "mixed"


def pairability_status(
    candidate: AuthenticatedRun,
    baseline: AuthenticatedRun,
) -> tuple[str, str]:
    if (
        candidate.dataset_version != baseline.dataset_version
        or candidate.data_manifest_hash != baseline.data_manifest_hash
        or candidate.target_definition != baseline.target_definition
        or candidate.split_contract_identity != baseline.split_contract_identity
        or candidate.preprocessing_identity != baseline.preprocessing_identity
    ):
        return (
            "UNAVAILABLE_EXPERIMENT_CONTRACT_MISMATCH",
            "Dataset, target, split, or preprocessing contract differs.",
        )
    if candidate.model_config_identity != baseline.model_config_identity:
        return (
            "UNAVAILABLE_MODEL_CONFIGURATION_MISMATCH",
            "Downstream model configuration identity differs.",
        )
    if not math.isclose(
        candidate.feature_budget,
        baseline.feature_budget,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        return (
            "UNAVAILABLE_EXPERIMENT_CONTRACT_MISMATCH",
            "Final feature budgets differ without a documented intentional contrast.",
        )
    candidate_folds = {int(row["fold_id"]): row for row in candidate.master_rows}
    baseline_folds = {int(row["fold_id"]): row for row in baseline.master_rows}
    if set(candidate_folds) != {1, 2, 3, 4, 5} or set(baseline_folds) != {
        1,
        2,
        3,
        4,
        5,
    }:
        return (
            "UNAVAILABLE_MISSING_FOLD_RESULTS",
            "One or both runs do not have all five folds.",
        )
    mismatches = [
        fold
        for fold in range(1, 6)
        if candidate_folds[fold]["fold_identity"]
        != baseline_folds[fold]["fold_identity"]
    ]
    if mismatches:
        return (
            "UNAVAILABLE_FOLD_IDENTITY_NOT_AUTHENTICATED",
            f"Fold identity differs for folds {mismatches}.",
        )
    return "COMPUTED", "All pairability contract fields match."


def interpretation_text(
    candidate: str,
    wins_a: int,
    wins_b: int,
    ties: int,
    exact_p: float,
    rating: str,
) -> str:
    if wins_a > wins_b:
        observed = f"{PIPELINE_LABELS[candidate]} had higher AUC in {wins_a} folds"
    elif wins_b > wins_a:
        observed = f"mRMR had higher AUC in {wins_b} folds"
    else:
        observed = "the fold directions were evenly balanced"
    return (
        f"{observed} ({ties} ties); fold consistency was {rating}. "
        f"The exact two-sided p-value was {exact_p:.4f}; this low-power result "
        "must not be interpreted as equivalence or as a replacement for OOT evidence."
    )


def build_comparisons(
    runs: dict[tuple[str, str, str], AuthenticatedRun],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for dataset, model, candidate_name in itertools.product(
        DATASETS, MODELS, CANDIDATES
    ):
        candidate = runs[(dataset, model, candidate_name)]
        baseline = runs[(dataset, model, "mrmr")]
        comp_id = comparison_id(dataset, model, candidate_name)
        status, pairing_note = pairability_status(candidate, baseline)
        base_result: dict[str, Any] = {
            "comparison_id": comp_id,
            "dataset": DATASET_LABELS[dataset],
            "dataset_version": candidate.dataset_version,
            "model": MODEL_LABELS[model],
            "pipeline_a": candidate_name,
            "pipeline_b": "mrmr",
            "planned_folds": 5,
            "paired_folds": 0,
            "nonzero_pairs": 0,
            "mean_delta_auc": np.nan,
            "median_delta_auc": np.nan,
            "std_delta_auc": np.nan,
            "min_delta_auc": np.nan,
            "max_delta_auc": np.nan,
            "pipeline_a_wins": 0,
            "pipeline_b_wins": 0,
            "ties": 0,
            "wilcoxon_w_positive": np.nan,
            "wilcoxon_w_negative": np.nan,
            "exact_two_sided_p": np.nan,
            "holm_adjusted_p": np.nan,
            "rank_biserial_correlation": np.nan,
            "raw_significant_0_05": False,
            "holm_significant_0_05": False,
            "direction": "not available",
            "consistency_rating": "not available",
            "comparison_status": status,
            "interpretation": pairing_note,
            "limitations": LIMITATION_TEXT,
        }
        metadata[comp_id] = {
            "candidate_run_id": candidate.run_id,
            "baseline_run_id": baseline.run_id,
            "oot_delta_auc": candidate.oot_auc - baseline.oot_auc,
            "pairing_status": status,
        }
        if status != "COMPUTED":
            result_rows.append(base_result)
            continue

        candidate_folds = {
            int(row["fold_id"]): row for row in candidate.master_rows
        }
        baseline_folds = {int(row["fold_id"]): row for row in baseline.master_rows}
        differences: list[float] = []
        for fold_id in range(1, 6):
            left = candidate_folds[fold_id]
            right = baseline_folds[fold_id]
            delta = float(left["validation_auc"]) - float(right["validation_auc"])
            if math.isclose(delta, 0.0, rel_tol=0, abs_tol=1e-15):
                delta = 0.0
            differences.append(delta)
            winner = (
                "pipeline_a"
                if delta > 0
                else "pipeline_b"
                if delta < 0
                else "tie"
            )
            detail_rows.append(
                {
                    "comparison_id": comp_id,
                    "dataset": DATASET_LABELS[dataset],
                    "model": MODEL_LABELS[model],
                    "pipeline_a": candidate_name,
                    "pipeline_b": "mrmr",
                    "fold_id": fold_id,
                    "fold_identity": left["fold_identity"],
                    "auc_pipeline_a": left["validation_auc"],
                    "auc_pipeline_b": right["validation_auc"],
                    "delta_auc": delta,
                    "winner": winner,
                    "source_path_a": candidate.cv_path,
                    "source_path_b": baseline.cv_path,
                    "pairing_status": "AUTHENTICATED",
                    "notes": pairing_note,
                }
            )
        delta_array = np.asarray(differences, dtype=float)
        wins_a = int((delta_array > 0).sum())
        wins_b = int((delta_array < 0).sum())
        ties = int((delta_array == 0).sum())
        (
            w_positive,
            w_negative,
            exact_p,
            rank_biserial,
            nonzero_count,
            tied_absolute,
            scipy_note,
        ) = exact_wilcoxon(delta_array)
        rating = consistency_rating(wins_a, wins_b, ties)
        mean_delta = float(delta_array.mean())
        median_delta = float(np.median(delta_array))
        direction = (
            "pipeline_a higher"
            if mean_delta > 0
            else "mRMR higher"
            if mean_delta < 0
            else "no mean difference"
        )
        base_result.update(
            {
                "paired_folds": 5,
                "nonzero_pairs": nonzero_count,
                "mean_delta_auc": mean_delta,
                "median_delta_auc": median_delta,
                "std_delta_auc": float(delta_array.std(ddof=1)),
                "min_delta_auc": float(delta_array.min()),
                "max_delta_auc": float(delta_array.max()),
                "pipeline_a_wins": wins_a,
                "pipeline_b_wins": wins_b,
                "ties": ties,
                "wilcoxon_w_positive": w_positive,
                "wilcoxon_w_negative": w_negative,
                "exact_two_sided_p": exact_p,
                "rank_biserial_correlation": rank_biserial,
                "raw_significant_0_05": bool(exact_p < 0.05),
                "direction": direction,
                "consistency_rating": rating,
                "interpretation": interpretation_text(
                    candidate_name, wins_a, wins_b, ties, exact_p, rating
                ),
            }
        )
        metadata[comp_id].update(
            {
                "tied_absolute_ranks": tied_absolute,
                "scipy_cross_check": scipy_note,
                "fold_deltas": differences,
                "fold_mean_direction": int(np.sign(mean_delta)),
                "oot_direction": int(
                    np.sign(metadata[comp_id]["oot_delta_auc"])
                ),
            }
        )
        result_rows.append(base_result)

    results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    computed = results["comparison_status"].eq("COMPUTED")
    tested_indices = results.index[computed].tolist()
    ordered = sorted(
        tested_indices,
        key=lambda index: (
            float(results.at[index, "exact_two_sided_p"]),
            str(results.at[index, "comparison_id"]),
        ),
    )
    running_max = 0.0
    m = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(
            1.0,
            (m - rank) * float(results.at[index, "exact_two_sided_p"]),
        )
        running_max = max(running_max, adjusted)
        results.at[index, "holm_adjusted_p"] = running_max
        results.at[index, "holm_significant_0_05"] = bool(running_max < 0.05)
    return results, pd.DataFrame(detail_rows, columns=DETAIL_COLUMNS), metadata


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.15g", lineterminator="\n")


def make_figure(results: pd.DataFrame, details: pd.DataFrame, path: Path) -> None:
    computed = results[results["comparison_status"].eq("COMPUTED")].copy()
    if computed.empty:
        raise ValueError("No computed comparisons are available for the figure.")
    order = computed["comparison_id"].tolist()
    labels = [
        f"{row.dataset} | {row.model} | {PIPELINE_LABELS[row.pipeline_a]}"
        for row in computed.itertuples(index=False)
    ]
    all_deltas = details["delta_auc"].astype(float).to_numpy()
    extent = max(abs(float(all_deltas.min())), abs(float(all_deltas.max())), 0.001)
    x_limit = extent * 1.15
    fig, ax = plt.subplots(figsize=(12, 8.5))
    offsets = np.linspace(-0.14, 0.14, 5)
    for position, comp_id in enumerate(order):
        group = details[details["comparison_id"].eq(comp_id)].sort_values("fold_id")
        values = group["delta_auc"].astype(float).to_numpy()
        ax.hlines(
            position,
            float(values.min()),
            float(values.max()),
            color="#4c566a",
            linewidth=2,
            zorder=1,
        )
        ax.scatter(
            values,
            position + offsets,
            s=20,
            color="#6b7280",
            alpha=0.9,
            zorder=2,
        )
        ax.scatter(
            [float(np.median(values))],
            [position],
            marker="D",
            s=48,
            color="#1f4e79",
            zorder=3,
        )
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(-x_limit, x_limit)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired validation AUC difference (pipeline − mRMR)")
    ax.set_title("Fold-level AUC consistency across authenticated comparisons")
    ax.grid(axis="x", color="#d1d5db", linewidth=0.6)
    fig.text(
        0.5,
        0.015,
        "Diamond: median; line: observed fold minimum–maximum (not a confidence "
        "interval); gray points: five saved folds.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(fig)


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def build_summary(
    runs: dict[tuple[str, str, str], AuthenticatedRun],
    results: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    generated_names: list[str],
) -> str:
    computed = results[results["comparison_status"].eq("COMPUTED")]
    unavailable = results[~results["comparison_status"].eq("COMPUTED")]
    lines = [
        "# Paired Fold Significance and Consistency Analysis",
        "",
        "## Objective",
        "",
        "This analysis tests whether three LLM-family feature-selection pipelines "
        "show consistent validation ROC AUC differences from the statistical mRMR "
        "baseline across the five saved cross-validation folds. It covers Home "
        "Credit and LendingClub v2 with Logistic Regression and CatBoost, yielding "
        "12 planned primary comparisons. No training, feature selection, "
        "prediction generation, preprocessing, dataset construction, or CLIP work "
        "was rerun.",
        "",
        "## Runs and sources",
        "",
    ]
    run_rows: list[list[Any]] = []
    for dataset, model, pipeline in itertools.product(DATASETS, MODELS, PIPELINES):
        run = runs[(dataset, model, pipeline)]
        run_rows.append(
            [
                DATASET_LABELS[dataset],
                MODEL_LABELS[model],
                PIPELINE_LABELS[pipeline],
                f"`{run.run_id}`",
                f"`{run.cv_path}`",
                "authenticated",
            ]
        )
    lines.extend(
        markdown_table(
            ["Dataset", "Model", "Pipeline", "Run ID", "Fold source", "Pairability"],
            run_rows,
        )
    )
    lines.extend(
        [
            "",
            "The objective requires 16 canonical runs: four pipelines in each of "
            "four dataset/model strata. This produces 80 master fold rows. The "
            "prompt's separate estimate of eight runs and 40 rows is arithmetically "
            "inconsistent with its 12-comparison design, so the complete design "
            "takes precedence.",
            "",
            "## Method",
            "",
            "For every run, exactly five numeric fold rows were extracted from "
            "`cv_results.csv`; aggregate `mean` and `std` rows were excluded. The "
            "`auc` field was accepted only after its five-fold mean matched "
            "`run_manifest.json` and each stored Gini value satisfied "
            "`gini = 2 × auc − 1`. Fold identity is a SHA-256 digest of dataset and "
            "data-manifest identity, target contract, split contract, fold number, "
            "validation index bounds, validation time bounds, and validation row "
            "count. Every candidate fold identity had to equal its mRMR counterpart.",
            "",
            "Differences are candidate AUC minus mRMR AUC. Zero differences are "
            "removed before ranking; tied absolute differences receive average "
            "ranks. The two-sided Wilcoxon p-value is calculated by exhaustively "
            "enumerating all sign assignments for the non-zero ranks. The "
            "rank-biserial correlation is `(W+ − W−) / (W+ + W−)`. Holm correction "
            f"is applied across the {len(computed)} comparisons actually computed. "
            "With only five folds, inference is low-powered and descriptive fold "
            "direction is more informative than a binary threshold.",
            "",
            "## Results",
            "",
        ]
    )
    result_rows: list[list[Any]] = []
    for row in results.itertuples(index=False):
        result_rows.append(
            [
                row.dataset,
                row.model,
                PIPELINE_LABELS[row.pipeline_a],
                fmt_float(row.mean_delta_auc),
                fmt_float(row.median_delta_auc),
                (
                    f"{row.pipeline_a_wins}-{row.pipeline_b_wins}-{row.ties}"
                    if row.comparison_status == "COMPUTED"
                    else "NA"
                ),
                fmt_float(row.exact_two_sided_p, 4),
                fmt_float(row.holm_adjusted_p, 4),
                fmt_float(row.rank_biserial_correlation, 3),
                row.consistency_rating,
                row.comparison_status,
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Dataset",
                "Model",
                "Pipeline vs mRMR",
                "Mean Δ",
                "Median Δ",
                "Wins-losses-ties",
                "Raw p",
                "Holm p",
                "Rank-biserial",
                "Consistency",
                "Status",
            ],
            result_rows,
        )
    )
    lines.extend(["", "## Interpretation by dataset", "", "### Home Credit", ""])
    for model in MODELS:
        subset = computed[
            computed["dataset"].eq(DATASET_LABELS["homecredit"])
            & computed["model"].eq(MODEL_LABELS[model])
        ]
        lines.append(f"**{MODEL_LABELS[model]}.**")
        for row in subset.itertuples(index=False):
            oot_delta = metadata[row.comparison_id]["oot_delta_auc"]
            alignment = (
                "aligned"
                if np.sign(float(row.mean_delta_auc)) == np.sign(oot_delta)
                else "not aligned"
            )
            lines.append(
                f"{PIPELINE_LABELS[row.pipeline_a]} had mean fold ΔAUC "
                f"{row.mean_delta_auc:+.6f}, median {row.median_delta_auc:+.6f}, "
                f"and wins-losses-ties {row.pipeline_a_wins}-"
                f"{row.pipeline_b_wins}-{row.ties}. Its direction was {alignment} "
                f"with the authenticated OOT ΔAUC ({oot_delta:+.6f})."
            )
        lines.append("")
    lines.extend(["### LendingClub v2", ""])
    for model in MODELS:
        subset = computed[
            computed["dataset"].eq(DATASET_LABELS["lendingclub_v2"])
            & computed["model"].eq(MODEL_LABELS[model])
        ]
        lines.append(f"**{MODEL_LABELS[model]}.**")
        for row in subset.itertuples(index=False):
            oot_delta = metadata[row.comparison_id]["oot_delta_auc"]
            alignment = (
                "aligned"
                if np.sign(float(row.mean_delta_auc)) == np.sign(oot_delta)
                else "not aligned"
            )
            lines.append(
                f"{PIPELINE_LABELS[row.pipeline_a]} had mean fold ΔAUC "
                f"{row.mean_delta_auc:+.6f}, median {row.median_delta_auc:+.6f}, "
                f"and wins-losses-ties {row.pipeline_a_wins}-"
                f"{row.pipeline_b_wins}-{row.ties}. Its direction was {alignment} "
                f"with the authenticated OOT ΔAUC ({oot_delta:+.6f})."
            )
        lines.append("")
    strongest = computed[computed["consistency_rating"].eq("strong")]
    strong_names = (
        ", ".join(
            f"{row.dataset}/{row.model}/{PIPELINE_LABELS[row.pipeline_a]}"
            for row in strongest.itertuples(index=False)
        )
        if not strongest.empty
        else "none"
    )
    raw_count = int(computed["raw_significant_0_05"].astype(bool).sum())
    holm_count = int(computed["holm_significant_0_05"].astype(bool).sum())
    lines.extend(
        [
            "## Scientific interpretation",
            "",
            "The LLM-family methods were not uniformly stronger than mRMR; "
            "direction depended on dataset, downstream model, and selector. "
            f"Strong same-direction fold consistency occurred for {strong_names}. "
            f"{raw_count} comparisons crossed raw p < 0.05 and {holm_count} crossed "
            "Holm-adjusted p < 0.05. With five non-zero differences, even a perfect "
            "five-fold direction normally yields a minimum exact two-sided p-value "
            "of 0.0625. Non-significance therefore reflects both the observed data "
            "and the low-power design and cannot establish equivalence. Fold and "
            "OOT directions were assessed separately; discordance is retained "
            "rather than reconciled away. These tests describe consistency under "
            "the saved CV design and do not replace authenticated OOT rankings.",
            "",
            "## Limitations",
            "",
            "1. Five folds provide very low inferential power.",
            "2. With five non-zero paired differences, the smallest attainable "
            "exact two-sided Wilcoxon p-value is normally 0.0625.",
            "3. Consequently, no comparison with only five non-zero folds can "
            "normally cross a two-sided 0.05 threshold.",
            "4. A non-significant result does not prove that two pipelines are "
            "equivalent.",
            "5. Cross-validation folds are not fully independent scientific "
            "replications because their training samples may overlap.",
            "6. The tests evaluate fold-level consistency under the saved CV design; "
            "they do not test the final OOT AUC difference.",
            "7. OOT conclusions remain based on authenticated OOT metrics and are "
            "not replaced by these CV results.",
            "8. Effect direction, fold wins, median difference, and observed "
            "variability are more informative here than a binary significance label.",
            "",
            "## Conclusion",
            "",
            f"All {len(computed)} of 12 planned comparisons were authenticated and "
            "computed from direct fold evidence. The results support restrained "
            "claims about directional consistency, not definitive superiority or "
            "equivalence. The observed ranges and rank-biserial values should be "
            "read alongside OOT metrics and the limitations of five overlapping CV "
            "training samples.",
            "",
            "## Reproducibility outputs",
            "",
        ]
    )
    for name in generated_names:
        lines.append(f"- `{name}`")
    if not unavailable.empty:
        lines.extend(
            [
                "",
                f"Unavailable planned comparisons: {len(unavailable)}. Reasons are "
                "recorded in the primary results table.",
            ]
        )
    return "\n".join(lines) + "\n"


def validate_outputs(
    root: Path,
    output_dir: Path,
    source_hashes_before: dict[str, str],
    master: pd.DataFrame,
    details: pd.DataFrame,
    results: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    summary_text: str,
    generated_paths: list[Path],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["canonical_run_count"] = len(master["run_id"].unique()) == 16
    checks["no_lendingclub_v1"] = set(master["dataset"]) == {
        "Home Credit",
        "LendingClub v2",
    }
    checks["no_invalid_failed_smoke_or_dry_runs"] = not master["run_id"].str.contains(
        "failed|incomplete|smoke|dry|clip", case=False, regex=True
    ).any()
    checks["five_unique_folds_per_run"] = bool(
        master.groupby("run_id")["fold_id"].agg(["count", "nunique"]).eq(5).all().all()
    )
    checks["all_auc_finite_and_bounded"] = bool(
        np.isfinite(master["validation_auc"].astype(float)).all()
        and master["validation_auc"].astype(float).between(0, 1).all()
    )
    checks["all_planned_comparisons_present"] = len(results) == 12
    checks["all_comparisons_computed"] = bool(
        results["comparison_status"].eq("COMPUTED").all()
    )
    checks["fold_identity_matches"] = bool(
        details["pairing_status"].eq("AUTHENTICATED").all()
        and len(details) == 60
    )
    reconstructed = []
    master_index = {
        (row.dataset, row.model, row.pipeline_id, int(row.fold_id)): float(
            row.validation_auc
        )
        for row in master.itertuples(index=False)
    }
    for row in details.itertuples(index=False):
        reconstructed.append(
            math.isclose(
                float(row.delta_auc),
                master_index[
                    (row.dataset, row.model, row.pipeline_a, int(row.fold_id))
                ]
                - master_index[
                    (row.dataset, row.model, row.pipeline_b, int(row.fold_id))
                ],
                rel_tol=0,
                abs_tol=1e-15,
            )
        )
    checks["paired_differences_reproduce"] = all(reconstructed)
    exact_reproduced = []
    for row in results.itertuples(index=False):
        delta = details.loc[
            details["comparison_id"].eq(row.comparison_id), "delta_auc"
        ].to_numpy(dtype=float)
        exact_reproduced.append(
            math.isclose(
                exact_wilcoxon(delta)[2],
                float(row.exact_two_sided_p),
                rel_tol=0,
                abs_tol=1e-12,
            )
        )
    checks["exact_p_values_reproduce"] = all(exact_reproduced)
    raw = results["exact_two_sided_p"].astype(float).to_numpy()
    adjusted = results["holm_adjusted_p"].astype(float).to_numpy()
    order = np.argsort(raw, kind="stable")
    expected = np.empty(len(raw))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * raw[index]))
        expected[index] = running
    checks["holm_values_reproduce"] = bool(np.allclose(adjusted, expected, atol=1e-12))
    rank_biserial_ok = []
    for row in results.itertuples(index=False):
        denominator = float(row.wilcoxon_w_positive) + float(
            row.wilcoxon_w_negative
        )
        expected_effect = (
            0.0
            if denominator == 0
            else (
                float(row.wilcoxon_w_positive)
                - float(row.wilcoxon_w_negative)
            )
            / denominator
        )
        rank_biserial_ok.append(
            math.isclose(
                expected_effect,
                float(row.rank_biserial_correlation),
                rel_tol=0,
                abs_tol=1e-12,
            )
        )
    checks["rank_biserial_reproduces"] = all(rank_biserial_ok)
    checks["wins_losses_ties_sum"] = bool(
        (
            results["pipeline_a_wins"].astype(int)
            + results["pipeline_b_wins"].astype(int)
            + results["ties"].astype(int)
        )
        .eq(results["paired_folds"].astype(int))
        .all()
    )
    checks["csv_files_parse"] = all(
        not pd.read_csv(path).empty
        for path in (
            output_dir / "fold_auc_master.csv",
            output_dir / "paired_difference_details.csv",
            output_dir / "paired_significance_results.csv",
        )
    )
    checks["summary_references_every_output"] = all(
        path.name in summary_text for path in generated_paths
    )
    checks["figure_contains_only_computed_comparisons"] = (
        len(metadata) == int(results["comparison_status"].eq("COMPUTED").sum()) == 12
    )
    source_hashes_after = {
        source: sha256_file(root / source) for source in source_hashes_before
    }
    mutations = [
        source
        for source, digest in source_hashes_before.items()
        if source_hashes_after[source] != digest
    ]
    checks["scientific_sources_unchanged"] = not mutations
    checks["no_placeholder_text"] = not any(
        token in summary_text.upper() for token in ("TODO", "TBD", "PLACEHOLDER")
    )
    git_check = subprocess.run(
        ["git", "diff", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    checks["git_diff_check"] = git_check.returncode == 0
    failures = [name for name, passed in checks.items() if passed is not True]
    if failures:
        raise ValueError(f"Validation checks failed: {failures}")
    return {
        "checks": checks,
        "failures": [],
        "scientific_source_mutations": mutations,
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        root
        / "results"
        / "finalized_research"
        / "pending_analyses"
        / "significance_tests"
    )
    if output_dir.resolve() != Path(__file__).resolve().parent:
        raise ValueError(
            "This analysis may write only to its canonical significance_tests directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory, manifest_index, starting_files = load_canonical_indexes(root)
    source_hashes: dict[str, str] = {
        source: sha256_file(root / source) for source in starting_files
    }
    runs = resolve_runs(
        root, inventory, manifest_index, source_hashes
    )
    source_hashes_before = dict(sorted(source_hashes.items()))
    master = pd.DataFrame(
        [
            row
            for key in itertools.product(DATASETS, MODELS, PIPELINES)
            for row in runs[key].master_rows
        ],
        columns=MASTER_COLUMNS,
    )
    results, details, comparison_metadata = build_comparisons(runs)

    master_path = output_dir / "fold_auc_master.csv"
    details_path = output_dir / "paired_difference_details.csv"
    results_path = output_dir / "paired_significance_results.csv"
    figure_path = output_dir / "figures" / "paired_auc_difference_forest.png"
    summary_path = output_dir / "significance_summary.md"
    manifest_path = output_dir / "significance_manifest.json"
    script_path = Path(__file__).resolve()
    save_csv(master, master_path)
    save_csv(details, details_path)
    save_csv(results, results_path)
    make_figure(results, details, figure_path)

    generated_relative = [
        relpath(root, master_path),
        relpath(root, details_path),
        relpath(root, results_path),
        relpath(root, figure_path),
        relpath(root, summary_path),
        relpath(root, manifest_path),
        relpath(root, script_path),
    ]
    summary_text = build_summary(
        runs, results, comparison_metadata, generated_relative
    )
    summary_path.write_text(summary_text, encoding="utf-8", newline="\n")

    validation = validate_outputs(
        root,
        output_dir,
        source_hashes_before,
        master,
        details,
        results,
        comparison_metadata,
        summary_text,
        [
            master_path,
            details_path,
            results_path,
            figure_path,
            summary_path,
            manifest_path,
            script_path,
        ],
    )
    non_manifest_generated = [
        master_path,
        details_path,
        results_path,
        figure_path,
        summary_path,
        script_path,
    ]
    generated_hashes = {
        relpath(root, path): sha256_file(path) for path in non_manifest_generated
    }
    canonical_mapping = [
        {
            "dataset": DATASET_LABELS[dataset],
            "model": MODEL_LABELS[model],
            "pipeline_id": pipeline,
            "pipeline_label": PIPELINE_LABELS[pipeline],
            "run_id": runs[(dataset, model, pipeline)].run_id,
            "cv_results_path": runs[(dataset, model, pipeline)].cv_path,
            "run_manifest_path": runs[(dataset, model, pipeline)].manifest_path,
            "split_manifest_path": runs[(dataset, model, pipeline)].split_path,
            "registry_status": runs[(dataset, model, pipeline)].source_status,
        }
        for dataset, model, pipeline in itertools.product(
            DATASETS, MODELS, PIPELINES
        )
    ]
    computed_ids = results.loc[
        results["comparison_status"].eq("COMPUTED"), "comparison_id"
    ].tolist()
    unavailable_records = results.loc[
        ~results["comparison_status"].eq("COMPUTED"),
        ["comparison_id", "comparison_status", "interpretation"],
    ].to_dict("records")
    manifest_payload = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_value(root, "rev-parse", "HEAD"),
        "repository_branch": git_value(root, "branch", "--show-current"),
        "source_files": list(source_hashes_before),
        "source_file_sha256": source_hashes_before,
        "canonical_run_mapping": canonical_mapping,
        "fold_identity_method": (
            "SHA-256 over canonical JSON containing dataset/data-manifest/target/"
            "split identities, fold ID, validation indices, time bounds, and row count."
        ),
        "auc_source_columns": {
            run.run_id: run.auc_column for run in runs.values()
        },
        "planned_comparisons": [
            comparison_id(dataset, model, candidate)
            for dataset, model, candidate in itertools.product(
                DATASETS, MODELS, CANDIDATES
            )
        ],
        "computed_comparisons": computed_ids,
        "unavailable_comparisons": unavailable_records,
        "wilcoxon_method": (
            "Manual exact two-sided paired Wilcoxon signed-rank test using average "
            "absolute ranks and exhaustive enumeration of every non-zero-rank sign "
            "assignment; min(W+, W-) is the test statistic."
        ),
        "zero_difference_method": (
            "Differences with absolute value <= 1e-15 are treated as zero and "
            "excluded from signed ranks; all-zero comparisons receive p=1 and "
            "rank-biserial correlation=0."
        ),
        "tie_handling": (
            "Average ranks for tied absolute differences; ties in AUC are counted "
            "separately from directional wins."
        ),
        "multiple_testing_method": (
            f"Holm step-down family-wise correction across {len(computed_ids)} "
            "successfully computed comparisons from a planned family of 12."
        ),
        "effect_size_method": (
            "Paired rank-biserial correlation = (W_positive - W_negative) / "
            "(W_positive + W_negative)."
        ),
        "generated_files": generated_relative,
        "generated_file_sha256": generated_hashes,
        "manifest_self_hash_policy": (
            "The manifest cannot contain its own final file SHA-256 without a "
            "self-referential hash paradox; every other generated file is hashed."
        ),
        "scientific_source_mutations": validation["scientific_source_mutations"],
        "validation_results": validation,
        "known_limitations": [
            "Five folds provide very low inferential power.",
            "With five non-zero pairs, the normal minimum attainable exact "
            "two-sided p-value is 0.0625.",
            "Non-significance is not evidence of equivalence.",
            "CV training samples may overlap and are not independent replications.",
            "Fold consistency does not replace authenticated OOT evaluation.",
            "The prompt's 8-run/40-row estimate conflicts with its 12-comparison "
            "design; 16 runs and 80 fold rows are required.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    read_json(manifest_path)
    print(
        json.dumps(
            {
                "canonical_runs": len(runs),
                "master_fold_rows": len(master),
                "planned_comparisons": 12,
                "computed_comparisons": len(computed_ids),
                "unavailable_comparisons": len(unavailable_records),
                "detail_rows": len(details),
                "raw_significant": int(
                    results["raw_significant_0_05"].astype(bool).sum()
                ),
                "holm_significant": int(
                    results["holm_significant_0_05"].astype(bool).sum()
                ),
                "validation_failures": validation["failures"],
                "scientific_source_mutations": validation[
                    "scientific_source_mutations"
                ],
                "output_directory": relpath(root, output_dir),
                "manifest_sha256": sha256_file(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
