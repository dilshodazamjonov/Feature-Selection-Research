"""Builders for each ``todo`` CSV.  Frozen evidence first, recomputation second."""

from __future__ import annotations

import math
import re
import shutil
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from scripts.todo_fill import frozen, llm_cache
from scripts.todo_fill.common import (
    BORUTA_STAGE_LABELS,
    BUDGETS,
    COPY_VERBATIM,
    FOLDS,
    FULL_DEV,
    HOMECREDIT,
    IV_THEN_BORUTA_POOL,
    LENDINGCLUB,
    PARTITIONS,
    REPO_ROOT,
    THIRD,
    UNIVERSE_SIZE,
    FillError,
    Manifest,
    Skeleton,
    fmt,
    log,
    method_for,
    nogueira,
    pairwise_jaccard_summary,
    write_json,
)
from scripts.todo_fill.selections import Request, SelectionEngine, fold_sets, full_dev_selection

STEP = {
    "01_obfuscation.csv": "01",
    "06_leakage.csv": "06",
    "08_boruta.csv": "08",
    "09_psi.csv": "09",
    "10_stability.csv": "10",
    "11_subsets.csv": "11",
    "12_overlap.csv": "12",
    "17_mrmr.csv": "17",
}


def _row_key(row: dict[str, str], *columns: str) -> dict[str, str]:
    return {column: row.get(column, "") for column in columns}


def _method(row: dict[str, str]) -> str:
    return method_for(row["selector"], row["dataset"])


# ---------------------------------------------------------------- verbatim copy


def copy_verbatim(todo_dir: Path, out_dir: Path, manifest: Manifest) -> None:
    for name in COPY_VERBATIM:
        source = todo_dir / name
        if not source.exists():
            manifest.skip("copy", {"file": name}, "not present in the todo folder")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, out_dir / name)
        skeleton = Skeleton.read(source)
        value_columns = [column for column in skeleton.columns if column in {"ho_auc", "dev_oof_auc", "brier", "n_features"}]
        filled, total = skeleton.fill_rate(value_columns)
        manifest.csv_summary[name] = {"filled": filled, "total": total, "status": "copied verbatim (computationally expensive; not recomputed)"}
        manifest.note("copy", f"{name} copied verbatim from todo/ ({filled}/{total} value cells were already filled)")


# --------------------------------------------------------------- planning helpers


def frozen_boruta_count(row: dict[str, str]) -> tuple[int | None, str | None]:
    """Frozen realised subset size for one ``08_boruta.csv`` row, if the audit layer has it."""

    dataset, backbone, label, partition = row["dataset"], row["backbone"], row["selector"], row["partition"]
    method = method_for(label, dataset)
    if dataset in (HOMECREDIT, LENDINGCLUB):
        if method == "boruta_random_forest":
            if partition == FULL_DEV:
                cell = frozen.final_table_row(dataset, method, backbone, f"k{BUDGETS[backbone]}")
                return (cell["realized_k"], cell["source"]) if cell else (None, None)
            item = frozen.baseline_stability().get((dataset, backbone, method))
            if item and len(item.fold_counts) == 5:
                return item.fold_counts[FOLDS.index(partition)], item.source
        if method == "iv_then_boruta":
            pool = IV_THEN_BORUTA_POOL[dataset]
            if partition == FULL_DEV:
                cell = frozen.final_table_row(dataset, method, backbone, f"pool{pool}")
                return (cell["realized_k"], cell["source"]) if cell else (None, None)
            item = frozen.combination_fold_support().get((dataset, method, pool, FOLDS.index(partition) + 1))
            if item and item.get("realized_selected_count") is not None:
                return item["realized_selected_count"], item["source"]
    if dataset == THIRD and partition != FULL_DEV:
        pool = IV_THEN_BORUTA_POOL[dataset] if method == "iv_then_boruta" else None
        folds = frozen.third_dev_accounting().get((method, backbone, pool))
        if folds:
            item = folds.get(FOLDS.index(partition) + 1)
            if item and item.get("status") == "complete" and item.get("selected_feature_count") is not None:
                return item["selected_feature_count"], item["source"]
            if item:
                return None, f"frozen status {item.get('status')} ({item.get('reason')})"
    return None, None


def frozen_stability(row: dict[str, str]) -> dict[str, Any] | None:
    dataset, backbone, label = row["dataset"], row["backbone"], row["selector"]
    method = method_for(label, dataset)
    if dataset not in (HOMECREDIT, LENDINGCLUB):
        return None
    if method in {"random_k", "iv_woe", "boruta_random_forest", "rfe_catboost", "catboost_shap", "mrmr_mutual_information", "lasso_l1_logistic"}:
        item = frozen.baseline_stability().get((dataset, backbone, method))
        if item:
            return {"d": UNIVERSE_SIZE[dataset], "nogueira": item.nogueira, "jaccard_mean": item.jaccard_mean, "jaccard_min": item.jaccard_min, "jaccard_max": item.jaccard_max, "source": f"frozen:{item.source}"}
    if method == "iv_then_boruta":
        item = frozen.combination_stability().get((dataset, method, IV_THEN_BORUTA_POOL[dataset], backbone))
        if item:
            return {"d": UNIVERSE_SIZE[dataset], "nogueira": item.nogueira, "jaccard_mean": item.jaccard_mean, "jaccard_min": item.jaccard_min, "jaccard_max": item.jaccard_max, "source": f"frozen:{item.source} (Nogueira rebuilt from frozen selection frequencies)"}
    return None


HEADLINE_STEPS = ("09_psi.csv", "11_subsets.csv", "12_overlap.csv")


def plan_requests(skeletons: dict[str, Skeleton], manifest: Manifest, leakage_headline_rows: list[dict[str, str]] | None = None) -> list[Request]:
    """Every (dataset, backbone, label, partition) the CSVs need and frozen evidence cannot supply.

    ``leakage_headline_rows`` are the ``11_subsets.csv`` rows: step 06 flags membership in the
    Home Credit headline subsets, so those full-DEV selections are requested whenever step 06 is
    planned, even when step 11 itself is excluded.
    """

    requests: list[Request] = []
    if "08_boruta.csv" in skeletons:
        for row in skeletons["08_boruta.csv"].rows:
            count, _ = frozen_boruta_count(row)
            if count is None:
                requests.append(Request(row["dataset"], row["backbone"], row["selector"], row["partition"]))
    if "10_stability.csv" in skeletons:
        for row in skeletons["10_stability.csv"].rows:
            if row["selector"] == "Pure LLM" or frozen_stability(row) is not None:
                continue
            if row["selector"] == "PCA":
                continue
            if row["selector"] == "Random K" and row["dataset"] == THIRD:
                continue
            for partition in FOLDS:
                requests.append(Request(row["dataset"], row["backbone"], row["selector"], partition))
    headline: set[tuple[str, str, str]] = set()
    for name in HEADLINE_STEPS:
        if name not in skeletons:
            continue
        for row in skeletons[name].rows:
            dataset = row.get("dataset") or LENDINGCLUB
            headline.add((dataset, row["backbone"], row["selector"]))
    if "06_leakage.csv" in skeletons:
        for row in leakage_headline_rows or []:
            if row["dataset"] == HOMECREDIT:
                headline.add((row["dataset"], row["backbone"], row["selector"]))
    for dataset, backbone, label in sorted(headline):
        requests.append(Request(dataset, backbone, label, FULL_DEV))
    unique = list(dict.fromkeys(requests))
    manifest.note("plan", f"{len(unique)} selector fits are not covered by frozen evidence and will be recomputed where feasible")
    return unique


# ------------------------------------------------------------------ 08 boruta


def fill_08(skeleton: Skeleton, engine: SelectionEngine, manifest: Manifest) -> Skeleton:
    step = "08"
    for row in skeleton.rows:
        key = _row_key(row, "dataset", "backbone", "selector", "K_budget", "partition")
        count, source = frozen_boruta_count(row)
        if count is not None:
            row["n_selected"] = fmt(count)
            manifest.cell(step, key, "n_selected", count, f"frozen:{source}")
            continue
        record = engine.lookup(Request(row["dataset"], row["backbone"], row["selector"], row["partition"]))
        if record is not None:
            row["n_selected"] = fmt(record["n_selected"])
            extra = {}
            if record.get("details", {}).get("boruta_confirmed_count") is not None:
                extra["boruta_confirmed_count"] = record["details"]["boruta_confirmed_count"]
            manifest.cell(step, key, "n_selected", record["n_selected"], f"recomputed:{record['protocol']}", **extra)
            continue
        row["n_selected"] = ""
        manifest.skip(step, key, source or "no frozen count and the selector could not be refitted here")
    return skeleton


# --------------------------------------------------------------- 10 stability


def fill_10(skeleton: Skeleton, engine: SelectionEngine, manifest: Manifest, pins: dict[tuple[str, int], dict[str, Any]]) -> Skeleton:
    step = "10"
    for row in skeleton.rows:
        key = _row_key(row, "dataset", "backbone", "selector", "K")
        dataset, backbone, label = row["dataset"], row["backbone"], row["selector"]
        k = BUDGETS[backbone]
        if label == "Pure LLM":
            if dataset == THIRD:
                manifest.skip(step, key, "third-dataset LLM ranking is not on this machine")
                continue
            pin = pins.get((dataset, k))
            sets = llm_cache.truncated_sets(dataset, k, budget=k)
            summary = pairwise_jaccard_summary(list(sets.values()))
            row["jaccard_min"] = fmt(summary["min"])
            row["jaccard_max"] = fmt(summary["max"])
            source = "cache:artifacts/llm_cache top-K fold truncations" + (" (fold files pinned to the pre-filled Table 9 values)" if pin and pin.get("matched") else " (WARNING: pre-filled values not reproduced)")
            manifest.cell(step, key, "jaccard_min", summary["min"], source, files=pin.get("chosen_files") if pin else None)
            manifest.cell(step, key, "jaccard_max", summary["max"], source)
            if pin and not pin.get("matched"):
                manifest.note(step, f"{dataset} K={k}: cached fold rankings do not reproduce the pre-filled Nogueira/Jaccard; kept the heuristic file choice ({pin.get('reproduced')})")
            continue
        if label == "PCA":
            values = {"d": UNIVERSE_SIZE[dataset], "nogueira": 1.0, "jaccard_mean": 1.0, "jaccard_min": 1.0, "jaccard_max": 1.0}
            if dataset == THIRD:
                manifest.skip(step, key, "PCA was never part of the frozen third-dataset matrix")
                continue
            for field_name, value in values.items():
                row[field_name] = fmt(value)
                manifest.cell(step, key, field_name, value, "analytic:PCASelector publishes component names PC1..PCk, identical in every fold")
            continue
        if label == "Random K" and dataset == THIRD:
            values = {"d": UNIVERSE_SIZE[dataset], "nogueira": 1.0, "jaccard_mean": 1.0, "jaccard_min": 1.0, "jaccard_max": 1.0}
            for field_name, value in values.items():
                row[field_name] = fmt(value)
                manifest.cell(step, key, field_name, value, "analytic:random_k draws one seeded permutation of the fixed candidate order, so every fold selects the same subset (matches the frozen two-dataset value 1.0)")
            continue
        found = frozen_stability(row)
        if found is not None:
            for field_name in ("d", "nogueira", "jaccard_mean", "jaccard_min", "jaccard_max"):
                row[field_name] = fmt(found[field_name])
                manifest.cell(step, key, field_name, found[field_name], found["source"])
            continue
        sets = fold_sets(engine, dataset, backbone, label)
        if sets is None:
            manifest.skip(step, key, "fold selections unavailable (frozen lists absent and refit not completed on this machine)")
            continue
        d = UNIVERSE_SIZE[dataset]
        nog = nogueira([sets[p] for p in FOLDS], d)
        summary = pairwise_jaccard_summary([sets[p] for p in FOLDS])
        values = {"d": d, "nogueira": nog, "jaccard_mean": summary["mean"], "jaccard_min": summary["min"], "jaccard_max": summary["max"]}
        legacy = frozen.legacy_values(dataset, backbone, method_for(label, dataset))
        check = ""
        if legacy and legacy.get("nogueira_stability") is not None and nog is not None:
            check = f"; archive report Nogueira={legacy['nogueira_stability']} meanJaccard={legacy.get('mean_pairwise_jaccard')} (archive used the model-matrix column count as d)"
        for field_name, value in values.items():
            row[field_name] = fmt(value)
            manifest.cell(step, key, field_name, value, f"recomputed:fold refits under the {engine.lookup(Request(dataset, backbone, label, 'fold1'))['protocol']} protocol{check}")
    return skeleton


# ----------------------------------------------------------------- 11 subsets


def _ordered_features(record: dict[str, Any]) -> list[str]:
    features = list(record["features"])
    ranked = record.get("ranked")
    if ranked:
        ordered = [item for item in ranked if item in set(features)]
        ordered += [item for item in features if item not in set(ordered)]
        return ordered
    return features


def fill_11(skeleton: Skeleton, engine: SelectionEngine, manifest: Manifest) -> Skeleton:
    step = "11"
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in skeleton.rows:
        groups.setdefault((row["dataset"], row["backbone"], row["selector"]), []).append(row)
    for (dataset, backbone, label), rows in groups.items():
        key = {"dataset": dataset, "backbone": backbone, "selector": label}
        record = full_dev_selection(engine, dataset, backbone, label)
        if record is None:
            manifest.skip(step, key, "full-DEV selection unavailable on this machine")
            continue
        ordered = _ordered_features(record)
        rows.sort(key=lambda item: int(item["rank"]))
        for row in rows:
            rank = int(row["rank"])
            row["feature"] = ordered[rank - 1] if rank <= len(ordered) else ""
        source = f"{'cache' if record['protocol'] == 'cached_llm_truncation' else 'recomputed'}:{record['protocol']}"
        manifest.cell(step, key, "feature", ordered, source, n_selected=record["n_selected"], ranked_order_available=bool(record.get("ranked")))
        if len(ordered) < len(rows):
            manifest.note(step, f"{key}: only {len(ordered)} features selected for {len(rows)} ranks (natural support below budget)")
    return skeleton


# -------------------------------------------------------------------- 09 PSI


def _dense_dev_ho(engine: SelectionEngine, dataset: str, raw_features: Sequence[str]):
    """Dense model-matrix representation of ``raw_features`` on DEV (fit) and HO (transform)."""

    import pandas as pd
    from credit_risk_fs.preprocessing.encoding import Preprocessor
    from scripts.todo_fill.common import PREPROCESSOR_KWARGS

    data = engine.data
    if dataset == THIRD:
        X_dev, _ = data.third().dev_frame(list(raw_features))
        X_ho, _ = data.third().holdout_frame(list(raw_features))
    else:
        X_dev = data.dev(dataset).X.loc[:, list(raw_features)]
        X_ho, _ = data.holdout(dataset, list(raw_features))
    preprocessor = Preprocessor(**PREPROCESSOR_KWARGS.get(dataset, {}))
    dense_dev = preprocessor.fit_transform(X_dev)
    dense_ho = preprocessor.transform(X_ho)
    if not isinstance(dense_dev, pd.DataFrame):
        dense_dev = pd.DataFrame(dense_dev)
        dense_ho = pd.DataFrame(dense_ho, columns=dense_dev.columns)
    return dense_dev, dense_ho


def _raw_bases(features: Iterable[str], universe: Iterable[str]) -> list[str]:
    return frozen.collapse_model_matrix_columns(features, universe)


def fill_09(skeleton: Skeleton, engine: SelectionEngine, manifest: Manifest, out_dir: Path) -> Skeleton:
    step = "09"
    from credit_risk_fs.evaluation.drift import calculate_psi

    detail_rows: list[dict[str, Any]] = []
    for row in skeleton.rows:
        key = _row_key(row, "dataset", "backbone", "selector", "K")
        dataset, backbone, label = row["dataset"], row["backbone"], row["selector"]
        record = full_dev_selection(engine, dataset, backbone, label)
        if record is None:
            manifest.skip(step, key, "full-DEV selection unavailable on this machine")
            continue
        try:
            universe = engine.data.third().predictors if dataset == THIRD else engine.data.dev(dataset).candidate_features
            raw = _raw_bases(record["features"], universe)
            dense_dev, dense_ho = _dense_dev_ho(engine, dataset, raw)
        except Exception as exc:  # noqa: BLE001
            manifest.skip(step, key, f"could not build DEV/HO frames: {type(exc).__name__}: {exc}")
            continue
        selected_dense = [column for column in dense_dev.columns if column in set(record["features"])]
        if not selected_dense:
            selected_dense = list(dense_dev.columns)
        psi_values: dict[str, float] = {}
        for column in selected_dense:
            value = calculate_psi(dense_dev[column], dense_ho[column], bins=10)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                psi_values[column] = float(value)
            detail_rows.append({**key, "feature": column, "raw_base": next((base for base in raw if column == base or column.startswith(base + "_")), column), "psi": value})
        if not psi_values:
            manifest.skip(step, key, "no finite PSI values")
            continue
        values = list(psi_values.values())
        psi_mean = statistics.fmean(values)
        psi_median = statistics.median(values)
        max_feature = max(psi_values, key=psi_values.get)
        psi_max = psi_values[max_feature]
        row["psi_mean"], row["psi_median"], row["psi_max"], row["psi_max_feature"] = fmt(psi_mean), fmt(psi_median), fmt(psi_max), max_feature
        consistent = psi_mean <= (psi_median + psi_max) / 2 + 1e-12
        source = f"recomputed:calculate_psi(10 DEV-quantile bins, eps 1e-6) on the dense model-matrix columns of the {'cached' if record['protocol']=='cached_llm_truncation' else 'refitted'} full-DEV subset"
        for field_name, value in (("psi_mean", psi_mean), ("psi_median", psi_median), ("psi_max", psi_max), ("psi_max_feature", max_feature)):
            manifest.cell(step, key, field_name, value, source, n_features=len(psi_values), delivery_check_mean_le_half_median_plus_max=consistent)
        frozen_psi = frozen.baseline_feature_psi().get((dataset, backbone, method_for(label, dataset)))
        if frozen_psi:
            frozen_map = {name: value for name, value in frozen_psi if value is not None}
            common = sorted(set(frozen_map) & set(psi_values))
            if common:
                max_diff = max(abs(frozen_map[name] - psi_values[name]) for name in common)
                manifest.note(step, f"{key}: frozen full_baseline_v1 PSI available for {len(common)} shared columns; max abs difference {max_diff:.2e}")
        legacy = frozen.legacy_values(dataset, backbone, method_for(label, dataset))
        if legacy and legacy.get("psi_mean") is not None:
            manifest.note(step, f"{key}: archive report psi_mean={legacy['psi_mean']} psi_median={legacy.get('psi_median')} psi_max={legacy.get('psi_max')} vs recomputed {psi_mean:.4f}/{psi_median:.4f}/{psi_max:.4f}")
    if detail_rows:
        import pandas as pd

        pd.DataFrame(detail_rows).to_csv(out_dir / "09_psi_features.csv", index=False, lineterminator="\n")
    return skeleton


# ---------------------------------------------------------------- 12 overlap

_FORMULA_FUNCTIONS = {
    "is_missing", "log1p", "sqrt", "max", "min", "mean", "fixed_bins", "group", "domain_cap",
    "derived_from_safe_fields", "abs", "log", "exp", "clip", "sum", "avg", "median", "mode", "var",
    "std", "count", "where", "if", "else", "and", "or", "not", "nan", "none", "true", "false",
}


def _inventory() -> dict[str, dict[str, str]]:
    import csv

    path = REPO_ROOT / "data/lendingclub_v2/metadata/feature_inventory.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["feature"]: row for row in csv.DictReader(handle)}


def _direct_bases(feature: str, inventory: dict[str, dict[str, str]]) -> set[str]:
    row = inventory.get(feature)
    if row is None:
        return {feature}
    formula = row.get("source_column_or_formula") or feature
    if formula.strip() == feature or row.get("feature_type") == "raw":
        return {feature}
    tokens = {token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula) if token.lower() not in _FORMULA_FUNCTIONS}
    tokens.discard(feature)
    return tokens or {feature}


def _root_bases(feature: str, inventory: dict[str, dict[str, str]], depth: int = 0) -> set[str]:
    direct = _direct_bases(feature, inventory)
    if depth > 6 or direct == {feature}:
        return direct
    roots: set[str] = set()
    for base in direct:
        if base == feature or base not in inventory or inventory[base].get("feature_type") == "raw":
            roots.add(base)
        else:
            roots |= _root_bases(base, inventory, depth + 1)
    return roots


def fill_12(skeleton: Skeleton, engine: SelectionEngine, manifest: Manifest, out_dir: Path) -> Skeleton:
    step = "12"
    inventory = _inventory()
    detail_rows: list[dict[str, Any]] = []
    for row in skeleton.rows:
        key = _row_key(row, "backbone", "selector", "K")
        record = full_dev_selection(engine, LENDINGCLUB, row["backbone"], row["selector"])
        if record is None:
            manifest.skip(step, key, "full-DEV selection unavailable on this machine")
            continue
        raw = _raw_bases(record["features"], engine.data.dev(LENDINGCLUB).candidate_features) if record["protocol"] != "cached_llm_truncation" else list(record["features"])
        bases = {feature: _root_bases(feature, inventory) for feature in raw}
        direct = {feature: _direct_bases(feature, inventory) for feature in raw}
        sharing = 0
        for feature, roots in bases.items():
            others = set().union(*(other for name, other in bases.items() if name != feature)) if len(bases) > 1 else set()
            shares = bool(roots & others)
            sharing += int(shares)
            detail_rows.append({**key, "feature": feature, "direct_bases": "|".join(sorted(direct[feature])), "root_bases": "|".join(sorted(roots)), "shares_root_base": shares})
        distinct = len(set().union(*bases.values())) if bases else 0
        row["n_features"], row["n_sharing_base_column"], row["n_distinct_base_columns"] = fmt(len(raw)), fmt(sharing), fmt(distinct)
        source = "recomputed:source_column_or_formula lineage in data/lendingclub_v2/metadata/feature_inventory.csv resolved recursively to raw source columns"
        for field_name, value in (("n_features", len(raw)), ("n_sharing_base_column", sharing), ("n_distinct_base_columns", distinct)):
            manifest.cell(step, key, field_name, value, source)
    if detail_rows:
        import pandas as pd

        pd.DataFrame(detail_rows).to_csv(out_dir / "12_overlap_features.csv", index=False, lineterminator="\n")
    return skeleton


# ---------------------------------------------------------------- 06 leakage


def fill_06(skeleton: Skeleton, engine: SelectionEngine, manifest: Manifest, headline_rows: list[dict[str, str]]) -> Skeleton:
    step = "06"
    from scripts.generate_homecredit_domain_rule_ranking import leakage_status

    try:
        universe = list(engine.data.dev(HOMECREDIT).candidate_features)
    except Exception as exc:  # noqa: BLE001
        manifest.skip(step, {"dataset": HOMECREDIT}, f"could not load the Home Credit candidate universe: {type(exc).__name__}: {exc}")
        return skeleton
    flagged = [name for name in universe if leakage_status(name) != "LEAKAGE_SAFE_LABEL_FREE"]
    candidate_sets = {item.partition: set(item.candidate_features) for item in llm_cache.canonical_files(HOMECREDIT)}
    union_candidates = set().union(*candidate_sets.values()) if candidate_sets else set()
    headline: set[str] = set()
    for row in headline_rows:
        if row["dataset"] != HOMECREDIT:
            continue
        record = full_dev_selection(engine, HOMECREDIT, row["backbone"], row["selector"])
        if record:
            headline |= set(_raw_bases(record["features"], universe))
    skeleton.rows = []
    for name in flagged:
        skeleton.rows.append(
            {
                "feature": name,
                "in_candidate_set_373": fmt(name in union_candidates),
                "in_headline_subset": fmt(name in headline),
            }
        )
        manifest.cell(step, {"feature": name}, "leakage_status", leakage_status(name), "recomputed:scripts/generate_homecredit_domain_rule_ranking.leakage_status over the 529 candidate universe")
    sizes = ", ".join(f"{partition}: {len(names)}" for partition, names in sorted(candidate_sets.items()))
    manifest.note(
        step,
        f"domain-rule name screen flagged {len(flagged)} of {len(universe)} Home Credit candidates "
        f"(markers TARGET/LABEL/BAD_RATE/OUTCOME/FUTURE); candidate-set membership uses the union of the cached "
        f"fold-local LLM candidate lists ({sizes}); the manuscript's 373 is not reproducible from any file on disk (TODO item 22a)",
    )
    return skeleton


# ---------------------------------------------------------------- 17 mRMR time


def fill_17(skeleton: Skeleton, engine: SelectionEngine, manifest: Manifest, third_ranking: list[str] | None) -> Skeleton:
    step = "17"
    import numpy as np

    from credit_risk_fs.selectors.lightweight.mi_mrmr import MutualInformationMRMRSelector
    from scripts.todo_fill.data import encode_for_selection
    from scripts.todo_fill.selectors import baseline_settings

    rows_by_pool = {int(row["pool_size"]): row for row in skeleton.rows}
    fold = skeleton.rows[0]["fold"] if skeleton.rows else "fold1"
    try:
        third = engine.data.third()
        X, y = third.training_frame(fold)
    except Exception as exc:  # noqa: BLE001
        for row in skeleton.rows:
            manifest.skip(step, _row_key(row, "dataset", "fold", "pool_size"), f"third dataset {fold} training slice unavailable: {type(exc).__name__}: {exc}")
        return skeleton
    predictors = list(X.columns)
    numeric = encode_for_selection(X, release_source=True)
    del X
    settings = baseline_settings(THIRD, "mrmr_mutual_information")
    if third_ranking:
        pool100 = [feature for feature in third_ranking if feature in numeric.columns][:100]
        pool_source = "top-100 of the regenerated target-free LLM ranking"
    else:
        rng = np.random.default_rng(42)
        pool100 = [predictors[i] for i in sorted(rng.permutation(len(predictors))[:100])]
        pool_source = "seeded random 100-feature pool (numpy default_rng(42)); no third-dataset LLM ranking is on this machine"
    reference = frozen.third_pilot_fit_seconds()
    for pool_size, columns in ((len(predictors), predictors), (100, pool100)):
        row = rows_by_pool.get(pool_size)
        if row is None:
            continue
        timings: dict[int, float] = {}
        for k in (40, 20):
            selector = MutualInformationMRMRSelector(k=k, n_bins=int(settings.get("n_bins", 10)), objective=str(settings.get("objective", "mid")), random_state=42, fit_scope="dev_fold_training_only")
            log(f"[17] timing MI-mRMR k={k} on {pool_size} features x {len(numeric)} rows")
            import time as _time

            started = _time.perf_counter()
            selector.fit(numeric.loc[:, columns], y)
            timings[k] = _time.perf_counter() - started
        row["seconds"] = fmt(timings[40], 2)
        manifest.cell(
            step,
            _row_key(row, "dataset", "fold", "pool_size"),
            "seconds",
            timings[40],
            "recomputed:MutualInformationMRMRSelector(n_bins=10, objective=mid, seed 42) on the fold-1 DEV training slice, k=40",
            k20_seconds=timings[20],
            pool_source=pool_source if pool_size == 100 else "all 1959 frozen predictors",
            frozen_machine_reference_seconds={"k20": reference.get(("mrmr_mutual_information", 20)), "k40": reference.get(("mrmr_mutual_information", 40))} if pool_size != 100 else None,
        )
    return skeleton
