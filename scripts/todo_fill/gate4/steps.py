"""Builders for the nine Gate-4 CSVs.  Each ``fill_XX`` resolves its own inputs, so any subset of
steps can run on its own (``--steps 40,45``)."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.todo_fill import llm_cache
from scripts.todo_fill.common import (
    BUDGETS,
    FOLDS,
    FULL_DEV,
    HOMECREDIT,
    LENDINGCLUB,
    PARTITIONS,
    REPO_ROOT,
    THIRD,
    FillError,
    Manifest,
    Skeleton,
    fmt,
    log,
    nogueira,
    pairwise_jaccard_summary,
    read_json,
    write_json,
)
from scripts.todo_fill.data import DataContext
from scripts.todo_fill.gate4 import config
from scripts.todo_fill.gate4.fits import FitExecutor, FitSpec, ho_identity, ho_scores, load_training, raw_bases
from scripts.todo_fill.gate4.inference import paired_auc_inference
from scripts.todo_fill.gate4.llm import Ranker, prompt_sha, scrubbed_prompt_sha
from scripts.todo_fill.gate4.records import RecordFactory
from scripts.todo_fill.gate4.subsets import SubsetResolver, apply_family_cap
from scripts.todo_fill.selections import Request, SelectionEngine

LC_IDENTITY = REPO_ROOT / "data/lendingclub_v2/processed/record_identity_v1.csv"
PIPE = "|"


@dataclass
class Context:
    data: DataContext
    engine: SelectionEngine
    executor: FitExecutor
    ranker: Ranker
    records: RecordFactory
    subsets: SubsetResolver
    manifest: Manifest
    work_dir: Path
    out_dir: Path
    options: dict[str, Any] = field(default_factory=dict)
    sidecar_dir: Path | None = None

    def sidecar(self, name: str) -> Path:
        folder = Path(self.sidecar_dir) if self.sidecar_dir is not None else self.out_dir
        folder.mkdir(parents=True, exist_ok=True)
        return folder / name


# ------------------------------------------------------------------ small helpers


def _key(row: dict[str, str], *columns: str) -> dict[str, str]:
    return {column: row.get(column, "") for column in columns}


def _ensure_holdout(ctx: Context, specs: Iterable[FitSpec]) -> None:
    """Materialise the locked HO frame for every dataset a full-DEV spec touches (one load per dataset)."""

    needed: dict[str, set[str]] = {}
    for spec in specs:
        if spec.partition == FULL_DEV and spec.dataset != THIRD:
            needed.setdefault(spec.dataset, set()).update(raw_bases(spec.features, ctx.records.universe(spec.dataset)))
    for dataset, features in needed.items():
        ctx.data.holdout(dataset, [name for name in ctx.records.universe(dataset) if name in features])


def _run_fits(ctx: Context, step: str, specs: Sequence[FitSpec]) -> dict[str, dict[str, Any]]:
    specs = [spec for spec in specs if spec is not None]
    if not specs:
        return {}
    _ensure_holdout(ctx, specs)
    return ctx.executor.run(specs, step)


def _ok(results: dict[str, dict[str, Any]], spec: FitSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    payload = results.get(spec.fit_id)
    return payload if payload and payload.get("status") == "ok" else None


def _fold_specs(dataset: str, backbone: str, rankings: dict[str, Sequence[str]], k: int, label: str, variant: str = "onehot") -> dict[str, FitSpec]:
    """One spec per partition from partition-specific rankings (fold-local top-K, full-DEV top-K)."""

    specs: dict[str, FitSpec] = {}
    for partition, ranking in rankings.items():
        prefix = tuple(dict.fromkeys(str(item) for item in ranking))[:k]
        if len(prefix) < k:
            continue
        specs[partition] = FitSpec(dataset, backbone, partition, prefix, variant=variant, label=f"{label}:{partition}")
    return specs


def _fill_folds(row: dict[str, str], specs: dict[str, FitSpec], results: dict[str, dict[str, Any]], manifest: Manifest, step: str, key: dict[str, str], source: str) -> list[float]:
    """Write ``foldN_auc`` where the schema has those columns; return the AUCs either way."""

    values: list[float] = []
    for fold in FOLDS:
        payload = _ok(results, specs.get(fold))
        if payload is None:
            continue
        values.append(float(payload["auc"]))
        if f"{fold}_auc" in row:
            row[f"{fold}_auc"] = fmt(payload["auc"])
        manifest.cell(step, key, f"{fold}_auc", payload["auc"], source, fit_id=payload["fit_id"])
    return values


def _write_range(row: dict[str, str], values: Sequence[float], low: str, high: str, manifest: Manifest, step: str, key: dict[str, str], source: str) -> None:
    """Fill a ``*_min`` / ``*_max`` pair when the schema asks for the range rather than each fold."""

    if not values:
        return
    for column, value in ((low, min(values)), (high, max(values))):
        if column in row:
            row[column] = fmt(value)
            manifest.cell(step, key, column, value, source, n_folds=len(values))


def _inference(ctx: Context, dataset: str, spec_a: FitSpec, spec_b: FitSpec, mask: np.ndarray | None = None) -> dict[str, Any]:
    identity = ho_identity(ctx.work_dir, dataset)
    target = identity["target"].to_numpy(dtype=int)
    a = ho_scores(ctx.work_dir, spec_a)
    b = ho_scores(ctx.work_dir, spec_b)
    if mask is not None:
        target, a, b = target[mask], a[mask], b[mask]
    return paired_auc_inference(target, a, b)


def _write_inference(row: dict[str, str], result: dict[str, Any], manifest: Manifest, step: str, key: dict[str, str], source: str, delta_column: str) -> None:
    for column, value in (("auc_A", result["auc_A"]), ("auc_B", result["auc_B"]), (delta_column, result["delta"]), ("ci95_low", result["ci95_low"]), ("ci95_high", result["ci95_high"]), ("n_ho", result["n_ho"])):
        if column in row:
            row[column] = fmt(value, 8) if column != "n_ho" else str(int(value))
            manifest.cell(step, key, column, value, source, p_value=result["p_value"])


def _pipe(names: Iterable[str]) -> str:
    return PIPE.join(str(name) for name in names)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


# =========================================================================== 38


def fill_38(skeleton: Skeleton, ctx: Context) -> None:
    step = "38"
    for row in skeleton.rows:
        key = _key(row, "rule", "dataset", "backbone", "method_A", "method_B")
        dataset, backbone = row["dataset"], row["backbone"]
        selections = {}
        for side in ("method_A", "method_B"):
            selections[side] = ctx.subsets.headline(dataset, backbone, row[side], step)
        if any(value is None for value in selections.values()):
            ctx.manifest.skip(step, key, "a full-DEV selection is unavailable; no refit possible")
            continue
        specs = {side: FitSpec(dataset, backbone, FULL_DEV, tuple(selections[side]["features"]), label=f"38:{row[side]}") for side in selections}
        results = _run_fits(ctx, step, list(specs.values()))
        if any(_ok(results, spec) is None for spec in specs.values()):
            ctx.manifest.skip(step, key, "a full-DEV refit failed (see fit skips)")
            continue
        result = _inference(ctx, dataset, specs["method_A"], specs["method_B"])
        source = "recomputed:frozen full-DEV refits of both selections scored on the locked HO population; paired stratified bootstrap (2,000 draws, seed 20260721)"
        _write_inference(row, result, ctx.manifest, step, key, source, "delta_auc_A_minus_B")
        for side, expected in (("method_A", config.STEP38_EXPECTED["auc_A"]), ("method_B", config.STEP38_EXPECTED["auc_B"])):
            observed = result["auc_A" if side == "method_A" else "auc_B"]
            ctx.manifest.note(step, f"{row[side]}: refit HO AUC {observed:.6f} vs full_matrix.csv {expected} (difference {observed - expected:+.6f}); the frozen prediction files are not on this machine, so the join check is against a refit under the frozen pipeline, not the paper's prediction vector")


# =========================================================================== 37


def _lc_filters(ctx: Context) -> dict[str, np.ndarray]:
    identity = ho_identity(ctx.work_dir, LENDINGCLUB)
    sidecar = pd.read_csv(LC_IDENTITY, usecols=["loan_id", "split", "issue_month"], dtype={"loan_id": "string"})
    sidecar = sidecar.loc[sidecar["split"].str.upper().eq("OOT")]
    merged = identity.merge(sidecar, left_on="stable_row_id", right_on="loan_id", how="left", validate="one_to_one")
    if merged["issue_month"].isna().any():
        raise FillError("LendingClub HO rows without an issue month in the identity sidecar")
    term_frame, _ = ctx.data.holdout(LENDINGCLUB, ["term_months"])
    if len(term_frame) != len(identity):
        raise FillError("LendingClub HO term column does not align with the scored identity")
    issue = pd.to_datetime(merged["issue_month"]).dt.strftime("%Y-%m").to_numpy()
    term = pd.to_numeric(term_frame["term_months"], errors="coerce").to_numpy()
    masks: dict[str, np.ndarray] = {}
    masks["matured_36m_2016-01_to_2016-05"] = (term == 36) & (issue >= "2016-01") & (issue <= "2016-05")
    for month in sorted(set(issue)):
        masks[f"month_{month}"] = issue == month
    masks["term_36"] = term == 36
    masks["term_60"] = term == 60
    return masks


def fill_37(skeleton: Skeleton, ctx: Context) -> None:
    step = "37"
    needed = sorted({(row["backbone"], row[side]) for row in skeleton.rows for side in ("method_A", "method_B")})
    selections: dict[tuple[str, str], dict[str, Any] | None] = {}
    for backbone, label in needed:
        selections[(backbone, label)] = ctx.subsets.headline(LENDINGCLUB, backbone, label, step)
    specs = {pair: FitSpec(LENDINGCLUB, pair[0], FULL_DEV, tuple(sel["features"]), label=f"37:{pair[1]}") for pair, sel in selections.items() if sel is not None}
    results = _run_fits(ctx, step, list(specs.values()))
    for pair, sel in selections.items():
        if sel is not None:
            payload = _ok(results, specs[pair])
            if payload is not None:
                ctx.manifest.note(step, f"LendingClub {pair[0]} {pair[1]}: full-DEV refit HO AUC {payload['ho_auc']:.6f} on {payload['n_ho']} rows ({len(sel['features'])} selected features; {sel['source']})")
    try:
        masks = _lc_filters(ctx)
    except Exception as exc:  # noqa: BLE001
        for row in skeleton.rows:
            ctx.manifest.skip(step, _key(row, "subset", "rule", "backbone"), f"row filters unavailable: {type(exc).__name__}: {exc}")
        return
    filter_rows = []
    identity = ho_identity(ctx.work_dir, LENDINGCLUB)
    for name, mask in masks.items():
        filter_rows.append({"subset": name, "n_ho": int(mask.sum()), "n_positive": int(identity["target"].to_numpy()[mask].sum())})
    _write_csv(ctx.sidecar("37_lc_maturity_filters.csv"), filter_rows, ["subset", "n_ho", "n_positive"])
    if masks["matured_36m_2016-01_to_2016-05"].sum() != 123_898:
        ctx.manifest.note(step, f"matured_36m_2016-01_to_2016-05 has {int(masks['matured_36m_2016-01_to_2016-05'].sum())} rows; TODO.md states 123,898")
    cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in skeleton.rows:
        key = _key(row, "subset", "rule", "backbone", "method_A", "method_B")
        mask = masks.get(row["subset"])
        if mask is None:
            ctx.manifest.skip(step, key, "unknown row filter")
            continue
        spec_a = specs.get((row["backbone"], row["method_A"]))
        spec_b = specs.get((row["backbone"], row["method_B"]))
        if _ok(results, spec_a) is None or _ok(results, spec_b) is None:
            ctx.manifest.skip(step, key, "a full-DEV refit or selection is unavailable")
            continue
        cache_key = (row["subset"], row["backbone"], row["method_A"], row["method_B"])
        if cache_key not in cache:
            cache[cache_key] = _inference(ctx, LENDINGCLUB, spec_a, spec_b, mask)
        source = f"recomputed:row filter on refit HO prediction vectors ({row['subset']}); paired stratified bootstrap (2,000 draws, seed 20260721)"
        _write_inference(row, cache[cache_key], ctx.manifest, step, key, source, "delta_auc_A_minus_B")


# =========================================================================== 39a


def _onehot_dimension(ctx: Context, dataset: str, features: Sequence[str]) -> dict[str, Any]:
    from credit_risk_fs.preprocessing.encoding import Preprocessor
    from scripts.todo_fill.common import PREPROCESSOR_KWARGS

    universe = ctx.records.universe(dataset)
    raw = raw_bases(features, universe)
    dense_selection = any(name not in set(universe) for name in features)
    X, _, _, _ = load_training(ctx.work_dir, dataset, FULL_DEV, raw)
    categorical = [name for name in raw if not pd.api.types.is_numeric_dtype(X[name])]
    transformed = Preprocessor(**PREPROCESSOR_KWARGS.get(dataset, {})).fit_transform(X)
    if dense_selection:
        received = [name for name in features if name in transformed.columns]
        columns = len(received)
    else:
        columns = int(transformed.shape[1])
    return {"n_original": len(raw), "n_categorical": len(categorical), "categorical": categorical, "columns_after_onehot": columns, "dense_selection": dense_selection, "expanded_columns_of_raw": int(transformed.shape[1])}


def fill_39a(skeleton: Skeleton, ctx: Context) -> None:
    step = "39a"
    detail_rows = []
    for row in skeleton.rows:
        key = _key(row, "dataset", "backbone", "selector", "K")
        selection = ctx.subsets.headline(row["dataset"], row["backbone"], row["selector"], step)
        if selection is None:
            continue
        try:
            dims = _onehot_dimension(ctx, row["dataset"], selection["features"])
        except Exception as exc:  # noqa: BLE001
            ctx.manifest.skip(step, key, f"could not build the full-DEV preprocessing map: {type(exc).__name__}: {exc}")
            continue
        row["n_categorical_in_subset"] = str(dims["n_categorical"])
        row["columns_after_onehot"] = str(dims["columns_after_onehot"])
        source = f"recomputed:Preprocessor(mean/standard, Missing token, one-hot min_frequency={'50' if row['dataset'] == LENDINGCLUB else '10'}) fitted on full DEV over the {selection['source']} subset"
        extra = {"n_original_features": dims["n_original"], "categorical_features": dims["categorical"], "dense_model_matrix_selection": dims["dense_selection"], "expanded_columns_of_all_raw_bases": dims["expanded_columns_of_raw"]}
        ctx.manifest.cell(step, key, "n_categorical_in_subset", dims["n_categorical"], source, **extra)
        ctx.manifest.cell(step, key, "columns_after_onehot", dims["columns_after_onehot"], source, **extra)
        if dims["dense_selection"]:
            ctx.manifest.note(step, f"{key}: the frozen selection names {len(selection['features'])} model-matrix columns of {dims['n_original']} original features; columns_after_onehot counts the columns the model actually received")
        if dims["n_original"] != int(row["K"]) and not dims["dense_selection"]:
            ctx.manifest.note(step, f"{key}: the selection holds {dims['n_original']} original features (natural support), not K={row['K']}")
        detail_rows.append({**key, **{k: v for k, v in dims.items() if k != "categorical"}, "categorical_features": _pipe(dims["categorical"]), "features": _pipe(selection["features"])})
    if detail_rows:
        _write_csv(ctx.sidecar("39a_onehot_dimension_details.csv"), detail_rows, list(detail_rows[0].keys()))


# =========================================================================== 39b


def fill_39b(skeleton: Skeleton, ctx: Context) -> None:
    step = "39b"
    plans: list[tuple[dict[str, str], dict[str, FitSpec], dict[str, Any]]] = []
    for row in skeleton.rows:
        key = _key(row, "dataset", "backbone", "selector", "K")
        if row["backbone"] != "catboost":
            ctx.manifest.skip(step, key, "native categorical handling is a CatBoost-only variant")
            continue
        selection = ctx.subsets.headline(row["dataset"], row["backbone"], row["selector"], step)
        if selection is None:
            continue
        features = tuple(selection["features"])
        specs = {partition: FitSpec(row["dataset"], "catboost", partition, features, variant="native_cat", label=f"39b:{row['selector']}:{partition}") for partition in PARTITIONS}
        plans.append((row, specs, selection))
    results = _run_fits(ctx, step, [spec for _, specs, _ in plans for spec in specs.values()])
    for row, specs, selection in plans:
        key = _key(row, "dataset", "backbone", "selector", "K")
        source = f"recomputed:CatBoost with cat_features on the raw categoricals of the {selection['source']} subset (numeric block preprocessed exactly as the one-hot pipeline; frozen configuration, seed 42)"
        full = _ok(results, specs[FULL_DEV])
        if full is not None:
            row["ho_auc_native"] = fmt(full["ho_auc"])
            ctx.manifest.cell(step, key, "ho_auc_native", full["ho_auc"], source, native_cat_features=full.get("native_cat_features"), n_model_columns=full.get("n_model_columns"), fit_id=full["fit_id"])
            if row.get("ho_auc_onehot"):
                ctx.manifest.note(step, f"{key}: native {full['ho_auc']:.6f} vs one-hot (paper) {row['ho_auc_onehot']}; {len(full.get('native_cat_features') or [])} categorical columns handed to CatBoost natively")
        _fill_folds(row, specs, results, ctx.manifest, step, key, source)


# =========================================================================== 40


def fill_40(skeleton: Skeleton, ctx: Context) -> None:
    step = "40"
    plans: list[tuple[dict[str, str], FitSpec, Any]] = []
    for row in skeleton.rows:
        key = _key(row, "dataset", "partition", "backbone", "K")
        k = int(row["K"])
        try:
            cached = llm_cache.ranking(row["dataset"], row["partition"], budget=k)
        except KeyError as exc:
            ctx.manifest.skip(step, key, str(exc))
            continue
        prefix = tuple(cached.features[:k])
        if len(prefix) < k:
            ctx.manifest.skip(step, key, f"cached {row['partition']} ranking holds only {len(prefix)} names")
            continue
        plans.append((row, FitSpec(row["dataset"], row["backbone"], FULL_DEV, prefix, label=f"40:{row['dataset']}:{row['partition']}"), cached))
    results = _run_fits(ctx, step, [spec for _, spec, _ in plans])
    for row, spec, cached in plans:
        key = _key(row, "dataset", "partition", "backbone", "K")
        payload = _ok(results, spec)
        if payload is None:
            continue
        row["ho_auc"] = fmt(payload["ho_auc"])
        ctx.manifest.cell(step, key, "ho_auc", payload["ho_auc"], f"recomputed:top-{row['K']} of the cached {row['partition']} LLM ranking refit on full DEV under the frozen pipeline and scored on HO", cache_file=cached.relative_path, cache_sha256=cached.sha256, fit_id=payload["fit_id"])
    by_case: dict[tuple[str, str], list[float]] = {}
    for row in skeleton.rows:
        if row.get("ho_auc"):
            by_case.setdefault((row["dataset"], row["backbone"]), []).append(float(row["ho_auc"]))
    for (dataset, backbone), values in by_case.items():
        ctx.manifest.note(step, f"{dataset}/{backbone}: five-draw HO AUC spread min {min(values):.6f} max {max(values):.6f} range {max(values) - min(values):.6f} (0.010 bar)")


# =========================================================================== 43


def fill_43(skeleton: Skeleton, ctx: Context) -> None:
    step = "43"
    plans: list[tuple[dict[str, str], dict[str, FitSpec], dict[str, Any]]] = []
    detail_rows: list[dict[str, Any]] = []
    for row in skeleton.rows:
        key = _key(row, "dataset", "backbone", "cap_rule", "K")
        dataset, backbone, k = row["dataset"], row["backbone"], int(row["K"])
        label = config.HO_RULE_CLASSICAL_LEADER[(dataset, backbone)]
        row["base_selector"] = label
        ctx.manifest.cell(step, key, "base_selector", label, "TODO.md:Table 4 HO-rule classical leader of the case")
        families = ctx.records.families(dataset)
        if row["cap_rule"] == "family_cap":
            n_families = ctx.records.n_families(dataset)
            cap = config.family_cap(k, n_families)
            columns = None
        elif row["cap_rule"] == "depth0_only":
            if dataset != THIRD:
                ctx.manifest.skip(step, key, "depth0_only is defined for the third dataset only")
                continue
            columns = ctx.records.third_depth0_universe()
            n_families, cap = len(config.THIRD_DEPTH0_FAMILIES), None
        else:
            ctx.manifest.skip(step, key, f"unknown cap rule {row['cap_rule']}")
            continue
        specs: dict[str, FitSpec] = {}
        for partition in PARTITIONS:
            ranking = ctx.subsets.base_ranking(dataset=dataset, backbone=backbone, label=label, partition=partition, columns=columns, step=step)
            if ranking is None:
                continue
            if cap is not None:
                capped = apply_family_cap(ranking["ranking"], families, k, cap)
            else:
                chosen = list(ranking["ranking"])[:k]
                capped = {"features": chosen, "family_counts": {fam: sum(families[name] == fam for name in chosen) for fam in set(families[name] for name in chosen)}, "cap": None, "k": k, "shortfall": k - len(chosen), "skipped_by_cap": []}
            if capped["shortfall"] > 0:
                ctx.manifest.note(step, f"{key} {partition}: cap {cap} per family over {n_families} families yields only {len(capped['features'])} of K={k} features")
            if not capped["features"]:
                continue
            specs[partition] = FitSpec(dataset, backbone, partition, tuple(capped["features"]), label=f"43:{row['cap_rule']}:{label}:{partition}")
            detail_rows.append({**key, "base_selector": label, "partition": partition, "n_families": n_families, "cap_per_family": cap if cap is not None else "", "n_selected": len(capped["features"]), "family_counts": json.dumps(capped["family_counts"], sort_keys=True), "features": _pipe(capped["features"]), "ranking_protocol": ranking.get("protocol", "")})
        plans.append((row, specs, {"cap": cap, "n_families": n_families, "label": label}))
    results = _run_fits(ctx, step, [spec for _, specs, _ in plans for spec in specs.values()])
    for row, specs, info in plans:
        key = _key(row, "dataset", "backbone", "cap_rule", "K")
        source = (
            f"recomputed:{info['label']} ranking with at most {info['cap']} features per source family ({info['n_families']} families), frozen backbone refit"
            if info["cap"] is not None
            else f"recomputed:{info['label']} restricted to the depth-0 candidates (static_0, static_cb_0), frozen backbone refit"
        )
        full = _ok(results, specs.get(FULL_DEV))
        if full is not None:
            row["ho_auc"] = fmt(full["ho_auc"])
            ctx.manifest.cell(step, key, "ho_auc", full["ho_auc"], source, fit_id=full["fit_id"])
        _fill_folds(row, specs, results, ctx.manifest, step, key, source)
    if detail_rows:
        _write_csv(ctx.sidecar("43_diversity_capped_subsets.csv"), detail_rows, list(detail_rows[0].keys()))


# =========================================================================== shared LLM-arm helpers


def _named_records(ctx: Context, dataset: str, partition: str, budget: int | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    names = ctx.records.candidate_set(dataset, partition, budget)
    return names, ctx.records.named_records(dataset, names)


def _call_name(dataset: str, partition: str, condition: str, budget: int | None = None, repeat: int | None = None) -> str:
    parts = [dataset, partition]
    if budget is not None:
        parts.append(f"b{budget}")
    if repeat is not None:
        parts.append(f"repeat{repeat:02d}")
    parts.append(condition)
    return "__".join(parts)


def _rankings_csv(path: Path, calls: list[dict[str, Any]], with_ids: bool = False) -> None:
    rows = []
    for call in calls:
        for rank, (feature, feature_id) in enumerate(zip(call["ranking"], call.get("ranking_ids", call["ranking"])), start=1):
            rows.append({"dataset": call["dataset"], "partition": call["partition"], "budget": call.get("budget", ""), "condition": call["condition"], "repeat_id": call.get("repeat_id", ""), "rank": rank, "feature_id": feature_id if with_ids else feature, "original_feature_name": feature, "response_id": call.get("response_id", "")})
    _write_csv(path, rows, ["dataset", "partition", "budget", "condition", "repeat_id", "rank", "feature_id", "original_feature_name", "response_id"])


def _calls_json(path: Path, calls: list[dict[str, Any]]) -> None:
    write_json(path, [{key: value for key, value in call.items() if key not in {"attempts", "ranking", "ranking_ids"}} | {"ranking": call["ranking"]} for call in calls])


def _cached_lc_ranking(partition: str, budget: int) -> list[str]:
    return list(llm_cache.ranking(LENDINGCLUB, partition, budget=budget).features)


# =========================================================================== 44


def _llm_then_mrmr_override(ctx: Context, partition: str, pool: Sequence[str], tag: str) -> dict[str, Any] | None:
    """LendingClub CatBoost ``LLM then mRMR`` with a replacement LLM pool (legacy protocol)."""

    from scripts.todo_fill import selectors as recipes

    path = ctx.work_dir / "gate4" / "selections_override" / f"lendingclub_v2__catboost__llm_then_mrmr__{tag}__{partition}.json"
    if path.exists():
        return read_json(path)
    fit_ctx = recipes.FitContext(dataset=LENDINGCLUB, backbone="catboost", partition=partition, data=ctx.data, ranking_override=list(pool))
    log(f"[44] LLM then mRMR ({tag}) on lendingclub_v2/catboost/{partition} with a {len(pool)}-name pool")
    try:
        output = recipes.fit_llm_then_mrmr(fit_ctx)
    except Exception as exc:  # noqa: BLE001
        ctx.manifest.skip("44", {"dataset": LENDINGCLUB, "backbone": "catboost", "selector": "LLM then mRMR", "partition": partition}, f"{type(exc).__name__}: {exc}")
        return None
    finally:
        fit_ctx.release()
    payload = {"features": list(output.features), "protocol": output.protocol, "details": output.details, "fit_seconds": output.fit_seconds}
    write_json(path, payload)
    return payload


def _cached_top100(ctx: Context, dataset: str, k: int, third_named: dict[str, Any] | None) -> list[str]:
    """The cached-description full-DEV ranking this cell is compared against, first 100 names."""

    if dataset == THIRD:
        return list(third_named["ranking"][:100]) if third_named else []
    try:
        return list(llm_cache.ranking(dataset, FULL_DEV, budget=k).features[:100])
    except Exception as exc:  # noqa: BLE001
        ctx.manifest.skip("44", {"dataset": dataset, "K": k}, f"cached full-DEV ranking unavailable for the top-100 overlap: {type(exc).__name__}: {exc}")
        return []


def fill_44(skeleton: Skeleton, ctx: Context) -> None:
    step = "44"
    calls: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    descriptions: dict[str, list[dict[str, Any]]] = {}
    datasets = sorted({row["dataset"] for row in skeleton.rows})

    def call(dataset: str, partition: str, budget: int | None) -> dict[str, Any] | None:
        key = (dataset, partition, budget)
        if key in calls:
            return calls[key]
        names = ctx.records.candidate_set(dataset, partition, budget)
        mechanical = ctx.records.mechanical_records(dataset, names)
        if partition == FULL_DEV and dataset not in descriptions:
            descriptions[dataset] = mechanical
        result = ctx.ranker.rank(step=step, name=_call_name(dataset, partition, "mechanical", budget), dataset=dataset, partition=partition, condition="named", records=mechanical, ids=names, universe=ctx.records.universe(dataset), extra={"budget": budget, "template": "mechanical"})
        if result is not None:
            calls[key] = result
        return result

    # ---- calls follow the cached run (HC one per partition, LC one per partition x budget),
    # narrowed to what this skeleton actually consumes: no fold columns means no fold call,
    # and only the budgets its rows name.
    wanted = PARTITIONS if any(f"{fold}_auc" in row for fold in FOLDS for row in skeleton.rows) else (FULL_DEV,)
    lc_budgets = sorted({100 if row["selector"] == "LLM then mRMR" else int(row["K"]) for row in skeleton.rows if row["dataset"] == LENDINGCLUB})
    if wanted != PARTITIONS or (LENDINGCLUB in datasets and set(lc_budgets) != set(config.LENDINGCLUB_CACHE_BUDGETS)):
        ctx.manifest.note(step, f"call plan narrowed to the skeleton: partitions={list(wanted)}, LendingClub budgets={lc_budgets}")
    if HOMECREDIT in datasets:
        for partition in wanted:
            call(HOMECREDIT, partition, None)
    if LENDINGCLUB in datasets:
        for partition in wanted:
            for budget in lc_budgets:
                call(LENDINGCLUB, partition, budget)
    third_named = None
    if THIRD in datasets:
        call(THIRD, FULL_DEV, None)
        third_named = ctx.ranker.third_named(ctx.records)
    for dataset, records in descriptions.items():
        write_json(ctx.sidecar(f"44_mechanical_descriptions_{dataset}.json"), records)
    if calls:
        _rankings_csv(ctx.sidecar("44_mechanical_rankings.csv"), list(calls.values()))
        _calls_json(ctx.sidecar("44_mechanical_calls.json"), list(calls.values()))

    # ---- per row: mechanical partitions + cached reference
    plans: list[dict[str, Any]] = []
    for row in skeleton.rows:
        key = _key(row, "dataset", "backbone", "selector", "K")
        dataset, backbone, label, k = row["dataset"], row["backbone"], row["selector"], int(row["K"])
        budget = None if dataset != LENDINGCLUB else (100 if label == "LLM then mRMR" else k)
        partitions = wanted if dataset != THIRD else (FULL_DEV,)
        mech_rankings = {partition: calls[(dataset, partition, budget)]["ranking"] for partition in partitions if (dataset, partition, budget) in calls}
        if FULL_DEV not in mech_rankings:
            ctx.manifest.skip(step, key, "mechanical full-DEV ranking unavailable (call not made)")
            continue
        if label == "Pure LLM":
            mech_specs = _fold_specs(dataset, backbone, mech_rankings, k, f"44:mechanical:{dataset}")
            if dataset == THIRD:
                if third_named is None:
                    ctx.manifest.skip(step, key, "cached Stability 2024 ranking unavailable; no reference for the paired comparison")
                    continue
                cached_features = tuple(third_named["ranking"][:k])
                cached_source = "regenerated named ranking (paper's cached Stability 2024 ranking is not on this machine)"
            else:
                cached_features = tuple(llm_cache.ranking(dataset, FULL_DEV, budget=k).features[:k])
                cached_source = "artifacts/llm_cache full-DEV ranking truncation"
            cached_spec = FitSpec(dataset, backbone, FULL_DEV, cached_features, label=f"44:cached:{dataset}:{label}")
        else:  # LLM then mRMR on LendingClub CatBoost
            mech_specs = {}
            for partition, ranking in mech_rankings.items():
                selection = _llm_then_mrmr_override(ctx, partition, list(ranking)[:100], "mechanical")
                if selection is not None and selection["features"]:
                    mech_specs[partition] = FitSpec(dataset, backbone, partition, tuple(selection["features"]), label=f"44:mechanical:{dataset}:llm_then_mrmr:{partition}")
            reference = ctx.subsets.headline(dataset, backbone, label, step)
            if reference is None or FULL_DEV not in mech_specs:
                ctx.manifest.skip(step, key, "cached or mechanical LLM then mRMR selection unavailable")
                continue
            cached_spec = FitSpec(dataset, backbone, FULL_DEV, tuple(reference["features"]), label=f"44:cached:{dataset}:{label}")
            cached_source = reference["source"]
        plans.append({"row": row, "key": key, "mech_specs": mech_specs, "cached_spec": cached_spec, "cached_source": cached_source, "top100": list(mech_rankings[FULL_DEV])[:100], "cached_top100": list(_cached_top100(ctx, dataset, budget or k, third_named))})
    results = _run_fits(ctx, step, [spec for plan in plans for spec in list(plan["mech_specs"].values()) + [plan["cached_spec"]]])
    for plan in plans:
        row, key = plan["row"], plan["key"]
        if "fulldev_top100_names_pipe_separated" in row:
            row["fulldev_top100_names_pipe_separated"] = _pipe(plan["top100"])
        ctx.manifest.cell(step, key, "fulldev_top100_names_pipe_separated", plan["top100"], "recomputed:mechanical-description full-DEV ranking (100 names)")
        if "overlap_top100_with_cached" in row and plan["cached_top100"]:
            overlap = len(set(plan["top100"]) & set(plan["cached_top100"]))
            row["overlap_top100_with_cached"] = str(overlap)
            ctx.manifest.cell(step, key, "overlap_top100_with_cached", overlap, "recomputed:names shared between the mechanical and cached full-DEV top-100 lists", n_mechanical=len(plan["top100"]), n_cached=len(plan["cached_top100"]))
        full = _ok(results, plan["mech_specs"].get(FULL_DEV))
        cached = _ok(results, plan["cached_spec"])
        source = "recomputed:mechanical-description arm refit on full DEV, paired against the cached-description refit on the same HO rows (2,000 stratified draws, seed 20260721)"
        if full is not None:
            row["ho_auc_mechanical"] = fmt(full["ho_auc"])
            ctx.manifest.cell(step, key, "ho_auc_mechanical", full["ho_auc"], source, fit_id=full["fit_id"])
        if full is not None and cached is not None:
            result = _inference(ctx, row["dataset"], plan["mech_specs"][FULL_DEV], plan["cached_spec"])
            for column, value in (("delta_mechanical_minus_cached", result["delta"]), ("ci95_low", result["ci95_low"]), ("ci95_high", result["ci95_high"])):
                row[column] = fmt(value, 8)
                ctx.manifest.cell(step, key, column, value, source, cached_reference=plan["cached_source"], cached_refit_ho_auc=cached["ho_auc"], p_value=result["p_value"])
            row["n_ho"] = str(result["n_ho"])
            ctx.manifest.cell(step, key, "n_ho", result["n_ho"], source)
            if row.get("ho_auc_cached"):
                ctx.manifest.note(step, f"{key}: cached-description refit HO AUC {cached['ho_auc']:.6f} vs prefilled ho_auc_cached {row['ho_auc_cached']} ({plan['cached_source']})")
        _fill_folds(row, plan["mech_specs"], results, ctx.manifest, step, key, "recomputed:fold-local mechanical ranking top-K, frozen backbone on fold training rows, validation AUC")


# =========================================================================== 45


def fill_45(skeleton: Skeleton, ctx: Context) -> None:
    step = "45"
    calls: dict[tuple[str, int | None, int], dict[str, Any]] = {}
    datasets = sorted({row["dataset"] for row in skeleton.rows})
    repeats = sorted({int(row["repeat_id"]) for row in skeleton.rows})
    for dataset in datasets:
        budgets: tuple[int | None, ...] = (None,) if dataset == HOMECREDIT else tuple(sorted({int(row["K"]) for row in skeleton.rows if row["dataset"] == dataset}))
        for budget in budgets:
            names, named = _named_records(ctx, dataset, FULL_DEV, budget)
            for repeat in repeats:
                result = ctx.ranker.rank(step=step, name=_call_name(dataset, FULL_DEV, "named", budget, repeat), dataset=dataset, partition=FULL_DEV, condition="named", records=named, ids=names, universe=ctx.records.universe(dataset), extra={"budget": budget, "repeat_id": repeat})
                if result is not None:
                    calls[(dataset, budget, repeat)] = result
    if calls:
        _rankings_csv(ctx.sidecar("45_repeated_rankings.csv"), list(calls.values()))
        _calls_json(ctx.sidecar("45_repeated_calls_log.json"), list(calls.values()))
    plans: list[tuple[dict[str, str], FitSpec, dict[str, Any]]] = []
    for row in skeleton.rows:
        key = _key(row, "dataset", "repeat_id", "backbone", "K")
        dataset, k, repeat = row["dataset"], int(row["K"]), int(row["repeat_id"])
        budget = None if dataset == HOMECREDIT else k
        result = calls.get((dataset, budget, repeat))
        if result is None:
            ctx.manifest.skip(step, key, "ranking call not made")
            continue
        row["response_id"] = str(result.get("response_id") or "")
        row["top100_names_pipe_separated"] = _pipe(result["ranking"][:100])
        ctx.manifest.cell(step, key, "response_id", row["response_id"], "recomputed:provider response id of the repeat call")
        ctx.manifest.cell(step, key, "top100_names_pipe_separated", result["ranking"][:100], "recomputed:repeat call ranking")
        prefix = tuple(result["ranking"][:k])
        plans.append((row, FitSpec(dataset, row["backbone"], FULL_DEV, prefix, label=f"45:{dataset}:repeat{repeat}:{row['backbone']}"), result))
    results = _run_fits(ctx, step, [spec for _, spec, _ in plans])
    for row, spec, _ in plans:
        key = _key(row, "dataset", "repeat_id", "backbone", "K")
        payload = _ok(results, spec)
        if payload is None:
            continue
        row["ho_auc"] = fmt(payload["ho_auc"])
        ctx.manifest.cell(step, key, "ho_auc", payload["ho_auc"], "recomputed:repeat ranking top-K refit on full DEV under the frozen pipeline, scored on HO", fit_id=payload["fit_id"])
    # repeat stability summary (Nogueira / Jaccard over the ten top-K sets) for the manifest and a sidecar
    summary_rows = []
    for dataset in datasets:
        for backbone, k in BUDGETS.items():
            budget = None if dataset == HOMECREDIT else k
            sets = [list(calls[(dataset, budget, repeat)]["ranking"][:k]) for repeat in repeats if (dataset, budget, repeat) in calls]
            aucs = [float(row["ho_auc"]) for row in skeleton.rows if row["dataset"] == dataset and row["backbone"] == backbone and row.get("ho_auc")]
            if len(sets) < 2:
                continue
            d = len(ctx.records.candidate_set(dataset, FULL_DEV, budget))
            jac = pairwise_jaccard_summary(sets)
            summary_rows.append({"dataset": dataset, "backbone": backbone, "K": k, "n_repeats": len(sets), "d_candidates": d, "nogueira": nogueira(sets, d), "jaccard_mean": jac["mean"], "jaccard_min": jac["min"], "jaccard_max": jac["max"], "ho_auc_median": float(np.median(aucs)) if aucs else "", "ho_auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else "", "ho_auc_min": min(aucs) if aucs else "", "ho_auc_max": max(aucs) if aucs else ""})
            ctx.manifest.note(step, f"{dataset}/{backbone}: {len(sets)} repeats, Nogueira {summary_rows[-1]['nogueira']}, mean Jaccard {jac['mean']}, HO AUC median {summary_rows[-1]['ho_auc_median']} sd {summary_rows[-1]['ho_auc_sd']}")
    if summary_rows:
        _write_csv(ctx.sidecar("45_repeat_stability.csv"), summary_rows, list(summary_rows[0].keys()))


# =========================================================================== 47


def fill_47(skeleton: Skeleton, ctx: Context) -> None:
    step = "47"
    calls: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
    datasets = sorted({row["dataset"] for row in skeleton.rows})
    arms = sorted({row["arm"] for row in skeleton.rows})
    prompt_hashes: dict[str, dict[str, str]] = {}

    def arm_call(dataset: str, partition: str, arm: str, budget: int | None) -> dict[str, Any] | None:
        key = (dataset, partition, arm, budget)
        if key in calls:
            return calls[key]
        names, named = _named_records(ctx, dataset, partition, budget)
        universe = ctx.records.universe(dataset)
        if arm.startswith("A"):
            records, ids = ctx.records.arm_a(dataset, named)
            condition, reverse = "obfuscated", ctx.records.reverse_mapping(dataset)
            named_sha, arm_sha = prompt_sha(ctx.records, named, names), prompt_sha(ctx.records, records, ids)
            if scrubbed_prompt_sha(ctx.records, dataset, named, names) != arm_sha:
                raise FillError(f"{dataset}/{partition}: named and Arm A prompts differ by more than literal name replacement")
            label = f"{dataset}/{partition}" + (f"/b{budget}" if budget else "")
            prompt_hashes.setdefault(label, {})["named"] = named_sha[:12]
            prompt_hashes[label]["arm_A"] = arm_sha[:12]
        else:
            records, ids = ctx.records.arm_b(named), list(names)
            condition, reverse = "named", None
        result = ctx.ranker.rank(step=step, name=_call_name(dataset, partition, arm, budget), dataset=dataset, partition=partition, condition=condition, records=records, ids=ids, universe=universe, reverse=reverse, extra={"budget": budget, "arm": arm})
        if result is not None:
            calls[key] = result
        return result

    for dataset in datasets:
        budgets: tuple[int | None, ...] = tuple(sorted({int(row["K"]) for row in skeleton.rows if row["dataset"] == dataset})) if dataset == LENDINGCLUB else (None,)
        partitions = PARTITIONS if dataset != THIRD else (FULL_DEV,)
        for budget in budgets:
            for partition in partitions:
                for arm in arms:
                    arm_call(dataset, partition, arm, budget)
    third_named = ctx.ranker.third_named(ctx.records) if THIRD in datasets else None
    for arm in arms:
        rows = [call for key, call in calls.items() if key[2] == arm]
        if rows:
            _rankings_csv(ctx.sidecar(f"47_obfuscated_lists_arm{'A' if arm.startswith('A') else 'B'}.csv"), rows, with_ids=arm.startswith("A"))
    if calls:
        _calls_json(ctx.sidecar("47_obfuscation_calls.json"), list(calls.values()))

    plans: list[dict[str, Any]] = []
    named_specs: dict[tuple[str, str], FitSpec] = {}
    for row in skeleton.rows:
        key = _key(row, "arm", "dataset", "backbone", "K")
        dataset, backbone, k, arm = row["dataset"], row["backbone"], int(row["K"]), row["arm"]
        budget = k if dataset == LENDINGCLUB else None
        partitions = PARTITIONS if dataset != THIRD else (FULL_DEV,)
        rankings = {partition: calls[(dataset, partition, arm, budget)]["ranking"] for partition in partitions if (dataset, partition, arm, budget) in calls}
        if FULL_DEV not in rankings:
            ctx.manifest.skip(step, key, "arm full-DEV ranking unavailable (call not made)")
            continue
        if dataset == THIRD and set(rankings) == {FULL_DEV}:
            # Stability gets one full-DEV call, so a fold AUC means that same top-K
            # re-evaluated on each fold, not a fold-local ranking
            rankings = {partition: rankings[FULL_DEV] for partition in PARTITIONS}
        specs = _fold_specs(dataset, backbone, rankings, k, f"47:{arm}:{dataset}")
        if dataset == THIRD:
            named_top = list(third_named["ranking"][:k]) if third_named else None
            named_source = "regenerated named ranking (paper's Stability 2024 ranking not on this machine)"
            if named_top:
                named_specs[(dataset, backbone)] = FitSpec(dataset, backbone, FULL_DEV, tuple(named_top), label=f"47:named-control:{dataset}:{backbone}")
        else:
            named_top = _cached_lc_ranking(FULL_DEV, k)[:k]
            named_source = f"artifacts/llm_cache full-DEV ranking (budget {k}) top-{k} (Appendix E subset)"
            named_specs[(dataset, backbone)] = FitSpec(dataset, backbone, FULL_DEV, tuple(named_top), label=f"47:named-control:{dataset}:{backbone}")
        named_folds = {} if dataset == THIRD else {partition: _cached_lc_ranking(partition, k)[:k] for partition in FOLDS} if dataset == LENDINGCLUB else {}
        arm_folds = {partition: list(ranking)[:k] for partition, ranking in rankings.items() if partition in FOLDS}
        plans.append({"row": row, "key": key, "specs": specs, "named_top": named_top, "named_source": named_source, "topk": list(rankings[FULL_DEV])[:k], "named_folds": named_folds, "arm_folds": arm_folds})
    results = _run_fits(ctx, step, [spec for plan in plans for spec in plan["specs"].values()] + list(named_specs.values()))
    for plan in plans:
        row, key = plan["row"], plan["key"]
        if "fulldev_topK_names_pipe_separated" in row:
            row["fulldev_topK_names_pipe_separated"] = _pipe(plan["topk"])
        ctx.manifest.cell(step, key, "fulldev_topK_names_pipe_separated", plan["topk"], "recomputed:arm full-DEV ranking top-K (original names)")
        if plan["named_top"] is not None:
            overlap = len(set(plan["topk"]) & set(plan["named_top"]))
            overlap_source = f"recomputed:names shared with the named full-DEV top-K ({plan['named_source']})"
            for column in ("overlap_with_named_fulldev", "overlap_fulldev"):
                if column in row:
                    row[column] = str(overlap)
            ctx.manifest.cell(step, key, "overlap_fulldev", overlap, overlap_source, named_topK=plan["named_top"])
        fold_overlaps = [len(set(plan["arm_folds"][partition]) & set(names)) for partition, names in plan["named_folds"].items() if partition in plan["arm_folds"]]
        if fold_overlaps:
            stats = (("overlap_folds_min", min(fold_overlaps)), ("overlap_folds_max", max(fold_overlaps)), ("overlap_folds_mean", sum(fold_overlaps) / len(fold_overlaps)))
            for column, value in stats:
                if column in row:
                    row[column] = str(value) if column != "overlap_folds_mean" else fmt(value, 4)
                    ctx.manifest.cell(step, key, column, value, "recomputed:per-fold overlap between the arm top-K and the cached named top-K of the same fold", n_folds=len(fold_overlaps), per_fold=fold_overlaps)
        full = _ok(results, plan["specs"].get(FULL_DEV))
        source = "recomputed:arm full-DEV top-K refit under the frozen pipeline, scored on HO"
        if full is not None:
            row["ho_auc_arm"] = fmt(full["ho_auc"])
            ctx.manifest.cell(step, key, "ho_auc_arm", full["ho_auc"], source, fit_id=full["fit_id"])
        fold_source = "recomputed:fold-local arm ranking top-K, frozen backbone on fold training rows, validation AUC" if row["dataset"] != THIRD else "recomputed:the arm full-DEV top-K re-evaluated on each fold (Stability 2024 has one full-DEV call)"
        fold_aucs = _fill_folds(row, plan["specs"], results, ctx.manifest, step, key, fold_source)
        _write_range(row, fold_aucs, "fold_auc_min", "fold_auc_max", ctx.manifest, step, key, fold_source)
    control_rows = []
    for (dataset, backbone), spec in named_specs.items():
        payload = _ok(results, spec)
        if payload is not None:
            control_rows.append({"dataset": dataset, "backbone": backbone, "K": BUDGETS[backbone], "ho_auc_named_refit": payload["ho_auc"], "source": "regenerated named ranking" if dataset == THIRD else "artifacts/llm_cache full-DEV ranking", "features": _pipe(spec.features)})
            ctx.manifest.note(step, f"{dataset}/{backbone}: named control refit HO AUC {payload['ho_auc']:.6f} (prefilled ho_auc_named in the skeleton is the paper value)")
    if control_rows:
        _write_csv(ctx.sidecar("47_named_control.csv"), control_rows, list(control_rows[0].keys()))
    import scripts.b6_obfuscation as helpers

    settings = (
        f"snapshot={helpers.MODEL_SNAPSHOT}; temperature={helpers.TEMPERATURE}; template=stability_expert_v3 target-free prompt (Appendix A), evidence_mode={helpers.EVIDENCE_MODE}; "
        f"response_format=json_schema target_free_feature_ranking (exactly 100 distinct names, up to 3 attempts, no fallback); shuffle_seed={helpers.MAPPING_SEED} "
        f"(numpy default_rng permutation of the sorted candidate universe -> F001.. per dataset, one global mapping per dataset as in the Home Credit run); "
        f"candidate_sets={ctx.records.candidate_source} (LendingClub: the cached run's fold-local lists, one call per reported budget 20/40 with the identical prompt; Stability 2024: fold-1 availability filter, full DEV only); "
        f"arm A scrubs every literal name occurrence from every text field; arm B blanks approved_definition and keeps every name/lineage/semantic-group field; "
        f"prompt sha256 (named/arm A) per partition: {json.dumps(prompt_hashes, sort_keys=True)}; refits: top-20 -> logistic regression, top-40 -> CatBoost, frozen pipeline."
    )
    ctx.sidecar("47_obfuscation_settings.txt").write_text(settings + "\n", encoding="utf-8")


BUILDERS = {"37": fill_37, "38": fill_38, "39a": fill_39a, "39b": fill_39b, "40": fill_40, "43": fill_43, "44": fill_44, "45": fill_45, "47": fill_47}
