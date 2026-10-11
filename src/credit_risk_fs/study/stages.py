"""Study stages: prepare, llm, main, robustness, voting and sweep.

Every unit of work (a ranking, a fitted pipeline on one partition, a robustness fit)
is written to the study output directory as soon as it finishes and is skipped when the
stage is run again, so an interrupted stage resumes where it stopped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from credit_risk_fs.study import llm
from credit_risk_fs.study.config import (
    BACKBONES,
    CLASSICAL,
    FOLDS,
    FULL,
    FULL_DEV,
    LLM_ASSISTED,
    PARTITIONS,
    SELECTORS,
    SLUGS,
    StudyError,
    budget,
    family_cap,
    output_dir,
)
from credit_risk_fs.study.data import (
    Prepared,
    Split,
    build_stability_index,
    build_tabular_cache,
    load_prepared,
    load_split,
    load_targets,
    make_folds,
    missing_mask,
    read_stability_columns,
    stability_categorical,
    stability_index,
    stability_missing_rates,
)
from credit_risk_fs.study.metrics import auc, secondary_metrics
from credit_risk_fs.study.modeling import FittedPreprocessing, fit_score, fit_score_native
from credit_risk_fs.study.records import candidate_records, source_families
from credit_risk_fs.study.selection import PartitionSelector, reference_features
from credit_risk_fs.study.utils import (
    ResourceMonitor,
    ensure_dir,
    log,
    read_json,
    save_array,
    utc_now,
    write_csv,
    write_json,
)


# ===================================================================== prepare
def run_prepare(cfg: dict[str, Any]) -> None:
    directory = ensure_dir(output_dir(cfg) / "prepare")
    stability = cfg["loader"] == "stability_2024"
    meta = build_stability_index(cfg) if stability else build_tabular_cache(cfg)
    universe = list(meta["universe"])
    if cfg.get("universe_size") and len(universe) != int(cfg["universe_size"]):
        raise StudyError(f"universe has {len(universe)} features; expected {cfg['universe_size']}")
    y_dev, ids_dev, times_dev = load_targets(cfg, "dev")
    y_ho, ids_ho, times_ho = load_targets(cfg, "ho")
    folds = make_folds(times_dev, int(cfg["folds"]["n_splits"]), int(cfg["folds"]["gap"]))
    np.savez(
        directory / "folds.npz",
        **{f"{name}_train": train for name, (train, _) in folds.items()},
        **{f"{name}_valid": valid for name, (_, valid) in folds.items()},
    )

    screen_rows = folds["fold1"][0]
    log(f"{cfg['dataset']}: missingness screen on {len(screen_rows):,} fold-1 training rows")
    if stability:
        rows = stability_index(cfg)["dev_rows"][screen_rows]
        rates = stability_missing_rates(cfg, universe, rows)
        categorical_universe = stability_categorical(cfg, universe)
    else:
        dev = load_split(cfg, "dev", universe)
        subset = dev.X.iloc[screen_rows]
        rates = pd.Series({name: float(missing_mask(subset[name]).mean()) for name in universe})
        categorical_universe = [
            name
            for name in universe
            if not pd.api.types.is_numeric_dtype(dev.X[name].dtype) or pd.api.types.is_bool_dtype(dev.X[name].dtype)
        ]
        del dev, subset
    patterns = [str(pattern).casefold() for pattern in cfg.get("index_derived_patterns", [])]
    index_derived = [name for name in universe if any(pattern in name.casefold() for pattern in patterns)]
    threshold = float(cfg["screen"]["max_missing_rate"])
    candidates = [name for name in universe if rates[name] <= threshold and name not in set(index_derived)]
    categorical = [name for name in candidates if name in set(categorical_universe)]

    families = source_families(cfg, universe)
    records = candidate_records(cfg, candidates, categorical)
    write_json(
        directory / "candidates.json",
        {
            "universe": universe,
            "candidates": candidates,
            "categorical": categorical,
            "index_derived_excluded": index_derived,
            "n_dev": int(len(y_dev)),
            "n_ho": int(len(y_ho)),
            "screen_partition": "fold1",
            "screen_rows": int(len(screen_rows)),
            "max_missing_rate": threshold,
        },
    )
    write_csv(
        directory / "screen.csv",
        pd.DataFrame(
            {
                "feature": universe,
                "missing_rate": [float(rates[name]) for name in universe],
                "retained": [name in set(candidates) for name in universe],
                "index_derived": [name in set(index_derived) for name in universe],
            }
        ),
    )
    write_json(
        directory / "families.json",
        {"families": families, "n_families": len(set(families.values()))},
    )
    write_json(directory / "records.json", records)
    write_json(directory / "meta.json", {**{k: v for k, v in meta.items() if k != "universe"}, "prepared_at_utc": utc_now()})
    _write_cohort(cfg, directory, meta, (y_dev, times_dev), (y_ho, ids_ho, times_ho), folds)
    log(f"{cfg['dataset']}: {len(candidates)} of {len(universe)} features retained; DEV {len(y_dev):,} rows, HO {len(y_ho):,} rows")


def _index_text(value: Any) -> str:
    """Ordering-index value as text (integral day counts without a decimal point)."""

    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _split_row(split: str, y: np.ndarray, times: np.ndarray, total: int) -> dict[str, Any]:
    return {
        "split": split,
        "window_start": _index_text(np.min(times)),
        "window_end": _index_text(np.max(times)),
        "unique_index_units": int(len(np.unique(times))),
        "n_rows": int(len(y)),
        "n_events": int(np.sum(y)),
        "event_rate": float(np.mean(y)),
        "share_of_rows": float(len(y) / total),
    }


def _write_cohort(
    cfg: dict[str, Any],
    directory: Path,
    meta: dict[str, Any],
    dev: tuple[np.ndarray, np.ndarray],
    ho: tuple[np.ndarray, np.ndarray, np.ndarray],
    folds: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    y_dev, times_dev = dev
    y_ho, ids_ho, times_ho = ho
    total = len(y_dev) + len(y_ho)
    rows = [_split_row("DEV", y_dev, times_dev, total), _split_row("HO", y_ho, times_ho, total)]
    rows.append(
        {
            "split": "total",
            "n_rows": int(total),
            "n_events": int(np.sum(y_dev) + np.sum(y_ho)),
            "event_rate": float((np.sum(y_dev) + np.sum(y_ho)) / total),
            "share_of_rows": 1.0,
        }
    )
    if meta.get("release_rows"):
        rows.append({"split": "release", "n_rows": int(meta["release_rows"])})
    for name, (train, valid) in folds.items():
        rows.append(
            {
                "split": f"{name}_train",
                "window_start": _index_text(times_dev[train].min()),
                "window_end": _index_text(times_dev[train].max()),
                "unique_index_units": int(len(np.unique(times_dev[train]))),
                "n_rows": int(len(train)),
                "n_events": int(y_dev[train].sum()),
                "event_rate": float(y_dev[train].mean()),
            }
        )
        rows.append(
            {
                "split": f"{name}_valid",
                "window_start": _index_text(times_dev[valid].min()),
                "window_end": _index_text(times_dev[valid].max()),
                "unique_index_units": int(len(np.unique(times_dev[valid]))),
                "n_rows": int(len(valid)),
                "n_events": int(y_dev[valid].sum()),
                "event_rate": float(y_dev[valid].mean()),
            }
        )
    extras = pd.DataFrame({"id": ids_ho, "target": y_ho})
    subpopulations = cfg.get("subpopulations", {})
    if subpopulations.get("kind") == "lendingclub":
        term_column = subpopulations["term_column"]
        dev_term = load_split(cfg, "dev", [term_column]).X[term_column].to_numpy()
        ho_term = load_split(cfg, "ho", [term_column]).X[term_column].to_numpy()
        for split, y, term in (("DEV", y_dev, dev_term), ("HO", y_ho, ho_term)):
            for months in subpopulations["terms"]:
                mask = term == months
                rows.append(
                    {
                        "split": f"{split}_term{int(months)}",
                        "n_rows": int(mask.sum()),
                        "n_events": int(y[mask].sum()),
                        "event_rate": float(y[mask].mean()) if mask.any() else None,
                    }
                )
        reference = pd.Timestamp(subpopulations["issue_reference_date"])
        issue = reference + pd.to_timedelta(np.asarray(times_ho, dtype=np.int64), unit="D")
        extras["term"] = ho_term
        extras["issue_month"] = issue.strftime("%Y-%m")
    elif subpopulations.get("kind") == "monthly":
        extras["month"] = [str(value)[:7] for value in times_ho]
    write_json(directory / "cohort.json", {"cohort_cutoff": str(cfg["cohort_cutoff"]), "rows": rows})
    extras.to_parquet(directory / "ho_extras.parquet", index=False)


# ===================================================================== llm
def run_llm(cfg: dict[str, Any], *, allow_request: bool = True) -> None:
    prepared = load_prepared(cfg)
    rows = []
    for spec in llm.plan_calls(cfg):
        cached = llm.ranking_path(cfg, spec.call_id).exists()
        payload = llm.ensure_ranking(cfg, spec, prepared.records, prepared.universe, allow_request=allow_request)
        rows.append(
            {
                "call_id": spec.call_id,
                "arm": spec.arm,
                "partition": spec.partition,
                "budget": spec.budget,
                "repeat": spec.repeat,
                "cache_hit": cached,
                "attempts": payload.get("attempts"),
                "prompt_tokens": payload.get("prompt_tokens"),
                "completion_tokens": payload.get("completion_tokens"),
                "response_model": payload.get("response_model"),
            }
        )
    write_csv(output_dir(cfg) / "llm" / "calls.csv", pd.DataFrame(rows))
    log(f"{cfg['dataset']}: {len(rows)} rankings available")


# ===================================================================== shared fitting helpers
def record_path(cfg: dict[str, Any], *parts: str) -> Path:
    return output_dir(cfg).joinpath(*parts).with_suffix(".json")


def _frames_for(subset: dict[str, Any], X_train: pd.DataFrame, X_eval: pd.DataFrame, candidates: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mode = subset["mode"]
    if mode == "pca":
        return X_train, X_eval, {"mode": "pca", "pca_components": int(subset["pca_components"])}
    if mode == "expanded":
        wanted = set(subset["pool"])
        pool = [name for name in candidates if name in wanted]
        return X_train.loc[:, pool], X_eval.loc[:, pool], {"mode": "expanded", "columns": list(subset["features"])}
    features = list(subset["features"])
    return X_train.loc[:, features], X_eval.loc[:, features], {"mode": mode}


def evaluate(
    cfg: dict[str, Any],
    path: Path,
    *,
    backbone: str,
    label: str,
    partition: str,
    subset: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    ids_eval: np.ndarray,
    candidates: Sequence[str],
    scores_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one pipeline on the training rows and score the evaluation rows once."""

    record: dict[str, Any] = {
        "label": label,
        "backbone": backbone,
        "partition": partition,
        "features": list(subset["features"]),
        "mode": subset["mode"],
        "selection_seconds": float(subset.get("seconds", 0.0)),
        **{key: value for key, value in subset.items() if key in {"n_confirmed", "core_size", "pool", "pca_components"}},
        **(extra or {}),
    }
    if not subset["features"]:
        record.update({"auc": None, "n_model_columns": 0, "model_seconds": 0.0, "note": "empty subset; no model fitted"})
        write_json(path, record)
        return record
    train_frame, eval_frame, options = _frames_for(subset, X_train, X_eval, candidates)
    with ResourceMonitor() as monitor:
        scores, n_columns, names = fit_score(
            cfg,
            backbone,
            train_frame,
            y_train,
            [eval_frame],
            mode=options["mode"],
            columns=options.get("columns"),
            pca_components=options.get("pca_components"),
        )
    score = scores[0]
    record.update(
        {
            "auc": auc(y_eval, score),
            "n_model_columns": int(n_columns),
            "model_columns": names if len(names) <= 400 else None,
            "model_seconds": float(monitor.seconds),
            "peak_rss_bytes": int(monitor.peak_rss_bytes),
            "n_eval_rows": int(len(y_eval)),
        }
    )
    if partition == FULL_DEV:
        record.update(secondary_metrics(y_eval, score, ids_eval))
        if scores_path is not None:
            save_array(scores_path, score)
            record["scores_file"] = scores_path.name
    write_json(path, record)
    return record


def _take(frame: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    taken = frame.iloc[rows]
    taken.index = pd.RangeIndex(len(taken))
    return taken


def _partition_frames(prepared: Prepared, dev: Split, partition: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame | None, np.ndarray | None, np.ndarray | None]:
    train = prepared.train_rows(partition)
    X_train = dev.X if partition == FULL_DEV else _take(dev.X, train)
    y_train = dev.y[train]
    if partition == FULL_DEV:
        return X_train, y_train, None, None, None
    valid = prepared.valid_rows(partition)
    return X_train, y_train, _take(dev.X, valid), dev.y[valid], dev.ids[valid]


def main_record_path(cfg: dict[str, Any], backbone: str, label: str, partition: str) -> Path:
    return record_path(cfg, "main", backbone, SLUGS[label], partition)


def main_scores_path(cfg: dict[str, Any], backbone: str, label: str) -> Path:
    return output_dir(cfg) / "main" / "scores" / f"{backbone}__{SLUGS[label]}.npy"


# ===================================================================== main
def run_main(cfg: dict[str, Any]) -> None:
    prepared = load_prepared(cfg)
    dev = load_split(cfg, "dev", prepared.candidates)
    ho: Split | None = None
    for partition in PARTITIONS:
        X_train, y_train, X_valid, y_valid, ids_valid = _partition_frames(prepared, dev, partition)
        if partition == FULL_DEV:
            if ho is None:
                ho = load_split(cfg, "ho", prepared.candidates)
            X_eval, y_eval, ids_eval = ho.X, ho.y, ho.ids
        else:
            X_eval, y_eval, ids_eval = X_valid, y_valid, ids_valid
        selector = PartitionSelector(cfg, partition, X_train, y_train)
        for backbone in BACKBONES:
            k = budget(cfg, backbone)
            for label in SELECTORS:
                path = main_record_path(cfg, backbone, label, partition)
                if path.exists():
                    continue
                log(f"{cfg['dataset']} main: {label} / {backbone} / {partition}")
                with ResourceMonitor() as monitor:
                    subset = selector.subset(label, backbone, k)
                record = evaluate(
                    cfg,
                    path,
                    backbone=backbone,
                    label=label,
                    partition=partition,
                    subset=subset,
                    X_train=X_train,
                    y_train=y_train,
                    X_eval=X_eval,
                    y_eval=y_eval,
                    ids_eval=ids_eval,
                    candidates=prepared.candidates,
                    scores_path=main_scores_path(cfg, backbone, label) if partition == FULL_DEV else None,
                    extra={"K": k},
                )
                record["selection_peak_rss_bytes"] = int(monitor.peak_rss_bytes)
                write_json(path, record)
        selector.release()
        del selector, X_train
    del dev, ho
    _run_full_reference(cfg, prepared)


def _run_full_reference(cfg: dict[str, Any], prepared: Prepared) -> None:
    features = reference_features(cfg, prepared.universe, prepared.candidates)
    pending = [
        (backbone, partition)
        for backbone in BACKBONES
        for partition in PARTITIONS
        if not main_record_path(cfg, backbone, FULL, partition).exists()
    ]
    if not pending:
        return
    dev = load_split(cfg, "dev", features)
    ho = load_split(cfg, "ho", features) if any(partition == FULL_DEV for _, partition in pending) else None
    subset = {"features": features, "mode": "full", "seconds": 0.0}
    for backbone, partition in pending:
        log(f"{cfg['dataset']} main: {FULL} ({len(features)} features) / {backbone} / {partition}")
        X_train, y_train, X_valid, y_valid, ids_valid = _partition_frames(prepared, dev, partition)
        if partition == FULL_DEV:
            X_eval, y_eval, ids_eval = ho.X, ho.y, ho.ids
        else:
            X_eval, y_eval, ids_eval = X_valid, y_valid, ids_valid
        evaluate(
            cfg,
            main_record_path(cfg, backbone, FULL, partition),
            backbone=backbone,
            label=FULL,
            partition=partition,
            subset=subset,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            ids_eval=ids_eval,
            candidates=features,
            scores_path=main_scores_path(cfg, backbone, FULL) if partition == FULL_DEV else None,
            extra={"K": len(features)},
        )


# ===================================================================== leaders
def load_main_records(cfg: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    records = {}
    for backbone in BACKBONES:
        for label in (*SELECTORS, FULL):
            for partition in PARTITIONS:
                path = main_record_path(cfg, backbone, label, partition)
                if path.exists():
                    records[(backbone, label, partition)] = read_json(path)
    return records


def dev_mean(records: dict[tuple[str, str, str], dict[str, Any]], backbone: str, label: str) -> float | None:
    values = [records.get((backbone, label, fold), {}).get("auc") for fold in FOLDS]
    if any(value is None for value in values):
        return None
    return float(np.mean(values))


def _best(family: Sequence[str], values: dict[str, float | None]) -> str | None:
    available = [label for label in family if values.get(label) is not None]
    if not available:
        return None
    return max(available, key=lambda label: (values[label], -SELECTORS.index(label)))


def leaders(cfg: dict[str, Any], records: dict[tuple[str, str, str], dict[str, Any]] | None = None) -> dict[str, dict[str, tuple[str | None, str | None]]]:
    """HO, DEV and mixed leader pairs ``(LLM-assisted, classical)`` for each backbone."""

    records = records if records is not None else load_main_records(cfg)
    output: dict[str, dict[str, tuple[str | None, str | None]]] = {}
    for backbone in BACKBONES:
        ho = {label: records.get((backbone, label, FULL_DEV), {}).get("auc") for label in SELECTORS}
        dev = {label: dev_mean(records, backbone, label) for label in SELECTORS}
        output[backbone] = {
            "HO": (_best(LLM_ASSISTED, ho), _best(CLASSICAL, ho)),
            "DEV": (_best(LLM_ASSISTED, dev), _best(CLASSICAL, dev)),
            "mixed": (_best(LLM_ASSISTED, dev), _best(CLASSICAL, ho)),
        }
    return output


def _require_main(cfg: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    records = load_main_records(cfg)
    missing = [
        (backbone, label, partition)
        for backbone in BACKBONES
        for label in (*SELECTORS, FULL)
        for partition in PARTITIONS
        if (backbone, label, partition) not in records
    ]
    if missing:
        raise StudyError(f"{len(missing)} main pipeline records are missing (first: {missing[0]}); run --stage main")
    return records


# ===================================================================== robustness
def run_robustness(cfg: dict[str, Any]) -> None:
    prepared = load_prepared(cfg)
    records = _require_main(cfg)
    pairs = leaders(cfg, records)
    write_json(output_dir(cfg) / "robustness" / "leaders.json", {backbone: {rule: list(pair) for rule, pair in rules.items()} for backbone, rules in pairs.items()})
    dev = load_split(cfg, "dev", prepared.candidates)
    ho = load_split(cfg, "ho", prepared.candidates)
    full = PartitionSelector(cfg, FULL_DEV, dev.X, dev.y)

    def fit_full(path: Path, backbone: str, label: str, subset: dict[str, Any], *, scores_name: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        if path.exists():
            return read_json(path)
        log(f"{cfg['dataset']} robustness: {path.parent.name}/{path.stem}")
        scores_path = output_dir(cfg) / "robustness" / "scores" / f"{scores_name}.npy" if scores_name else None
        return evaluate(
            cfg,
            path,
            backbone=backbone,
            label=label,
            partition=FULL_DEV,
            subset=subset,
            X_train=dev.X,
            y_train=dev.y,
            X_eval=ho.X,
            y_eval=ho.y,
            ids_eval=ho.ids,
            candidates=prepared.candidates,
            scores_path=scores_path,
            extra=extra,
        )

    # --- native categorical handling (CatBoost leaders, same features, configuration and seed)
    for label in pairs["catboost"]["HO"]:
        if label is None:
            continue
        path = record_path(cfg, "robustness", "native", SLUGS[label])
        if path.exists():
            continue
        main = records[("catboost", label, FULL_DEV)]
        variables = _original_variables(cfg, main, dev.X, prepared.candidates)
        log(f"{cfg['dataset']} robustness: native categorical {label}")
        scores = fit_score_native(cfg, dev.X.loc[:, variables], dev.y, ho.X.loc[:, variables])
        write_json(
            path,
            {
                "label": label,
                "variables": variables,
                "n_categorical": int(sum(not pd.api.types.is_numeric_dtype(dev.X[name].dtype) for name in variables)),
                "ho_auc_native": auc(ho.y, scores),
                "ho_auc_onehot": main["auc"],
            },
        )

    # --- family cap and depth-0 controls (HO-rule classical leader, full DEV)
    for backbone in BACKBONES:
        k = budget(cfg, backbone)
        classical = pairs[backbone]["HO"][1]
        if classical is None or classical == "PCA":
            # Components are not original variables: neither control is defined for PCA.
            continue
        cap = family_cap(k, prepared.n_families)
        path = record_path(cfg, "robustness", "family_cap", backbone)
        if not path.exists() and (classical == "mRMR" or full.base_ranking(classical, k) is not None):
            subset = full.capped_subset(classical, k, prepared.families, cap)
            fit_full(
                path,
                backbone,
                classical,
                subset,
                scores_name=f"family_cap__{backbone}",
                extra={"base_selector": classical, "cap_per_family": cap, "n_families": prepared.n_families, "K": k},
            )
        if cfg.get("depth0_control", False):
            path = record_path(cfg, "robustness", "depth0", backbone)
            if not path.exists():
                depth0 = [record["name"] for record in prepared.records if str(record.get("depth")) == "0"]
                restricted = PartitionSelector(cfg, FULL_DEV, dev.X.loc[:, depth0], dev.y, key_prefix="depth0__")
                subset = restricted.subset(classical, backbone, k)
                fit_full(
                    path,
                    backbone,
                    classical,
                    subset,
                    scores_name=f"depth0__{backbone}",
                    extra={"base_selector": classical, "n_depth0_candidates": len(depth0), "K": k},
                )
                restricted.release()

    # --- descriptions rebuilt from lineage (Pure LLM and the HO-rule LLM-assisted leader)
    if cfg["llm"].get("lineage", True):
        for backbone in BACKBONES:
            k = budget(cfg, backbone)
            llm_leader = pairs[backbone]["HO"][0]
            for label in dict.fromkeys(["Pure LLM", llm_leader]):
                if label is None:
                    continue
                path = record_path(cfg, "robustness", "lineage", f"{backbone}__{SLUGS[label]}")
                if path.exists():
                    continue
                subset = full.subset(label, backbone, k, arm=llm.LINEAGE)
                fit_full(path, backbone, label, subset, scores_name=f"lineage__{backbone}__{SLUGS[label]}", extra={"K": k})

    # --- response sampling: per-fold rankings (or repeated full-DEV calls) through the full-DEV pipeline
    repeats = int(cfg["llm"].get("repeats", 0))
    for backbone in BACKBONES:
        k = budget(cfg, backbone)
        draws = [("repeat", f"r{index:02d}", FULL_DEV, index) for index in range(1, repeats + 1)] if repeats else [
            ("fold_ranking", fold, fold, 0) for fold in FOLDS
        ]
        for source, draw_id, partition, repeat in draws:
            path = record_path(cfg, "robustness", "response", f"{backbone}__{draw_id}")
            if path.exists():
                continue
            names = llm.ranking(cfg, llm.NAMED, partition, k, repeat=repeat)[:k]
            fit_full(path, backbone, "Pure LLM", {"features": names, "mode": "subset", "seconds": 0.0}, extra={"draw_source": source, "draw_id": draw_id, "K": k})

    # --- name obfuscation (two arms; fold models and the full-DEV refit)
    if cfg["llm"].get("obfuscation", False):
        for partition in PARTITIONS:
            X_train, y_train, X_valid, y_valid, ids_valid = _partition_frames(prepared, dev, partition)
            X_eval, y_eval, ids_eval = (ho.X, ho.y, ho.ids) if partition == FULL_DEV else (X_valid, y_valid, ids_valid)
            for arm in (llm.ARM_A, llm.ARM_B):
                for backbone in BACKBONES:
                    k = budget(cfg, backbone)
                    path = record_path(cfg, "robustness", "obfuscation", f"{arm}__{backbone}__{partition}")
                    if path.exists():
                        continue
                    names = llm.ranking(cfg, arm, partition, k)[:k]
                    log(f"{cfg['dataset']} robustness: obfuscation arm {arm} / {backbone} / {partition}")
                    evaluate(
                        cfg,
                        path,
                        backbone=backbone,
                        label="Pure LLM",
                        partition=partition,
                        subset={"features": names, "mode": "subset", "seconds": 0.0},
                        X_train=X_train,
                        y_train=y_train,
                        X_eval=X_eval,
                        y_eval=y_eval,
                        ids_eval=ids_eval,
                        candidates=prepared.candidates,
                        scores_path=output_dir(cfg) / "robustness" / "scores" / f"obfuscation__{arm}__{backbone}.npy" if partition == FULL_DEV else None,
                        extra={"arm": arm, "K": k},
                    )

    # --- L1-penalised selector (five folds and the full-DEV refit)
    for partition in PARTITIONS:
        X_train, y_train, X_valid, y_valid, ids_valid = _partition_frames(prepared, dev, partition)
        X_eval, y_eval, ids_eval = (ho.X, ho.y, ho.ids) if partition == FULL_DEV else (X_valid, y_valid, ids_valid)
        selector = PartitionSelector(cfg, partition, X_train, y_train) if partition != FULL_DEV else full
        for backbone in BACKBONES:
            k = budget(cfg, backbone)
            path = record_path(cfg, "robustness", "l1", f"{backbone}__{partition}")
            if path.exists():
                continue
            ranking = selector.l1(k)
            log(f"{cfg['dataset']} robustness: L1 selector / {backbone} / {partition}")
            evaluate(
                cfg,
                path,
                backbone=backbone,
                label="L1",
                partition=partition,
                subset={"features": ranking["selected"], "mode": "subset", "seconds": ranking["seconds"]},
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=y_eval,
                ids_eval=ids_eval,
                candidates=prepared.candidates,
                extra={"K": k, "budget_status": ranking["budget_status"]},
            )
        if selector is not full:
            selector.release()
    full.release()

    # --- matched mRMR timing on fold 1: unscreened universe against a 100-feature semantic pool
    timing = cfg.get("matched_mrmr_timing")
    if timing:
        _matched_mrmr_timing(cfg, prepared, timing)


def _original_variables(cfg: dict[str, Any], main: dict[str, Any], X_dev: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    """Original variables behind a subset (one-hot slots of the expanded scale map back to their source)."""

    if main.get("mode") != "expanded":
        return list(main["features"])
    pool = [name for name in candidates if name in set(main["pool"])]
    preprocessing = FittedPreprocessing(cfg, X_dev.loc[:, pool])
    variables: list[str] = []
    for column in main["features"]:
        source = preprocessing.source_of(column)
        if source not in variables:
            variables.append(source)
    return variables


def _matched_mrmr_timing(cfg: dict[str, Any], prepared: Prepared, timing: dict[str, Any]) -> None:
    import time

    from credit_risk_fs.selectors.lightweight.mi_mrmr import MutualInformationMRMRSelector

    path = output_dir(cfg) / "robustness" / "matched_mrmr_timing.json"
    if path.exists():
        return
    partition = str(timing.get("partition", "fold1"))
    k = int(timing.get("k", 40))
    pool_size = int(timing.get("pool", 100))
    train = prepared.train_rows(partition)
    if cfg["loader"] == "stability_2024":
        rows = stability_index(cfg)["dev_rows"][train]
        (X,) = read_stability_columns(cfg, prepared.universe, [rows])
        y = stability_index(cfg)["dev_target"][train]
    else:
        split = load_split(cfg, "dev", prepared.universe)
        X, y = _take(split.X, train), split.y[train]
    settings = cfg["selectors"]["mrmr"]

    def timed(frame: pd.DataFrame) -> float:
        selector = MutualInformationMRMRSelector(
            k=k,
            estimator=str(settings["estimator"]),
            sample_rows=settings.get("sample_rows"),
            n_neighbors=int(settings["n_neighbors"]),
            objective=str(settings["objective"]),
            random_state=int(cfg["seed"]),
            n_jobs=int(settings.get("n_jobs", 1)),
        )
        started = time.perf_counter()
        selector.fit(frame, pd.Series(np.asarray(y, dtype=int)))
        return time.perf_counter() - started

    pool = llm.ranking(cfg, llm.NAMED, partition, pool_size)[:pool_size]
    log(f"{cfg['dataset']} robustness: matched mRMR timing over {X.shape[1]} features")
    universe_seconds = timed(X)
    log(f"{cfg['dataset']} robustness: matched mRMR timing over the {pool_size}-feature pool")
    pool_seconds = timed(X.loc[:, pool])
    write_json(
        path,
        {
            "partition": partition,
            "k": k,
            "universe_features": int(X.shape[1]),
            "pool_features": len(pool),
            "universe_seconds": universe_seconds,
            "pool_seconds": pool_seconds,
            "rows": int(len(X)),
        },
    )


# ===================================================================== voting
def run_voting(cfg: dict[str, Any]) -> None:
    if not cfg.get("voting", {}).get("enabled", False):
        log(f"{cfg['dataset']}: the voting family is not configured for this dataset")
        return
    prepared = load_prepared(cfg)
    dev = load_split(cfg, "dev", prepared.candidates)
    ho: Split | None = None
    pools = [int(value) for value in cfg["voting"]["pools"]]
    d = len(prepared.candidates)
    for partition in PARTITIONS:
        X_train, y_train, _, _, _ = _partition_frames(prepared, dev, partition)
        selector = PartitionSelector(cfg, partition, X_train, y_train)
        log(f"{cfg['dataset']} voting: voters on {partition}")
        voters = {
            "rf_correlation": selector.rf_corr_full()["ranking"],
            "boruta": selector.boruta()["ranking"],
            "rfe": selector.rfe_full()["ranking"],
        }
        positions = {voter: {name: rank for rank, name in enumerate(ranking, start=1)} for voter, ranking in voters.items()}
        average = {
            name: float(np.mean([(positions[voter][name] - 1) / (d - 1) for voter in voters]))
            for name in prepared.candidates
        }
        order = sorted(prepared.candidates, key=lambda name: (average[name], name))
        for backbone in BACKBONES:
            k = budget(cfg, backbone)
            path = record_path(cfg, "voting", backbone, partition)
            if path.exists() and partition != FULL_DEV:
                continue
            subsets: dict[str, list[str]] = {}
            for pool in pools:
                members = order[:pool]

                def compute(members: list[str] = members) -> dict[str, Any]:
                    return {"subset": selector._rfe(selector._ordered(members), k)[:k], "pool": members}

                subsets[str(pool)] = selector.cached(f"vote_{backbone}_P{pool}", compute)["subset"]
            reference = voters["rf_correlation"][:k]
            payload = {"partition": partition, "K": k, "voted": subsets, "reference": reference, "average_rank_order": order[: max(pools)]}
            if partition == FULL_DEV:
                if ho is None:
                    ho = load_split(cfg, "ho", prepared.candidates)
                payload["ho"] = {}
                for name, features in [*((f"P{pool}", subsets[str(pool)]) for pool in pools), ("reference", reference)]:
                    scores_path = output_dir(cfg) / "voting" / "scores" / f"{backbone}__{name}.npy"
                    record_file = record_path(cfg, "voting", "ho", f"{backbone}__{name}")
                    if record_file.exists():
                        payload["ho"][name] = read_json(record_file)
                        continue
                    log(f"{cfg['dataset']} voting: {backbone} {name} full-DEV refit")
                    payload["ho"][name] = evaluate(
                        cfg,
                        record_file,
                        backbone=backbone,
                        label=f"voting {name}",
                        partition=FULL_DEV,
                        subset={"features": features, "mode": "subset", "seconds": 0.0},
                        X_train=X_train,
                        y_train=y_train,
                        X_eval=ho.X,
                        y_eval=ho.y,
                        ids_eval=ho.ids,
                        candidates=prepared.candidates,
                        scores_path=scores_path,
                    )
            write_json(path, payload)
        selector.release()


# ===================================================================== sweep
def run_sweep(cfg: dict[str, Any]) -> None:
    prepared = load_prepared(cfg)
    dev = load_split(cfg, "dev", prepared.candidates)
    budgets = [int(value) for value in cfg["sweep"]["budgets"]]
    for partition in FOLDS:
        X_train, y_train, X_valid, y_valid, ids_valid = _partition_frames(prepared, dev, partition)
        selector = PartitionSelector(cfg, partition, X_train, y_train)
        for backbone in BACKBONES:
            for k in budgets:
                if k == budget(cfg, backbone):
                    continue  # the documented budget is the main run
                for label in SELECTORS:
                    path = record_path(cfg, "sweep", backbone, f"K{k}", SLUGS[label], partition)
                    if path.exists():
                        continue
                    log(f"{cfg['dataset']} sweep: {label} / {backbone} / K={k} / {partition}")
                    subset = selector.subset(label, backbone, k)
                    evaluate(
                        cfg,
                        path,
                        backbone=backbone,
                        label=label,
                        partition=partition,
                        subset=subset,
                        X_train=X_train,
                        y_train=y_train,
                        X_eval=X_valid,
                        y_eval=y_valid,
                        ids_eval=ids_valid,
                        candidates=prepared.candidates,
                        extra={"K": k},
                    )
        selector.release()


__all__ = [
    "dev_mean",
    "leaders",
    "load_main_records",
    "main_record_path",
    "main_scores_path",
    "record_path",
    "run_llm",
    "run_main",
    "run_prepare",
    "run_robustness",
    "run_sweep",
    "run_voting",
]
