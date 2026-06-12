from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk_fs.feature_metadata.builder import infer_semantic_group


DATASETS = ("homecredit", "lendingclub")
RESULTS_ROOT = Path("results")
REPORTS_ROOT = Path("reports")
OUTPUT_SUBDIR = Path("analysis") / "clip_readiness"
STABLE_CORE_THRESHOLD = 0.8

EVIDENCE_COLUMNS = [
    "dataset",
    "feature",
    "description",
    "semantic_group",
    "source_table",
    "dtype_if_available",
    "missing_rate_dev",
    "missing_rate_oot_if_available",
    "iv_score_if_available",
    "psi_dev_oot_if_available",
    "psi_available_flag",
    "psi_missing_reason",
    "bootstrap_selection_frequency_if_available",
    "mrmr_selection_frequency",
    "boruta_selection_frequency",
    "llm_best_rank",
    "llm_mean_rank_if_available",
    "stable_core_membership",
    "selected_by_any_pipeline",
    "selected_by_mrmr",
    "selected_by_llm",
    "selected_by_llm_then_mrmr",
    "selected_by_stable_core_llm_fill",
    "mean_dev_if_available",
    "mean_oot_if_available",
    "std_dev_if_available",
    "std_oot_if_available",
    "evidence_source_files",
    "usable_for_clip_training_flag",
    "exclusion_reason_for_clip_if_any",
    "oot_fields_are_evaluation_only",
]

SUMMARY_COLUMNS = [
    "dataset",
    "total_features",
    "features_with_description",
    "features_with_semantic_group",
    "features_with_psi",
    "features_missing_psi",
    "features_with_iv",
    "features_with_llm_rank",
    "features_with_mrmr_frequency",
    "features_with_boruta_frequency",
    "features_with_stable_core_membership",
    "usable_for_clip_training_count",
    "not_usable_for_clip_training_count",
    "main_missing_reason",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _normalize_path(path_text: Any) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return Path(str(path_text).replace("\\", "/"))


def _repo_rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def _first_nonempty(series: pd.Series) -> Any:
    if series.empty:
        return pd.NA
    clean = series.dropna()
    clean = clean[clean.astype(str).str.len() > 0]
    if clean.empty:
        return pd.NA
    return clean.iloc[0]


def _max_numeric(series: pd.Series) -> Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return pd.NA
    return float(values.max())


def _mean_numeric(series: pd.Series) -> Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return pd.NA
    return float(values.mean())


def _load_matrix(dataset: str) -> pd.DataFrame:
    matrix = _read_csv(RESULTS_ROOT / dataset / "matrix_runs.csv")
    if matrix.empty:
        return matrix
    matrix = matrix[matrix["status"].eq("completed")].copy()
    matrix["output_folder"] = matrix["output_folder"].map(_normalize_path)
    return matrix


def _feature_col(frame: pd.DataFrame) -> str | None:
    for col in ("feature", "feature_name", "name", "Unnamed: 0"):
        if col in frame.columns:
            return col
    return None


def _empty_base(dataset: str, features: set[str]) -> pd.DataFrame:
    return pd.DataFrame({"dataset": dataset, "feature": sorted(features)})


def _collect_features_and_sources(dataset: str, matrix: pd.DataFrame) -> tuple[set[str], dict[str, set[str]]]:
    features: set[str] = set()
    sources: dict[str, set[str]] = defaultdict(set)

    candidate_paths = [
        RESULTS_ROOT / dataset / "feature_level_evidence.csv",
        RESULTS_ROOT / dataset / "analysis" / "feature_level_drift" / "feature_level_psi_by_run.csv",
        RESULTS_ROOT / dataset / "analysis" / "feature_level_drift" / "llm_top100_candidate_psi.csv",
    ]
    if not matrix.empty:
        for run_folder in matrix["output_folder"].tolist():
            folder = Path(run_folder)
            candidate_paths.extend(
                [
                    folder / "features" / "final_selected_features.csv",
                    folder / "features" / "selection_frequency.csv",
                    folder / "features" / "llm_rankings_summary.csv",
                    folder / "results" / "selected_feature_psi.csv",
                    folder / "llm_responses" / "final_dev" / "missing_filter_summary.csv",
                    folder / "llm_responses" / "final_dev" / "llm" / "missing_filter_summary.csv",
                    folder / "llm_responses" / "final_dev" / "feature_metadata.csv",
                    folder / "llm_responses" / "final_dev" / "llm" / "feature_metadata.csv",
                    folder / "llm_responses" / "final_dev" / "iv_prefilter" / "feature_audit.csv",
                    folder / "llm_responses" / "final_dev" / "llm" / "iv_prefilter" / "feature_audit.csv",
                    folder / "llm_responses" / "final_dev" / "statistical" / "stable_core_frequency.csv",
                ]
            )

    for path in candidate_paths:
        frame = _read_csv(path)
        if frame.empty:
            continue
        col = _feature_col(frame)
        if col is None:
            continue
        for feature in frame[col].dropna().astype(str):
            features.add(feature)
            sources[feature].add(_repo_rel(path))
    return features, sources


def _metadata_frame(dataset: str, matrix: pd.DataFrame, sources: dict[str, set[str]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base_path = RESULTS_ROOT / dataset / "feature_level_evidence.csv"
    base = _read_csv(base_path)
    if not base.empty:
        base = base.rename(
            columns={
                "dataset_name": "dataset",
                "feature_name": "feature",
                "dtype": "dtype_if_available",
                "missing_rate_mean": "missing_rate_dev",
            }
        )
        cols = [
            "feature",
            "description",
            "semantic_group",
            "source_table",
            "dtype_if_available",
            "missing_rate_dev",
        ]
        frames.append(base[[col for col in cols if col in base.columns]].copy())

    if not matrix.empty:
        for run_folder in matrix["output_folder"].tolist():
            folder = Path(run_folder)
            for rel in [
                "llm_responses/final_dev/feature_metadata.csv",
                "llm_responses/final_dev/llm/feature_metadata.csv",
            ]:
                path = folder / rel
                meta = _read_csv(path)
                if meta.empty:
                    continue
                meta = meta.rename(
                    columns={
                        "name": "feature",
                        "table": "source_table",
                        "dtype": "dtype_if_available",
                        "missing_rate": "missing_rate_dev",
                        "mean": "mean_dev_if_available",
                        "std": "std_dev_if_available",
                    }
                )
                cols = [
                    "feature",
                    "description",
                    "semantic_group",
                    "source_table",
                    "dtype_if_available",
                    "missing_rate_dev",
                    "mean_dev_if_available",
                    "std_dev_if_available",
                ]
                frames.append(meta[[col for col in cols if col in meta.columns]].copy())

            for rel in [
                "llm_responses/final_dev/missing_filter_summary.csv",
                "llm_responses/final_dev/llm/missing_filter_summary.csv",
            ]:
                path = folder / rel
                missing = _read_csv(path)
                if missing.empty:
                    continue
                missing = missing.rename(columns={"missing_rate": "missing_rate_dev"})
                if "feature" in missing.columns and "missing_rate_dev" in missing.columns:
                    frames.append(missing[["feature", "missing_rate_dev"]].copy())

    if not frames:
        return pd.DataFrame(columns=["feature"])

    combined = pd.concat(frames, ignore_index=True, sort=False)
    grouped = combined.groupby("feature", dropna=False).agg(
        description=("description", _first_nonempty) if "description" in combined.columns else ("feature", lambda _: pd.NA),
        semantic_group=("semantic_group", _first_nonempty) if "semantic_group" in combined.columns else ("feature", lambda _: pd.NA),
        source_table=("source_table", _first_nonempty) if "source_table" in combined.columns else ("feature", lambda _: pd.NA),
        dtype_if_available=("dtype_if_available", _first_nonempty)
        if "dtype_if_available" in combined.columns
        else ("feature", lambda _: pd.NA),
        missing_rate_dev=("missing_rate_dev", _mean_numeric) if "missing_rate_dev" in combined.columns else ("feature", lambda _: pd.NA),
        mean_dev_if_available=("mean_dev_if_available", _mean_numeric)
        if "mean_dev_if_available" in combined.columns
        else ("feature", lambda _: pd.NA),
        std_dev_if_available=("std_dev_if_available", _mean_numeric)
        if "std_dev_if_available" in combined.columns
        else ("feature", lambda _: pd.NA),
    )
    grouped = grouped.reset_index()

    if dataset == "lendingclub":
        grouped["semantic_group"] = grouped["feature"].map(lambda feature: infer_semantic_group(str(feature)))
    else:
        missing_semantic = grouped["semantic_group"].isna() | grouped["semantic_group"].astype(str).eq("")
        grouped.loc[missing_semantic, "semantic_group"] = grouped.loc[missing_semantic, "feature"].map(
            lambda feature: infer_semantic_group(str(feature))
        )
    return grouped


def _iv_frame(dataset: str, matrix: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_folder in matrix["output_folder"].tolist() if not matrix.empty else []:
        folder = Path(run_folder)
        for rel in [
            "llm_responses/final_dev/iv_prefilter/feature_audit.csv",
            "llm_responses/final_dev/llm/iv_prefilter/feature_audit.csv",
        ]:
            path = folder / rel
            frame = _read_csv(path)
            if frame.empty or "feature" not in frame.columns or "IV" not in frame.columns:
                continue
            subset = frame[["feature", "IV"]].copy()
            subset["iv_source"] = _repo_rel(path)
            frames.append(subset)
    if not frames:
        return pd.DataFrame(columns=["feature", "iv_score_if_available"])
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("feature", as_index=False)
        .agg(
            iv_score_if_available=("IV", _max_numeric),
            iv_source=("iv_source", _first_nonempty),
        )
        .copy()
    )


def _psi_frame(dataset: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in [
        RESULTS_ROOT / dataset / "analysis" / "feature_level_drift" / "feature_level_psi_by_run.csv",
        RESULTS_ROOT / dataset / "analysis" / "feature_level_drift" / "llm_top100_candidate_psi.csv",
    ]:
        frame = _read_csv(path)
        if frame.empty or "feature" not in frame.columns:
            continue
        cols = ["feature"]
        for col in ["psi_dev_oot", "feature_missing_oot", "missing_from_dev_oot_reason"]:
            if col in frame.columns:
                cols.append(col)
        subset = frame[cols].copy()
        subset["psi_source"] = _repo_rel(path)
        frames.append(subset)
    if not frames:
        return pd.DataFrame(
            columns=[
                "feature",
                "psi_dev_oot_if_available",
                "missing_rate_oot_if_available",
                "psi_missing_reason",
            ]
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    grouped = (
        combined.groupby("feature", as_index=False)
        .agg(
            psi_dev_oot_if_available=("psi_dev_oot", _mean_numeric)
            if "psi_dev_oot" in combined.columns
            else ("feature", lambda _: pd.NA),
            missing_rate_oot_if_available=("feature_missing_oot", _mean_numeric)
            if "feature_missing_oot" in combined.columns
            else ("feature", lambda _: pd.NA),
            psi_missing_reason=("missing_from_dev_oot_reason", _first_nonempty)
            if "missing_from_dev_oot_reason" in combined.columns
            else ("feature", lambda _: pd.NA),
            psi_source=("psi_source", _first_nonempty),
        )
        .copy()
    )
    return grouped


def _ranking_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_folder in matrix["output_folder"].tolist() if not matrix.empty else []:
        path = Path(run_folder) / "features" / "llm_rankings_summary.csv"
        frame = _read_csv(path)
        if frame.empty or "feature_name" not in frame.columns or "rank" not in frame.columns:
            continue
        frame = frame[frame.get("scope", pd.Series(index=frame.index, dtype=object)).astype(str).eq("final_dev")].copy()
        if frame.empty:
            continue
        frame = frame.rename(columns={"feature_name": "feature"})
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame["llm_rank_source"] = _repo_rel(path)
        frames.append(frame[["feature", "rank", "llm_rank_source"]])
    if not frames:
        return pd.DataFrame(columns=["feature", "llm_best_rank", "llm_mean_rank_if_available"])
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("feature", as_index=False)
        .agg(
            llm_best_rank=("rank", _max_rank_inverse_min),
            llm_mean_rank_if_available=("rank", _mean_numeric),
            llm_rank_source=("llm_rank_source", _first_nonempty),
        )
        .copy()
    )


def _max_rank_inverse_min(series: pd.Series) -> Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return pd.NA
    return float(values.min())


def _selection_maps(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    stable_rows: list[dict[str, Any]] = []

    for run in matrix.to_dict("records") if not matrix.empty else []:
        folder = Path(run["output_folder"])
        selector = str(run["selector"])

        for rel in ["features/final_selected_features.csv", "selected_feature_sets/final_selected_features.csv"]:
            path = folder / rel
            selected = _read_csv(path)
            if selected.empty:
                continue
            col = _feature_col(selected)
            if col is None:
                continue
            for feature in selected[col].dropna().astype(str).unique():
                selected_rows.append(
                    {
                        "feature": feature,
                        "selector": selector,
                        "selected_source": _repo_rel(path),
                    }
                )
            break

        freq_path = folder / "features" / "selection_frequency.csv"
        freq = _read_csv(freq_path)
        if not freq.empty and "feature_name" in freq.columns and "selection_frequency" in freq.columns:
            for item in freq[["feature_name", "selection_frequency"]].to_dict("records"):
                frequency_rows.append(
                    {
                        "feature": str(item["feature_name"]),
                        "selector": selector,
                        "selection_frequency": item["selection_frequency"],
                        "frequency_source": _repo_rel(freq_path),
                    }
                )

        if selector == "stable_core_llm_fill":
            for rel in [
                "llm_responses/final_dev/statistical/stable_core_frequency.csv",
                "llm_responses/final_dev/llm/statistical/stable_core_frequency.csv",
            ]:
                stable_path = folder / rel
                stable = _read_csv(stable_path)
                if stable.empty or "feature_name" not in stable.columns or "selection_frequency" not in stable.columns:
                    continue
                for item in stable[["feature_name", "selection_frequency"]].to_dict("records"):
                    stable_rows.append(
                        {
                            "feature": str(item["feature_name"]),
                            "bootstrap_selection_frequency_if_available": item["selection_frequency"],
                            "stable_source": _repo_rel(stable_path),
                        }
                    )

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        selected_summary = pd.DataFrame(columns=["feature"])
    else:
        selected_summary = selected.groupby("feature", as_index=False).agg(
            selected_by_any_pipeline=("selector", lambda values: True),
            selected_by_mrmr=("selector", lambda values: "mrmr" in set(values)),
            selected_by_llm=("selector", lambda values: "llm" in set(values)),
            selected_by_llm_then_mrmr=("selector", lambda values: "llm_then_mrmr" in set(values)),
            selected_by_stable_core_llm_fill=("selector", lambda values: "stable_core_llm_fill" in set(values)),
            selected_source=("selected_source", _first_nonempty),
        )

    frequencies = pd.DataFrame(frequency_rows)
    if frequencies.empty:
        frequency_summary = pd.DataFrame(columns=["feature"])
    else:
        mrmr = (
            frequencies[frequencies["selector"].eq("mrmr")]
            .groupby("feature", as_index=False)
            .agg(
                mrmr_selection_frequency=("selection_frequency", _max_numeric),
                mrmr_frequency_source=("frequency_source", _first_nonempty),
            )
        )
        boruta = (
            frequencies[frequencies["selector"].eq("boruta")]
            .groupby("feature", as_index=False)
            .agg(
                boruta_selection_frequency=("selection_frequency", _max_numeric),
                boruta_frequency_source=("frequency_source", _first_nonempty),
            )
        )
        frequency_summary = mrmr.merge(boruta, on="feature", how="outer")

    stable = pd.DataFrame(stable_rows)
    if stable.empty:
        stable_summary = pd.DataFrame(columns=["feature"])
    else:
        stable_summary = stable.groupby("feature", as_index=False).agg(
            bootstrap_selection_frequency_if_available=("bootstrap_selection_frequency_if_available", _max_numeric),
            stable_source=("stable_source", _first_nonempty),
        )
        stable_summary["stable_core_membership"] = (
            pd.to_numeric(stable_summary["bootstrap_selection_frequency_if_available"], errors="coerce")
            >= STABLE_CORE_THRESHOLD
        )
    return selected_summary, frequency_summary, stable_summary


def _merge_sources(row: pd.Series, existing_sources: dict[str, set[str]]) -> str:
    sources = set(existing_sources.get(str(row["feature"]), set()))
    for col in [
        "iv_source",
        "psi_source",
        "llm_rank_source",
        "selected_source",
        "mrmr_frequency_source",
        "boruta_frequency_source",
        "stable_source",
    ]:
        value = row.get(col)
        if pd.notna(value) and str(value):
            sources.add(str(value))
    return ";".join(sorted(sources))


def _training_exclusion_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if pd.isna(row.get("description")) or not str(row.get("description")).strip():
        reasons.append("missing_description")
    if pd.isna(row.get("semantic_group")) or not str(row.get("semantic_group")).strip():
        reasons.append("missing_semantic_group")

    dev_signal_cols = [
        "iv_score_if_available",
        "bootstrap_selection_frequency_if_available",
        "mrmr_selection_frequency",
        "boruta_selection_frequency",
        "llm_best_rank",
    ]
    has_numeric_signal = any(pd.notna(row.get(col)) for col in dev_signal_cols)
    has_selection_signal = any(
        bool(row.get(col))
        for col in [
            "selected_by_any_pipeline",
            "selected_by_mrmr",
            "selected_by_llm",
            "selected_by_llm_then_mrmr",
            "selected_by_stable_core_llm_fill",
        ]
    )
    if not has_numeric_signal and not has_selection_signal:
        reasons.append("no_dev_training_signal_saved")
    return ";".join(reasons)


def _main_missing_reason(frame: pd.DataFrame) -> str:
    reasons = Counter()
    for value in frame["exclusion_reason_for_clip_if_any"].fillna("").astype(str):
        if not value:
            continue
        for reason in value.split(";"):
            if reason:
                reasons[reason] += 1
    if not reasons:
        return ""
    return reasons.most_common(1)[0][0]


def build_dataset(dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = _load_matrix(dataset)
    features, sources = _collect_features_and_sources(dataset, matrix)
    evidence = _empty_base(dataset, features)

    for extra in [
        _metadata_frame(dataset, matrix, sources),
        _iv_frame(dataset, matrix),
        _psi_frame(dataset),
        _ranking_frame(matrix),
    ]:
        if not extra.empty:
            evidence = evidence.merge(extra, on="feature", how="left")

    selected, frequencies, stable = _selection_maps(matrix)
    for extra in [selected, frequencies, stable]:
        if not extra.empty:
            evidence = evidence.merge(extra, on="feature", how="left")

    bool_cols = [
        "selected_by_any_pipeline",
        "selected_by_mrmr",
        "selected_by_llm",
        "selected_by_llm_then_mrmr",
        "selected_by_stable_core_llm_fill",
        "stable_core_membership",
    ]
    for col in bool_cols:
        if col not in evidence.columns:
            evidence[col] = False
        evidence[col] = evidence[col].fillna(False).astype(bool)

    if "semantic_group" not in evidence.columns:
        evidence["semantic_group"] = pd.NA
    missing_semantic = evidence["semantic_group"].isna() | evidence["semantic_group"].astype(str).eq("")
    evidence.loc[missing_semantic, "semantic_group"] = evidence.loc[missing_semantic, "feature"].map(
        lambda feature: infer_semantic_group(str(feature))
    )
    if dataset == "lendingclub":
        evidence["semantic_group"] = evidence["feature"].map(lambda feature: infer_semantic_group(str(feature)))

    evidence["psi_available_flag"] = evidence["psi_dev_oot_if_available"].notna()
    evidence["psi_missing_reason"] = evidence["psi_missing_reason"].fillna("")
    missing_psi = ~evidence["psi_available_flag"]
    default_missing_reason = "not_selected_or_no_saved_candidate_psi_artifact"
    if dataset == "homecredit":
        default_missing_reason = "rejected_candidate_or_unselected_feature_psi_not_saved"
    evidence.loc[missing_psi & evidence["psi_missing_reason"].eq(""), "psi_missing_reason"] = default_missing_reason

    evidence["evidence_source_files"] = evidence.apply(lambda row: _merge_sources(row, sources), axis=1)
    evidence["exclusion_reason_for_clip_if_any"] = evidence.apply(_training_exclusion_reason, axis=1)
    evidence["usable_for_clip_training_flag"] = evidence["exclusion_reason_for_clip_if_any"].eq("")
    evidence["oot_fields_are_evaluation_only"] = True

    for col in [
        "description",
        "source_table",
        "dtype_if_available",
        "missing_rate_dev",
        "missing_rate_oot_if_available",
        "iv_score_if_available",
        "psi_dev_oot_if_available",
        "bootstrap_selection_frequency_if_available",
        "mrmr_selection_frequency",
        "boruta_selection_frequency",
        "llm_best_rank",
        "llm_mean_rank_if_available",
        "mean_dev_if_available",
        "mean_oot_if_available",
        "std_dev_if_available",
        "std_oot_if_available",
    ]:
        if col not in evidence.columns:
            evidence[col] = pd.NA

    evidence = evidence[EVIDENCE_COLUMNS].sort_values("feature").reset_index(drop=True)
    out_dir = RESULTS_ROOT / dataset / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_level_evidence_for_clip.csv"
    evidence.to_csv(out_path, index=False)

    summary = {
        "dataset": dataset,
        "total_features": len(evidence),
        "features_with_description": int(evidence["description"].notna().sum()),
        "features_with_semantic_group": int(evidence["semantic_group"].notna().sum()),
        "features_with_psi": int(evidence["psi_available_flag"].sum()),
        "features_missing_psi": int((~evidence["psi_available_flag"]).sum()),
        "features_with_iv": int(evidence["iv_score_if_available"].notna().sum()),
        "features_with_llm_rank": int(evidence["llm_best_rank"].notna().sum()),
        "features_with_mrmr_frequency": int(evidence["mrmr_selection_frequency"].notna().sum()),
        "features_with_boruta_frequency": int(evidence["boruta_selection_frequency"].notna().sum()),
        "features_with_stable_core_membership": int(evidence["stable_core_membership"].sum()),
        "usable_for_clip_training_count": int(evidence["usable_for_clip_training_flag"].sum()),
        "not_usable_for_clip_training_count": int((~evidence["usable_for_clip_training_flag"]).sum()),
        "main_missing_reason": _main_missing_reason(evidence),
    }
    return evidence, summary


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows available."
    text = frame.fillna("").astype(str)
    cols = list(text.columns)
    rows = text.values.tolist()
    widths = [max(len(col), *(len(row[idx]) for row in rows)) for idx, col in enumerate(cols)]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(values[idx].ljust(widths[idx]) for idx in range(len(values))) + " |"

    return "\n".join(
        [
            render(cols),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *[render(row) for row in rows],
        ]
    )


def write_report(summary: pd.DataFrame, datasets: dict[str, pd.DataFrame]) -> None:
    REPORTS_ROOT.mkdir(exist_ok=True)
    missing_tables = []
    for dataset, frame in datasets.items():
        missing = (
            frame.assign(missing_field_count=0)
            .pipe(
                lambda df: pd.DataFrame(
                    {
                        "field": [
                            "description",
                            "psi_dev_oot_if_available",
                            "iv_score_if_available",
                            "llm_best_rank",
                            "mrmr_selection_frequency",
                            "boruta_selection_frequency",
                            "bootstrap_selection_frequency_if_available",
                            "mean_oot_if_available",
                            "std_oot_if_available",
                        ],
                        "missing_count": [
                            int(df["description"].isna().sum()),
                            int(df["psi_dev_oot_if_available"].isna().sum()),
                            int(df["iv_score_if_available"].isna().sum()),
                            int(df["llm_best_rank"].isna().sum()),
                            int(df["mrmr_selection_frequency"].isna().sum()),
                            int(df["boruta_selection_frequency"].isna().sum()),
                            int(df["bootstrap_selection_frequency_if_available"].isna().sum()),
                            int(df["mean_oot_if_available"].isna().sum()),
                            int(df["std_oot_if_available"].isna().sum()),
                        ],
                    }
                )
            )
        )
        missing.insert(0, "dataset", dataset)
        missing_tables.append(missing)
    missing_summary = pd.concat(missing_tables, ignore_index=True)

    lines = [
        "# CLIP Readiness Feature Evidence Report",
        "",
        "This report builds CLIP-readiness evidence from saved baseline artifacts only. It does not implement CLIP, train a CLIP model, generate contrastive pairs, retrain selectors/models, or rerun the experiment matrix.",
        "",
        "## Evidence Tables Created",
        "",
        "- `results/homecredit/analysis/clip_readiness/feature_level_evidence_for_clip.csv`",
        "- `results/lendingclub/analysis/clip_readiness/feature_level_evidence_for_clip.csv`",
        "- `results/cross_dataset/analysis/clip_readiness/feature_level_evidence_summary.csv`",
        "",
        "## Cross-Dataset Summary",
        "",
        _frame_to_markdown(summary),
        "",
        "## Missing Fields By Dataset",
        "",
        _frame_to_markdown(missing_summary),
        "",
        "## Readiness Answers",
        "",
        "1. Baseline evidence is complete enough for CLIP planning. The tables consolidate descriptions, semantic groups, DEV missingness, IV where saved, LLM ranks, fold-selection frequencies, stable-core bootstrap frequencies, selected-pipeline flags, and available PSI support.",
        "2. Baseline evidence is not complete enough for CLIP training. Several fields needed for a clean training design are missing for material subsets, especially OOT-independent empirical summaries for rejected candidates and complete PSI/IV coverage.",
        "3. Missing fields vary by dataset, but both datasets lack saved OOT mean/std feature summaries. Features outside saved LLM/IV/selection artifacts also lack IV, LLM rank, and selector-frequency fields.",
        "4. Home Credit still lacks rejected-candidate PSI. Selected-feature PSI exists, and some LLM top-100 rows have PSI when the candidate was selected, but rejected-candidate DEV/OOT design matrices were not saved for complete PSI recovery.",
        "5. LendingClub still has unavailable PSI for categorical or missing-frame features where numeric DEV/OOT values were unavailable or the feature was not present in the processed safe frame.",
        "6. Missing values can mostly be fixed by targeted artifact generation: save DEV-only per-feature descriptive stats for the full candidate pool, save OOT support stats separately, compute candidate PSI from saved design matrices or regenerate design-matrix diagnostics, and persist IV for the full candidate universe.",
        "7. A full experiment rerun is not required. The missing pieces are diagnostic/evidence artifacts, not changed feature-selection or model-training results.",
        "8. Before CLIP training, generate training-safe DEV-only evidence tables, explicit train/evaluation field manifests, full candidate-pool IV, full candidate-pool missingness and numeric moments, complete LLM candidate ranks/reasons, and optional OOT PSI/mean/std support artifacts kept out of selector training.",
        "",
        "## OOT Field Policy",
        "",
        "`oot_fields_are_evaluation_only` is set to `true` in the per-feature evidence tables. OOT PSI and OOT summary statistics may be used for evaluation/support diagnostics only and must not be used to train a selector unless explicitly approved later.",
        "",
        "## Missing Artifacts",
        "",
        "- Complete saved DEV/OOT design matrices for rejected Home Credit LLM candidates.",
        "- Full candidate-pool PSI for Home Credit rejected or unselected features.",
        "- Full candidate-pool PSI for LendingClub categorical or missing-frame features that do not have numeric DEV/OOT values in saved artifacts.",
        "- Saved OOT mean/std feature-summary artifacts for both datasets.",
        "- Full candidate-pool IV artifacts for every feature in the unioned candidate universe.",
        "",
        "## Commands Run",
        "",
        "- `python scripts/build_clip_readiness_feature_evidence.py`",
        "",
        "## Rerun Decision",
        "",
        "No full rerun is required. Targeted artifact generation is sufficient before CLIP training, provided it does not alter selector/model outputs.",
    ]
    (REPORTS_ROOT / "clip_readiness_feature_evidence_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    datasets: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        frame, summary = build_dataset(dataset)
        datasets[dataset] = frame
        summaries.append(summary)
        print(f"{dataset}: wrote {len(frame)} feature rows")

    summary_frame = pd.DataFrame(summaries)[SUMMARY_COLUMNS]
    out_dir = RESULTS_ROOT / "cross_dataset" / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(out_dir / "feature_level_evidence_summary.csv", index=False)
    write_report(summary_frame, datasets)
    print("wrote cross-dataset summary and readiness report")


if __name__ == "__main__":
    main()
