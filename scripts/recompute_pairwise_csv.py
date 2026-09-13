"""Recompute pairwise HO AUC inference from authenticated row-level scores.

The input comparison rows are preserved, one provenance column named ``score``
is appended, and every metric that depends on the paired scores is recomputed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numba as nb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(r"D:\python projects\Research")
BACKUP = Path(r"D:\python projects\Research_pre_cleanup_backup_20260704")
INPUT = Path(r"C:\Users\DILSHOD\Downloads\Telegram Desktop\pairwise (2).csv")
OUTPUT = ROOT / "pairwise.csv"
SEED = 20260721
REPETITIONS = 2_000

sys.path.insert(0, str(ROOT / "src"))
from credit_risk_fs.evaluation.paired_inference import paired_delong_test  # noqa: E402


def _backup_prediction(relative: str) -> Path:
    return BACKUP / "results/final_experiments/id_preserved_predictions/predictions" / relative


SOURCES: dict[tuple[str, str, str], Path] = {
    ("homecredit", "lr", "Stable core + LLM fill"): _backup_prediction(
        "homecredit/lr/hybrid_stable_core_llm_fill/oot_predictions.csv"
    ),
    ("homecredit", "lr", "mRMR"): _backup_prediction(
        "homecredit/lr/statistical_mrmr/oot_predictions.csv"
    ),
    ("homecredit", "catboost", "Pure LLM"): _backup_prediction(
        "homecredit/catboost/llm/oot_predictions.csv"
    ),
    ("homecredit", "catboost", "RFE CatBoost"): ROOT
    / "results/full_baseline_v1/runs/homecredit/"
    "fbv1-034-homecredit-catboost-rfe-catboost-s42/predictions_oot.csv",
    ("homecredit", "lr", "LLM then mRMR"): _backup_prediction(
        "homecredit/lr/hybrid_llm_then_mrmr/oot_predictions.csv"
    ),
    # IV -> Boruta is configuration-specific. The Home Credit supplied values
    # are closest to the registered pool-100 runs; the LendingClub values match
    # the registered pool-300 runs.
    ("homecredit", "lr", "IV then Boruta"): ROOT
    / "results/selector_combinations_v1/oot/evaluations/"
    "scv1-oot-005-homecredit-iv-then-boruta-lr-full-dev-pool100-s42.oot_predictions.csv",
    ("homecredit", "catboost", "IV then Boruta"): ROOT
    / "results/selector_combinations_v1/oot/evaluations/"
    "scv1-oot-006-homecredit-iv-then-boruta-catboost-full-dev-pool100-s42.oot_predictions.csv",
    ("lendingclub_v2", "lr", "Pure LLM"): _backup_prediction(
        "lendingclub_v2/lr/llm/oot_predictions.csv"
    ),
    ("lendingclub_v2", "lr", "IV then Boruta"): ROOT
    / "results/selector_combinations_v1/oot/evaluations/"
    "scv1-oot-015-lendingclub_v2-iv-then-boruta-lr-full-dev-pool300-s42.oot_predictions.csv",
    ("lendingclub_v2", "catboost", "LLM then mRMR"): _backup_prediction(
        "lendingclub_v2/catboost/hybrid_llm_then_mrmr/oot_predictions.csv"
    ),
    ("lendingclub_v2", "catboost", "IV then Boruta"): ROOT
    / "results/selector_combinations_v1/oot/evaluations/"
    "scv1-oot-016-lendingclub_v2-iv-then-boruta-catboost-full-dev-pool300-s42.oot_predictions.csv",
    ("lendingclub_v2", "lr", "RFE CatBoost"): ROOT
    / "results/full_baseline_v1/runs/lendingclub_v2/"
    "fbv1-035-lendingclub_v2-lr-rfe-catboost-s42/predictions_oot.csv",
    ("lendingclub_v2", "catboost", "RFE CatBoost"): ROOT
    / "results/full_baseline_v1/runs/lendingclub_v2/"
    "fbv1-036-lendingclub_v2-catboost-rfe-catboost-s42/predictions_oot.csv",
    ("homecredit_model_stability_2024", "lr", "Pure LLM"): ROOT
    / "results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1/"
    "supplemental/evaluations/cell_031/predictions.parquet",
    ("homecredit_model_stability_2024", "lr", "RFE CatBoost"): ROOT
    / "results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1/"
    "classical/evaluations/cell_017/predictions.parquet",
    ("homecredit_model_stability_2024", "catboost", "Pure LLM"): ROOT
    / "results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1/"
    "supplemental/evaluations/cell_032/predictions.parquet",
    ("homecredit_model_stability_2024", "catboost", "CatBoost SHAP"): ROOT
    / "results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1/"
    "classical/evaluations/cell_014/predictions.parquet",
    ("homecredit_model_stability_2024", "catboost", "RFE CatBoost"): ROOT
    / "results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1/"
    "classical/evaluations/cell_018/predictions.parquet",
}


@nb.njit(cache=True)
def _weighted_auc(
    positive_weights: np.ndarray,
    negative_weights: np.ndarray,
    negative_order: np.ndarray,
    left_edges: np.ndarray,
    right_edges: np.ndarray,
) -> float:
    """AUC of a bootstrap multiset, including the sklearn 0.5 tie rule."""

    cumulative_negative = np.empty(len(negative_weights) + 1, dtype=np.int64)
    cumulative_negative[0] = 0
    for position in range(len(negative_order)):
        cumulative_negative[position + 1] = (
            cumulative_negative[position] + negative_weights[negative_order[position]]
        )
    numerator = 0.0
    for positive_index in range(len(positive_weights)):
        before = cumulative_negative[left_edges[positive_index]]
        tied = cumulative_negative[right_edges[positive_index]] - before
        numerator += positive_weights[positive_index] * (before + 0.5 * tied)
    return numerator / (positive_weights.sum() * negative_weights.sum())


def _auc_lookup(score: np.ndarray, event_indices: np.ndarray, non_event_indices: np.ndarray):
    positive_score = score[event_indices]
    negative_score = score[non_event_indices]
    negative_order = np.argsort(negative_score, kind="mergesort")
    sorted_negative = negative_score[negative_order]
    left_edges = np.searchsorted(sorted_negative, positive_score, side="left")
    right_edges = np.searchsorted(sorted_negative, positive_score, side="right")
    return negative_order, left_edges, right_edges


def _bootstrap_auc_difference(
    target: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> tuple[float, float]:
    event_indices = np.flatnonzero(target == 1)
    non_event_indices = np.flatnonzero(target == 0)
    if not len(event_indices) or not len(non_event_indices):
        raise ValueError("paired bootstrap requires both target classes")

    event_slot = np.full(len(target), -1, dtype=np.int64)
    non_event_slot = np.full(len(target), -1, dtype=np.int64)
    event_slot[event_indices] = np.arange(len(event_indices), dtype=np.int64)
    non_event_slot[non_event_indices] = np.arange(len(non_event_indices), dtype=np.int64)
    lookup_a = _auc_lookup(score_a, event_indices, non_event_indices)
    lookup_b = _auc_lookup(score_b, event_indices, non_event_indices)

    # Compile and independently verify the weighted implementation on the
    # unresampled data before using it for the 2,000 requested draws.
    unit_events = np.ones(len(event_indices), dtype=np.int64)
    unit_non_events = np.ones(len(non_event_indices), dtype=np.int64)
    weighted_full_a = _weighted_auc(unit_events, unit_non_events, *lookup_a)
    weighted_full_b = _weighted_auc(unit_events, unit_non_events, *lookup_b)
    np.testing.assert_allclose(
        [weighted_full_a, weighted_full_b],
        [roc_auc_score(target, score_a), roc_auc_score(target, score_b)],
        rtol=0.0,
        atol=1e-12,
    )

    rng = np.random.default_rng(SEED)
    differences = np.empty(REPETITIONS, dtype=float)
    for repetition in range(REPETITIONS):
        sampled_events = rng.choice(event_indices, len(event_indices), replace=True)
        sampled_non_events = rng.choice(
            non_event_indices, len(non_event_indices), replace=True
        )
        event_weights = np.bincount(
            event_slot[sampled_events], minlength=len(event_indices)
        ).astype(np.int64, copy=False)
        non_event_weights = np.bincount(
            non_event_slot[sampled_non_events], minlength=len(non_event_indices)
        ).astype(np.int64, copy=False)
        auc_a = _weighted_auc(event_weights, non_event_weights, *lookup_a)
        auc_b = _weighted_auc(event_weights, non_event_weights, *lookup_b)
        differences[repetition] = auc_a - auc_b

        # Check one actual draw against the literal roc_auc_score algorithm.
        if repetition == 0:
            sampled = np.concatenate([sampled_events, sampled_non_events])
            literal = roc_auc_score(target[sampled], score_a[sampled]) - roc_auc_score(
                target[sampled], score_b[sampled]
            )
            np.testing.assert_allclose(differences[repetition], literal, rtol=0.0, atol=1e-12)

    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_sidecar() -> pd.DataFrame:
    path = ROOT / "data/lendingclub_v2/processed/record_identity_v1.csv"
    sidecar = pd.read_csv(
        path,
        usecols=["loan_id", "split", "target", "processed_row_position"],
        dtype={"loan_id": "string"},
    )
    return sidecar.loc[sidecar["split"].str.upper().eq("OOT")].copy()


def _load_prediction(
    key: tuple[str, str, str], path: Path, sidecar: pd.DataFrame | None
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        result = pd.read_parquet(path, columns=["case_id", "target", "score"]).rename(
            columns={"case_id": "observation_id"}
        )
    else:
        columns = pd.read_csv(path, nrows=0).columns
        if "borrower_id" in columns:
            requested = ["borrower_id", "target", "score"]
            if key[0] == "lendingclub_v2":
                requested.append("source_row_index")
            result = pd.read_csv(path, usecols=requested)
            if key[0] == "lendingclub_v2":
                if sidecar is None:
                    raise ValueError("LendingClub identity sidecar was not loaded")
                result = result.merge(
                    sidecar,
                    left_on="source_row_index",
                    right_on="processed_row_position",
                    how="left",
                    validate="one_to_one",
                    suffixes=("_prediction", "_identity"),
                )
                if result["loan_id"].isna().any():
                    raise ValueError(f"unmatched LendingClub identities in {path}")
                if not np.array_equal(
                    result["target_prediction"].to_numpy(),
                    result["target_identity"].to_numpy(),
                ):
                    raise ValueError(f"LendingClub target mismatch in {path}")
                result = result.rename(
                    columns={"loan_id": "observation_id", "target_prediction": "target"}
                )[["observation_id", "target", "score"]]
            else:
                result = result.rename(columns={"borrower_id": "observation_id"})
        else:
            result = pd.read_csv(
                path,
                usecols=["stable_row_id", "target", "prediction_probability"],
            ).rename(
                columns={
                    "stable_row_id": "observation_id",
                    "prediction_probability": "score",
                }
            )

    result["observation_id"] = result["observation_id"].astype("string")
    if result["observation_id"].isna().any() or result["observation_id"].duplicated().any():
        raise ValueError(f"missing or duplicate observation IDs in {path}")
    if set(result["target"].unique()) != {0, 1}:
        raise ValueError(f"invalid target values in {path}")
    if not np.isfinite(result["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite scores in {path}")
    return result.sort_values("observation_id", kind="mergesort").reset_index(drop=True)


def main() -> None:
    comparisons = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    original_columns = list(comparisons.columns)
    required_columns = {
        "rule",
        "dataset",
        "backbone",
        "method_A",
        "method_B",
        "auc_A",
        "auc_B",
        "delta_auc_A_minus_B",
        "ci95_low",
        "ci95_high",
        "p_value",
        "n_ho",
    }
    if required_columns - set(original_columns):
        raise ValueError("input pairwise CSV schema is incomplete")

    needed_keys: set[tuple[str, str, str]] = set()
    for row in comparisons.itertuples(index=False):
        needed_keys.add((row.dataset, row.backbone, row.method_A))
        needed_keys.add((row.dataset, row.backbone, row.method_B))
    missing_keys = needed_keys - set(SOURCES)
    if missing_keys:
        raise KeyError(f"no prediction source mapping for: {sorted(missing_keys)}")

    sidecar = _load_sidecar() if any(key[0] == "lendingclub_v2" for key in needed_keys) else None
    loaded = {key: _load_prediction(key, SOURCES[key], sidecar) for key in needed_keys}
    inference_cache: dict[tuple[Path, Path], dict[str, float | int]] = {}
    output_rows: list[dict[str, str]] = []

    for row_number, row in enumerate(comparisons.to_dict(orient="records"), start=1):
        key_a = (row["dataset"], row["backbone"], row["method_A"])
        key_b = (row["dataset"], row["backbone"], row["method_B"])
        source_a, source_b = SOURCES[key_a], SOURCES[key_b]
        cache_key = (source_a, source_b)
        if cache_key not in inference_cache:
            paired = loaded[key_a].merge(
                loaded[key_b],
                on="observation_id",
                how="inner",
                validate="one_to_one",
                suffixes=("_a", "_b"),
            )
            if len(paired) != len(loaded[key_a]) or len(paired) != len(loaded[key_b]):
                raise ValueError(f"incomplete prediction join for row {row_number}")
            if not np.array_equal(
                paired["target_a"].to_numpy(), paired["target_b"].to_numpy()
            ):
                raise ValueError(f"target mismatch after prediction join for row {row_number}")

            target = paired["target_a"].to_numpy(dtype=int)
            score_a = paired["score_a"].to_numpy(dtype=float)
            score_b = paired["score_b"].to_numpy(dtype=float)
            delong = paired_delong_test(target, score_a, score_b)
            ci_low, ci_high = _bootstrap_auc_difference(target, score_a, score_b)
            inference_cache[cache_key] = {
                "auc_a": float(delong["auc_a"]),
                "auc_b": float(delong["auc_b"]),
                "delta": float(delong["auc_difference_a_minus_b"]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": float(delong["two_sided_p_value"]),
                "n_ho": len(paired),
            }
        result = inference_cache[cache_key]
        updated = dict(row)
        updated["auc_A"] = f"{result['auc_a']:.6f}"
        updated["auc_B"] = f"{result['auc_b']:.6f}"
        updated["delta_auc_A_minus_B"] = f"{result['delta']:+.6f}"
        updated["ci95_low"] = f"{result['ci_low']:+.6f}"
        updated["ci95_high"] = f"{result['ci_high']:+.6f}"
        updated["p_value"] = f"{result['p_value']:.12g}"
        updated["n_ho"] = str(result["n_ho"])
        updated["score"] = (
            f"A={_display_path(source_a)}; B={_display_path(source_b)}"
        )
        output_rows.append(updated)
        print(
            f"row {row_number:02d}: n={result['n_ho']}; "
            f"delta={result['delta']:+.6f}; "
            f"CI=[{result['ci_low']:+.6f}, {result['ci_high']:+.6f}]; "
            f"p={result['p_value']:.12g}",
            flush=True,
        )

    output = pd.DataFrame(output_rows, columns=[*original_columns, "score"])
    output.to_csv(OUTPUT, index=False, lineterminator="\n")
    print(f"wrote {OUTPUT} with {len(output)} rows and {len(output.columns)} columns")


if __name__ == "__main__":
    main()
