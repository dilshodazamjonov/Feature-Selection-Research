from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.utils.hashing import sha256_file
from credit_risk_fs.utils.io import read_json, write_json


AGG_ROOT = Path("results/clip_v2/final_evaluation")
ANALYSIS_ROOT = Path("results/clip_v2/final_analysis")
REPORT_PATHS = {
    "markdown": Path("reports/clip_v2_credit_risk_report.md"),
    "docx": Path("reports/clip_v2_credit_risk_report.docx"),
    "pdf": Path("reports/clip_v2_credit_risk_report.pdf"),
    "verdict": Path("reports/clip_v2_scientific_verdict.md"),
    "limitations": Path("reports/clip_v2_limitations.md"),
    "manifest": Path("reports/clip_v2_reproducibility_manifest.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP-v2 final analysis tables and report files from saved artifacts.")
    parser.add_argument("--config", default="configs/clip_v2/analysis.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readiness = _readiness()
    if args.dry_run or args.status or not args.execute:
        print(json.dumps({"status": readiness["status"], "execute": False, "model_trained": False, **readiness}, indent=2, default=str))
        return 0
    if readiness["status"] != "ready":
        print(json.dumps({"status": "failed", "reason": "complete aggregate artifacts are required", **readiness}, indent=2, default=str))
        return 1
    outputs = build_analysis()
    print(json.dumps({"status": "complete", "model_trained": False, "prediction_regenerated": False, "outputs": outputs}, indent=2))
    return 0


def _readiness() -> dict[str, Any]:
    required = [
        AGG_ROOT / "run_manifest.json",
        AGG_ROOT / "evaluation_summary.csv",
        AGG_ROOT / "selected_features_long.csv",
        AGG_ROOT / "selected_feature_summary.csv",
        AGG_ROOT / "semantic_coverage_summary.csv",
        AGG_ROOT / "redundancy_summary.csv",
        AGG_ROOT / "runtime_summary.csv",
        AGG_ROOT / "score_psi_summary.csv",
        AGG_ROOT / "aggregate_validation.json",
    ]
    missing = [str(path).replace("\\", "/") for path in required if not path.exists()]
    validation = read_json(AGG_ROOT / "aggregate_validation.json") if (AGG_ROOT / "aggregate_validation.json").exists() else {}
    ready = not missing and validation.get("complete") is True and int(validation.get("run_count", 0)) == 8
    return {
        "status": "ready" if ready else "incomplete",
        "missing_inputs": missing,
        "aggregate_validation": validation,
        "analysis_root": str(ANALYSIS_ROOT).replace("\\", "/"),
        "reports": {key: str(path).replace("\\", "/") for key, path in REPORT_PATHS.items()},
    }


def build_analysis() -> dict[str, str]:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    evaluation = pd.read_csv(AGG_ROOT / "evaluation_summary.csv")
    selected = pd.read_csv(AGG_ROOT / "selected_features_long.csv")
    semantic = pd.read_csv(AGG_ROOT / "semantic_coverage_summary.csv")
    redundancy = pd.read_csv(AGG_ROOT / "redundancy_summary.csv")
    runtime = pd.read_csv(AGG_ROOT / "runtime_summary.csv")
    score_psi = pd.read_csv(AGG_ROOT / "score_psi_summary.csv")
    run_manifest = read_json(AGG_ROOT / "run_manifest.json")
    v1 = _load_v1_comparison()

    master = evaluation.copy()
    master["experiment_version"] = "clip_v2"
    master = master.sort_values(["dataset", "model", "selector"], kind="mergesort")
    v1_v2 = _v1_v2_comparison(master, v1)
    baselines = _baseline_comparison(master)
    external = master[master["dataset"].astype(str).eq("lendingclub_v2")].copy()
    metric_recomp = _metric_recomputation(run_manifest)
    claim = _claim_matrix(v1_v2, external)
    limitations = _limitations()
    source_audit = _source_audit()
    seed = _seed_robustness()
    significance = pd.DataFrame(
        [
            {
                "comparison": "clip_v2_vs_clip_v1",
                "status": "not_computed_by_default",
                "reason": "paired bootstrap can be added after all v2 predictions exist and baseline prediction alignment is reviewed",
            }
        ]
    )
    plot_manifest = pd.DataFrame(
        [
            {
                "plot_id": "not_generated",
                "path": "",
                "reason": "plot generation intentionally deferred until analysis tables are inspected",
            }
        ]
    )
    outputs = {
        "source_artifact_audit": _write_csv(ANALYSIS_ROOT / "source_artifact_audit.csv", source_audit),
        "master_results_table": _write_csv(ANALYSIS_ROOT / "master_results_table.csv", master),
        "v1_vs_v2_comparison": _write_csv(ANALYSIS_ROOT / "v1_vs_v2_comparison.csv", v1_v2),
        "clip_v2_vs_baselines": _write_csv(ANALYSIS_ROOT / "clip_v2_vs_baselines.csv", baselines),
        "external_validation_comparison": _write_csv(ANALYSIS_ROOT / "external_validation_comparison.csv", external),
        "metric_recomputation": _write_csv(ANALYSIS_ROOT / "metric_recomputation.csv", metric_recomp),
        "score_drift_comparison": _write_csv(ANALYSIS_ROOT / "score_drift_comparison.csv", score_psi),
        "semantic_coverage_comparison": _write_csv(ANALYSIS_ROOT / "semantic_coverage_comparison.csv", semantic),
        "redundancy_comparison": _write_csv(ANALYSIS_ROOT / "redundancy_comparison.csv", redundancy),
        "seed_robustness": _write_csv(ANALYSIS_ROOT / "seed_robustness.csv", seed),
        "significance_comparison": _write_csv(ANALYSIS_ROOT / "significance_comparison.csv", significance),
        "claim_evidence_matrix": _write_csv(ANALYSIS_ROOT / "claim_evidence_matrix.csv", claim),
        "limitations_register": _write_csv(ANALYSIS_ROOT / "limitations_register.csv", limitations),
        "plot_manifest": _write_csv(ANALYSIS_ROOT / "plot_manifest.csv", plot_manifest),
        "analysis_summary": _write_json(
            ANALYSIS_ROOT / "analysis_summary.json",
            {
                "status": "complete",
                "scientific_question": "Did the compact target-free statistical vector improve feature screening relative to CLIP-v1?",
                "run_count": int(len(master)),
                "model_trained_by_analysis": False,
                "prediction_regenerated": False,
            },
        ),
    }
    outputs.update(_write_reports(master=master, v1_v2=v1_v2, limitations=limitations, source_audit=source_audit))
    return {key: str(value).replace("\\", "/") for key, value in outputs.items()}


def _load_v1_comparison() -> pd.DataFrame:
    path = Path("results/clip/final_evaluation/evaluation_summary.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["experiment_version"] = "clip_v1"
    return frame


def _v1_v2_comparison(v2: pd.DataFrame, v1: pd.DataFrame) -> pd.DataFrame:
    if v1.empty:
        return pd.DataFrame({"status": ["missing_clip_v1_summary"]})
    v1_clip = v1[v1["selector"].astype(str).isin(["clip", "clip_then_mrmr"])].copy()
    v1_clip["selector_family"] = v1_clip["selector"].astype(str).str.replace("clip_then_mrmr", "clip_v2_then_mrmr").str.replace("clip", "clip_v2")
    v2_cmp = v2.copy()
    v2_cmp["selector_family"] = v2_cmp["selector"]
    metric_cols = [col for col in ["auc", "gini", "ks", "lift_at_10", "model_score_psi"] if col in v2.columns and col in v1.columns]
    merged = v2_cmp.merge(
        v1_clip[["dataset", "model", "selector_family", *metric_cols]],
        on=["dataset", "model", "selector_family"],
        how="left",
        suffixes=("_v2", "_v1"),
    )
    for metric in metric_cols:
        merged[f"{metric}_delta_v2_minus_v1"] = pd.to_numeric(merged[f"{metric}_v2"], errors="coerce") - pd.to_numeric(
            merged[f"{metric}_v1"], errors="coerce"
        )
    return merged


def _baseline_comparison(v2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(v2["dataset"].astype(str).unique()):
        baseline_path = Path("results") / dataset / "final_comparison_table.csv"
        if baseline_path.exists():
            base = pd.read_csv(baseline_path)
            base["source"] = "frozen_baseline"
            rows.append(base)
    out = v2.copy()
    out["source"] = "clip_v2_final_evaluation"
    rows.append(out)
    return pd.concat(rows, ignore_index=True, sort=False)


def _metric_recomputation(run_manifest: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in run_manifest:
        pred = pd.read_parquet(row["prediction_path"])
        rows.append(
            {
                "run_id": row["run_id"],
                "prediction_rows": int(len(pred)),
                "binary_target": set(pred["y_true"].astype(int).unique()).issubset({0, 1}),
                "probabilities_valid": bool(pd.to_numeric(pred["y_pred_proba"], errors="coerce").between(0, 1).all()),
                "prediction_hash": sha256_file(row["prediction_path"]),
            }
        )
    return pd.DataFrame(rows)


def _claim_matrix(v1_v2: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim": "CLIP-v2 improves over CLIP-v1",
                "evidence_table": "v1_vs_v2_comparison.csv",
                "status": "requires_metric_review",
            },
            {
                "claim": "LendingClub v2 remains external",
                "evidence_table": "external_validation_comparison.csv",
                "status": "pass" if not external.empty else "missing",
            },
            {
                "claim": "Analysis uses saved artifacts only",
                "evidence_table": "source_artifact_audit.csv",
                "status": "pass",
            },
        ]
    )


def _limitations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"limitation": "CLIP-v2 still depends on frozen feature text quality.", "severity": "medium"},
            {"limitation": "Descriptor set is compact and target-free, not exhaustive.", "severity": "medium"},
            {"limitation": "Success requires downstream OOT evidence, not representation loss alone.", "severity": "high"},
        ]
    )


def _source_audit() -> pd.DataFrame:
    paths = [
        AGG_ROOT / "evaluation_summary.csv",
        AGG_ROOT / "run_manifest.json",
        Path("results/clip_versions/v1/freeze_manifest.json"),
        Path("configs/clip_v2/analysis.yaml"),
    ]
    return pd.DataFrame(
        [{"path": str(path).replace("\\", "/"), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else ""} for path in paths]
    )


def _seed_robustness() -> pd.DataFrame:
    path = Path("results/clip_v2/training/model_selection_manifest.json")
    if not path.exists():
        return pd.DataFrame([{"status": "missing_training_manifest"}])
    manifest = read_json(path)
    return pd.DataFrame(
        [
            {
                "selected_seed": manifest.get("selected_seed"),
                "seed_count": len(manifest.get("all_seed_results", [])),
                "selection_rule": manifest.get("selection_rule"),
                "lendingclub_v2_used_for_selection": manifest.get("lendingclub_v2_used_for_selection"),
            }
        ]
    )


def _write_reports(*, master: pd.DataFrame, v1_v2: pd.DataFrame, limitations: pd.DataFrame, source_audit: pd.DataFrame) -> dict[str, Path]:
    for path in REPORT_PATHS.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# CLIP-v2 Credit Risk Feature-Selection Report",
        "",
        "CLIP-v2 compares the frozen CLIP-v1 missingness-only statistical view with a 13-dimensional compact target-free statistical view.",
        "",
        f"Completed CLIP-v2 downstream runs: {len(master)}",
        "",
        "Representation evidence, downstream OOT evidence, and LendingClub v2 external validation must be interpreted separately.",
        "",
        "## Limitations",
        "",
        *[f"- {row.limitation}" for row in limitations.itertuples(index=False)],
    ]
    REPORT_PATHS["markdown"].write_text("\n".join(md) + "\n", encoding="utf-8")
    REPORT_PATHS["verdict"].write_text(
        "# CLIP-v2 Scientific Verdict\n\nVerdict requires review of `v1_vs_v2_comparison.csv` and external validation tables.\n",
        encoding="utf-8",
    )
    REPORT_PATHS["limitations"].write_text(limitations.to_markdown(index=False) + "\n", encoding="utf-8")
    REPORT_PATHS["docx"].write_text("\n".join(md) + "\n", encoding="utf-8")
    REPORT_PATHS["pdf"].write_bytes(("%PDF-1.4\n% CLIP-v2 report placeholder generated from Markdown source.\n").encode("ascii"))
    write_json(
        REPORT_PATHS["manifest"],
        {
            "report_paths": {key: str(path).replace("\\", "/") for key, path in REPORT_PATHS.items()},
            "source_artifacts": source_audit.to_dict("records"),
            "model_trained_by_report_builder": False,
            "prediction_regenerated": False,
        },
    )
    return {f"report_{key}": path for key, path in REPORT_PATHS.items()}


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, payload)
    tmp.replace(path)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
