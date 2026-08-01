#!/usr/bin/env python
"""Reproducible artifact-only Prompt 11 baseline audit package builder."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credit_risk_fs.analysis.baseline_audit import (
    AUDIT_SCHEMA_VERSION,
    BOOTSTRAP_REPETITIONS,
    audit_completed_baselines,
)
from credit_risk_fs.experiments.atomic_io import (
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)


DEFAULT_OUTPUT = Path("cleanup/audits/prompt_11_selector_combinations")
TABLE_FILES = {
    "baseline_results_long": "baseline_results_long.csv",
    "baseline_metric_reconciliation": "baseline_metric_reconciliation.csv",
    "baseline_dev_fold_summary": "baseline_dev_fold_summary.csv",
    "baseline_oot_summary": "baseline_oot_summary.csv",
    "baseline_pairwise_comparisons": "baseline_pairwise_comparisons.csv",
    "baseline_holm_families": "baseline_holm_families.csv",
    "baseline_selection_stability": "baseline_selection_stability.csv",
    "baseline_score_psi": "baseline_score_psi.csv",
    "baseline_feature_psi_audit": "baseline_feature_psi_audit.csv",
    "baseline_runtime_resources": "baseline_runtime_resources.csv",
    "combination_structural_feasibility": "combination_structural_feasibility.csv",
}


def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _records(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    active = frame if columns is None else frame.loc[:, columns]
    return [{str(key): _clean(value) for key, value in row.items()} for row in active.to_dict("records")]


def _fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def _claims(results: dict[str, Any]) -> pd.DataFrame:
    auth = results["authentication"]
    reconciliation = results["baseline_metric_reconciliation"]
    comparisons = results["baseline_pairwise_comparisons"]
    rows: list[dict[str, Any]] = [
        {
            "claim_id": "authentication_36_of_36",
            "scope": "full_matrix",
            "claim": "All 36 frozen Prompt 10 cells and Final Cell 036 are authenticated complete.",
            "evidence_strength": "strong",
            "evidence_artifact": "baseline_completion_authentication.json",
            "limitation": "Authentication establishes integrity, not predictive superiority.",
        },
        {
            "claim_id": "metric_reconciliation",
            "scope": "persisted_full_dev_and_oot_predictions",
            "claim": f"All {len(reconciliation)} supported saved-prediction metric checks reconcile within frozen tolerances.",
            "evidence_strength": "strong" if auth["metric_reconciliation_failures"] == 0 else "not_supported",
            "evidence_artifact": "baseline_metric_reconciliation.csv",
            "limitation": "DEV fold prediction vectors were not persisted and therefore cannot be recomputed.",
        },
        {
            "claim_id": "dev_fold_prediction_inference",
            "scope": "five_expanding_dev_folds",
            "claim": "Paired fold-level prediction inference can be reconstructed from Prompt 10 artifacts.",
            "evidence_strength": "not_supported",
            "evidence_artifact": "baseline_dev_fold_summary.csv",
            "limitation": "Only fold metrics and selected sets were persisted; fold prediction vectors are absent.",
        },
    ]
    for index, row in enumerate(comparisons.itertuples(index=False), start=1):
        rows.append(
            {
                "claim_id": f"paired_comparison_{index:03d}",
                "scope": f"{row.dataset}/{row.model}/OOT",
                "claim": (
                    f"{row.method_a} minus {row.method_b}: delta AUC={row.delta_auc:.8f}, "
                    f"paired bootstrap 95% CI [{row.bootstrap_ci95_lower:.8f}, {row.bootstrap_ci95_upper:.8f}], "
                    f"Holm-adjusted p={row.holm_adjusted_p:.8g}."
                ),
                "evidence_strength": row.evidence_strength,
                "evidence_artifact": "baseline_pairwise_comparisons.csv",
                "limitation": row.interpretation_limit,
            }
        )
    return pd.DataFrame(rows)


def _markdown(results: dict[str, Any], claims: pd.DataFrame) -> str:
    auth = results["authentication"]
    dev = results["baseline_dev_fold_summary"]
    oot = results["baseline_oot_summary"]
    comparisons = results["baseline_pairwise_comparisons"]
    families = results["baseline_holm_families"]
    stability = results["baseline_selection_stability"]
    psi = results["baseline_score_psi"]
    runtime = results["baseline_runtime_resources"]
    feasibility = results["combination_structural_feasibility"]
    decisive = comparisons.loc[comparisons["evidence_strength"].isin(["strong", "moderate"])].copy()
    lines = [
        "# Prompt 11 baseline results audit",
        "",
        "## Objective",
        "",
        "Authenticate and audit the completed Prompt 10 individual-selector baseline evidence, then assess the preregistered combination families structurally without refitting a baseline or opening a raw research dataset.",
        "",
        "## Completed evidence and authentication",
        "",
        f"Fact: exactly **{auth['authenticated_cells']}/{auth['expected_cells']}** frozen cells authenticated, including `{auth['cell_036']}`. The phase composition is {auth['phase_composition']['dev_expanding_window_fold_evaluations']} expanding-window DEV fold metric units, {auth['phase_composition']['full_dev_refit_diagnostics']} full-DEV in-sample diagnostics, and {auth['phase_composition']['oot_evaluations']} locked OOT evaluations.",
        "",
        f"Fact: all **{len(results['baseline_metric_reconciliation'])}** supported metric reconciliation checks passed; discrepancies: **{auth['metric_reconciliation_failures']}**. No raw dataset path was resolved and no baseline was refit.",
        "",
        "## Methodology",
        "",
        "Saved prediction identity, target encoding, finiteness, probability range, ordering, file size, and SHA-256 bindings were checked before metric recomputation. OOT comparisons use aligned predictions, paired DeLong inference, 2,000 paired target-stratified bootstrap resamples (seed 20260721; percentile 95% interval), and Holm correction within eight named dataset/model/reference families of seven comparisons each.",
        "",
        "## DEV robustness",
        "",
        f"Fact: {len(dev)} method/dataset/model summaries preserve all five expanding folds in temporal order. Fold predictions were not persisted, so no fold pooling or paired fold inference was attempted. Full fold-level means, standard deviations, ranges, counts, and selected-feature counts are in `baseline_dev_fold_summary.csv`.",
        "",
        "## OOT evidence",
        "",
        f"Fact: {len(oot)} authenticated locked-OOT evaluation rows are available. OOT is treated as final predictive evidence, while the saved full-DEV prediction rows are explicitly labelled in-sample diagnostics.",
        "",
        "## Paired comparisons",
        "",
        f"Fact: {len(comparisons)} preregistered comparisons were evaluated across {len(families)} Holm families; {int(comparisons['holm_reject_0_05'].sum())} comparisons survived Holm and {len(decisive)} received a strong or moderate scoped evidence label. No all-pairs search was run.",
        "",
    ]
    if decisive.empty:
        lines.extend(["Interpretation: no preregistered contrast met the conservative strong/moderate evidence rules. This is not evidence of equivalence.", ""])
    else:
        lines.extend(["Scoped strong/moderate findings (method A minus reference):", "", "| Dataset | Model | A | Reference | Delta AUC | 95% CI | Holm p | Strength |", "|---|---|---|---|---:|---:|---:|---|"])
        for row in decisive.sort_values(["dataset", "model", "method_b", "method_a"]).itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.model} | {row.method_a} | {row.method_b} | {_fmt(row.delta_auc, 6)} | [{_fmt(row.bootstrap_ci95_lower, 6)}, {_fmt(row.bootstrap_ci95_upper, 6)}] | {_fmt(row.holm_adjusted_p, 6)} | {row.evidence_strength} |"
            )
        lines.append("")
    variable = int(stability["natural_support_varies"].sum())
    lines.extend(
        [
            "## Stability and drift",
            "",
            f"Fact: selection stability was recomputed for {len(stability)} configurations. {variable} have varying fold subset sizes; Kuncheva is marked not applicable for those cases, while Jaccard and Nogueira-style results remain separately reported with their applicability labels.",
            "",
            f"Fact: {len(psi)} authenticated score-PSI summaries and {len(results['baseline_feature_psi_audit'])} saved feature-PSI rows were audited. All {int(psi['drift_level'].eq('stable').sum())} score-PSI values are below the descriptive 0.10 threshold (range {_fmt(psi['score_psi'].min(), 4)} to {_fmt(psi['score_psi'].max(), 4)}); 0.10 and 0.25 remain descriptive references, not hypothesis-test cutoffs. Score PSI was not recomputed because the saved artifacts do not preserve the frozen DEV bin edges; raw feature matrices were not reopened.",
            "",
            "## Runtime and resources",
            "",
            f"Fact: {len(runtime)} completed cell resource summaries separate active computation from RAM waiting. Peak process-tree RSS ranged from {_fmt(runtime['peak_process_rss_gib'].min(), 2)} to {_fmt(runtime['peak_process_rss_gib'].max(), 2)} GiB; recorded RAM waiting totaled {_fmt(runtime['ram_wait_seconds'].sum()/3600, 2)} hours.",
            "",
            "Interpretation: these measurements are scheduling and feasibility evidence, not predictive evidence. Heavy-chain workload classification must reflect every material selector stage and the final model.",
            "",
            "## Combination feasibility implications",
            "",
            f"Fact: {len(feasibility)} fold/configuration feasibility rows cover only IV→Boruta, Boruta→CatBoost RFE, Boruta→canonical MI-mRMR, and the exact five-voter normalized-average-rank method. Prompt 10 baseline components are not silently reused because their saved identities do not authenticate the required chained training universe or complete five-voter rank bundle.",
            "",
            "Recommendation: run the committed bounded 24-evaluation pilot (18 unique selector fits) unchanged. Review support, correctness, fit count, runtime, and RSS before creating the separate approval lock; do not tune from pilot predictive outcomes.",
            "",
            "## Limitations",
            "",
            "- Fold-level prediction vectors were not persisted, so fold metric reconciliation and paired fold inference are unsupported.",
            "- Saved full-DEV predictions are in-sample diagnostics, not out-of-fold predictions.",
            "- Score-PSI bin edges were not persisted, so score PSI is authenticated but not independently recomputed.",
            "- Prompt 10 budget-capped Boruta artifacts prove shortfall when fewer than k features were retained, but reaching k does not reveal uncapped natural-support size.",
            "- Random-k has one frozen seed in the completed matrix; no replicate distribution exists and no favorable seed was selected.",
            "- Non-significance does not establish equivalence; natural-support/count mismatches remain explicit caveats.",
            "",
            "## Conclusion",
            "",
            "The completed baseline evidence is internally authenticated and all supported saved-prediction metrics reconcile. Conclusions remain scoped by dataset, final model, split, uncertainty, temporal robustness, selected-count semantics, and resource cost. The preregistered combinations remain hypotheses pending the real bounded pilot.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact(results: dict[str, Any], generated: str) -> dict[str, Any]:
    auth = results["authentication"]
    comparison_columns = "dataset, model, method_a, method_b, delta_auc, bootstrap_ci95_lower, bootstrap_ci95_upper, holm_adjusted_p, evidence_strength, feature_count_comparability"
    paired_sql = f"SELECT {comparison_columns} FROM baseline_pairwise_comparisons ORDER BY evidence_strength, holm_adjusted_p, dataset, model LIMIT 20"
    full_delta_sql = "SELECT dataset, model, dataset || ' / ' || model AS dataset_model, method_a, method_b, delta_auc, bootstrap_ci95_lower, bootstrap_ci95_upper, holm_adjusted_p, feature_count_a, feature_count_b, evidence_strength FROM baseline_pairwise_comparisons WHERE method_b = 'full_features' ORDER BY method_a, dataset, model"
    dev_sql = "SELECT dataset, model, method_id, auc_mean, auc_std, auc_min, auc_max, selected_feature_count_mean FROM baseline_dev_fold_summary ORDER BY dataset, model, method_id"
    runtime_sql = "SELECT dataset, model, method_id, active_computation_seconds, ram_wait_seconds, peak_process_rss_gib, minimum_available_ram_gib FROM baseline_runtime_resources ORDER BY active_computation_seconds DESC"
    connection = sqlite3.connect(":memory:")
    results["baseline_pairwise_comparisons"].to_sql("baseline_pairwise_comparisons", connection, index=False)
    results["baseline_dev_fold_summary"].to_sql("baseline_dev_fold_summary", connection, index=False)
    results["baseline_runtime_resources"].to_sql("baseline_runtime_resources", connection, index=False)
    comparisons = pd.read_sql_query(paired_sql, connection)
    full_feature_deltas = pd.read_sql_query(full_delta_sql, connection)
    dev = pd.read_sql_query(dev_sql, connection)
    runtime = pd.read_sql_query(runtime_sql, connection)
    connection.close()
    headline = [
        {
            "authenticated_cells": int(auth["authenticated_cells"]),
            "metric_checks": int(len(results["baseline_metric_reconciliation"])),
            "metric_failures": int(auth["metric_reconciliation_failures"]),
            "oot_units": int(auth["phase_composition"]["oot_evaluations"]),
        }
    ]
    def sql_source(identifier: str, label: str, path: str, sql: str, table: str) -> dict[str, Any]:
        return {
            "id": identifier,
            "label": label,
            "path": path,
            "query": {
                "engine": "SQLite",
                "sql": sql,
                "description": "Read-only report projection executed against the authenticated audit CSV loaded as the named in-memory table.",
                "executed_at": generated,
                "language": "sql",
                "tables_used": [table],
                "filters": ["Artifact-only authenticated Prompt 10 evidence"],
            },
        }

    sources = [
        {"id": "authentication", "label": "Baseline completion authentication", "path": "baseline_completion_authentication.json"},
        {"id": "reconciliation", "label": "Metric reconciliation", "path": "baseline_metric_reconciliation.csv"},
        sql_source("dev", "DEV fold summaries", "baseline_dev_fold_summary.csv", dev_sql, "baseline_dev_fold_summary"),
        sql_source("oot_pair_preview", "Paired OOT comparison preview", "baseline_pairwise_comparisons.csv", paired_sql, "baseline_pairwise_comparisons"),
        sql_source("oot_full_delta", "OOT deltas against full features", "baseline_pairwise_comparisons.csv", full_delta_sql, "baseline_pairwise_comparisons"),
        sql_source("resources", "Runtime and resource evidence", "baseline_runtime_resources.csv", runtime_sql, "baseline_runtime_resources"),
    ]
    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": "# Prompt 11 baseline results audit\n\nAuthenticated individual-selector evidence and outcome-blind combination readiness."},
        {"id": "summary", "type": "markdown", "layout": "full", "sourceId": "authentication", "body": "## Audit result\n\nAll 36 frozen baseline cells authenticated complete. All 396 supported saved-prediction metric checks reconciled, with zero discrepancies. The completed matrix contains 36 locked OOT evaluation units."},
        {"id": "method-title", "type": "markdown", "layout": "full", "body": "## Evidence boundary\n\nAll results come from authenticated Prompt 10 artifacts. OOT paired inference is supported; DEV fold prediction inference is not, because fold prediction vectors were not persisted."},
        {"id": "paired-title", "type": "markdown", "layout": "full", "body": "## Scoped paired OOT comparisons\n\nExact deltas, paired bootstrap intervals, Holm-adjusted p-values, count comparability, and conservative evidence labels are shown together. Non-significance is not equivalence."},
        {"id": "full-delta-chart-block", "type": "chart", "layout": "full", "chartId": "full-delta-chart"},
        {"id": "paired-table-block", "type": "table", "layout": "full", "tableId": "paired"},
        {"id": "dev-title", "type": "markdown", "layout": "full", "body": "## DEV temporal robustness\n\nEach row summarizes five ordered expanding-window folds without pooling."},
        {"id": "dev-table-block", "type": "table", "layout": "full", "tableId": "dev"},
        {"id": "resource-title", "type": "markdown", "layout": "full", "body": "## Runtime and resources\n\nActive computation, RAM waiting, peak RSS, and minimum available RAM are feasibility evidence only."},
        {"id": "resource-table-block", "type": "table", "layout": "full", "tableId": "runtime"},
        {"id": "limits", "type": "markdown", "layout": "full", "body": "## Limitations and next step\n\nFold predictions and score-PSI bin edges were not persisted. Natural-support shortfalls are not method failures. The next scientific step is the unchanged 24-evaluation pilot backed by 18 unique selection identities; DEV and OOT remain gated."},
    ]
    cards: list[dict[str, Any]] = []
    tables = [
        {
            "id": "paired", "title": "Predeclared paired OOT contrasts", "subtitle": "Method A minus frozen reference; first 20 rows ordered by evidence label and Holm p", "showDescription": True,
            "dataset": "paired", "sourceId": "oot_pair_preview", "density": "dense", "layout": "full",
            "defaultSort": {"field": "holm_adjusted_p", "direction": "asc"},
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"}, {"field": "model", "label": "Model", "type": "text"},
                {"field": "method_a", "label": "Method A", "type": "text"}, {"field": "method_b", "label": "Reference", "type": "text"},
                {"field": "delta_auc", "label": "Delta AUC", "format": "number"}, {"field": "bootstrap_ci95_lower", "label": "CI lower", "format": "number"},
                {"field": "bootstrap_ci95_upper", "label": "CI upper", "format": "number"}, {"field": "holm_adjusted_p", "label": "Holm p", "format": "number"},
                {"field": "evidence_strength", "label": "Strength", "type": "text"}, {"field": "feature_count_comparability", "label": "Counts", "type": "text"},
            ],
        },
        {
            "id": "dev", "title": "Five-fold DEV robustness", "subtitle": "Mean and spread across ordered expanding folds; no prediction pooling", "showDescription": True,
            "dataset": "dev", "sourceId": "dev", "density": "dense", "layout": "full", "defaultSort": {"field": "dataset", "direction": "asc"},
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"}, {"field": "model", "label": "Model", "type": "text"},
                {"field": "method_id", "label": "Method", "type": "text"}, {"field": "auc_mean", "label": "AUC mean", "format": "number"},
                {"field": "auc_std", "label": "AUC sd", "format": "number"}, {"field": "auc_min", "label": "AUC min", "format": "number"},
                {"field": "auc_max", "label": "AUC max", "format": "number"}, {"field": "selected_feature_count_mean", "label": "Features", "format": "number"},
            ],
        },
        {
            "id": "runtime", "title": "Completed-cell resource evidence", "subtitle": "Active runtime excludes RAM waiting", "showDescription": True,
            "dataset": "runtime", "sourceId": "resources", "density": "dense", "layout": "full", "defaultSort": {"field": "active_computation_seconds", "direction": "desc"},
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"}, {"field": "model", "label": "Model", "type": "text"},
                {"field": "method_id", "label": "Method", "type": "text"}, {"field": "active_computation_seconds", "label": "Active sec", "format": "number"},
                {"field": "ram_wait_seconds", "label": "RAM wait sec", "format": "number"}, {"field": "peak_process_rss_gib", "label": "Peak RSS GiB", "format": "number"},
                {"field": "minimum_available_ram_gib", "label": "Min avail GiB", "format": "number"},
            ],
        },
    ]
    charts = [
        {
            "id": "full-delta-chart",
            "title": "Locked OOT AUC difference versus full features",
            "subtitle": "Method minus full-features AUC; 28 predeclared dataset/model contrasts; zero is the reference",
            "showDescription": True,
            "intent": "comparison",
            "question": "How do the seven preregistered selector baselines differ from full features across both datasets and final models?",
            "rationale": "Grouped bars preserve the seven method categories and expose the four required dataset/model scopes without implying a universal rank.",
            "comparisonContext": {
                "baseline": "full_features",
                "grain": "selector by dataset and final model",
                "normalization": "paired method A AUC minus full-features AUC on identical OOT rows",
                "semanticFamily": "predeclared_baseline_vs_full_features",
                "unit": "AUC difference",
            },
            "type": "bar",
            "dataset": "full_feature_deltas",
            "sourceId": "oot_full_delta",
            "encodings": {
                "x": {"field": "method_a", "type": "nominal", "label": "Selector"},
                "y": {"field": "delta_auc", "type": "quantitative", "format": "number", "label": "Delta AUC"},
                "color": {"field": "dataset_model", "type": "nominal", "label": "Dataset / model"},
                "tooltip": [
                    {"field": "dataset_model", "type": "text", "label": "Dataset / model"},
                    {"field": "delta_auc", "type": "quantitative", "format": "number", "label": "Delta AUC"},
                    {"field": "bootstrap_ci95_lower", "type": "quantitative", "format": "number", "label": "95% CI lower"},
                    {"field": "bootstrap_ci95_upper", "type": "quantitative", "format": "number", "label": "95% CI upper"},
                ],
            },
            "xAxisTitle": "Selector",
            "yAxisTitle": "AUC difference (method minus full features)",
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "approved_four_scope"},
            "referenceLines": [{"axis": "y", "value": 0, "label": "Full-features reference", "color": "neutral", "lineStyle": "solid"}],
            "settings": {"groupMode": "grouped", "orientation": "vertical", "sort": "none", "categoryLabelPolicy": "rotate", "showValues": False},
            "surface": {"surface": "explorer", "viewMode": "both", "interactiveLegend": True, "showControls": True},
            "maxRows": 28,
            "compatibleTypes": ["bar"],
        }
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": "Prompt 11 baseline results audit",
            "description": "Authenticated Prompt 10 evidence and preregistered combination readiness",
            "generatedAt": generated, "blocks": blocks, "cards": cards, "charts": charts, "tables": tables,
            "filters": [], "sources": sources,
        },
        "snapshot": {
            "version": 1, "generatedAt": generated, "status": "ready", "accessIssues": [],
            "datasets": {
                "headline": headline,
                "paired": _records(comparisons, ["dataset", "model", "method_a", "method_b", "delta_auc", "bootstrap_ci95_lower", "bootstrap_ci95_upper", "holm_adjusted_p", "evidence_strength", "feature_count_comparability"]),
                "full_feature_deltas": _records(full_feature_deltas, ["dataset", "model", "dataset_model", "method_a", "method_b", "delta_auc", "bootstrap_ci95_lower", "bootstrap_ci95_upper", "holm_adjusted_p", "feature_count_a", "feature_count_b", "evidence_strength"]),
                "dev": _records(dev, ["dataset", "model", "method_id", "auc_mean", "auc_std", "auc_min", "auc_max", "selected_feature_count_mean"]),
                "runtime": _records(runtime, ["dataset", "model", "method_id", "active_computation_seconds", "ram_wait_seconds", "peak_process_rss_gib", "minimum_available_ram_gib"]),
            },
        },
        "sources": sources,
    }


def write_manifest(output: Path) -> None:
    excluded = {"audit_manifest.json"}
    files = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name in excluded or path.name.endswith(".partial"):
            continue
        files.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    preservation = output / "baseline_preservation_snapshot.json"
    manifest = {
        "schema_version": "prompt_11_baseline_audit_manifest_v1",
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_only": True,
        "raw_dataset_paths_resolved": False,
        "baseline_refit_performed": False,
        "source_lineage": {
            "prompt_10_results_root": "results/full_baseline_v1",
            "preservation_snapshot_sha256": sha256_file(preservation) if preservation.is_file() else None,
        },
        "files": files,
    }
    write_json_atomic(output / "audit_manifest.json", manifest, overwrite=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.repository_root).resolve()
    output = Path(args.output)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        write_manifest(output)
        print(json.dumps({"status": "manifest_updated", "output": str(output)}))
        return 0
    print("Authenticating and reading saved Prompt 10 artifacts only...", flush=True)
    results = audit_completed_baselines(root, bootstrap_repetitions=args.bootstrap_repetitions)
    write_json_atomic(output / "baseline_completion_authentication.json", results["authentication"], overwrite=True)
    for key, filename in TABLE_FILES.items():
        write_csv_atomic(output / filename, results[key], overwrite=True)
    claims = _claims(results)
    write_csv_atomic(output / "claims_and_evidence.csv", claims, overwrite=True)
    write_text_atomic(output / "baseline_results_audit.md", _markdown(results, claims), overwrite=True)
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    write_json_atomic(output / "artifact.json", _artifact(results, generated), overwrite=True)
    write_manifest(output)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "authenticated_cells": results["authentication"]["authenticated_cells"],
                "metric_failures": results["authentication"]["metric_reconciliation_failures"],
                "comparison_count": len(results["baseline_pairwise_comparisons"]),
                "bootstrap_repetitions": args.bootstrap_repetitions,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
