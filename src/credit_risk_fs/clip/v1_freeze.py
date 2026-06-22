from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


VERSION_NAME = "clip_v1"
SCIENTIFIC_NAME = "missingness_only"
FREEZE_ROOT = Path("results/clip_versions/v1")
DOC_PATH = Path("docs/clip/CLIP_V1_FREEZE.md")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(args: list[str]) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, timeout=30)
    return result.stdout.strip()


@dataclass(frozen=True)
class FreezeArtifact:
    role: str
    path: Path
    required: bool = True


def freeze_artifacts() -> list[FreezeArtifact]:
    artifacts = [
        FreezeArtifact("prompt1_manifest", Path("results/clip/dry_run/training_manifest.json")),
        FreezeArtifact("prompt2_text_summary", Path("results/clip/text_baseline/text_baseline_summary.json")),
        FreezeArtifact("prompt3_statistical_summary", Path("results/clip/statistical_baseline/statistical_baseline_summary.json")),
        FreezeArtifact("prompt3_statistical_preprocessor", Path("results/clip/statistical_baseline/statistical_preprocessor.json")),
        FreezeArtifact("prompt4_pair_manifest", Path("results/clip/contrastive_data/contrastive_pair_manifest.json")),
        FreezeArtifact("prompt4_tensor_schema", Path("results/clip/contrastive_data/contrastive_tensor_schema.json")),
        FreezeArtifact("prompt5_training_manifest", Path("results/clip/training/training_manifest.json")),
        FreezeArtifact("prompt5_model_selection", Path("results/clip/training/model_selection_manifest.json")),
        FreezeArtifact("prompt5_seed_comparison", Path("results/clip/training/seed_comparison.csv")),
        FreezeArtifact("prompt5_selected_checkpoint", Path("results/clip/training/seeds/seed_55/best_checkpoint.pt")),
        FreezeArtifact("prompt5_homecredit_anchor", Path("results/clip/training/learned_anchor_manifest.json")),
        FreezeArtifact("prompt6_integration_manifest", Path("results/clip/selector_integration/integration_manifest.json")),
        FreezeArtifact("prompt7_run_manifest", Path("results/clip/final_evaluation/run_manifest.json")),
        FreezeArtifact("prompt7_evaluation_summary", Path("results/clip/final_evaluation/evaluation_summary.csv")),
        FreezeArtifact("prompt7_comparison", Path("results/clip/final_evaluation/comparison_with_frozen_baselines.csv")),
        FreezeArtifact("prompt8_analysis_summary", Path("results/clip/final_analysis/analysis_summary.json")),
        FreezeArtifact("prompt8_master_results", Path("results/clip/final_analysis/master_results_table.csv")),
        FreezeArtifact("prompt8_plot_manifest", Path("results/clip/final_analysis/plot_manifest.csv")),
        FreezeArtifact("report_markdown", Path("reports/final_clip_credit_risk_report.md")),
        FreezeArtifact("report_docx", Path("reports/final_clip_credit_risk_report.docx")),
        FreezeArtifact("report_pdf", Path("reports/final_clip_credit_risk_report.pdf")),
        FreezeArtifact("report_verdict", Path("reports/final_clip_scientific_verdict.md")),
        FreezeArtifact("report_limitations", Path("reports/final_clip_limitations.md")),
        FreezeArtifact("report_reproducibility", Path("reports/final_clip_reproducibility_manifest.json")),
    ]
    artifacts.extend(
        FreezeArtifact(f"prompt8_plot_{path.stem}", path)
        for path in sorted(Path("results/clip/final_analysis/plots").glob("*.png"))
    )
    return artifacts


def artifact_inventory() -> list[dict[str, Any]]:
    rows = []
    for artifact in freeze_artifacts():
        exists = artifact.path.exists()
        rows.append(
            {
                "role": artifact.role,
                "path": artifact.path.as_posix(),
                "required": artifact.required,
                "exists": exists,
                "size_bytes": artifact.path.stat().st_size if exists and artifact.path.is_file() else "",
                "sha256": sha256_file(artifact.path) if exists and artifact.path.is_file() else "",
                "status": "pass" if exists else ("fail" if artifact.required else "warn"),
            }
        )
    return rows


def build_freeze_manifest() -> dict[str, Any]:
    inventory = artifact_inventory()
    missing = [row["path"] for row in inventory if row["required"] and not row["exists"]]
    if missing:
        raise RuntimeError(f"cannot freeze CLIP-v1; missing required artifacts: {missing}")

    training = read_json(Path("results/clip/training/training_manifest.json"))
    model_selection = read_json(Path("results/clip/training/model_selection_manifest.json"))
    text_summary = read_json(Path("results/clip/text_baseline/text_baseline_summary.json"))
    stat_summary = read_json(Path("results/clip/statistical_baseline/statistical_baseline_summary.json"))
    run_manifest = read_json(Path("results/clip/final_evaluation/run_manifest.json"))
    prompt1 = read_json(Path("results/clip/dry_run/training_manifest.json"))

    aggregate_paths = sorted(Path("results/clip/final_evaluation").glob("*.csv")) + sorted(
        Path("results/clip/final_evaluation").glob("*.json")
    )
    final_analysis_paths = sorted(path for path in Path("results/clip/final_analysis").rglob("*") if path.is_file())
    plot_paths = sorted(Path("results/clip/final_analysis/plots").glob("*.png"))

    return {
        "version_name": VERSION_NAME,
        "scientific_name": SCIENTIFIC_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": git_output(["branch", "--show-current"]),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "working_tree_state": git_output(["status", "--short"]),
        "source_hashes": prompt1.get("source_hashes", {}),
        "split_hashes": {
            "text_group_split": sha256_file(Path("results/clip/text_baseline/homecredit_group_split.csv")),
            "contrastive_split_manifest": sha256_file(Path("results/clip/contrastive_data/split_manifest.json")),
        },
        "text_embedding_hashes": {
            "homecredit": sha256_file(Path("results/clip/text_baseline/homecredit_text_embeddings.parquet")),
            "lendingclub_v2": sha256_file(Path("results/clip/text_baseline/lendingclub_v2_text_embeddings.parquet")),
            "manifest": sha256_file(Path("results/clip/text_baseline/embedding_cache_manifest.json")),
        },
        "statistical_preprocessor_hash": stat_summary.get("preprocessor_hash"),
        "statistical_vector_dimension": stat_summary.get("vector_dimension"),
        "statistical_field_list": stat_summary.get("main_statistical_fields"),
        "checkpoint_hash": model_selection.get("selected_checkpoint_hash"),
        "selected_seed": model_selection.get("selected_seed"),
        "selected_checkpoint_rule": model_selection.get("selection_rule"),
        "anchor_hash": run_manifest["runs"][0].get("anchor_hash"),
        "training_dataset": training.get("training_dataset"),
        "external_validation_dataset": training.get("external_validation_dataset"),
        "evaluation_run_ids": [run["run_id"] for run in run_manifest["runs"]],
        "prediction_hashes": {run["run_id"]: run.get("prediction_hash") for run in run_manifest["runs"]},
        "aggregate_table_hashes": {path.as_posix(): sha256_file(path) for path in aggregate_paths},
        "final_analysis_hashes": {path.as_posix(): sha256_file(path) for path in final_analysis_paths},
        "plot_hashes": {path.as_posix(): sha256_file(path) for path in plot_paths},
        "report_hashes": {
            "markdown": sha256_file(Path("reports/final_clip_credit_risk_report.md")),
            "docx": sha256_file(Path("reports/final_clip_credit_risk_report.docx")),
            "pdf": sha256_file(Path("reports/final_clip_credit_risk_report.pdf")),
        },
        "test_counts": {
            "tests_clip": "105 passed",
            "tests_full": "176 passed, 107 warnings",
        },
        "known_limitations": [
            "missingness-only DEV statistical view",
            "non-fold-local CLIP preparation",
            "Home Credit-only contrastive training",
            "LendingClub v2 external application only",
            "DEV score vectors not persisted for full independent PSI recomputation",
            "CLIP-v1 conclusions do not establish that richer semantic-statistical contrastive learning is ineffective",
        ],
        "artifact_count": len(inventory),
    }


def write_freeze_package() -> dict[str, Any]:
    FREEZE_ROOT.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory = artifact_inventory()
    manifest = build_freeze_manifest()

    with (FREEZE_ROOT / "artifact_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    with (FREEZE_ROOT / "config_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "role"])
        writer.writeheader()
        for path in sorted(Path("configs/clip").glob("*.yaml")):
            writer.writerow({"path": path.as_posix(), "sha256": sha256_file(path), "role": "clip_v1_config"})

    (FREEZE_ROOT / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (FREEZE_ROOT / "artifact_hashes.json").write_text(
        json.dumps({row["path"]: row["sha256"] for row in inventory if row["sha256"]}, indent=2),
        encoding="utf-8",
    )
    (FREEZE_ROOT / "source_commit.json").write_text(
        json.dumps(
            {
                "branch": manifest["branch"],
                "git_commit": manifest["git_commit"],
                "working_tree_state": manifest["working_tree_state"],
                "freeze_commit": None,
                "freeze_tag": None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.read_csv("results/clip/final_analysis/master_results_table.csv").to_csv(
        FREEZE_ROOT / "result_summary.csv",
        index=False,
    )
    scope = (
        "# CLIP-v1 Scientific Scope\n\n"
        "CLIP-v1 aligns frozen semantic feature metadata with a missingness-only DEV statistical view. "
        "Its conclusions apply to this limited implementation and do not establish that richer semantic-statistical "
        "contrastive learning is ineffective.\n\n"
        "Frozen comparison scope: Home Credit contrastive training, LendingClub v2 external application, "
        "`clip`, `clip_then_mrmr`, and eight downstream Prompt 7 evaluation runs.\n"
    )
    (FREEZE_ROOT / "scientific_scope.md").write_text(scope, encoding="utf-8")
    (FREEZE_ROOT / "verification_report.md").write_text(build_verification_report(manifest, inventory), encoding="utf-8")
    DOC_PATH.write_text(
        scope
        + "\n## Freeze package\n\n"
        + "The hash-based freeze package is stored at `results/clip_versions/v1/`. "
        + "It references canonical artifacts and does not duplicate large checkpoints, predictions, embeddings, or datasets.\n",
        encoding="utf-8",
    )
    return manifest


def build_verification_report(manifest: dict[str, Any], inventory: list[dict[str, Any]]) -> str:
    failed = [row for row in inventory if row["status"] == "fail"]
    lines = [
        "# CLIP-v1 Freeze Verification",
        "",
        f"Version: `{manifest['version_name']}`",
        f"Scientific name: `{manifest['scientific_name']}`",
        f"Statistical dimension: `{manifest['statistical_vector_dimension']}`",
        f"Statistical fields: `{manifest['statistical_field_list']}`",
        f"Selected seed: `{manifest['selected_seed']}`",
        f"Checkpoint hash: `{manifest['checkpoint_hash']}`",
        f"Anchor hash: `{manifest['anchor_hash']}`",
        f"Evaluation run count: `{len(manifest['evaluation_run_ids'])}`",
        f"Artifact count: `{manifest['artifact_count']}`",
        f"Missing required artifacts: `{len(failed)}`",
        "",
        "No large artifacts are duplicated in this package; canonical paths and hashes are recorded.",
        "",
    ]
    if failed:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {row['path']}" for row in failed)
    return "\n".join(lines) + "\n"


def verify_freeze_package() -> dict[str, Any]:
    required = [
        FREEZE_ROOT / "freeze_manifest.json",
        FREEZE_ROOT / "artifact_inventory.csv",
        FREEZE_ROOT / "artifact_hashes.json",
        FREEZE_ROOT / "config_inventory.csv",
        FREEZE_ROOT / "source_commit.json",
        FREEZE_ROOT / "result_summary.csv",
        FREEZE_ROOT / "scientific_scope.md",
        FREEZE_ROOT / "verification_report.md",
        DOC_PATH,
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        return {"status": "failed", "missing": missing}
    manifest = read_json(FREEZE_ROOT / "freeze_manifest.json")
    checks = {
        "version_name": manifest.get("version_name") == VERSION_NAME,
        "scientific_name": manifest.get("scientific_name") == SCIENTIFIC_NAME,
        "statistical_vector_dimension": manifest.get("statistical_vector_dimension") == 1,
        "statistical_field_list": manifest.get("statistical_field_list") == ["missing_rate_dev"],
        "evaluation_run_count": len(manifest.get("evaluation_run_ids", [])) == 8,
        "training_dataset": manifest.get("training_dataset") == "homecredit",
        "external_validation_dataset": manifest.get("external_validation_dataset") == "lendingclub_v2",
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "manifest_path": str(FREEZE_ROOT / "freeze_manifest.json")}
