"""Build the canonical artifact manifest without copying scientific payloads."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CANONICAL_ROOTS = {
    "results/corrected_homecredit_clip": (
        "canonical_active",
        "corrected Home Credit CLIP and forward-transfer evidence",
        "results/corrected_homecredit_clip/task_manifest.json",
    ),
    "results/corrected_lendingclub_to_homecredit_transfer": (
        "canonical_active",
        "corrected reverse-transfer evidence",
        "results/corrected_lendingclub_to_homecredit_transfer/manifests/"
        "register_stage_manifest.json",
    ),
    "results/final_research_package_v2": (
        "canonical_active",
        "current final report package",
        "results/final_research_package_v2/final_package_manifest.json",
    ),
    "results/research_summary": (
        "canonical_active",
        "central registry and compact migration evidence",
        "results/research_summary/summary_manifest.json",
    ),
}
PENDING_TOKENS = (
    "cv_results",
    "fold",
    "selected_feature",
    "selection_frequency",
    "psi",
    "drift",
    "llm",
    "runtime",
    "token",
    "prompt",
    "response",
    "data_split_manifest",
    "semantic_group",
    "feature_metadata",
    "paired_",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path.cwd().resolve()
    destination = root / "results" / "finalized_research"
    registry_root = root / "results" / "research_summary"
    artifacts = pd.read_csv(registry_root / "artifact_registry.csv")
    runs = pd.read_csv(registry_root / "run_index.csv").set_index("run_id")
    registry_by_path = {
        str(row.relative_path).replace("\\", "/"): row
        for row in artifacts.itertuples(index=False)
        if str(row.file_exists).lower() == "true"
    }

    selected: dict[str, tuple[str, str, str]] = {}
    for relative_root, metadata in CANONICAL_ROOTS.items():
        for path in (root / relative_root).rglob("*"):
            if path.is_file():
                selected[path.relative_to(root).as_posix()] = metadata
    for relative in registry_by_path:
        path = root / relative
        if path.is_file():
            selected.setdefault(
                relative,
                (
                    "canonical_active",
                    "active central-registry artifact",
                    "results/research_summary/summary_manifest.json",
                ),
            )
    for base in (
        root / "results" / "homecredit",
        root / "results" / "lendingclub_v2",
        root / "results" / "cross_dataset_v2",
        root / "artifacts" / "llm_cache",
    ):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            lower = relative.lower()
            if any(token in lower for token in PENDING_TOKENS):
                selected.setdefault(
                    relative,
                    (
                        "required_pending_analysis_input",
                        "retained input for significance, feature drift, or LLM cost analysis",
                        "",
                    ),
                )

    rows = []
    for relative, (status, purpose, source_manifest) in sorted(selected.items()):
        path = root / relative
        registry_row = registry_by_path.get(relative)
        if registry_row is not None:
            purpose = str(registry_row.human_description) or purpose
            status = str(registry_row.reuse_status)
            owner = str(registry_row.created_by_run_id or "").strip()
            if owner and owner in runs.index:
                run_manifest = runs.loc[owner].get("manifest_path")
                if pd.notna(run_manifest) and str(run_manifest).strip():
                    source_manifest = str(run_manifest).replace("\\", "/")
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "purpose": purpose,
                "status": status,
                "source_manifest": source_manifest,
            }
        )

    inventory_path = destination / "canonical_artifact_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "manifest_version": "canonical_artifact_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(rows),
        "payload_policy": (
            "large scientific payloads remain at authenticated repository-relative paths"
        ),
        "canonical_entry_point": "results/finalized_research/README.md",
        "inventory_path": inventory_path.relative_to(root).as_posix(),
        "artifacts": rows,
    }
    manifest_path = destination / "canonical_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact_count": len(rows),
                "inventory": inventory_path.relative_to(root).as_posix(),
                "manifest": manifest_path.relative_to(root).as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
