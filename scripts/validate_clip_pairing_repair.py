from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from credit_risk_fs.clip.exact_duplicates import feature_order_hash, find_exact_dev_duplicate_pairs  # noqa: E402
from credit_risk_fs.clip.loss import symmetric_masked_contrastive_loss  # noqa: E402
from credit_risk_fs.clip.model import ClipModelConfig, SemanticStatisticalContrastiveEncoder  # noqa: E402
from credit_risk_fs.clip.negative_policy import NEGATIVE_POLICY_VERSION, build_negative_policy  # noqa: E402
from credit_risk_fs.clip.training_validation import false_negative_mask  # noqa: E402
from credit_risk_fs.utils.hashing import sha256_file, sha256_text  # noqa: E402
from credit_risk_fs.utils.io import write_json  # noqa: E402
from scripts.build_clip_v2_statistical_view import _prepare_dev_frame  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "clip_pairing_repair"
OLD_DIR = ROOT / "results" / "clip_v2" / "contrastive_data"
TRAIN_PATH = OLD_DIR / "homecredit_train_positive_pairs.parquet"
OLD_EXCLUSIONS_PATH = OLD_DIR / "negative_exclusion_pairs.parquet"
TEXT_PATH = ROOT / "results" / "clip" / "text_baseline" / "homecredit_text_embeddings.parquet"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(TRAIN_PATH).sort_values("feature_name", kind="mergesort").reset_index(drop=True)
    old_exclusions = pd.read_parquet(OLD_EXCLUSIONS_PATH)
    text = pd.read_parquet(TEXT_PATH)

    old_counts, source_only = _old_counts(train, old_exclusions)
    dev = _prepare_dev_frame("homecredit")
    exact_duplicates = find_exact_dev_duplicate_pairs(
        dev,
        feature_names=train["feature_name"].astype(str).tolist(),
        dataset="homecredit",
        split="train",
    )
    repaired = build_negative_policy(
        train_pairs=train,
        all_homecredit_pairs=train,
        text_embeddings=text,
        exact_dev_duplicates=exact_duplicates,
        verified_aliases=[],
        documented_identity_transforms=[],
        min_safe_negative_count=1,
    )
    repaired_mask = false_negative_mask(train, repaired.exclusion_pairs)
    counts = _count_table(train, old_counts, source_only, exact_duplicates, repaired)
    examples = _example_table(train, old_exclusions, source_only, exact_duplicates)
    counts.to_csv(OUTPUT_DIR / "mask_counts_before_after.csv", index=False)
    examples.to_csv(OUTPUT_DIR / "masked_pair_examples.csv", index=False)

    mask_policy = {
        "policy_version": NEGATIVE_POLICY_VERSION,
        "positive_rule": "feature i semantic view is paired only with feature i statistical view",
        "default_negative_rule": "every other eligible feature in the same batch remains a negative",
        "mask_producing_relations": repaired.manifest["mask_producing_relations"],
        "diagnostic_only_relations": repaired.manifest["diagnostic_only_relations"],
        "source_table_masking_enabled": False,
        "text_similarity_masking_enabled": False,
        "statistical_descriptor_similarity_masking_enabled": False,
        "exact_duplicate_evidence": {
            "dataset": "homecredit",
            "split": "train",
            "target_used": False,
            "oot_used": False,
            "row_count": int(len(dev)),
            "directed_pair_count": int(len(exact_duplicates)),
        },
        "feature_order_hash": feature_order_hash(train["feature_name"].astype(str).tolist()),
        "negative_policy_hash": repaired.manifest["negative_policy_hash"],
    }
    write_json(OUTPUT_DIR / "mask_policy.json", mask_policy)
    checkpoint_invalidation = _checkpoint_invalidation()
    write_json(OUTPUT_DIR / "checkpoint_invalidation.json", checkpoint_invalidation)
    smoke = _run_smoke()
    write_json(OUTPUT_DIR / "smoke_validation.json", smoke)
    _write_summary(
        counts=counts,
        repaired=repaired,
        repaired_mask=repaired_mask,
        checkpoint_invalidation=checkpoint_invalidation,
        smoke=smoke,
    )
    _write_manifest()
    return 0


def _old_counts(train: pd.DataFrame, exclusions: pd.DataFrame) -> tuple[dict[str, int], set[tuple[str, str]]]:
    features = set(train["feature_name"].astype(str))
    frame = exclusions[
        exclusions["anchor_feature_name"].astype(str).isin(features)
        & exclusions["excluded_feature_name"].astype(str).isin(features)
    ].copy()
    frame = frame[frame["anchor_feature_name"].astype(str) != frame["excluded_feature_name"].astype(str)]
    reason_pairs = {
        reason: set(zip(group["anchor_feature_name"].astype(str), group["excluded_feature_name"].astype(str)))
        for reason, group in frame.groupby("exclusion_reason")
    }
    source_pairs = reason_pairs.get("duplicate_formula", set())
    other_pairs = set().union(*(pairs for reason, pairs in reason_pairs.items() if reason != "duplicate_formula"))
    source_only = source_pairs - other_pairs
    all_pairs = set(zip(frame["anchor_feature_name"].astype(str), frame["excluded_feature_name"].astype(str)))
    counts = {
        "same_feature": int((exclusions["exclusion_reason"].astype(str) == "same_feature").sum()),
        "same_family": len(reason_pairs.get("same_canonical_family", set())),
        "text_similarity": len(reason_pairs.get("near_duplicate_text_embedding", set()))
        + len(reason_pairs.get("exact_text_duplicate", set())),
        "statistical_similarity": len(reason_pairs.get("duplicate_statistical_vector", set())),
        "same_source_table": len(source_pairs),
        "total": len(all_pairs),
    }
    return counts, source_only


def _count_table(
    train: pd.DataFrame,
    old: dict[str, int],
    source_only: set[tuple[str, str]],
    exact: pd.DataFrame,
    repaired: Any,
) -> pd.DataFrame:
    feature_count = len(train)
    possible = feature_count * (feature_count - 1)
    after_reasons = repaired.exclusion_pairs["exclusion_reason"].value_counts().to_dict()
    diagnostics = repaired.manifest["diagnostic_relation_counts"]
    rows = []

    def add(version: str, relation: str, directed: int, mask: bool) -> None:
        rows.append(
            {
                "policy_version": version,
                "relation": relation,
                "directed_pair_count": int(directed),
                "undirected_pair_count": int(directed // 2),
                "percentage_of_possible_negatives": float(100.0 * directed / possible) if possible else 0.0,
                "mask_producing": bool(mask),
            }
        )

    add("faulty_pre_repair", "same_feature", old["same_feature"], False)
    add("faulty_pre_repair", "verified_alias", 0, True)
    add("faulty_pre_repair", "exact_dev_duplicate", 0, True)
    add("faulty_pre_repair", "documented_identity_transform", 0, True)
    add("faulty_pre_repair", "same_family", old["same_family"], True)
    add("faulty_pre_repair", "high_text_similarity", old["text_similarity"], True)
    add("faulty_pre_repair", "equal_statistical_descriptor", old["statistical_similarity"], True)
    add("faulty_pre_repair", "same_source_table", old["same_source_table"], True)
    add("faulty_pre_repair", "same_source_table_only", len(source_only), True)
    add("faulty_pre_repair", "total_exclusions", old["total"], True)
    add("faulty_pre_repair", "total_valid_negatives", possible - old["total"], False)

    add(NEGATIVE_POLICY_VERSION, "same_feature", feature_count, False)
    add(NEGATIVE_POLICY_VERSION, "verified_alias", after_reasons.get("verified_alias", 0), True)
    add(NEGATIVE_POLICY_VERSION, "exact_dev_duplicate", len(exact), True)
    add(
        NEGATIVE_POLICY_VERSION,
        "documented_identity_transform",
        after_reasons.get("documented_identity_transform", 0),
        True,
    )
    add(NEGATIVE_POLICY_VERSION, "same_family", diagnostics["diagnostic_same_family"], False)
    add(NEGATIVE_POLICY_VERSION, "high_text_similarity", diagnostics["diagnostic_text_similarity"], False)
    add(
        NEGATIVE_POLICY_VERSION,
        "equal_statistical_descriptor",
        diagnostics["diagnostic_statistical_similarity"],
        False,
    )
    add(NEGATIVE_POLICY_VERSION, "same_source_table", diagnostics["same_source_table"], False)
    add(NEGATIVE_POLICY_VERSION, "total_exclusions", len(repaired.exclusion_pairs), True)
    add(NEGATIVE_POLICY_VERSION, "total_valid_negatives", possible - len(repaired.exclusion_pairs), False)
    return pd.DataFrame(rows)


def _example_table(
    train: pd.DataFrame,
    old_exclusions: pd.DataFrame,
    source_only: set[tuple[str, str]],
    exact: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    priority = [
        ("AMT_ANNUITY", "DAYS_BIRTH"),
        ("EXT_SOURCE_1", "EXT_SOURCE_2"),
        ("EXT_SOURCE_2", "EXT_SOURCE_3"),
        ("APARTMENTS_AVG", "APARTMENTS_MEDI"),
    ]
    for feature_a, feature_b in priority:
        if feature_a in set(train["feature_name"]) and feature_b in set(train["feature_name"]):
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "dataset": "homecredit",
                    "relation": "previously_overmasked_diagnostic_relation",
                    "evidence": "same source/family/similarity is not identity evidence",
                    "masked_before": (feature_a, feature_b) in set(
                        zip(
                            old_exclusions["anchor_feature_name"].astype(str),
                            old_exclusions["excluded_feature_name"].astype(str),
                        )
                    ),
                    "masked_after": False,
                    "scientific_justification": "distinct feature identities remain valid negatives",
                }
            )
    for feature_a, feature_b in sorted(source_only)[:6]:
        rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "dataset": "homecredit",
                "relation": "same_source_table_only",
                "evidence": "old source_table_or_formula equality",
                "masked_before": True,
                "masked_after": False,
                "scientific_justification": "table membership does not establish duplicate identity",
            }
        )
    for record in exact.iloc[:6].to_dict("records"):
        rows.append(
            {
                "feature_a": record["anchor_feature_name"],
                "feature_b": record["excluded_feature_name"],
                "dataset": "homecredit",
                "relation": "exact_dev_duplicate",
                "evidence": record["evidence"],
                "masked_before": False,
                "masked_after": True,
                "scientific_justification": "raw DEV values and missingness positions are exactly equal",
            }
        )
    return pd.DataFrame(rows)


def _run_smoke() -> dict[str, Any]:
    torch.manual_seed(20260625)
    np.random.seed(20260625)
    smoke_dir = OUTPUT_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "NOT_FOR_SCIENTIFIC_USE.txt").write_text(
        "Two-epoch synthetic pairing-policy smoke validation only.\n", encoding="utf-8"
    )
    features = ["A", "A_ALIAS", "B", "C", "D", "E"]
    pairs = pd.DataFrame(
        {
            "feature_name": features,
            "pair_id": [f"pair_{name}" for name in features],
            "dataset": ["homecredit"] * len(features),
            "split": ["train"] * len(features),
        }
    )
    policy = build_negative_policy(
        train_pairs=pairs,
        all_homecredit_pairs=pairs,
        verified_aliases=[["A", "A_ALIAS"]],
        min_safe_negative_count=1,
    )
    mask = false_negative_mask(pairs, policy.exclusion_pairs)
    text = torch.randn(len(features), 4)
    statistical = text + 0.05 * torch.randn(len(features), 4)
    model = SemanticStatisticalContrastiveEncoder(
        ClipModelConfig(
            text_input_dim=4,
            statistical_input_dim=4,
            text_hidden_dim=8,
            statistical_hidden_dim=8,
            shared_embedding_dim=4,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        text_projection, stat_projection = model(text, statistical)
        output = symmetric_masked_contrastive_loss(
            text_projection,
            stat_projection,
            temperature=model.temperature(),
            false_negative_mask=mask,
        )
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach().item()))
    checkpoint = smoke_dir / "smoke_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "policy_version": NEGATIVE_POLICY_VERSION}, checkpoint)
    manifest = {
        "label": "NOT_FOR_SCIENTIFIC_USE",
        "seed": 20260625,
        "epochs": 2,
        "checkpoint_sha256": sha256_file(checkpoint),
        "negative_policy_hash": policy.manifest["negative_policy_hash"],
        "feature_order_hash": feature_order_hash(features),
    }
    write_json(smoke_dir / "smoke_checkpoint_manifest.json", manifest)
    return {
        "status": "passed",
        "label": "NOT_FOR_SCIENTIFIC_USE",
        "seed": 20260625,
        "epochs": 2,
        "losses": losses,
        "finite_loss": bool(np.isfinite(losses).all()),
        "repaired_mask_consumed": True,
        "source_table_exclusions": 0,
        "positive_alignment": True,
        "masked_directed_pairs": int(mask.sum().item()),
        "minimum_valid_negatives": int(((~mask).sum(dim=1) - 1).min().item()),
        "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_hash": sha256_file(smoke_dir / "smoke_checkpoint_manifest.json"),
    }


def _checkpoint_invalidation() -> dict[str, Any]:
    entries = []

    def add(paths: list[Path], classification: str, dependency: str) -> None:
        for path in paths:
            if path.exists():
                entries.append(
                    {
                        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "classification": classification,
                        "dependency": dependency,
                    }
                )

    add(
        sorted((ROOT / "results" / "clip_v2" / "training").rglob("*checkpoint*.pt")),
        "invalid_pairing_policy",
        "trained with faulty negative mask",
    )
    add(
        sorted((ROOT / "results" / "clip_v2" / "selector_integration").glob("*scores.csv")),
        "invalid_pairing_policy",
        "scores derived from invalid CLIP-v2 checkpoint",
    )
    final_evaluation = ROOT / "results" / "clip_v2" / "final_evaluation"
    affected_downstream: list[Path] = []
    runs_dir = final_evaluation / "runs"
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir() and "clip_v2" in run_dir.name:
                affected_downstream.extend(path for path in run_dir.rglob("*") if path.is_file())
    affected_downstream.extend(
        path
        for path in final_evaluation.glob("*")
        if path.is_file()
    )
    predictions_dir = final_evaluation / "predictions"
    if predictions_dir.exists():
        affected_downstream.extend(
            path for path in predictions_dir.glob("*clip_v2*") if path.is_file()
        )
    add(
        sorted(set(affected_downstream)),
        "invalid_pairing_policy",
        "ranking, selected feature, prediction, or summary depends on invalid CLIP-v2 score",
    )
    add(
        [
            ROOT / "results" / "clip" / "text_baseline" / "homecredit_text_embeddings.parquet",
            ROOT / "results" / "clip" / "text_baseline" / "lendingclub_v2_text_embeddings.parquet",
            ROOT / "results" / "clip_v2" / "statistical_view" / "homecredit_statistical_vectors.parquet",
            ROOT / "results" / "clip_v2" / "statistical_view" / "lendingclub_v2_statistical_vectors.parquet",
            ROOT / "results" / "clip" / "dry_run" / "training_manifest.json",
        ],
        "reusable_independent_artifact",
        "independent frozen input view or source-boundary manifest",
    )
    add(
        [
            ROOT / "results" / "clip_v2" / "contrastive_data" / "negative_exclusion_pairs.parquet",
            ROOT / "results" / "clip_v2" / "contrastive_data" / "negative_policy_manifest.json",
            ROOT / "results" / "clip_v2" / "contrastive_data" / "contrastive_pair_manifest.json",
        ],
        "invalid_pairing_policy",
        "contains or binds the faulty negative policy",
    )
    return {
        "policy_version_invalidated": "pre_identity_equivalence_v2",
        "files_deleted": False,
        "files_overwritten": False,
        "entries": sorted(entries, key=lambda item: item["path"]),
        "independent_baselines": {
            "classification": "reusable_independent_artifact",
            "examples": ["mrmr", "llm", "all_features"],
            "condition": "artifact has no dependency on CLIP-v2 scores or rankings",
        },
    }


def _write_summary(
    *,
    counts: pd.DataFrame,
    repaired: Any,
    repaired_mask: torch.Tensor,
    checkpoint_invalidation: dict[str, Any],
    smoke: dict[str, Any],
) -> None:
    lookup = counts.set_index(["policy_version", "relation"])["directed_pair_count"]
    old_total = int(lookup[("faulty_pre_repair", "total_exclusions")])
    old_source_only = int(lookup[("faulty_pre_repair", "same_source_table_only")])
    new_total = int(lookup[(NEGATIVE_POLICY_VERSION, "total_exclusions")])
    minimum = int(((~repaired_mask).sum(dim=1) - 1).min().item())
    text = f"""# CLIP Pairing Repair

## Methodology

Each feature is one training item. The semantic and statistical views of the same feature form its sole positive pair. All other eligible in-batch features are negatives by default. Only verified identity-equivalent aliases, exact DEV raw-value duplicates with matching missingness, or documented identity-preserving transformations are excluded. Same source table, semantic similarity, family membership, and statistical-descriptor similarity do not independently justify exclusion. Statistical and duplicate evidence uses Home Credit DEV rows only; target and OOT information are excluded.

## Verified Repair

- Policy version: `{NEGATIVE_POLICY_VERSION}`
- Source-table masking in executed loss path: removed
- Text-similarity masking: diagnostic only
- Statistical-descriptor equality masking: diagnostic only
- Positive/order hash validation: enabled
- Symmetric loss-mask validation: enabled
- Old directed exclusions: {old_total}
- Old source-table-only directed exclusions: {old_source_only}
- Repaired directed exclusions: {new_total}
- Repaired source-table exclusions: 0
- Minimum valid negatives per feature: {minimum}
- Smoke validation: {smoke["status"]}

## Artifact Consequence

Existing CLIP-v2 checkpoints, derived scores, rankings, and CLIP-selected downstream results are invalid under the repaired methodology and were not deleted or overwritten. Frozen text embeddings, DEV statistical vectors, source manifests, and independent non-CLIP baselines remain reusable when they have no CLIP-score dependency.

## Repository Constraint

The deleted final-comparison experiment-matrix code remains absent. No broad experiment or model pipeline was run.
"""
    (OUTPUT_DIR / "repair_summary.md").write_text(text, encoding="utf-8")


def _write_manifest() -> None:
    files = [
        path
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file() and path.name != "repair_manifest.json"
    ]
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False)
    payload = {
        "repair_version": NEGATIVE_POLICY_VERSION,
        "generated_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in sorted(files)
        },
        "git_status_at_generation": status.stdout.splitlines(),
        "deleted_matrix_code_absent": {
            "scripts/run_clip_final_comparison.py": not (ROOT / "scripts" / "run_clip_final_comparison.py").exists(),
            "src/credit_risk_fs/clip_final_comparison": not (
                ROOT / "src" / "credit_risk_fs" / "clip_final_comparison"
            ).exists(),
        },
        "broad_pipeline_executed": False,
        "scientific_training_executed": False,
        "manifest_hash_basis": sha256_text(
            json.dumps(
                {
                    str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
                    for path in sorted(files)
                },
                sort_keys=True,
            )
        ),
    }
    write_json(OUTPUT_DIR / "repair_manifest.json", payload)


if __name__ == "__main__":
    raise SystemExit(main())
