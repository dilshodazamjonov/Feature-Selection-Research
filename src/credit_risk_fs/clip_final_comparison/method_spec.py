from __future__ import annotations

from pathlib import Path
from typing import Any

from credit_risk_fs.clip.statistical_schema_v2 import DESCRIPTOR_COLUMNS_V2
from credit_risk_fs.clip_final_comparison.constants import (
    ABLATIONS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CLIP_V2_SEEDS,
    CORRELATION_FILTER_THRESHOLD,
    MODEL_BUDGETS,
    POOL_MULTIPLIERS,
    RANDOM_SEEDS,
)
from credit_risk_fs.clip_final_comparison.io import atomic_write_json, atomic_write_text


def build_method_specification() -> dict[str, Any]:
    return {
        "candidate_universe": "eligible DEV feature columns after existing leakage, ID, target, split, OOT, PSI, prediction, and post-origination exclusions",
        "exclusion_rules": [
            "no OOT fields in screening or mRMR",
            "no target or label fields in screening text/statistics",
            "legacy LendingClub excluded from CLIP-v2 representation training",
        ],
        "text_template": "existing frozen CLIP feature text builder; no LLM rank, OOT, PSI, target, ID, or outcome fields",
        "text_encoder": "frozen sentence-transformer encoder recorded in existing CLIP artifacts",
        "text_embedding_dimension": 384,
        "statistical_fields": DESCRIPTOR_COLUMNS_V2,
        "statistical_transformations": "Home Credit DEV-only robust preprocessing, then transform-only for LendingClub v2",
        "scaler_fitting_boundary": "Home Credit training-split feature vectors only",
        "projection_architecture": "existing CLIP-v2 statistical/text projection modules",
        "fusion_rule": "frozen CLIP-v2 learned similarity ranking; ablations isolate branch inputs",
        "normalization": "L2/cosine similarity for embedding and anchor comparisons",
        "contrastive_loss": "existing symmetric contrastive loss with false-negative masking",
        "temperature": "existing CLIP-v2 training config",
        "positive_pair_construction": "Home Credit train/validation positives plus LendingClub v2 external positives for validation only",
        "false_negative_masking": "family/text/vector duplicate masks from existing pair builder",
        "optimizer": "existing CLIP-v2 training optimizer",
        "learning_rate": "existing CLIP-v2 training config",
        "batch_size": "existing CLIP-v2 training config",
        "epochs": "existing CLIP-v2 training config with early stopping",
        "early_stopping": "existing validation-based checkpoint selection",
        "seed_list": list(CLIP_V2_SEEDS),
        "checkpoint_selection_rule": "selected seed remains 55; other seeds are robustness-only",
        "anchor_construction": "unchanged Home Credit-fitted CLIP-v2 anchor",
        "ranking_formula": "descending screening score, deterministic feature-name tie break",
        "pool_size_rules": {"budgets": MODEL_BUDGETS, "multipliers": list(POOL_MULTIPLIERS), "cap": "eligible candidate count"},
        "mrmr_implementation": "existing DEV-only MRMR selector after screening pool construction",
        "lr_hyperparameters": "reuse frozen downstream LR configuration",
        "catboost_hyperparameters": "reuse frozen downstream CatBoost configuration",
        "llm_model": "existing LLM workflow only as frozen comparator; no behavior changes",
        "llm_prompt_hash": "recorded from existing LLM artifacts where available",
        "random_seeds": list(RANDOM_SEEDS),
        "correlation_threshold": CORRELATION_FILTER_THRESHOLD,
        "bootstrap_settings": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "primary": "month-cluster if reliable month exists else paired stratified row bootstrap"},
        "temporal_cutoff_rules": "up to three chronological cutoffs per dataset; no invented unsupported cutoffs; frozen representation throughout",
        "ablations": ABLATIONS,
    }


def write_method_specification(output_dir: Path) -> dict[str, str]:
    spec = build_method_specification()
    json_path = output_dir / "manifests" / "full_method_specification.json"
    md_path = output_dir / "manifests" / "full_method_specification.md"
    atomic_write_json(json_path, spec)
    atomic_write_text(md_path, _to_markdown(spec))
    return {"json": json_path.as_posix(), "markdown": md_path.as_posix()}


def _to_markdown(spec: dict[str, Any]) -> str:
    lines = ["# Full Method Specification", ""]
    for key, value in spec.items():
        lines.append(f"## {key}")
        if isinstance(value, (dict, list, tuple)):
            lines.append("```json")
            import json

            lines.append(json.dumps(value, indent=2, default=str))
            lines.append("```")
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines)

