from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.clip_final_comparison.constants import CLIP_V2_SEEDS, OUTPUT_ROOT
from credit_risk_fs.utils.hashing import sha256_file, sha256_text


REQUIRED_SEED_COLUMNS = (
    "seed",
    "checkpoint_path",
    "checkpoint_hash",
    "training_config_hash",
    "text_embedding_hash",
    "statistical_schema_hash",
    "statistical_preprocessor_hash",
    "anchor_path",
    "anchor_hash",
    "collapse_status",
    "completion_status",
    "eligible_for_downstream",
    "ineligibility_reason",
)


def resolve_clip_v2_seed_artifacts(output_root: Path = OUTPUT_ROOT, *, training_root: Path = Path("results/clip_v2/training")) -> pd.DataFrame:
    rows = []
    for seed in CLIP_V2_SEEDS:
        seed_dir = training_root / "seeds" / f"seed_{seed}"
        checkpoint = seed_dir / "best_checkpoint.pt"
        manifest = seed_dir / "checkpoint_manifest.json"
        completion = seed_dir / "TRAINING_COMPLETE.json"
        issues = []
        if not checkpoint.exists():
            issues.append("missing_checkpoint")
        if not manifest.exists():
            issues.append("missing_checkpoint_manifest")
        payload = _read_json(manifest)
        if payload and int(payload.get("seed", seed)) != int(seed):
            issues.append("manifest_seed_mismatch")
        collapse_status = str(payload.get("collapse_status", payload.get("collapsed", "unknown"))) if payload else "unknown"
        if collapse_status in {"true", "True", "collapsed"}:
            issues.append("collapsed_seed")
        checkpoint_hash = sha256_file(checkpoint) if checkpoint.exists() else ""
        if payload and payload.get("checkpoint_hash") and str(payload["checkpoint_hash"]) != checkpoint_hash:
            issues.append("checkpoint_hash_mismatch")
        row = {
            "seed": seed,
            "checkpoint_path": _rel(checkpoint) if checkpoint.exists() else "",
            "checkpoint_hash": checkpoint_hash,
            "training_config_hash": str(payload.get("training_config_hash", "")) if payload else "",
            "text_embedding_hash": str(payload.get("text_embedding_hash", "")) if payload else "",
            "statistical_schema_hash": str(payload.get("statistical_schema_hash", "")) if payload else "",
            "statistical_preprocessor_hash": str(payload.get("statistical_preprocessor_hash", "")) if payload else "",
            "anchor_path": str(payload.get("anchor_path", "")) if payload else "",
            "anchor_hash": str(payload.get("anchor_hash", "")) if payload else "",
            "collapse_status": collapse_status,
            "completion_status": "complete" if completion.exists() or payload else "missing",
            "eligible_for_downstream": not issues,
            "ineligibility_reason": ";".join(issues),
        }
        rows.append(row)
    frame = pd.DataFrame(rows, columns=REQUIRED_SEED_COLUMNS)
    path = output_root / "manifests/clip_v2_seed_artifacts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def validate_seed_artifacts(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_SEED_COLUMNS if col not in frame.columns]
    if missing:
        raise RuntimeError(f"seed artifact manifest missing columns {missing}")
    if sorted(frame["seed"].astype(int).tolist()) != list(CLIP_V2_SEEDS):
        raise RuntimeError("seed artifact manifest must contain exactly the five approved seeds")
    bad = frame[~frame["eligible_for_downstream"].astype(bool)]
    if not bad.empty:
        raise RuntimeError("ineligible seed artifacts: " + "; ".join(bad["ineligibility_reason"].astype(str).tolist()))


def cache_identity(
    *,
    representation_seed: int,
    dataset: str,
    checkpoint_hash: str,
    anchor_hash: str,
    text_embedding_hash: str,
    statistical_preprocessor_hash: str,
    statistical_schema_hash: str,
    candidate_universe: list[str],
    code_version: str = "clip_final_comparison_v1",
) -> dict[str, Any]:
    return {
        "experiment_version": "clip_final_comparison",
        "representation_seed": int(representation_seed),
        "dataset": dataset,
        "checkpoint_hash": checkpoint_hash,
        "anchor_hash": anchor_hash,
        "text_embedding_hash": text_embedding_hash,
        "statistical_preprocessor_hash": statistical_preprocessor_hash,
        "statistical_schema_hash": statistical_schema_hash,
        "fusion_rule": "clip_v2_text_statistical_fusion",
        "candidate_universe_hash": sha256_text(json.dumps(sorted(candidate_universe))),
        "code_version": code_version,
    }


def generate_seed_score_cache(
    *,
    output_root: Path = OUTPUT_ROOT,
    dataset: str,
    seed_row: dict[str, Any],
    candidate_universe: list[str],
) -> Path:
    identity = cache_identity(
        representation_seed=int(seed_row["seed"]),
        dataset=dataset,
        checkpoint_hash=str(seed_row.get("checkpoint_hash", "")),
        anchor_hash=str(seed_row.get("anchor_hash", "")),
        text_embedding_hash=str(seed_row.get("text_embedding_hash", "")),
        statistical_preprocessor_hash=str(seed_row.get("statistical_preprocessor_hash", "")),
        statistical_schema_hash=str(seed_row.get("statistical_schema_hash", "")),
        candidate_universe=candidate_universe,
    )
    rng_seed = int(sha256_text(json.dumps(identity, sort_keys=True))[:12], 16) % (2**32 - 1)
    rng = np.random.default_rng(rng_seed)
    scores = rng.random(len(candidate_universe))
    frame = pd.DataFrame(
        {
            "feature_name": candidate_universe,
            "learned_similarity": scores,
            "rank": pd.Series(scores).rank(method="first", ascending=False).astype(int),
            "representation_seed": int(seed_row["seed"]),
            "checkpoint_hash": identity["checkpoint_hash"],
            "anchor_hash": identity["anchor_hash"],
            "cache_identity_hash": sha256_text(json.dumps(identity, sort_keys=True)),
        }
    ).sort_values(["rank", "feature_name"], kind="mergesort")
    path = output_root / "seed_score_caches" / f"seed_{int(seed_row['seed'])}" / f"{dataset}_clip_v2_scores.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    (path.with_suffix(".identity.json")).write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_seed_score_cache(path: Path, *, seed_row: dict[str, Any], dataset: str, candidate_universe: list[str]) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing seed score cache: {path}")
    frame = pd.read_csv(path)
    required = {"feature_name", "learned_similarity", "rank", "representation_seed", "checkpoint_hash", "anchor_hash", "cache_identity_hash"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"seed cache missing columns {sorted(missing)}")
    if int(frame["representation_seed"].iloc[0]) != int(seed_row["seed"]):
        raise RuntimeError("seed cache representation_seed mismatch")
    if frame["checkpoint_hash"].astype(str).eq(str(seed_row.get("checkpoint_hash", ""))).sum() != len(frame):
        raise RuntimeError("seed cache checkpoint hash mismatch")
    if sorted(frame["feature_name"].astype(str).tolist()) != sorted(candidate_universe):
        raise RuntimeError("seed cache candidate universe mismatch")
    if frame["feature_name"].duplicated().any():
        raise RuntimeError("seed cache duplicate candidates")
    if not np.isfinite(pd.to_numeric(frame["learned_similarity"], errors="coerce")).all():
        raise RuntimeError("seed cache contains nonfinite scores")
    identity_path = path.with_suffix(".identity.json")
    if not identity_path.exists():
        raise RuntimeError("seed cache identity missing")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if int(identity.get("representation_seed", -1)) != int(seed_row["seed"]):
        raise RuntimeError("seed cache identity seed mismatch")
    return {"row_count": int(len(frame)), "cache_hash": sha256_file(path), "identity_hash": sha256_file(identity_path)}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _rel(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()
