from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_fs.clip.score_cache import (
    build_score_cache_frame,
    select_cache_columns,
    validate_score_cache,
    write_score_cache,
)
from credit_risk_fs.clip.selector_validation import (
    ClipSelectorConfig,
    load_clip_selector_config,
    validate_clip_selector_binding,
)
from credit_risk_fs.utils.hashing import sha256_file


class ClipScoreAdapter:
    def __init__(self, config_path: str | Path = "configs/clip/selector.yaml", *, dataset: str = "homecredit") -> None:
        self.config_path = Path(config_path)
        self.config = load_clip_selector_config(self.config_path)
        self.dataset = str(dataset)
        self.binding = validate_clip_selector_binding(self.config)
        self.checkpoint_hash = str(self.binding["checkpoint_hash"])
        self.anchor_hash = str(self.binding["anchor_hash"])
        self.experiment_version = str(self.binding.get("experiment_version", "clip_v1"))

    def score_frame(self, *, use_cache: bool = True, write_cache: bool = False) -> pd.DataFrame:
        if self.dataset == "lendingclub":
            raise RuntimeError("legacy LendingClub is forbidden for CLIP scoring")
        if self.dataset not in {"homecredit", "lendingclub_v2"}:
            raise RuntimeError(f"unsupported CLIP scoring dataset: {self.dataset}")
        cache_path = self._cache_path()
        if use_cache and cache_path.exists():
            frame = pd.read_csv(cache_path)
            validate_score_cache(frame, checkpoint_hash=self.checkpoint_hash, anchor_hash=self.anchor_hash)
            return select_cache_columns(frame)

        source_path = self.config.homecredit_scores_path if self.dataset == "homecredit" else self.config.lendingclub_v2_scores_path
        joint_path = source_path.parent / (
            "homecredit_joint_embeddings.parquet"
            if self.dataset == "homecredit"
            else "lendingclub_v2_joint_embeddings.parquet"
        )
        scores = pd.read_csv(source_path)
        joint = pd.read_parquet(joint_path)
        cache = build_score_cache_frame(
            scores=scores,
            joint_embeddings=joint,
            config=self.config,
            checkpoint_hash=self.checkpoint_hash,
            anchor_hash=self.anchor_hash,
            preprocessor_hash=sha256_file(self.config.statistical_preprocessor_path),
            experiment_version=self.experiment_version,
            code_version=self._code_version(),
        )
        validate_score_cache(cache, checkpoint_hash=self.checkpoint_hash, anchor_hash=self.anchor_hash)
        if write_cache:
            write_score_cache(cache, cache_path)
        return select_cache_columns(cache)

    def rank_candidates(self, candidates: list[str], *, missing_feature_policy: str | None = None) -> pd.DataFrame:
        policy = missing_feature_policy or self.config.missing_feature_policy
        if len(candidates) != len(set(candidates)):
            raise RuntimeError("duplicate candidate feature names are not allowed")
        frame = self.score_frame(use_cache=True, write_cache=False)
        by_feature = frame.set_index("feature_name", drop=False)
        missing = [feature for feature in candidates if feature not in by_feature.index]
        if missing and policy == "error":
            raise RuntimeError(f"candidates missing CLIP scores: {missing[:20]}")
        if missing and policy != "exclude_with_manifest":
            raise RuntimeError(f"unsupported missing feature policy: {policy}")
        rows = []
        for feature in candidates:
            if feature not in by_feature.index:
                if policy == "exclude_with_manifest":
                    rows.append({"feature_name": feature, "exclusion_reason": "missing_clip_score"})
                continue
            record = by_feature.loc[feature].to_dict()
            record["exclusion_reason"] = ""
            rows.append(record)
        ranked = pd.DataFrame(rows)
        if ranked.empty:
            raise RuntimeError("no candidates have CLIP scores")
        scored = ranked[ranked["exclusion_reason"].astype(str).eq("")].copy()
        scored = scored.sort_values(["learned_similarity", "feature_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
        scored["clip_rank"] = range(1, len(scored) + 1)
        excluded = ranked[ranked["exclusion_reason"].astype(str).ne("")].copy()
        if len(excluded):
            excluded["clip_rank"] = pd.NA
            return pd.concat([scored, excluded], ignore_index=True, sort=False)
        return scored

    def _cache_path(self) -> Path:
        prefix = "clip_v2" if self.experiment_version == "clip_v2" else "clip"
        name = f"homecredit_{prefix}_scores.csv" if self.dataset == "homecredit" else f"lendingclub_v2_{prefix}_scores.csv"
        return self.config.output_dir / name

    def _code_version(self) -> str:
        parts = []
        for path in [
            Path("src/credit_risk_fs/selectors/clip_screening.py"),
            Path("src/credit_risk_fs/selectors/clip_then_mrmr.py"),
            Path("src/credit_risk_fs/clip/selector_adapter.py"),
            Path("src/credit_risk_fs/clip/score_cache.py"),
        ]:
            if path.exists():
                parts.append(sha256_file(path))
        return "|".join(parts)


def materialize_score_caches(config_path: str | Path = "configs/clip/selector.yaml") -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for dataset in ["homecredit", "lendingclub_v2"]:
        adapter = ClipScoreAdapter(config_path, dataset=dataset)
        frame = adapter.score_frame(use_cache=False, write_cache=True)
        paths[dataset] = adapter._cache_path()
        if len(frame) == 0:
            raise RuntimeError(f"{dataset}: empty CLIP score cache")
    return paths


def score_coverage(config_path: str | Path = "configs/clip/selector.yaml") -> dict[str, Any]:
    coverage = {}
    for dataset in ["homecredit", "lendingclub_v2"]:
        adapter = ClipScoreAdapter(config_path, dataset=dataset)
        frame = adapter.score_frame(use_cache=True, write_cache=False)
        coverage[dataset] = {
            "row_count": int(len(frame)),
            "feature_count": int(frame["feature_name"].nunique()),
            "checkpoint_hash": adapter.checkpoint_hash,
            "anchor_hash": adapter.anchor_hash,
        }
    return coverage
