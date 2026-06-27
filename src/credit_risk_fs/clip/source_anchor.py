from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from credit_risk_fs.clip.reverse_transfer import PAIRING_POLICY_VERSION
from credit_risk_fs.clip.statistical_view_v2 import (
    TYPE_NUMERIC,
    resolve_feature_type,
)
from credit_risk_fs.utils.hashing import sha256_file, sha256_text
from credit_risk_fs.utils.io import write_json


ANCHOR_RULE_VERSION = "lendingclub_dev_temporal_stable_core_v1"
EXPECTED_RANKING = (
    "max_adjacent_window_psi",
    "max_missing_rate_difference",
    "feature_id",
)


@dataclass(frozen=True)
class SourceAnchorConfig:
    source_dataset: str
    dev_start: float
    dev_end: float
    temporal_subwindows: int
    subwindow_strategy: str
    subwindow_boundaries: tuple[float, ...]
    training_split_only: bool
    target_used: bool
    oot_used: bool
    external_data_used: bool
    max_adjacent_window_psi: float
    max_missing_rate_difference: float
    member_count: int
    ranking: tuple[str, ...]
    fail_if_insufficient_members: bool
    min_non_missing_per_subwindow: int
    numeric_bins: int
    categorical_min_count: int
    psi_epsilon: float

    def validate(self) -> None:
        if self.source_dataset != "lendingclub_v2":
            raise ValueError("source anchor must be LendingClub v2")
        if self.target_used or self.oot_used or self.external_data_used:
            raise ValueError("source anchor must be target-free, OOT-free and source-only")
        if not self.training_split_only:
            raise ValueError("source anchor must be training-split-only")
        if self.temporal_subwindows != 4 or self.subwindow_strategy != "equal_duration":
            raise ValueError("source anchor requires four equal-duration subwindows")
        expected = equal_duration_boundaries(
            self.dev_start, self.dev_end, self.temporal_subwindows
        )
        if not np.allclose(self.subwindow_boundaries, expected, rtol=0, atol=1e-12):
            raise ValueError(
                f"configured subwindow boundaries differ from the fixed rule: {expected}"
            )
        if self.dev_start != -1795 or self.dev_end != -1065:
            raise ValueError("source anchor DEV interval must be [-1795, -1065)")
        if self.max_adjacent_window_psi != 0.10:
            raise ValueError("source anchor PSI threshold must be 0.10")
        if self.max_missing_rate_difference != 0.05:
            raise ValueError("source anchor missingness threshold must be 0.05")
        if self.member_count != 23:
            raise ValueError("source anchor must contain exactly 23 members")
        if self.ranking != EXPECTED_RANKING:
            raise ValueError(f"source anchor ranking must be {EXPECTED_RANKING}")
        if not self.fail_if_insufficient_members:
            raise ValueError("source anchor must fail when fewer than 23 members qualify")
        if self.min_non_missing_per_subwindow <= 0:
            raise ValueError("minimum subwindow support must be positive")
        if self.numeric_bins < 2 or self.categorical_min_count < 1:
            raise ValueError("invalid frozen PSI bucket configuration")

    def to_manifest(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["subwindow_boundaries"] = list(self.subwindow_boundaries)
        payload["ranking"] = list(self.ranking)
        payload["pairing_policy_version"] = PAIRING_POLICY_VERSION
        payload["anchor_rule_version"] = ANCHOR_RULE_VERSION
        return payload


@dataclass(frozen=True)
class FrozenBucketizer:
    feature_type: str
    numeric_edges: tuple[float, ...] = ()
    reference_constant: str = ""
    categorical_levels: tuple[str, ...] = ()
    categorical_min_count: int = 1
    fit_window_index: int = 0

    def transform(self, values: pd.Series) -> pd.Series:
        missing = values.isna()
        if self.feature_type == "numeric":
            numeric = pd.to_numeric(values, errors="coerce")
            if len(self.numeric_edges) >= 2:
                labels = pd.cut(
                    numeric,
                    bins=np.asarray(self.numeric_edges, dtype=float),
                    include_lowest=True,
                ).astype("string")
                labels = labels.fillna("MISSING")
                labels.loc[missing] = "MISSING"
                return labels.astype(str)
            constant = float(self.reference_constant)
            output = pd.Series("OTHER", index=values.index, dtype="string")
            output.loc[numeric.eq(constant)] = "VALUE"
            output.loc[missing] = "MISSING"
            return output.astype(str)
        text = values.astype("string")
        output = pd.Series("OTHER", index=values.index, dtype="string")
        output.loc[text.isin(self.categorical_levels)] = text.loc[
            text.isin(self.categorical_levels)
        ]
        output.loc[missing] = "MISSING"
        return output.astype(str)

    def manifest(self) -> dict[str, Any]:
        return {
            "feature_type": self.feature_type,
            "numeric_edges": list(self.numeric_edges),
            "reference_constant": self.reference_constant,
            "categorical_levels": list(self.categorical_levels),
            "categorical_min_count": self.categorical_min_count,
            "fit_window_index": self.fit_window_index,
            "missing_bucket": "MISSING",
            "unseen_or_rare_bucket": "OTHER",
        }


def load_source_anchor_config(data: Mapping[str, Any]) -> SourceAnchorConfig:
    anchor = data.get("source_anchor")
    if not isinstance(anchor, Mapping):
        raise ValueError("source_anchor configuration is required")
    dev = anchor.get("dev_window")
    if not isinstance(dev, Mapping):
        raise ValueError("source_anchor.dev_window is required")
    return SourceAnchorConfig(
        source_dataset=str(anchor.get("source_dataset", "")),
        dev_start=float(dev.get("start")),
        dev_end=float(dev.get("end")),
        temporal_subwindows=int(anchor.get("temporal_subwindows", 0)),
        subwindow_strategy=str(anchor.get("subwindow_strategy", "")),
        subwindow_boundaries=tuple(
            float(value) for value in anchor.get("subwindow_boundaries", [])
        ),
        training_split_only=bool(anchor.get("training_split_only", False)),
        target_used=bool(anchor.get("target_used", True)),
        oot_used=bool(anchor.get("oot_used", True)),
        external_data_used=bool(anchor.get("external_data_used", True)),
        max_adjacent_window_psi=float(anchor.get("max_adjacent_window_psi")),
        max_missing_rate_difference=float(
            anchor.get("max_missing_rate_difference")
        ),
        member_count=int(anchor.get("member_count", 0)),
        ranking=tuple(str(value) for value in anchor.get("ranking", [])),
        fail_if_insufficient_members=bool(
            anchor.get("fail_if_insufficient_members", False)
        ),
        min_non_missing_per_subwindow=int(
            anchor.get("min_non_missing_per_subwindow", 0)
        ),
        numeric_bins=int(anchor.get("numeric_bins", 0)),
        categorical_min_count=int(anchor.get("categorical_min_count", 0)),
        psi_epsilon=float(anchor.get("psi_epsilon", 1e-6)),
    )


def equal_duration_boundaries(
    start: float, end: float, windows: int = 4
) -> tuple[float, ...]:
    if windows <= 0 or end <= start:
        raise ValueError("invalid temporal subwindow interval")
    return tuple(float(value) for value in np.linspace(start, end, windows + 1))


def fit_frozen_bucketizer(
    values: pd.Series,
    *,
    numeric_bins: int,
    categorical_min_count: int,
) -> FrozenBucketizer:
    resolution = resolve_feature_type(values, feature_name=str(values.name or "feature"))
    if resolution.resolved_type == TYPE_NUMERIC:
        numeric = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        observed = numeric.dropna()
        if observed.empty:
            raise ValueError("cannot fit numeric PSI bins without observed values")
        edges = np.unique(
            np.quantile(observed.to_numpy(dtype=float), np.linspace(0, 1, numeric_bins + 1))
        )
        if len(edges) >= 2:
            edges[0] = -np.inf
            edges[-1] = np.inf
            return FrozenBucketizer(
                feature_type="numeric",
                numeric_edges=tuple(float(value) for value in edges),
                categorical_min_count=categorical_min_count,
            )
        return FrozenBucketizer(
            feature_type="numeric",
            reference_constant=str(float(edges[0])),
            categorical_min_count=categorical_min_count,
        )
    text = values.dropna().astype(str)
    counts = text.value_counts(dropna=False)
    levels = tuple(
        sorted(str(level) for level, count in counts.items() if count >= categorical_min_count)
    )
    return FrozenBucketizer(
        feature_type="categorical",
        categorical_levels=levels,
        categorical_min_count=categorical_min_count,
    )


def psi_from_frozen_buckets(
    expected_buckets: pd.Series,
    actual_buckets: pd.Series,
    *,
    epsilon: float,
) -> float:
    labels = sorted(
        set(expected_buckets.astype(str)).union(actual_buckets.astype(str))
    )
    expected = (
        expected_buckets.astype(str).value_counts(normalize=True).reindex(labels, fill_value=0)
    )
    actual = (
        actual_buckets.astype(str).value_counts(normalize=True).reindex(labels, fill_value=0)
    )
    return float(
        np.sum(
            (expected - actual)
            * np.log((expected + float(epsilon)) / (actual + float(epsilon)))
        )
    )


def build_feature_stability_evidence(
    data: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    time_column: str,
    config: SourceAnchorConfig,
    exact_duplicates: pd.DataFrame | None = None,
    verified_aliases: Iterable[Iterable[str]] = (),
    documented_identity_transforms: Iterable[Iterable[str]] = (),
) -> tuple[pd.DataFrame, dict[str, FrozenBucketizer]]:
    config.validate()
    if time_column not in data:
        raise ValueError("source time column is missing")
    if "dataset" in data and set(data["dataset"].dropna().astype(str)) != {
        "lendingclub_v2"
    }:
        raise ValueError("Home Credit or mixed-domain data cannot enter source anchor evidence")
    if "dataset" in candidates and set(
        candidates["dataset"].dropna().astype(str)
    ) != {"lendingclub_v2"}:
        raise ValueError("anchor candidates must belong to LendingClub v2")
    forbidden_candidate_names = {
        name
        for name in candidates["feature_name"].astype(str)
        if any(token in name.lower() for token in ("target", "label", "bad_flag"))
    }
    if forbidden_candidate_names:
        raise ValueError(
            f"target-like fields cannot enter anchor evidence: {sorted(forbidden_candidate_names)}"
        )
    dev = data[
        (pd.to_numeric(data[time_column], errors="coerce") >= config.dev_start)
        & (pd.to_numeric(data[time_column], errors="coerce") < config.dev_end)
    ].copy()
    if dev.empty:
        raise ValueError("source DEV interval is empty")
    windows = [
        dev[
            (dev[time_column] >= left) & (dev[time_column] < right)
        ]
        for left, right in zip(
            config.subwindow_boundaries[:-1], config.subwindow_boundaries[1:]
        )
    ]
    if any(window.empty for window in windows):
        raise ValueError("one or more fixed DEV subwindows are empty")
    exact_groups = _identity_groups_from_duplicate_frame(exact_duplicates)
    alias_groups = _identity_groups_from_pairs(verified_aliases)
    transform_groups = _identity_groups_from_pairs(documented_identity_transforms)
    rows: list[dict[str, Any]] = []
    bucketizers: dict[str, FrozenBucketizer] = {}
    for candidate in candidates.sort_values("feature_id", kind="mergesort").to_dict(
        "records"
    ):
        feature_id = str(candidate["feature_id"])
        feature_name = str(candidate["feature_name"])
        split = str(
            candidate.get("split_assignment", candidate.get("split", ""))
        )
        base_eligible = bool(candidate.get("eligible_for_pairing", True))
        exclusion = ""
        if not base_eligible:
            exclusion = str(candidate.get("exclusion_reason") or "not_pairing_eligible")
        elif split != "train":
            exclusion = "not_in_contrastive_training_split"
        elif feature_name not in dev:
            exclusion = "missing_approved_dev_statistical_evidence"
        counts = [
            int(window[feature_name].notna().sum()) if feature_name in window else 0
            for window in windows
        ]
        missing_rates = [
            float(window[feature_name].isna().mean()) if feature_name in window else 1.0
            for window in windows
        ]
        max_missing_difference = float(max(missing_rates) - min(missing_rates))
        psi_values: list[float] = []
        bucketizer: FrozenBucketizer | None = None
        if not exclusion and min(counts) < config.min_non_missing_per_subwindow:
            exclusion = "insufficient_non_missing_support"
        if not exclusion:
            bucketizer = fit_frozen_bucketizer(
                windows[0][feature_name],
                numeric_bins=config.numeric_bins,
                categorical_min_count=config.categorical_min_count,
            )
            transformed = [
                bucketizer.transform(window[feature_name]) for window in windows
            ]
            psi_values = [
                psi_from_frozen_buckets(
                    transformed[index],
                    transformed[index + 1],
                    epsilon=config.psi_epsilon,
                )
                for index in range(len(transformed) - 1)
            ]
            bucketizers[feature_id] = bucketizer
        max_psi = float(max(psi_values)) if psi_values else np.nan
        threshold_pass = (
            not exclusion
            and np.isfinite(max_psi)
            and max_psi <= config.max_adjacent_window_psi
            and max_missing_difference <= config.max_missing_rate_difference
        )
        if not exclusion and not threshold_pass:
            exclusion = (
                "stability_threshold_failed"
                if np.isfinite(max_psi)
                else "invalid_stability_measure"
            )
        rows.append(
            {
                "feature_id": feature_id,
                "feature_name": feature_name,
                "semantic_group": str(candidate.get("semantic_group", "unknown")),
                "subwindow_non_missing_counts": json.dumps(counts),
                "subwindow_missing_rates": json.dumps(missing_rates),
                "max_missing_rate_difference": max_missing_difference,
                "adjacent_window_psi_values": json.dumps(psi_values),
                "max_adjacent_window_psi": max_psi,
                "psi_bucket_manifest": json.dumps(
                    bucketizer.manifest() if bucketizer else {}, sort_keys=True
                ),
                "exact_duplicate_group": exact_groups.get(feature_name, ""),
                "alias_group": alias_groups.get(feature_name, ""),
                "identity_transform_group": transform_groups.get(feature_name, ""),
                "contrastive_split": split,
                "eligibility_status": "eligible" if threshold_pass else "excluded",
                "exclusion_reason": exclusion,
                "target_used": False,
                "oot_used": False,
                "external_data_used": False,
            }
        )
    return pd.DataFrame(rows), bucketizers


def select_anchor_members(
    evidence: pd.DataFrame,
    *,
    config: SourceAnchorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.validate()
    qualified = evidence[
        evidence["eligibility_status"].astype(str).eq("eligible")
        & evidence["contrastive_split"].astype(str).eq("train")
        & evidence["max_adjacent_window_psi"].le(config.max_adjacent_window_psi)
        & evidence["max_missing_rate_difference"].le(
            config.max_missing_rate_difference
        )
    ].sort_values(list(config.ranking), kind="mergesort")
    selected_rows: list[dict[str, Any]] = []
    used_identity_groups: set[str] = set()
    audit = evidence.copy()
    audit["qualifies_thresholds"] = audit["feature_id"].isin(qualified["feature_id"])
    audit["selected"] = False
    audit["selection_rank"] = pd.NA
    audit["selection_exclusion_reason"] = audit["exclusion_reason"].astype(str)
    for row in qualified.to_dict("records"):
        groups = {
            str(row.get(column, ""))
            for column in (
                "exact_duplicate_group",
                "alias_group",
                "identity_transform_group",
            )
            if str(row.get(column, ""))
        }
        if groups & used_identity_groups:
            audit.loc[
                audit["feature_id"].astype(str).eq(str(row["feature_id"])),
                "selection_exclusion_reason",
            ] = "identity_equivalent_to_higher_ranked_member"
            continue
        selected_rows.append(row)
        used_identity_groups.update(groups)
        if len(selected_rows) == config.member_count:
            break
    if len(selected_rows) != config.member_count:
        raise RuntimeError(
            "BLOCKED — insufficient leakage-safe LendingClub stable-core members: "
            f"required={config.member_count}, observed={len(selected_rows)}"
        )
    members = pd.DataFrame(selected_rows).reset_index(drop=True)
    members["anchor_rank"] = range(1, len(members) + 1)
    rank_by_id = dict(zip(members["feature_id"].astype(str), members["anchor_rank"]))
    audit["selected"] = audit["feature_id"].astype(str).isin(rank_by_id)
    audit["selection_rank"] = audit["feature_id"].astype(str).map(rank_by_id)
    audit.loc[audit["selected"], "selection_exclusion_reason"] = ""
    return members, audit.sort_values(
        [
            "selected",
            "max_adjacent_window_psi",
            "max_missing_rate_difference",
            "feature_id",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_seed_anchor(
    source_embeddings: pd.DataFrame,
    members: pd.DataFrame,
    *,
    seed: int,
    checkpoint_hash: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(members) != 23:
        raise ValueError("per-seed source anchor requires exactly 23 members")
    member_ids = members["feature_id"].astype(str).tolist()
    selected = source_embeddings[
        source_embeddings["feature_id"].astype(str).isin(member_ids)
    ].copy()
    if set(selected["feature_id"].astype(str)) != set(member_ids):
        raise ValueError("source embeddings are missing approved anchor members")
    if set(selected["dataset"].astype(str)) != {"lendingclub_v2"}:
        raise ValueError("Home Credit-trained/source embeddings cannot define this anchor")
    if not selected["split"].astype(str).eq("train").all():
        raise ValueError("source anchor contains validation-split features")
    columns = sorted(
        column for column in selected.columns if str(column).startswith("joint_")
    )
    values = selected.set_index("feature_id").loc[member_ids, columns].to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if bool((norms <= 0).any()):
        raise ValueError("source anchor member has zero-norm embedding")
    normalized_members = values / norms
    centroid = normalized_members.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm <= 0:
        raise ValueError("source anchor centroid has zero norm")
    centroid = (centroid / centroid_norm).astype("float32")
    anchor_hash = vector_hash(centroid)
    return centroid, {
        "seed": int(seed),
        "source_dataset": "lendingclub_v2",
        "checkpoint_hash": checkpoint_hash,
        "member_count": len(member_ids),
        "anchor_member_feature_ids": member_ids,
        "member_normalization": "l2_per_seed_space",
        "centroid_normalization": "l2",
        "anchor_hash": anchor_hash,
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "training_split_only": True,
        "target_used": False,
        "oot_used": False,
        "external_data_used": False,
    }


def build_source_anchor_manifest(
    *,
    config: SourceAnchorConfig,
    members: pd.DataFrame,
    configuration_hash: str,
    data_manifest_hash: str,
    evidence_hash: str,
    anchor_hashes_by_seed: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    config.validate()
    if len(members) != config.member_count:
        raise ValueError("source anchor manifest member count mismatch")
    return {
        "anchor_rule_version": ANCHOR_RULE_VERSION,
        "source_dataset": config.source_dataset,
        "dev_window": [config.dev_start, config.dev_end],
        "subwindow_boundaries": list(config.subwindow_boundaries),
        "target_used": False,
        "oot_used": False,
        "external_data_used": False,
        "stability_metrics": [
            "max_adjacent_window_psi",
            "max_missing_rate_difference",
        ],
        "psi_bin_fit_scope": "first_lendingclub_dev_subwindow_only_frozen_for_all_later_windows",
        "thresholds": {
            "max_adjacent_window_psi": config.max_adjacent_window_psi,
            "max_missing_rate_difference": config.max_missing_rate_difference,
            "min_non_missing_per_subwindow": config.min_non_missing_per_subwindow,
        },
        "ranking_rule": list(config.ranking),
        "required_member_count": config.member_count,
        "actual_member_count": len(members),
        "anchor_member_feature_ids": members["feature_id"].astype(str).tolist(),
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "training_split_only": True,
        "configuration_hash": configuration_hash,
        "data_manifest_hash": data_manifest_hash,
        "feature_stability_evidence_hash": evidence_hash,
        "anchor_members_hash": sha256_text(members.to_csv(index=False)),
        "anchor_hashes_by_seed": dict(anchor_hashes_by_seed or {}),
        "selected_using_downstream_performance": False,
        "external_results_used_for_selection": False,
    }


def validate_source_anchor_artifacts(
    *,
    config: SourceAnchorConfig,
    members: pd.DataFrame,
    manifest: Mapping[str, Any],
    training_feature_ids: set[str],
    configuration_hash: str,
    data_manifest_hash: str,
    member_path: str | Path | None = None,
    evidence_path: str | Path | None = None,
) -> None:
    config.validate()
    errors: list[str] = []
    expected = {
        "source_dataset": "lendingclub_v2",
        "target_used": False,
        "oot_used": False,
        "external_data_used": False,
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "training_split_only": True,
        "required_member_count": 23,
        "actual_member_count": 23,
        "configuration_hash": configuration_hash,
        "data_manifest_hash": data_manifest_hash,
        "selected_using_downstream_performance": False,
        "external_results_used_for_selection": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            errors.append(f"{key} mismatch")
    if list(manifest.get("subwindow_boundaries", [])) != list(
        config.subwindow_boundaries
    ):
        errors.append("subwindow boundaries mismatch")
    if manifest.get("thresholds", {}).get(
        "max_adjacent_window_psi"
    ) != config.max_adjacent_window_psi or manifest.get("thresholds", {}).get(
        "max_missing_rate_difference"
    ) != config.max_missing_rate_difference:
        errors.append("stability thresholds mismatch")
    if len(members) != 23 or members["feature_id"].duplicated().any():
        errors.append("anchor must contain 23 unique members")
    member_ids = set(members["feature_id"].astype(str))
    if not member_ids.issubset(training_feature_ids):
        errors.append("anchor members fall outside the LendingClub training split")
    for column in (
        "exact_duplicate_group",
        "alias_group",
        "identity_transform_group",
    ):
        nonempty = members[column].fillna("").astype(str)
        if nonempty[nonempty.ne("")].duplicated().any():
            errors.append(f"anchor contains duplicate identity group in {column}")
    if member_path is not None and manifest.get("anchor_members_hash") != sha256_text(
        Path(member_path).read_text(encoding="utf-8")
    ):
        errors.append("anchor members hash mismatch")
    if evidence_path is not None and manifest.get(
        "feature_stability_evidence_hash"
    ) != sha256_file(evidence_path):
        errors.append("feature stability evidence hash mismatch")
    if errors:
        raise ValueError("; ".join(errors))


def validate_seed_anchor(
    *,
    vector: np.ndarray,
    seed_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    expected_seed: int,
    expected_checkpoint_hash: str,
) -> None:
    errors = []
    if seed_manifest.get("source_dataset") != "lendingclub_v2":
        errors.append("seed anchor source dataset mismatch")
    if seed_manifest.get("seed") != expected_seed:
        errors.append("seed anchor seed mismatch")
    if seed_manifest.get("checkpoint_hash") != expected_checkpoint_hash:
        errors.append("seed anchor checkpoint hash mismatch")
    if seed_manifest.get("member_count") != 23:
        errors.append("seed anchor member count mismatch")
    if seed_manifest.get("target_used") or seed_manifest.get("oot_used") or seed_manifest.get(
        "external_data_used"
    ):
        errors.append("seed anchor leakage metadata is invalid")
    observed_hash = vector_hash(np.asarray(vector, dtype="float32"))
    if seed_manifest.get("anchor_hash") != observed_hash:
        errors.append("seed anchor hash mismatch")
    expected_hash = source_manifest.get("anchor_hashes_by_seed", {}).get(
        str(expected_seed)
    )
    if expected_hash != observed_hash:
        errors.append("source-manifest seed anchor hash mismatch")
    if not np.isclose(np.linalg.norm(vector), 1.0, rtol=0, atol=1e-6):
        errors.append("seed anchor is not L2-normalized")
    if errors:
        raise ValueError("; ".join(errors))


def write_anchor_selection_artifacts(
    *,
    output_dir: str | Path,
    config: SourceAnchorConfig,
    evidence: pd.DataFrame,
    audit: pd.DataFrame,
    members: pd.DataFrame,
    configuration_hash: str,
    data_manifest_hash: str,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = write_json(
        output / "stability_subwindow_config.json", config.to_manifest()
    )
    evidence_path = output / "feature_stability_evidence.csv"
    audit_path = output / "anchor_candidate_audit.csv"
    members_path = output / "anchor_members.csv"
    evidence.to_csv(evidence_path, index=False)
    audit.to_csv(audit_path, index=False)
    members.to_csv(members_path, index=False)
    manifest = build_source_anchor_manifest(
        config=config,
        members=members,
        configuration_hash=configuration_hash,
        data_manifest_hash=data_manifest_hash,
        evidence_hash=sha256_file(evidence_path),
    )
    manifest_path = write_json(output / "source_anchor_manifest.json", manifest)
    seed_manifest_path = output / "seed_anchor_manifest.csv"
    pd.DataFrame(
        columns=[
            "seed",
            "checkpoint_hash",
            "anchor_hash",
            "member_count",
            "source_dataset",
            "pairing_policy_version",
        ]
    ).to_csv(seed_manifest_path, index=False)
    return {
        "stability_subwindow_config": config_path,
        "feature_stability_evidence": evidence_path,
        "anchor_candidate_audit": audit_path,
        "anchor_members": members_path,
        "seed_anchor_manifest": seed_manifest_path,
        "source_anchor_manifest": manifest_path,
    }


def vector_hash(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="float32")
    return sha256_text(json.dumps(array.tolist(), separators=(",", ":")))


def _identity_groups_from_duplicate_frame(
    frame: pd.DataFrame | None,
) -> dict[str, str]:
    if frame is None or frame.empty:
        return {}
    return _union_groups(
        [
            (str(row.anchor_feature_name), str(row.excluded_feature_name))
            for row in frame.itertuples(index=False)
        ],
        prefix="exact",
    )


def _identity_groups_from_pairs(
    pairs: Iterable[Iterable[str]],
) -> dict[str, str]:
    normalized = []
    for pair in pairs:
        values = [str(value) for value in pair]
        if len(values) != 2 or values[0] == values[1]:
            raise ValueError("identity relation must contain two distinct feature names")
        normalized.append((values[0], values[1]))
    return _union_groups(normalized, prefix="identity")


def _union_groups(
    pairs: list[tuple[str, str]],
    *,
    prefix: str,
) -> dict[str, str]:
    parents: dict[str, str] = {}

    def find(value: str) -> str:
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    for left, right in pairs:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)
    members: dict[str, list[str]] = {}
    for value in list(parents):
        members.setdefault(find(value), []).append(value)
    output: dict[str, str] = {}
    for values in members.values():
        group_hash = sha256_text("|".join(sorted(values)))[:16]
        for value in values:
            output[value] = f"{prefix}:{group_hash}"
    return output
