from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from credit_risk_fs.clip.feature_family import (
    build_feature_family_audit,
    derive_canonical_feature_family,
    derive_feature_family,
)
from credit_risk_fs.utils.hashing import sha256_text
from credit_risk_fs.utils.io import write_json


@dataclass(frozen=True)
class GroupSplitResult:
    split: pd.DataFrame
    audit: dict[str, object]
    family_audit: pd.DataFrame
    family_audit_summary: dict[str, object]


def build_group_split(
    frame: pd.DataFrame,
    *,
    dataset: str = "homecredit",
    seed: int = 42,
    validation_fraction: float = 0.2,
    derived_family_aliases: Mapping[str, str] | None = None,
) -> GroupSplitResult:
    if dataset != "homecredit":
        raise ValueError("group-aware split is only fit on homecredit")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    working = frame.copy()
    family_audit, family_summary = build_feature_family_audit(working, aliases=derived_family_aliases)
    family_meta = family_audit.set_index("feature_name", drop=False)
    group_keys: list[str] = []
    group_sources: list[str] = []
    canonical_families: list[str] = []
    resolution_rules: list[str] = []
    resolution_sources: list[str] = []
    for row in working.to_dict("records"):
        feature = str(row["feature_name"])
        source = str(row.get("source_table", "") or "")
        semantic = str(row.get("semantic_group", "") or "")
        family_row = family_meta.loc[feature]
        canonical_family = str(family_row["canonical_feature_family"])
        resolution_source = str(family_row["family_resolution_source"])
        resolution_rule = str(family_row["family_resolution_rule"])
        family_member_count = int(family_row["family_member_count"])
        canonical_families.append(canonical_family)
        resolution_sources.append(resolution_source)
        resolution_rules.append(resolution_rule)
        if canonical_family and (canonical_family != feature or family_member_count > 1):
            group_keys.append(f"family:{canonical_family}")
            group_sources.append("canonical_feature_family")
        elif source and source.lower() != "nan":
            group_keys.append(f"source:{source}")
            group_sources.append("source_table")
        elif semantic and semantic.lower() != "nan":
            group_keys.append(f"semantic:{semantic}")
            group_sources.append("semantic_group")
        else:
            group_keys.append(f"name:{family}")
            group_sources.append("feature_name_fallback")

    working["group_key"] = group_keys
    working["group_source"] = group_sources
    working["canonical_feature_family"] = canonical_families
    working["family_resolution_source"] = resolution_sources
    working["family_resolution_rule"] = resolution_rules
    groups = sorted(working["group_key"].unique())
    scored = sorted(
        [(group, sha256_text(f"{seed}|{group}")) for group in groups],
        key=lambda item: item[1],
    )
    target_validation = max(1, round(len(working) * validation_fraction))
    validation_groups = set()
    validation_rows = 0
    for group, _ in scored:
        if validation_rows >= target_validation:
            break
        validation_groups.add(group)
        validation_rows += int(working["group_key"].eq(group).sum())

    working["split"] = working["group_key"].map(lambda group: "validation" if group in validation_groups else "train")
    output = working[
        [
            "dataset",
            "feature_name",
            "split",
            "group_key",
            "group_source",
            "canonical_feature_family",
            "family_resolution_source",
            "family_resolution_rule",
        ]
    ].copy()
    family_counts = output.groupby("canonical_feature_family")["feature_name"].transform("size")
    output["family_member_count"] = family_counts.astype(int)
    output["seed"] = int(seed)
    output = output.sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)
    family_audit = family_audit.merge(
        output[["feature_name", "split", "group_key"]], on="feature_name", how="left"
    ).sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)

    train_groups = set(output.loc[output["split"].eq("train"), "group_key"])
    validation_groups_observed = set(output.loc[output["split"].eq("validation"), "group_key"])
    overlap = sorted(train_groups.intersection(validation_groups_observed))
    train_families = set(output.loc[output["split"].eq("train"), "canonical_feature_family"].astype(str))
    validation_families = set(output.loc[output["split"].eq("validation"), "canonical_feature_family"].astype(str))
    family_overlap = sorted(train_families.intersection(validation_families))
    weak_sources = int(output["group_source"].eq("feature_name_fallback").sum())
    region_features = output[output["feature_name"].isin(["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"])]
    region_same_split = (
        len(region_features) == 2
        and region_features["split"].nunique() == 1
        and region_features["canonical_feature_family"].nunique() == 1
    )
    audit = {
        "dataset": dataset,
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "row_count": int(len(output)),
        "train_rows": int(output["split"].eq("train").sum()),
        "validation_rows": int(output["split"].eq("validation").sum()),
        "group_count": int(output["group_key"].nunique()),
        "group_overlap_count": len(overlap),
        "group_overlap": overlap,
        "canonical_family_count": int(output["canonical_feature_family"].nunique()),
        "multi_feature_family_count": int(
            output.groupby("canonical_feature_family").size().gt(1).sum()
        ),
        "train_validation_family_overlap_count": len(family_overlap),
        "train_validation_family_overlap": family_overlap,
        "region_rating_pair_same_split": bool(region_same_split),
        "region_rating_pair": region_features[
            ["feature_name", "split", "canonical_feature_family", "family_resolution_source", "family_resolution_rule"]
        ].to_dict("records"),
        "family_resolution": family_summary,
        "weak_grouping_row_count": weak_sources,
        "warnings": (
            ["weak feature-name fallback grouping used"] if weak_sources else []
        )
        + (["canonical family overlap exists"] if family_overlap else [])
        + ([] if region_same_split else ["REGION_RATING_CLIENT family pair is not in one split"]),
    }
    return GroupSplitResult(split=output, audit=audit, family_audit=family_audit, family_audit_summary=family_summary)


def save_group_split(result: GroupSplitResult, *, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_path = out / "homecredit_group_split.csv"
    audit_path = out / "group_split_audit.json"
    family_audit_path = out / "feature_family_audit.csv"
    family_audit_json_path = out / "feature_family_audit.json"
    result.split.to_csv(split_path, index=False)
    result.family_audit.to_csv(family_audit_path, index=False)
    write_json(audit_path, result.audit)
    family_payload = dict(result.family_audit_summary)
    family_payload.update(
        {
            "row_count": int(len(result.family_audit)),
            "train_validation_family_overlap_count": int(result.audit["train_validation_family_overlap_count"]),
            "train_validation_family_overlap": result.audit["train_validation_family_overlap"],
            "region_rating_pair_same_split": bool(result.audit["region_rating_pair_same_split"]),
            "region_rating_pair": result.audit["region_rating_pair"],
        }
    )
    write_json(family_audit_json_path, family_payload)
    return {
        "homecredit_group_split": split_path,
        "group_split_audit": audit_path,
        "feature_family_audit_csv": family_audit_path,
        "feature_family_audit_json": family_audit_json_path,
    }
