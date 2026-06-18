from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import pandas as pd

from credit_risk_fs.utils.hashing import sha256_text
from credit_risk_fs.utils.io import write_json


@dataclass(frozen=True)
class GroupSplitResult:
    split: pd.DataFrame
    audit: dict[str, object]


def derive_feature_family(feature_name: str) -> str:
    name = str(feature_name)
    for token in ["_MEAN", "_MAX", "_MIN", "_SUM", "_VAR", "_AVG", "_MEDI", "_MODE"]:
        if name.endswith(token):
            return name[: -len(token)]
    name = re.sub(r"(_is_zero|_missing_flag|_flag|_ratio|_share)$", "", name)
    parts = name.split("_")
    if len(parts) > 3:
        return "_".join(parts[:3])
    return name


def build_group_split(
    frame: pd.DataFrame,
    *,
    dataset: str = "homecredit",
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> GroupSplitResult:
    if dataset != "homecredit":
        raise ValueError("group-aware split is only fit on homecredit")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    working = frame.copy()
    group_keys = []
    group_sources = []
    for row in working.to_dict("records"):
        feature = str(row["feature_name"])
        source = str(row.get("source_table", "") or "")
        semantic = str(row.get("semantic_group", "") or "")
        family = derive_feature_family(feature)
        if family and family != feature:
            group_keys.append(f"family:{family}")
            group_sources.append("derived_feature_family")
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
    output = working[["dataset", "feature_name", "split", "group_key", "group_source"]].copy()
    output["seed"] = int(seed)
    output = output.sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)

    train_groups = set(output.loc[output["split"].eq("train"), "group_key"])
    validation_groups_observed = set(output.loc[output["split"].eq("validation"), "group_key"])
    overlap = sorted(train_groups.intersection(validation_groups_observed))
    weak_sources = int(output["group_source"].eq("feature_name_fallback").sum())
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
        "weak_grouping_row_count": weak_sources,
        "warnings": ["weak feature-name fallback grouping used"] if weak_sources else [],
    }
    return GroupSplitResult(split=output, audit=audit)


def save_group_split(result: GroupSplitResult, *, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_path = out / "homecredit_group_split.csv"
    audit_path = out / "group_split_audit.json"
    result.split.to_csv(split_path, index=False)
    write_json(audit_path, result.audit)
    return {"homecredit_group_split": split_path, "group_split_audit": audit_path}
