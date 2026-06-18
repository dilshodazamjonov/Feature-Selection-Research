from __future__ import annotations

from typing import Mapping


def build_feature_training_text(row: Mapping[str, object]) -> str:
    """Build a compact DEV-only feature text record for future CLIP encoders."""
    parts = [
        f"feature={row.get('feature', '')}",
        f"semantic_group={row.get('semantic_group', '')}",
        f"source_table={row.get('source_table', '')}",
        f"dtype={row.get('dtype_if_available', '')}",
        f"description={row.get('description', '')}",
    ]
    for source, label in [
        ("missing_rate_dev", "dev_missing_rate"),
        ("iv_score_if_available", "dev_iv"),
        ("llm_best_rank", "dev_llm_best_rank"),
    ]:
        value = row.get(source)
        if value not in (None, ""):
            parts.append(f"{label}={value}")
    return " | ".join(str(part) for part in parts)

