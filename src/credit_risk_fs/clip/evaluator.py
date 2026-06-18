from __future__ import annotations

import pandas as pd


def summarize_training_evidence(frame: pd.DataFrame) -> dict[str, int]:
    allowed = frame["allowed_for_clip_training"].astype(bool) if "allowed_for_clip_training" in frame else pd.Series(dtype=bool)
    return {
        "total_rows": int(len(frame)),
        "allowed_for_clip_training": int(allowed.sum()) if len(allowed) else 0,
        "blocked_for_clip_training": int((~allowed).sum()) if len(allowed) else int(len(frame)),
    }

