from __future__ import annotations

from pathlib import Path

import pandas as pd


def apply_leakage_blacklist(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    return df.drop(columns=[column for column in columns if column in df.columns], errors="ignore")


def read_leakage_blacklist(path: str | Path) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
