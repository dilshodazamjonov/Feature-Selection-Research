"""Description-file helpers for metadata generation."""

from pathlib import Path

import pandas as pd

from credit_risk_fs.feature_metadata.builder import build_feature_metadata


def load_descriptions(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


__all__ = ["build_feature_metadata", "load_descriptions"]
