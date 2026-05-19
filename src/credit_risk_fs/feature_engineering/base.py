from __future__ import annotations

from typing import Protocol

import pandas as pd


class FeatureAssembler(Protocol):
    def __call__(self, dataframes: dict[str, pd.DataFrame]) -> list[pd.DataFrame]: ...
