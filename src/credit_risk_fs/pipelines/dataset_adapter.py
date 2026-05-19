from __future__ import annotations

from dataclasses import dataclass

from credit_risk_fs.data.dataset_registry import get_dataset_config


@dataclass(slots=True)
class DatasetAdapter:
    name: str
    mode: str


def resolve_dataset_mode(*, config, loaded_frames: dict | None = None) -> str:
    if getattr(config, "dataset_name", None):
        try:
            return get_dataset_config(str(config.dataset_name)).mode
        except ValueError:
            pass
    frame_names = set((loaded_frames or {}).keys())
    if {"bureau", "previous_application"} & frame_names:
        return "homecredit_multitable"
    return "single_table"
