from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def src_root() -> Path:
    return project_root() / "src"


def dataset_root(dataset_name: str) -> Path:
    return project_root() / "data" / dataset_name.lower()


def results_root(dataset_name: str) -> Path:
    return project_root() / "results" / dataset_name.lower()
