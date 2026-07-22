from __future__ import annotations

import pytest

from credit_risk_fs.experiments.result_paths import (
    append_run_index_row,
    initialize_results_layout,
    update_run_index_row,
)


def test_interrupted_run_index_replacement_preserves_previous_registry(tmp_path, monkeypatch):
    results = initialize_results_layout(tmp_path)
    append_run_index_row(
        results,
        {
            "run_id": "run-1",
            "dataset": "synthetic",
            "selector": "none",
            "model": "lr",
            "status": "running",
        },
    )
    before = (results / "run_index.csv").read_bytes()

    def fail_replace(_source, _target):
        raise OSError("simulated interrupted registry replace")

    monkeypatch.setattr("credit_risk_fs.experiments.atomic_io.os.replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        update_run_index_row(results, "run-1", {"status": "completed"})
    assert (results / "run_index.csv").read_bytes() == before
