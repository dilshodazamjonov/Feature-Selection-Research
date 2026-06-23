from __future__ import annotations

import json
import subprocess
import sys

from credit_risk_fs.clip_final_comparison.constants import STAGES
from credit_risk_fs.clip_final_comparison.io import assert_isolated_output_path


def test_pipeline_plan_command_is_dry_and_reports_matrix():
    result = subprocess.run(
        [sys.executable, "scripts/run_clip_final_comparison.py", "--plan"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["execute"] is False
    assert payload["implementation_mode"] == "executable_research_pipeline"
    assert payload["matrix"]["core_candidate_pool_runs"] == 184
    assert payload["stage_order"] == list(STAGES)


def test_pipeline_status_command_is_safe():
    result = subprocess.run(
        [sys.executable, "scripts/run_clip_final_comparison.py", "--status"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["output_root"] == "results/clip_final_comparison"
    assert payload["completed_scientific_runs"] == 0


def test_output_path_isolation_rejects_canonical_result_roots():
    assert assert_isolated_output_path("results/clip_final_comparison/audit/x.json")
    try:
        assert_isolated_output_path("results/clip_v2/audit/x.json")
    except ValueError as exc:
        assert "results/clip_final_comparison" in str(exc)
    else:
        raise AssertionError("canonical CLIP-v2 path should be rejected")
