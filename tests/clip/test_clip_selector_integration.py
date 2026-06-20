from __future__ import annotations

import subprocess
import sys

from credit_risk_fs.clip.selector_validation import load_clip_selector_config, validate_clip_selector_binding


def test_selector_binding_uses_homecredit_anchor_and_lendingclub_v2_external_only():
    config = load_clip_selector_config("configs/clip/selector.yaml")
    binding = validate_clip_selector_binding(config)

    assert set(config.active_datasets) == {"homecredit", "lendingclub_v2"}
    assert "lendingclub" not in set(config.active_datasets)
    assert config.no_refit is True
    assert binding["checkpoint_hash"] == config.checkpoint_hash
    assert binding["anchor_hash"] == config.anchor_hash
    assert binding["statistical_view_scope"] == "missingness_only"


def test_selector_integration_dry_run_command_succeeds():
    completed = subprocess.run(
        [sys.executable, "scripts/validate_clip_selector_integration.py", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert '"status": "passed"' in completed.stdout
    assert '"lendingclub_v2_used_for_selection": false' in completed.stdout
