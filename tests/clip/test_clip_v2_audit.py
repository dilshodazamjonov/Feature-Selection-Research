from __future__ import annotations

from types import SimpleNamespace

from scripts import audit_clip_v2


def test_git_status_is_recorded_not_blocking(monkeypatch, tmp_path):
    monkeypatch.setattr(audit_clip_v2, "verify_freeze_package", lambda: {"status": "passed"})
    monkeypatch.setattr(audit_clip_v2, "validate_clip_v2_config", lambda raw: [])
    monkeypatch.setattr(audit_clip_v2.Path, "glob", lambda self, pattern: [])
    monkeypatch.setattr(
        audit_clip_v2,
        "_safe_json",
        lambda path: {
            "fit_dataset": "homecredit",
            "fit_split": "train",
            "lendingclub_v2_used_for_selection": False,
            "complete": True,
            "run_count": 8,
            "status": "complete",
            "source_artifacts": [{"path": "x", "sha256": "h"}],
        },
    )
    monkeypatch.setattr(
        audit_clip_v2.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=" M changed.py\n?? generated.csv\n"),
    )
    monkeypatch.setattr(audit_clip_v2, "EXPECTED_RUNS", set())

    checks = audit_clip_v2.run_audit(tmp_path, include_git=True)
    git_check = next(check for check in checks if check["check"] == "git_status_recorded")

    assert git_check["passed"] is True
    assert git_check["details"] == "dirty_entries=2"
    assert "git_clean" not in {check["check"] for check in checks}
