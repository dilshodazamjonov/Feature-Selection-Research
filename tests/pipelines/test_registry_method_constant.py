from __future__ import annotations

import inspect
from pathlib import Path

from credit_risk_fs.clip.reverse_transfer import REVERSE_METHOD
from credit_risk_fs.pipelines import reverse_transfer


def test_register_payload_construction_resolves_canonical_method() -> None:
    closure = inspect.getclosurevars(reverse_transfer._register)
    assert closure.globals["REVERSE_METHOD"] == REVERSE_METHOD
    assert REVERSE_METHOD == "lendingclub_clip_to_homecredit_mrmr"


def test_pipeline_has_no_duplicate_reverse_method_literal() -> None:
    source = Path(
        "src/credit_risk_fs/pipelines/reverse_transfer.py"
    ).read_text(encoding="utf-8")
    assert "lendingclub_clip_to_homecredit_mrmr" not in source
    assert source.count("REVERSE_METHOD") >= 5


def test_register_dry_run_uses_real_payload_builder_without_execute(
    monkeypatch,
    capsys,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        reverse_transfer,
        "load_config_dir",
        lambda path: {"loaded": True},
    )
    monkeypatch.setattr(
        reverse_transfer,
        "resolve_plan",
        lambda **kwargs: {"resolved_stages": ["register"]},
    )

    def fake_register(**kwargs):
        calls.append(kwargs)
        return {
            "transaction_outcome": "NEW_TRANSACTION",
            "canonical_method": REVERSE_METHOD,
            "writes_performed": False,
            "success_transaction_manifest_written": False,
        }

    monkeypatch.setattr(reverse_transfer, "_register", fake_register)
    monkeypatch.setattr(
        reverse_transfer,
        "execute_plan",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not execute stages")
        ),
    )
    result = reverse_transfer.run_cli(
        [
            "--stage",
            "register",
            "--config-dir",
            "unused",
            "--output-dir",
            "unused",
            "--dry-run",
        ]
    )
    assert result == 0
    assert len(calls) == 1
    assert calls[0]["dry_run"] is True
    output = capsys.readouterr().out
    assert f'"canonical_method": "{REVERSE_METHOD}"' in output
    assert '"writes_performed": false' in output


def test_commit_path_calls_same_register_builder_without_dry_run(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        reverse_transfer,
        "load_config_dir",
        lambda path: {"loaded": True},
    )
    monkeypatch.setattr(
        reverse_transfer,
        "resolve_plan",
        lambda **kwargs: {"resolved_stages": ["register"]},
    )
    monkeypatch.setattr(
        reverse_transfer,
        "execute_plan",
        lambda **kwargs: calls.append(kwargs),
    )
    assert (
        reverse_transfer.run_cli(
            [
                "--stage",
                "register",
                "--config-dir",
                "unused",
                "--output-dir",
                "unused",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0]["stages"] == ("register",)
