from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from credit_risk_fs.experiments import manual_research


ROOT = Path(__file__).resolve().parents[1]

#: The frozen canonical cross_dataset_rank_voting_v1 matrix, written out in full so
#: the preservation guard compares identities rather than a bare count. Cross-checked
#: against results/comparisons/cross_dataset_voting_configuration_lock.json, whose
#: status is `all_dev_validated_oot_configuration_locked`.
CANONICAL_CDV1_RUN_IDS = frozenset(
    {
        "cdv1-001-homecredit-reference-rf-corr-mrmr-lr-s42",
        "cdv1-002-homecredit-voting-k100-lr-s42",
        "cdv1-003-homecredit-voting-k200-lr-s42",
        "cdv1-004-homecredit-voting-k300-lr-s42",
        "cdv1-005-homecredit-reference-rf-corr-mrmr-catboost-s42",
        "cdv1-006-homecredit-voting-k100-catboost-s42",
        "cdv1-007-homecredit-voting-k200-catboost-s42",
        "cdv1-008-homecredit-voting-k300-catboost-s42",
        "cdv1-009-lendingclub-v2-reference-rf-corr-mrmr-lr-s42",
        "cdv1-010-lendingclub-v2-voting-k100-lr-s42",
        "cdv1-011-lendingclub-v2-voting-k200-lr-s42",
        "cdv1-012-lendingclub-v2-voting-k300-lr-s42",
        "cdv1-013-lendingclub-v2-reference-rf-corr-mrmr-catboost-s42",
        "cdv1-014-lendingclub-v2-voting-k100-catboost-s42",
        "cdv1-015-lendingclub-v2-voting-k200-catboost-s42",
        "cdv1-016-lendingclub-v2-voting-k300-catboost-s42",
    }
)


def _provenance() -> manual_research.ReleaseProvenance:
    return manual_research.ReleaseProvenance(
        git_commit="a" * 40,
        git_tag=manual_research.EXPECTED_TAG,
        git_tag_object_type="tag",
        git_dirty=False,
        python_version="fixture",
        platform="fixture",
        pyproject_sha256="b" * 64,
        dependency_lock_path="uv.lock",
        dependency_lock_sha256="c" * 64,
    )


class _Backend:
    def __init__(self, states=None, *, freeze_hash=None, fail_dev=None):
        self.states = dict(states or {})
        self.freeze_hash = freeze_hash
        self.fail_dev = fail_dev
        self.events: list[tuple[str, str | None]] = []

    def preflight(self, plan, provenance):
        self.events.append(("preflight", None))

    def ensure_ready(self, previous_run_id, next_run_id, phase):
        self.events.append((f"ready_{phase}", f"{previous_run_id}->{next_run_id}"))

    def run_state(self, spec):
        self.events.append(("state", spec.run_id))
        return self.states.get(spec.run_id, "missing")

    def execute_dev(self, spec):
        self.events.append(("dev", spec.run_id))
        state = "failed" if spec.run_id == self.fail_dev else "dev_complete"
        self.states[spec.run_id] = state
        return state

    def validate_dev(self, spec):
        self.events.append(("validate_dev", spec.run_id))

    def freeze_configuration_set(self, plan):
        self.events.append(("freeze", None))
        return self.freeze_hash or plan.configuration_set_sha256

    def execute_oot(self, spec, frozen_set_sha256):
        self.events.append(("oot", spec.run_id))
        self.states[spec.run_id] = "completed"
        return "completed"

    def validate_oot(self, spec):
        self.events.append(("validate_oot", spec.run_id))

    def finalize(self, plan, frozen_set_sha256):
        self.events.append(("finalize", None))

    def validate_complete(self, plan):
        self.events.append(("validate_complete", None))


def test_manual_plan_expands_exact_frozen_matrix_and_limits():
    plan = manual_research.build_manual_research_plan(ROOT)
    assert plan.counts["total_registered_runs"] == 16
    assert plan.counts["voting_runs"] == 12
    assert plan.counts["reference_reruns"] == 4
    assert plan.counts["total_dev_fold_executions"] == 80
    assert plan.counts["final_full_dev_oot_fits"] == 16
    assert manual_research.REQUIRED_PARALLELISM == {
        "concurrent_experiment_runs": 1,
        "concurrent_folds": 1,
        "data_loader_workers": 0,
        "estimator_threads": 4,
        "allow_nested_parallelism": False,
    }


def test_plan_mode_creates_no_index_rows_or_run_directories(
    capsys, monkeypatch, tmp_path
):
    monkeypatch.setattr(manual_research, "authenticate_release", lambda root: _provenance())
    index = ROOT / "results/run_index.csv"
    before_index = index.read_bytes()
    before_dirs = sorted(path.resolve() for path in (ROOT / "results/runs").glob("*/*"))
    assert manual_research.main(
        [
            "--plan",
            "--repository-root",
            str(ROOT),
            "--log-file",
            str(tmp_path / "logs" / "runs.log"),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution"] == "not_started_planning_only"
    assert index.read_bytes() == before_index
    assert sorted(path.resolve() for path in (ROOT / "results/runs").glob("*/*")) == before_dirs


def test_all_dev_validation_precedes_first_oot_and_configuration_is_frozen():
    plan = manual_research.build_manual_research_plan(ROOT)
    backend = _Backend()
    manual_research.execute_manual_workflow(plan, _provenance(), backend)
    first_oot = next(index for index, event in enumerate(backend.events) if event[0] == "oot")
    dev_validations = [event for event in backend.events[:first_oot] if event[0] == "validate_dev"]
    assert len(dev_validations) == 16
    assert backend.events[first_oot - 2][0] == "ready_oot"
    assert backend.events[first_oot - 1] == ("state", plan.run_specs[0].run_id)
    assert ("freeze", None) in backend.events[:first_oot]
    assert backend.events[-2:] == [("finalize", None), ("validate_complete", None)]


def test_one_incomplete_dev_configuration_keeps_oot_closed():
    plan = manual_research.build_manual_research_plan(ROOT)
    backend = _Backend(fail_dev=plan.run_specs[7].run_id)
    with pytest.raises(manual_research.ManualResearchStop) as error:
        manual_research.execute_manual_workflow(plan, _provenance(), backend)
    assert error.value.code == "DEV_PHASE_INCOMPLETE"
    assert not any(event[0] == "oot" for event in backend.events)


def test_oot_receives_only_the_frozen_validated_configuration_set():
    plan = manual_research.build_manual_research_plan(ROOT)
    backend = _Backend(freeze_hash="d" * 64)
    with pytest.raises(manual_research.ManualResearchStop) as error:
        manual_research.execute_manual_workflow(plan, _provenance(), backend)
    assert error.value.code == "CONFIGURATION_SET_DRIFT"
    assert not any(event[0] == "oot" for event in backend.events)


def test_resume_reuses_valid_dev_and_completed_runs_without_repeating_stages():
    plan = manual_research.build_manual_research_plan(ROOT)
    states = {
        **{item.run_id: "completed" for item in plan.run_specs[:4]},
        **{item.run_id: "dev_complete" for item in plan.run_specs[4:]},
    }
    backend = _Backend(states)
    manual_research.execute_manual_workflow(plan, _provenance(), backend)
    assert not any(event[0] == "dev" for event in backend.events)
    assert [event[1] for event in backend.events if event[0] == "oot"] == [
        item.run_id for item in plan.run_specs[4:]
    ]
    assert len([event for event in backend.events if event[0] == "validate_oot"]) == 16


def test_release_authentication_rejects_tag_or_commit_drift(monkeypatch):
    def fake_git(root, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "a" * 40
        if args[:2] == ("status", "--porcelain"):
            return ""
        if args[:3] == ("rev-list", "-n", "1"):
            return "b" * 40
        raise AssertionError(args)

    monkeypatch.setattr(manual_research, "_git", fake_git)
    with pytest.raises(manual_research.ManualResearchStop) as error:
        manual_research.authenticate_release(ROOT)
    assert error.value.code == "GIT_TAG_MISMATCH"


def test_ignored_runtime_logs_do_not_break_clean_release_authentication(tmp_path):
    (tmp_path / ".gitignore").write_text(
        "/logs/runs.log\n/logs/events.jsonl\n/logs/debug.log\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'logging-auth-fixture'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    identity = [
        "-c",
        "user.name=Authentication Test",
        "-c",
        "user.email=authentication-test@example.invalid",
    ]
    subprocess.run(
        ["git", *identity, "commit", "--quiet", "-m", "fixture release"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            *identity,
            "tag",
            "--annotate",
            manual_research.EXPECTED_TAG,
            "--message",
            "fixture observability release",
        ],
        cwd=tmp_path,
        check=True,
    )
    with manual_research.ResearchLogSession(
        Path("logs/runs.log"),
        repository_root=tmp_path,
        command_arguments=[],
    ) as session:
        session.finish("session_completed", message="authentication fixture complete")
    provenance = manual_research.authenticate_release(tmp_path)
    assert provenance.git_tag == manual_research.EXPECTED_TAG
    assert subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""


def test_controlled_failure_is_nonzero_and_reports_stable_reason(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        manual_research,
        "build_manual_research_plan",
        lambda root: (_ for _ in ()).throw(
            manual_research.ManualResearchStop("FIXTURE_STOP", "fixture failure")
        ),
    )
    log_path = tmp_path / "logs" / "runs.log"
    assert manual_research.main(
        ["--repository-root", str(ROOT), "--log-file", str(log_path)]
    ) == 2
    terminal = capsys.readouterr().err
    assert "STOP  | fixture failure" in terminal
    assert "CONTROLLED_STOP" not in terminal
    assert "Traceback (most recent call last)" not in terminal
    assert "ManualResearchStop" not in (
        tmp_path / "logs" / "debug.log"
    ).read_text(encoding="utf-8")


def test_unexpected_failure_prints_one_error_and_routes_traceback(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        manual_research,
        "build_manual_research_plan",
        lambda root: (_ for _ in ()).throw(RuntimeError("fixture explosion")),
    )
    log_path = tmp_path / "logs" / "runs.log"
    assert manual_research.main(
        ["--repository-root", str(ROOT), "--log-file", str(log_path)]
    ) == 3
    terminal = capsys.readouterr().err
    assert terminal.count("ERROR |") == 1
    assert "RuntimeError: fixture explosion" in terminal
    assert "Traceback (most recent call last)" not in terminal
    assert "Traceback (most recent call last)" in (
        tmp_path / "logs" / "debug.log"
    ).read_text(encoding="utf-8")


def test_keyboard_interrupt_prints_one_short_stop_without_traceback(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        manual_research,
        "build_manual_research_plan",
        lambda root: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    log_path = tmp_path / "logs" / "runs.log"
    assert manual_research.main(
        ["--repository-root", str(ROOT), "--log-file", str(log_path)]
    ) == 130
    terminal = capsys.readouterr().err
    assert terminal.count("STOP  | Research run interrupted manually") == 1
    assert "Traceback (most recent call last)" not in terminal
    assert "KeyboardInterrupt" not in (
        tmp_path / "logs" / "debug.log"
    ).read_text(encoding="utf-8")


def test_orchestration_contract_never_reaches_real_loaders_or_estimators(monkeypatch):
    from credit_risk_fs import pipelines
    from credit_risk_fs.models import registry as model_registry
    from credit_risk_fs.pipelines import common

    def forbidden(*args, **kwargs):
        raise AssertionError("real scientific path reached")

    monkeypatch.setattr(common, "prepare_voting_pilot_dev_data", forbidden)
    monkeypatch.setattr(common, "prepare_voting_research_oot_data", forbidden)
    monkeypatch.setattr(model_registry, "get_model_bundle", forbidden)
    plan = manual_research.build_manual_research_plan(ROOT)
    manual_research.execute_manual_workflow(plan, _provenance(), _Backend())
    assert pipelines is not None


def test_runbook_has_exactly_one_supported_manual_launch_command():
    runbook = (
        ROOT / "docs/research_extension/cross_dataset_voting_research_runbook_v1.md"
    ).read_text(encoding="utf-8")
    assert "## One-command manual execution" in runbook
    assert runbook.count(manual_research.MANUAL_COMMAND) == 1
    assert manual_research.EXPECTED_TAG in runbook
    assert "Prompt 6 did **not** execute the command" in runbook


def test_resume_handoff_has_one_supported_command_and_exact_run_014_boundary():
    handoff = (
        ROOT
        / "docs/research_extension/cross_dataset_voting_resume_after_run_014_v1.md"
    ).read_text(encoding="utf-8")
    assert "## One-command manual resume" in handoff
    assert handoff.count(manual_research.MANUAL_COMMAND) == 1
    assert manual_research.EXPECTED_TAG in handoff
    assert "DEV fold 5 at `dev_data_loading`" in handoff
    assert "Prompt 6.2 did not execute the resume command" in handoff


def test_existing_pilots_and_isolated_capacity_evidence_remain_unchanged():
    rows = (ROOT / "results/run_index.csv").read_text(encoding="utf-8").splitlines()[1:]
    pilot_rows = [row for row in rows if row.startswith("cdv1-pilot-")]
    assert len(pilot_rows) == 4
    pilot_dirs = sorted((ROOT / "results/runs").glob("*/cdv1-pilot-*"))
    assert len(pilot_dirs) == 4
    capacity = ROOT / "cleanup/audits/lendingclub_memory_refinement_capacity_gate"
    validation = json.loads((capacity / "validation_summary.json").read_text(encoding="utf-8"))
    assert validation["final_gate"] == "PASS"
    assert len(list((capacity / "capacity_execution/runs/lendingclub_v2").glob("*"))) == 3

    # The canonical matrix contains exactly these 16 runs. The expectation is
    # written out literally rather than derived from the directory listing, so the
    # guard still fails if a run directory is added, removed, or renamed. The
    # frozen configuration lock is cross-checked against the literal set, so a
    # divergence between this test and the authoritative registry also fails.
    observed = {path.name for path in (ROOT / "results/runs").glob("*/cdv1-0[01][0-9]-*")}
    assert observed == CANONICAL_CDV1_RUN_IDS
    locked = json.loads(
        (ROOT / "results/comparisons/cross_dataset_voting_configuration_lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(locked["run_ids"]) == CANONICAL_CDV1_RUN_IDS
    assert len(locked["run_ids"]) == 16
