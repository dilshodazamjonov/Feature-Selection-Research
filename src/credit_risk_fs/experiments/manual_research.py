"""Fail-closed orchestration for the frozen cross-dataset voting research run.

Planning is deliberately pure.  The production path delegates every run to the
existing registered lifecycle and only coordinates the global DEV-to-OOT gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from credit_risk_fs.experiments.atomic_io import sha256_file
from credit_risk_fs.experiments.matrix import (
    CrossDatasetVotingRunSpec,
    cross_dataset_matrix_expansion_summary,
    expand_cross_dataset_voting_matrix,
)


EXPECTED_TAG = "cross-dataset-voting-pre-execution-v1"
MATRIX_PATH = "configs/experiments/cross_dataset_rank_voting_matrix_v1.yaml"
POLICY_PATH = "configs/execution/local_laptop_safe_v1.yaml"
REFINEMENT_PATH = "configs/execution/lendingclub_memory_safe_refinement_v1.yaml"
MANUAL_COMMAND = r".\.venv\Scripts\python.exe scripts\run_cross_dataset_voting_research.py"
FROZEN_HASHES = {
    "scientific_protocol": (
        "configs/protocols/credit_scoring_extension_v1.yaml",
        "f4137e98b7c89a63e9a73bc495190858f416c586d237f8ac003c8fdb9e40bde0",
    ),
    "row_alignment_contract": (
        "configs/protocols/row_alignment_contract_v1.json",
        "fc1064069f5cc45d76fd34060bb506869683e953c354d3f1f1a13327d99e71a0",
    ),
    "voting_protocol": (
        "configs/protocols/cross_dataset_rank_voting_v1.yaml",
        "51030e49716fae0a9c09b52628784a237e9581bb14c1a027e9c8238a575f0b49",
    ),
    "execution_policy": (
        POLICY_PATH,
        "1b77add8bf55096864934f6553aeba174cd22e2b112f53c7a45e5df327934012",
    ),
    "memory_refinement": (
        REFINEMENT_PATH,
        "4e2a17b93a751bbcb7443d8e82b15781f8a0467a07aa0037a3c298abff4132d7",
    ),
}
REQUIRED_PARALLELISM = {
    "concurrent_experiment_runs": 1,
    "concurrent_folds": 1,
    "data_loader_workers": 0,
    "estimator_threads": 4,
    "allow_nested_parallelism": False,
}


class ManualResearchStop(RuntimeError):
    """A stable, user-actionable stop from the manual workflow."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ManualResearchPlan:
    repository_root: str
    matrix_path: str
    matrix_sha256: str
    run_specs: tuple[CrossDatasetVotingRunSpec, ...]
    counts: dict[str, Any]
    frozen_hashes: dict[str, str]
    configuration_set_sha256: str
    expected_tag: str = EXPECTED_TAG
    manual_command: str = MANUAL_COMMAND

    def public_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "cross_dataset_voting_manual_plan_v1",
            "repository_root": self.repository_root,
            "matrix_path": self.matrix_path,
            "matrix_sha256": self.matrix_sha256,
            "run_ids": [item.run_id for item in self.run_specs],
            "counts": self.counts,
            "frozen_hashes": self.frozen_hashes,
            "configuration_set_sha256": self.configuration_set_sha256,
            "expected_tag": self.expected_tag,
            "manual_command": self.manual_command,
            "execution": "not_started_planning_only",
        }


@dataclass(frozen=True, slots=True)
class ReleaseProvenance:
    git_commit: str
    git_tag: str
    git_tag_object_type: str
    git_dirty: bool
    python_version: str
    platform: str
    pyproject_sha256: str
    dependency_lock_path: str
    dependency_lock_sha256: str


class WorkflowBackend(Protocol):
    def preflight(self, plan: ManualResearchPlan, provenance: ReleaseProvenance) -> None: ...
    def run_state(self, spec: CrossDatasetVotingRunSpec) -> str: ...
    def execute_dev(self, spec: CrossDatasetVotingRunSpec) -> str: ...
    def validate_dev(self, spec: CrossDatasetVotingRunSpec) -> None: ...
    def freeze_configuration_set(self, plan: ManualResearchPlan) -> str: ...
    def execute_oot(self, spec: CrossDatasetVotingRunSpec, frozen_set_sha256: str) -> str: ...
    def validate_oot(self, spec: CrossDatasetVotingRunSpec) -> None: ...
    def finalize(self, plan: ManualResearchPlan, frozen_set_sha256: str) -> None: ...
    def validate_complete(self, plan: ManualResearchPlan) -> None: ...


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manual_research_plan(repository_root: str | Path) -> ManualResearchPlan:
    """Authenticate frozen files and expand the matrix without any writes."""

    root = Path(repository_root).resolve()
    matrix = root / MATRIX_PATH
    specs = expand_cross_dataset_voting_matrix(matrix)
    counts = cross_dataset_matrix_expansion_summary(specs)
    observed_hashes: dict[str, str] = {}
    for name, (relative, expected) in FROZEN_HASHES.items():
        path = root / relative
        if not path.is_file():
            raise ManualResearchStop("FROZEN_INPUT_MISSING", f"missing frozen input: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ManualResearchStop(
                "FROZEN_INPUT_HASH_MISMATCH",
                f"{relative} expected {expected}, observed {observed}",
            )
        observed_hashes[name] = observed
    matrix_hash = sha256_file(matrix)
    configuration_set = {
        "matrix_sha256": matrix_hash,
        "frozen_hashes": observed_hashes,
        "run_specs": [asdict(item) for item in specs],
        "parallelism": REQUIRED_PARALLELISM,
        "accelerator": "cpu",
    }
    return ManualResearchPlan(
        repository_root=str(root),
        matrix_path=MATRIX_PATH,
        matrix_sha256=matrix_hash,
        run_specs=specs,
        counts=counts,
        frozen_hashes=observed_hashes,
        configuration_set_sha256=_canonical_hash(configuration_set),
    )


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise ManualResearchStop("GIT_PROVENANCE_UNAVAILABLE", detail.strip()) from exc
    return result.stdout.strip()


def authenticate_release(repository_root: str | Path) -> ReleaseProvenance:
    """Require the exact clean annotated release tag before data access."""

    root = Path(repository_root).resolve()
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise ManualResearchStop("GIT_DIRTY", "manual research requires a clean worktree")
    tag_commit = _git(root, "rev-list", "-n", "1", EXPECTED_TAG)
    if tag_commit != head:
        raise ManualResearchStop(
            "GIT_TAG_MISMATCH", f"{EXPECTED_TAG} does not resolve to HEAD {head}"
        )
    tag_type = _git(root, "cat-file", "-t", f"refs/tags/{EXPECTED_TAG}")
    if tag_type != "tag":
        raise ManualResearchStop("GIT_TAG_NOT_ANNOTATED", f"{EXPECTED_TAG} is not annotated")
    lock_path = root / "uv.lock"
    if not lock_path.is_file():
        lock_path = root / "requirements.txt"
    if not lock_path.is_file():
        raise ManualResearchStop("DEPENDENCY_LOCK_MISSING", "no dependency lock file found")
    return ReleaseProvenance(
        git_commit=head,
        git_tag=EXPECTED_TAG,
        git_tag_object_type=tag_type,
        git_dirty=False,
        python_version=sys.version,
        platform=platform.platform(),
        pyproject_sha256=sha256_file(root / "pyproject.toml"),
        dependency_lock_path=lock_path.relative_to(root).as_posix(),
        dependency_lock_sha256=sha256_file(lock_path),
    )


def execute_manual_workflow(
    plan: ManualResearchPlan,
    provenance: ReleaseProvenance,
    backend: WorkflowBackend,
) -> None:
    """Run every DEV configuration, close the global gate, then permit OOT."""

    backend.preflight(plan, provenance)
    for spec in plan.run_specs:
        state = backend.run_state(spec)
        if state == "completed":
            backend.validate_dev(spec)
            continue
        if state != "dev_complete":
            state = backend.execute_dev(spec)
        if state != "dev_complete":
            raise ManualResearchStop(
                "DEV_PHASE_INCOMPLETE", f"{spec.run_id} stopped with state {state}"
            )
        backend.validate_dev(spec)

    # This hash is recorded only after all sixteen DEV configurations validate.
    frozen_set_sha256 = backend.freeze_configuration_set(plan)
    if frozen_set_sha256 != plan.configuration_set_sha256:
        raise ManualResearchStop(
            "CONFIGURATION_SET_DRIFT",
            "validated DEV configuration set differs from the frozen plan",
        )

    for spec in plan.run_specs:
        state = backend.run_state(spec)
        if state == "completed":
            backend.validate_oot(spec)
            continue
        if state != "dev_complete":
            raise ManualResearchStop(
                "OOT_BARRIER_CLOSED", f"{spec.run_id} is not DEV-complete"
            )
        state = backend.execute_oot(spec, frozen_set_sha256)
        if state != "completed":
            raise ManualResearchStop(
                "OOT_PHASE_INCOMPLETE", f"{spec.run_id} stopped with state {state}"
            )
        backend.validate_oot(spec)

    backend.finalize(plan, frozen_set_sha256)
    backend.validate_complete(plan)


class CanonicalWorkflowBackend:
    """Adapter to canonical runner/validator owners; no scientific logic lives here."""

    def __init__(self, root: Path, provenance: ReleaseProvenance) -> None:
        self.root = root.resolve()
        self.provenance = provenance

    @staticmethod
    def _runner() -> Any:
        from credit_risk_fs.experiments import runner

        return runner

    def preflight(self, plan: ManualResearchPlan, provenance: ReleaseProvenance) -> None:
        self._runner().preflight_cross_dataset_research(self.root, plan, provenance)

    def run_state(self, spec: CrossDatasetVotingRunSpec) -> str:
        return self._runner().cross_dataset_research_run_state(self.root, spec)

    def execute_dev(self, spec: CrossDatasetVotingRunSpec) -> str:
        return self._runner().execute_cross_dataset_research_phase(
            self.root, spec, phase="dev", provenance=self.provenance
        )

    def validate_dev(self, spec: CrossDatasetVotingRunSpec) -> None:
        self._runner().validate_cross_dataset_research_run(self.root, spec, phase="dev")

    def freeze_configuration_set(self, plan: ManualResearchPlan) -> str:
        return self._runner().freeze_cross_dataset_configuration_set(
            self.root, plan, self.provenance
        )

    def execute_oot(self, spec: CrossDatasetVotingRunSpec, frozen_set_sha256: str) -> str:
        return self._runner().execute_cross_dataset_research_phase(
            self.root,
            spec,
            phase="oot",
            provenance=self.provenance,
            frozen_set_sha256=frozen_set_sha256,
        )

    def validate_oot(self, spec: CrossDatasetVotingRunSpec) -> None:
        self._runner().validate_cross_dataset_research_run(self.root, spec, phase="oot")

    def finalize(self, plan: ManualResearchPlan, frozen_set_sha256: str) -> None:
        self._runner().finalize_cross_dataset_research(
            self.root, plan, frozen_set_sha256
        )

    def validate_complete(self, plan: ManualResearchPlan) -> None:
        self._runner().validate_completed_cross_dataset_research(self.root, plan)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete frozen cross-dataset voting research workflow."
    )
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Authenticate and print the frozen matrix without creating results or loading data.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    try:
        plan = build_manual_research_plan(root)
        if args.plan:
            print(json.dumps(plan.public_payload(), indent=2, sort_keys=True))
            return 0
        provenance = authenticate_release(root)
        backend = CanonicalWorkflowBackend(root, provenance)
        execute_manual_workflow(plan, provenance, backend)
    except ManualResearchStop as exc:
        print(f"CONTROLLED_STOP {exc.code}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"CONTROLLED_STOP UNEXPECTED_WORKFLOW_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print("CROSS_DATASET_VOTING_RESEARCH_COMPLETE")
    return 0


__all__ = [
    "CanonicalWorkflowBackend",
    "EXPECTED_TAG",
    "FROZEN_HASHES",
    "MANUAL_COMMAND",
    "ManualResearchPlan",
    "ManualResearchStop",
    "ReleaseProvenance",
    "REQUIRED_PARALLELISM",
    "authenticate_release",
    "build_manual_research_plan",
    "execute_manual_workflow",
    "main",
]
