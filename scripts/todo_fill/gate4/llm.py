"""Target-free ranking calls (snapshot ``gpt-4.1-mini-2025-04-14``, temperature 0, exactly 100 names).

Every call goes through the B6 guarded client (``scripts/run_b6_obfuscation_ablation.py``):
the rendered request is checked for outcome evidence, for the presence of every candidate
identifier and, in the obfuscated arm, for the absence of every original name.  Responses
are validated with the strict contract (100 distinct known names, up to three application
attempts, no fallback) and checkpointed so a crash never repeats a paid call.  The provider
response id is kept in the clear because step 45 owes it.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from scripts.todo_fill.common import FULL_DEV, THIRD, Manifest, load_dotenv_if_present, log, read_json, sha256_text, write_json
from scripts.todo_fill.gate4.records import RecordFactory

#: Columns of the contract-issue sidecar.  The counting block mirrors the paper's
#: ``priority_2/repeated_calls_contract_issues.csv`` so the two files read side by
#: side; the rest identifies the call this run made.
CONTRACT_ISSUE_COLUMNS: tuple[str, ...] = (
    "step",
    "call",
    "dataset",
    "partition",
    "condition",
    "budget",
    "repeat_id",
    "issue",
    "attempt",
    "n_names",
    "n_distinct",
    "n_unknown",
    "n_duplicates",
    "accepted",
    "validation_error",
)


def _attempt_counts(record: dict[str, Any], ids: Sequence[str]) -> dict[str, Any]:
    """Recover the contract counters from one recorded attempt's raw content."""

    content = str(((record.get("response") or {}).get("raw_content")) or "")
    try:
        selected = json.loads(content).get("selected_features")
    except (json.JSONDecodeError, AttributeError, TypeError):
        selected = None
    if not isinstance(selected, list):
        return {"n_names": "", "n_distinct": "", "n_unknown": "", "n_duplicates": ""}
    names = [str(value) for value in selected]
    known = set(map(str, ids))
    return {
        "n_names": len(names),
        "n_distinct": len(set(names)),
        "n_unknown": len([name for name in names if name not in known]),
        "n_duplicates": len(names) - len(set(names)),
    }


def _b6():
    import scripts.b6_obfuscation as helpers
    import scripts.run_b6_obfuscation_ablation as runner

    return helpers, runner


class Ranker:
    def __init__(self, work_dir: Path, manifest: Manifest, enabled: bool | None = None, retry_failed: int = 0) -> None:
        load_dotenv_if_present()
        self.work_dir = Path(work_dir)
        self.manifest = manifest
        self.enabled = bool(os.getenv("OPENAI_API_KEY")) if enabled is None else enabled
        self._client: Any = None
        self.calls_made = 0
        #: extra re-issues of a call that exhausted the frozen three attempts; 0 keeps
        #: the frozen protocol exactly (three attempts, no fallback, then the cell stays blank)
        self.retry_failed = max(0, int(retry_failed))
        self.calls_failed = 0
        self.contract_issues: list[dict[str, Any]] = []

    def _openai(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def path(self, step: str, name: str) -> Path:
        path = self.work_dir / "gate4" / "rankings" / step / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def cached(self, step: str, name: str) -> dict[str, Any] | None:
        path = self.path(step, name)
        return read_json(path) if path.exists() else None

    def rank(
        self,
        *,
        step: str,
        name: str,
        dataset: str,
        partition: str,
        condition: str,
        records: Sequence[dict[str, Any]],
        ids: Sequence[str],
        universe: Sequence[str],
        reverse: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """One guarded call; ``condition`` is ``named`` or ``obfuscated`` (arm A) for the payload guard."""

        found = self.cached(step, name)
        if found is not None:
            return found
        if not self.enabled:
            self.manifest.skip(step, {"call": name}, "OPENAI_API_KEY is not set (fill .env) or API calls are disabled; ranking call not made")
            return None
        helpers, runner = _b6()
        selector = runner._selector()
        selector._client = runner._GuardedClient(
            self._openai(), condition=condition, candidate_records=list(records), candidate_ids=list(ids), original_feature_names=list(universe)
        )
        log(f"[{step}] ranking call {name}: {dataset}/{partition}, {len(ids)} candidates, condition={condition}")
        started = time.perf_counter()
        meta = {"step": step, "call": name, "dataset": dataset, "partition": partition, "condition": condition, "budget": (extra or {}).get("budget", ""), "repeat_id": (extra or {}).get("repeat_id", "")}
        payload: dict[str, Any] | None = None
        ranked: list[str] = []
        attempts: list[dict[str, Any]] = []
        errors: list[str] = []
        # One issue = the frozen contract (three application attempts, no fallback).  A
        # failed issue costs this one cell, never the rest of the step; ``--retry-failed-calls``
        # re-issues it, which is a disclosed deviation and is off by default.
        for issue in range(1, self.retry_failed + 2):
            attempts = []
            try:
                payload = selector.rank_target_free(
                    list(records),
                    expected_features=list(ids),
                    expected_response_model=helpers.MODEL_SNAPSHOT,
                    attempt_recorder=lambda record: attempts.append({key: value for key, value in record.items() if key != "request"}),
                    maximum_attempts=3,
                )
                ranked = list(
                    helpers.validate_ranking_response(
                        payload, candidate_ids=list(ids), original_feature_names=list(universe) if condition == "obfuscated" else ()
                    )
                )
            except Exception as exc:  # noqa: BLE001  (contract failure, transport error or guard rejection)
                payload = None
                errors.append(f"issue={issue}: {type(exc).__name__}: {exc}")
                self._record_issue(meta, issue, attempts, ids, accepted=False, fallback_error=f"{type(exc).__name__}: {exc}")
                log(f"[{step}] ranking call {name} failed issue {issue}/{self.retry_failed + 1}: {type(exc).__name__}: {exc}")
                continue
            self._record_issue(meta, issue, attempts, ids, accepted=True)
            break
        if payload is None:
            self.calls_failed += 1
            write_json(self.path(step, f"{name}.failed"), {**meta, "n_candidates": len(ids), "issues": self.retry_failed + 1, "errors": errors, "attempts": attempts, "failed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            self.manifest.skip(step, {"call": name, "dataset": dataset, "partition": partition, "condition": condition}, f"ranking call failed the frozen target-free contract; cells left blank: {errors[-1] if errors else 'unknown error'}")
            return None
        self.calls_made += 1
        original = [reverse[item] for item in ranked] if reverse else list(ranked)
        result = {
            "step": step,
            "name": name,
            "dataset": dataset,
            "partition": partition,
            "condition": condition,
            "n_candidates": len(ids),
            "ranking_ids": ranked,
            "ranking": original,
            "response_id": payload.get("response_id"),
            "response_model": payload.get("response_model"),
            "request_model": payload.get("request_model"),
            "prompt_sha256": payload.get("prompt_sha256"),
            "application_attempt": payload.get("application_attempt"),
            "attempt_count": len(attempts),
            "prompt_tokens": payload.get("prompt_tokens"),
            "completion_tokens": payload.get("completion_tokens"),
            "seconds": time.perf_counter() - started,
            "accepted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attempts": attempts,
            **(extra or {}),
        }
        write_json(self.path(step, name), result)
        return result

    # ------------------------------------------------------------------ contract issues
    def _record_issue(self, meta: dict[str, Any], issue: int, attempts: Sequence[dict[str, Any]], ids: Sequence[str], *, accepted: bool, fallback_error: str | None = None) -> None:
        """One row per application attempt, accepted or not (the paper logs these per call)."""

        if not attempts and fallback_error is not None:
            self.contract_issues.append({**meta, "issue": issue, "attempt": "", "n_names": "", "n_distinct": "", "n_unknown": "", "n_duplicates": "", "accepted": False, "validation_error": fallback_error})
            return
        for index, record in enumerate(attempts, start=1):
            error = record.get("validation_error") or ""
            valid = bool(record.get("valid"))
            self.contract_issues.append({**meta, "issue": issue, "attempt": record.get("attempt", index), **_attempt_counts(record, ids), "accepted": bool(accepted and valid), "validation_error": error})
        if accepted and fallback_error is None and not attempts:
            self.contract_issues.append({**meta, "issue": issue, "attempt": "", "n_names": "", "n_distinct": "", "n_unknown": "", "n_duplicates": "", "accepted": True, "validation_error": ""})

    def write_contract_issues(self, path: Path) -> int:
        """Sidecar in the shape of ``priority_2/repeated_calls_contract_issues.csv``."""

        if not self.contract_issues:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CONTRACT_ISSUE_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            for row in self.contract_issues:
                writer.writerow(row)
        return len(self.contract_issues)

    # ------------------------------------------------------ shared regenerated third-dataset control
    def third_named(self, records: RecordFactory) -> dict[str, Any] | None:
        """The paper's Stability 2024 full-DEV ranking is not on this machine; regenerate it once.

        Same snapshot, template, temperature and candidate set as the frozen run, so it is the
        same-run control the Home Credit two-arm design used.  Every consumer flags it.
        """

        found = self.cached("shared", "third_named_full_dev")
        if found is not None:
            return found
        names = records.candidate_set(THIRD, FULL_DEV)
        named = records.named_records(THIRD, names)
        result = self.rank(step="shared", name="third_named_full_dev", dataset=THIRD, partition=FULL_DEV, condition="named", records=named, ids=names, universe=records.universe(THIRD), extra={"note": "regenerated named control; the frozen Stability 2024 ranking is not on this machine"})
        if result is not None:
            self.manifest.note("shared", f"regenerated the Stability 2024 full-DEV named ranking ({len(names)} candidates; prompt sha256 {str(result['prompt_sha256'])[:12]}); used wherever the paper's cached ranking is required")
        return result


def prompt_sha(records: RecordFactory, records_list: Sequence[dict[str, Any]], ids: Sequence[str]) -> str:
    """Prompt digest without calling the API (for settings lines and equivalence checks)."""

    _, runner = _b6()
    return sha256_text(runner._selector().build_target_free_prompt(list(records_list), expected_features=list(ids)))


def scrubbed_prompt_sha(records: RecordFactory, dataset: str, named: Sequence[dict[str, Any]], names: Sequence[str]) -> str:
    """Digest of the named prompt with every original name literally replaced (Arm A equivalence check)."""

    helpers, runner = _b6()
    named_prompt = runner._selector().build_target_free_prompt(list(named), expected_features=list(names))
    return sha256_text(helpers.scrub_text(named_prompt, records.mapping(dataset)))
