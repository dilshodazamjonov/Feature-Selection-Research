"""LLM call plan, cached ranking requests and ranking access.

Every call renders the unified prompt (:mod:`credit_risk_fs.selectors.llm_prompt`)
over the definition-only records of the screened candidate set, in candidate order, at
temperature 0 with the strict JSON-schema response format. A response is accepted only
when it holds exactly the requested number of distinct names copied from the records; a
failing response is re-requested up to the configured number of times and no fallback
ranking exists. Accepted responses are cached under ``<output>/llm/rankings`` and every
later stage reads the cache, never a fresh call.

Call plan (configured per dataset):

* ``named``: one call per configured partition and ranking budget (Home Credit: six
  partitions x 100 names, prefixes give every budget; LendingClub v2: six partitions x
  budgets 20, 40, 60 and 100; Stability 2024: one full-DEV call of 100 names serving
  every fold), plus ``repeats`` repetitions of the full-DEV call with the same prompt;
* ``A`` / ``B``: the name-obfuscation arms (Home Credit), one call per partition;
* ``lineage``: the calls repeated with descriptions rebuilt from lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from credit_risk_fs.study.config import FULL_DEV, StudyError, output_dir
from credit_risk_fs.study.records import blanked_records, lineage_records, obfuscated_records
from credit_risk_fs.study.utils import log, read_json, sha256_text, utc_now, write_json

NAMED, ARM_A, ARM_B, LINEAGE = "named", "A", "B", "lineage"
ARMS = (NAMED, ARM_A, ARM_B, LINEAGE)


@dataclass(frozen=True)
class CallSpec:
    arm: str
    partition: str
    budget: int
    repeat: int = 0

    @property
    def call_id(self) -> str:
        suffix = f"__r{self.repeat:02d}" if self.repeat else ""
        return f"{self.arm}__{self.partition}__b{self.budget}{suffix}"


def arm_partitions(cfg: dict[str, Any], arm: str) -> list[str]:
    llm = cfg["llm"]
    if arm == LINEAGE:
        return list(llm.get("lineage_partitions", [FULL_DEV]))
    return list(llm["partitions"])


def arm_budgets(cfg: dict[str, Any], arm: str) -> list[int]:
    llm = cfg["llm"]
    if arm == LINEAGE:
        return sorted(int(value) for value in llm.get("lineage_budgets", llm["budgets"]))
    if arm in (ARM_A, ARM_B):
        return [int(llm.get("obfuscation_budget", max(llm["budgets"])))]
    return sorted(int(value) for value in llm["budgets"])


def plan_calls(cfg: dict[str, Any]) -> list[CallSpec]:
    llm = cfg["llm"]
    calls = [
        CallSpec(NAMED, partition, budget)
        for partition in arm_partitions(cfg, NAMED)
        for budget in arm_budgets(cfg, NAMED)
    ]
    repeat_budget = max(arm_budgets(cfg, NAMED))
    calls += [CallSpec(NAMED, FULL_DEV, repeat_budget, repeat) for repeat in range(1, int(llm.get("repeats", 0)) + 1)]
    if llm.get("obfuscation", False):
        for arm in (ARM_A, ARM_B):
            calls += [CallSpec(arm, partition, budget) for partition in arm_partitions(cfg, arm) for budget in arm_budgets(cfg, arm)]
    if llm.get("lineage", True):
        calls += [
            CallSpec(LINEAGE, partition, budget)
            for partition in arm_partitions(cfg, LINEAGE)
            for budget in arm_budgets(cfg, LINEAGE)
        ]
    return calls


def call_records(
    cfg: dict[str, Any],
    spec: CallSpec,
    records: Sequence[dict[str, Any]],
    universe: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    if spec.arm == NAMED:
        return [dict(record) for record in records], None
    if spec.arm == ARM_A:
        return obfuscated_records(records, universe, spec.partition, int(cfg["obfuscation"]["seed"]))
    if spec.arm == ARM_B:
        return blanked_records(records), None
    if spec.arm == LINEAGE:
        return lineage_records(records), None
    raise StudyError(f"unknown arm {spec.arm!r}")


def ranking_path(cfg: dict[str, Any], call_id: str) -> Path:
    return output_dir(cfg) / "llm" / "rankings" / f"{call_id}.json"


def ensure_ranking(
    cfg: dict[str, Any],
    spec: CallSpec,
    records: Sequence[dict[str, Any]],
    universe: Sequence[str],
    *,
    allow_request: bool,
) -> dict[str, Any]:
    """Return the cached ranking of ``spec``; request it only when ``allow_request``."""

    from credit_risk_fs.selectors.llm_prompt import PROMPT_VERSION, render_prompt
    from credit_risk_fs.selectors.llm_screening import request_ranking

    call_records_, reverse = call_records(cfg, spec, records, universe)
    evidence_mode = str(cfg["llm"].get("evidence_mode", "definitions_only"))
    prompt = render_prompt(ranking_budget=spec.budget, records=call_records_, evidence_mode=evidence_mode)
    if spec.arm == ARM_A:
        leaked = [name for name in universe if str(name) in prompt]
        if leaked:
            raise StudyError(f"the obfuscated request for {spec.call_id} contains {len(leaked)} original name(s)")
    prompt_sha = sha256_text(prompt)
    path = ranking_path(cfg, spec.call_id)
    if path.exists():
        payload = read_json(path)
        if payload.get("prompt_sha256") != prompt_sha:
            raise StudyError(
                f"{path} was produced from a different prompt; delete it to request this call again"
            )
        return payload
    if not allow_request:
        raise StudyError(f"ranking {spec.call_id} is not cached; run --stage llm first")
    names = [str(record["name"]) for record in call_records_]
    log(f"{cfg['dataset']}: requesting {spec.call_id} ({len(names)} candidate records, {spec.budget} names)")
    response = request_ranking(
        prompt=prompt,
        candidate_names=names,
        ranking_budget=spec.budget,
        model=str(cfg["llm"]["model"]),
        temperature=float(cfg["llm"]["temperature"]),
        max_rerequests=int(cfg["llm"]["max_rerequests"]),
    )
    identifiers = list(response["selected_features"])
    ranking = [reverse[item] for item in identifiers] if reverse else identifiers
    payload = {
        "call_id": spec.call_id,
        "dataset": cfg["dataset"],
        "arm": spec.arm,
        "partition": spec.partition,
        "budget": spec.budget,
        "repeat": spec.repeat,
        "ranking": ranking,
        "ranking_identifiers": identifiers if reverse else None,
        "n_candidate_records": len(names),
        "candidate_names_sha256": sha256_text("\n".join(names)),
        "prompt_version": PROMPT_VERSION,
        "evidence_mode": evidence_mode,
        "prompt_sha256": prompt_sha,
        "request_sha256": response["request_sha256"],
        "request_model": response["request_model"],
        "response_model": response["response_model"],
        "response_id": response["response_id"],
        "temperature": response["temperature"],
        "attempts": response["attempts"],
        "rejected_attempts": response["rejected_attempts"],
        "prompt_tokens": response["prompt_tokens"],
        "completion_tokens": response["completion_tokens"],
        "total_tokens": response["total_tokens"],
        "reasoning_summary": response["reasoning_summary"],
        "selection_principles": response["selection_principles"],
        "raw_response": response["raw_response"],
        "accepted_at_utc": utc_now(),
    }
    write_json(path, payload)
    return payload


def _resolve_partition(cfg: dict[str, Any], arm: str, partition: str) -> str:
    available = arm_partitions(cfg, arm)
    if partition in available:
        return partition
    if available == [FULL_DEV]:
        # One ranking serves every fold when only the full-DEV call is configured.
        return FULL_DEV
    raise StudyError(f"no {arm} ranking is configured for partition {partition}")


def ranking(cfg: dict[str, Any], arm: str, partition: str, length: int, *, repeat: int = 0) -> list[str]:
    """The cached ranking of a partition from the smallest call budget covering ``length``."""

    budgets = arm_budgets(cfg, arm)
    covering = [value for value in budgets if value >= int(length)]
    chosen = min(covering) if covering else max(budgets)
    if repeat:
        chosen = max(arm_budgets(cfg, NAMED))
    spec = CallSpec(arm, _resolve_partition(cfg, arm, partition), chosen, repeat)
    path = ranking_path(cfg, spec.call_id)
    if not path.exists():
        raise StudyError(f"ranking {spec.call_id} is not cached; run --stage llm first")
    return [str(name) for name in read_json(path)["ranking"]]


def ranking_call_id(cfg: dict[str, Any], arm: str, partition: str, length: int, *, repeat: int = 0) -> str:
    budgets = arm_budgets(cfg, arm)
    covering = [value for value in budgets if value >= int(length)]
    chosen = min(covering) if covering else max(budgets)
    if repeat:
        chosen = max(arm_budgets(cfg, NAMED))
    return CallSpec(arm, _resolve_partition(cfg, arm, partition), chosen, repeat).call_id


__all__ = [
    "ARMS",
    "ARM_A",
    "ARM_B",
    "CallSpec",
    "LINEAGE",
    "NAMED",
    "arm_budgets",
    "arm_partitions",
    "call_records",
    "ensure_ranking",
    "plan_calls",
    "ranking",
    "ranking_call_id",
    "ranking_path",
]
