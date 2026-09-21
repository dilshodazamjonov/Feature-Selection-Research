"""Read the cached target-free LLM rankings under ``artifacts/llm_cache``.

The cache files carry no dataset field.  Home Credit files are recognised by
their ``EXT_SOURCE_*`` candidates; the canonical LendingClub v2 run is the
``config_hash`` group that also stores ``feature_budget`` in its cache key.
Both identifications are verified at run time against the pre-filled Table 9
values in ``todo/10_stability.csv`` (Nogueira and mean Jaccard of the top-K
fold truncations), and the verification outcome is written to the manifest.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from scripts.todo_fill.common import (
    FULL_DEV,
    HOMECREDIT,
    LENDINGCLUB,
    REPO_ROOT,
    THIRD,
    sha256_file,
)

CACHE_DIR = REPO_ROOT / "artifacts/llm_cache"
#: The canonical LendingClub v2 legacy-matrix cache group (V2 schema, 27 files).
CANONICAL_LENDINGCLUB_CONFIG_PREFIX = "f6f9c772"


@dataclass(frozen=True)
class CachedRanking:
    dataset: str
    partition: str
    features: tuple[str, ...]
    candidate_features: tuple[str, ...]
    n_candidates: int
    feature_budget: int | None
    prompt_hash: str
    config_hash: str
    path: Path
    sha256: str

    @property
    def relative_path(self) -> str:
        return str(self.path.relative_to(REPO_ROOT))


def _partition(payload: dict[str, Any]) -> str:
    if payload.get("scope") == "final_dev" or payload.get("fold_id") in (None, "None", ""):
        return FULL_DEV
    return f"fold{int(payload['fold_id'])}"


def _dataset(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidate_features") or []
    if any(str(name).startswith("EXT_SOURCE") for name in candidates):
        return HOMECREDIT
    description = str((payload.get("cache_key") or {}).get("description_csv_path", ""))
    if "lendingclub_v2" in description or candidates:
        return LENDINGCLUB
    return "unknown"


@lru_cache(maxsize=None)
def load_all() -> list[CachedRanking]:
    out: list[CachedRanking] = []
    for path in sorted(CACHE_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = [str(item) for item in payload.get("selected_features") or []]
        selected = list(dict.fromkeys(selected))
        budget = payload.get("feature_budget", (payload.get("cache_key") or {}).get("feature_budget"))
        out.append(
            CachedRanking(
                dataset=_dataset(payload),
                partition=_partition(payload),
                features=tuple(selected),
                candidate_features=tuple(str(item) for item in payload.get("candidate_features") or []),
                n_candidates=int(payload.get("n_candidates") or len(payload.get("candidate_features") or [])),
                feature_budget=int(budget) if budget not in (None, "") else None,
                prompt_hash=str(payload.get("prompt_hash", "")),
                config_hash=str(payload.get("config_hash", "")),
                path=path,
                sha256=sha256_file(path),
            )
        )
    return out


def canonical_files(dataset: str) -> list[CachedRanking]:
    files = [item for item in load_all() if item.dataset == dataset]
    if dataset == LENDINGCLUB:
        files = [item for item in files if item.config_hash.startswith(CANONICAL_LENDINGCLUB_CONFIG_PREFIX)]
    return files


def ranking(dataset: str, partition: str, budget: int | None = None) -> CachedRanking:
    """Return the cached ranking for one partition.

    ``budget`` is the ``feature_budget`` the historical run requested from the
    LLM selector (K for pure LLM and stable-core runs, the LLM candidate pool
    for ``llm_then_*`` hybrids).  Home Credit has one file per partition, so
    ``budget`` is ignored there.  Ties are broken towards complete 100-name
    rankings, then the prompt rendering shared by most partitions, then the
    file name, and the choice is deterministic.
    """

    if dataset == THIRD:
        raise KeyError("no target-free LLM ranking for the third dataset exists in artifacts/llm_cache")
    files = [item for item in canonical_files(dataset) if item.partition == partition]
    if not files:
        raise KeyError(f"no cached LLM ranking for {dataset}/{partition}")
    if dataset == HOMECREDIT or budget is None:
        candidates = files
    else:
        candidates = [item for item in files if item.feature_budget == budget] or files
    if len(candidates) > 1:
        prompt_counts = collections.Counter(item.prompt_hash for item in canonical_files(dataset))
        candidates = sorted(
            candidates,
            key=lambda item: (-(len(item.features) >= 100), -prompt_counts[item.prompt_hash], item.path.name),
        )
    return candidates[0]


_FOLD_OVERRIDES: dict[tuple[str, int], dict[str, CachedRanking]] = {}


def fold_rankings(dataset: str, budget: int | None = None) -> dict[str, CachedRanking]:
    if budget is not None and (dataset, budget) in _FOLD_OVERRIDES:
        return dict(_FOLD_OVERRIDES[(dataset, budget)])
    return {f"fold{i}": ranking(dataset, f"fold{i}", budget) for i in range(1, 6)}


def pin_fold_files_to_targets(
    dataset: str,
    k: int,
    *,
    target_nogueira: float,
    target_jaccard_mean: float,
    d: int,
    tolerance: float = 6e-4,
) -> dict[str, Any]:
    """Choose one cached file per fold so the top-K truncations reproduce Table 9.

    The legacy matrix issued several LLM calls per fold (one per run that shared
    the fold), so LendingClub folds can carry more than one ranking with the same
    ``feature_budget``.  The paper's published Nogueira / mean Jaccard pair
    identifies the run; we enumerate the per-fold combinations, keep those that
    reproduce both numbers, and pin the lexically first match.  If nothing
    matches, the heuristic choice stands and the caller records the mismatch.
    """

    import itertools

    from scripts.todo_fill.common import nogueira, pairwise_jaccard_summary

    per_fold: list[list[CachedRanking]] = []
    for i in range(1, 6):
        files = [item for item in canonical_files(dataset) if item.partition == f"fold{i}" and len(item.features) >= k]
        preferred = [item for item in files if item.feature_budget == k] or files
        per_fold.append(sorted(preferred, key=lambda item: item.path.name))
    matches: list[tuple[str, ...]] = []
    for combo in itertools.product(*per_fold):
        sets = [list(item.features[:k]) for item in combo]
        nog = nogueira(sets, d)
        jac = pairwise_jaccard_summary(sets)["mean"]
        if nog is None or jac is None:
            continue
        if abs(nog - target_nogueira) <= tolerance and abs(jac - target_jaccard_mean) <= tolerance:
            matches.append(tuple(item.path.name for item in combo))
            if len(matches) == 1:
                _FOLD_OVERRIDES[(dataset, k)] = {f"fold{i}": item for i, item in enumerate(combo, start=1)}
    chosen = fold_rankings(dataset, k)
    sets = [list(item.features[:k]) for item in chosen.values()]
    return {
        "dataset": dataset,
        "k": k,
        "target": {"nogueira": target_nogueira, "jaccard_mean": target_jaccard_mean, "d": d},
        "matched": bool(matches),
        "match_count": len(matches),
        "chosen_files": {partition: item.relative_path for partition, item in chosen.items()},
        "reproduced": {"nogueira": nogueira(sets, d), **{f"jaccard_{key}": value for key, value in pairwise_jaccard_summary(sets).items()}},
    }


def truncated_sets(dataset: str, k: int, budget: int | None = None) -> dict[str, list[str]]:
    """Top-K truncation of the cached fold rankings (the pure LLM fold subsets)."""

    return {partition: list(item.features[:k]) for partition, item in fold_rankings(dataset, budget).items()}


def inventory() -> list[dict[str, Any]]:
    return [
        {
            "file": item.relative_path,
            "dataset": item.dataset,
            "partition": item.partition,
            "n_candidates": item.n_candidates,
            "n_ranked": len(item.features),
            "feature_budget": item.feature_budget,
            "config_hash": item.config_hash[:12],
            "prompt_hash": item.prompt_hash[:12],
            "canonical": item in canonical_files(item.dataset) if item.dataset in (HOMECREDIT, LENDINGCLUB) else False,
        }
        for item in load_all()
    ]
