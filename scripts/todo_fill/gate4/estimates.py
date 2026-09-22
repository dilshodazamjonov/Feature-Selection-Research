"""Wall-clock estimates per step from the unit costs measured on this machine (dry-run output)."""

from __future__ import annotations

from scripts.todo_fill.common import FOLDS, FULL_DEV, HOMECREDIT, LENDINGCLUB, PARTITIONS, THIRD
from scripts.todo_fill.gate4 import config


def _parallel(seconds: list[float], jobs: int) -> float:
    """Greedy longest-first packing of independent fits onto ``jobs`` workers."""

    if not seconds:
        return 0.0
    lanes = [0.0] * max(1, jobs)
    for value in sorted(seconds, reverse=True):
        index = lanes.index(min(lanes))
        lanes[index] += value
    return max(lanes)


def _fits(dataset: str, backbone: str, partitions=PARTITIONS, scale: float = 1.0) -> list[float]:
    return [config.fit_seconds(dataset, backbone, partition) * scale for partition in partitions]


def estimate(step: str, jobs: int, third_refits: str = "run") -> dict[str, float | str]:
    calls = 0
    serial = 0.0  # selector fits and bootstraps run in the driver process
    fits: list[float] = []
    note = ""
    if step == "38":
        serial += config.selector_seconds(HOMECREDIT, "LLM then mRMR", FULL_DEV) + config.bootstrap_seconds(config.HO_ROWS[HOMECREDIT]) + 150
        fits += _fits(HOMECREDIT, "lr", (FULL_DEV,)) * 2
        note = "includes the one-off load of the Home Credit HO universe (~2-3 min)"
    elif step == "39a":
        calls = 1
        serial += 12 * 25
        note = "one API call regenerates the Stability 2024 named ranking if not cached"
    elif step == "40":
        fits += _fits(HOMECREDIT, "lr", (FULL_DEV,)) * 5 + _fits(HOMECREDIT, "catboost", (FULL_DEV,)) * 5
        fits += _fits(LENDINGCLUB, "lr", (FULL_DEV,)) * 5 + _fits(LENDINGCLUB, "catboost", (FULL_DEV,)) * 5
        note = "includes the one-off load of the LendingClub HO universe (~3 min)"
        serial += 180
    elif step == "37":
        serial += 2 * config.selector_seconds(LENDINGCLUB, "RFE CatBoost", FULL_DEV)
        fits += _fits(LENDINGCLUB, "lr", (FULL_DEV,)) * 3 + _fits(LENDINGCLUB, "catboost", (FULL_DEV,)) * 2 + _fits(LENDINGCLUB, "catboost", (FULL_DEV,), scale=2.0)
        serial += sum(config.bootstrap_seconds(n) for n in [123_898, 232_361, 60_744] * 4 + [25_000] * 48)
        note = "two RFE CatBoost refits on 599k LendingClub rows dominate; 60 paired bootstraps"
    elif step == "39b":
        fits += _fits(HOMECREDIT, "catboost") * 2 + _fits(LENDINGCLUB, "catboost") + _fits(LENDINGCLUB, "catboost", scale=2.5) + _fits(THIRD, "catboost")
        if third_refits == "run":
            # the CatBoost SHAP subset now comes from Appendix E, so only its backbone
            # refits cost anything; the full-universe selector refit is gone
            fits += _fits(THIRD, "catboost")
            note = "the Stability 2024 CatBoost SHAP subset comes from Appendix E, so no full-universe selector refit is needed"
        calls = 1
    elif step == "44":
        calls = 6 + 24 + 2
        serial += 6 * config.selector_seconds(LENDINGCLUB, "LLM then mRMR", FULL_DEV) + 7 * config.bootstrap_seconds(200_000)
        fits += _fits(HOMECREDIT, "lr") + _fits(HOMECREDIT, "catboost") + _fits(LENDINGCLUB, "lr") + _fits(LENDINGCLUB, "catboost") * 2
        fits += _fits(THIRD, "lr", (FULL_DEV,)) * 2 + _fits(THIRD, "catboost", (FULL_DEV,)) * 2
        fits += _fits(HOMECREDIT, "lr", (FULL_DEV,)) + _fits(HOMECREDIT, "catboost", (FULL_DEV,)) + _fits(LENDINGCLUB, "lr", (FULL_DEV,)) + _fits(LENDINGCLUB, "catboost", (FULL_DEV,)) * 2
    elif step == "47":
        calls = 24 + 2 + 1
        fits += _fits(LENDINGCLUB, "lr") * 2 + _fits(LENDINGCLUB, "catboost") * 2 + _fits(THIRD, "lr", (FULL_DEV,)) * 3 + _fits(THIRD, "catboost", (FULL_DEV,)) * 3
        fits += _fits(LENDINGCLUB, "lr", (FULL_DEV,)) + _fits(LENDINGCLUB, "catboost", (FULL_DEV,))
    elif step == "43":
        serial += sum(config.selector_seconds(HOMECREDIT, "mRMR", p) for p in PARTITIONS)
        serial += sum(config.selector_seconds(HOMECREDIT, "RFE CatBoost", p) for p in FOLDS)
        serial += sum(config.selector_seconds(LENDINGCLUB, "IV then Boruta", p) for p in FOLDS) + 300
        serial += sum(config.selector_seconds(THIRD, "RFE CatBoost", p) for p in PARTITIONS) * 0.25  # depth-0 universe (219 of 1,959 columns)
        serial += sum(config.selector_seconds(THIRD, "CatBoost SHAP", p) for p in PARTITIONS) * 0.25
        if third_refits == "run":
            serial += config.selector_seconds(THIRD, "RFE CatBoost", "fold1") + config.selector_seconds(THIRD, "RFE CatBoost", "fold2")
            serial += config.selector_seconds(THIRD, "CatBoost SHAP", "fold1") + config.selector_seconds(THIRD, "CatBoost SHAP", "fold2")
            note = "full-universe Stability 2024 rankings beyond folds 1-2 exceed 19 GiB and are skipped by the RAM guard"
        fits += _fits(HOMECREDIT, "lr") + _fits(HOMECREDIT, "catboost") + _fits(LENDINGCLUB, "lr") + _fits(LENDINGCLUB, "catboost")
        fits += _fits(THIRD, "lr") * 2 + _fits(THIRD, "catboost") * 2
    elif step == "45":
        calls = 30
        fits += _fits(HOMECREDIT, "lr", (FULL_DEV,)) * 10 + _fits(HOMECREDIT, "catboost", (FULL_DEV,)) * 10
        fits += _fits(LENDINGCLUB, "lr", (FULL_DEV,)) * 10 + _fits(LENDINGCLUB, "catboost", (FULL_DEV,)) * 10
    total = serial + calls * config.LLM_CALL_SECONDS + _parallel(fits, jobs)
    return {"step": step, "api_calls": calls, "backbone_fits": len(fits), "driver_seconds": serial, "fit_wall_seconds": _parallel(fits, jobs), "total_minutes": total / 60.0, "note": note}
