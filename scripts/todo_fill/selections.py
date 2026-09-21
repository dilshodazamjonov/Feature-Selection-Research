"""Resume-safe store of per-partition selected feature sets plus the fit engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from scripts.todo_fill import selectors as recipes
from scripts.todo_fill.common import (
    BUDGETS,
    FULL_DEV,
    PARTITIONS,
    THIRD,
    FillError,
    Manifest,
    ResourceSkip,
    heartbeat,
    log,
    method_for,
    read_json,
    slug,
    write_json,
)
from scripts.todo_fill.data import DataContext


@dataclass(frozen=True)
class Request:
    dataset: str
    backbone: str
    label: str
    partition: str

    @property
    def method(self) -> str:
        return method_for(self.label, self.dataset)

    @property
    def k(self) -> int:
        return BUDGETS[self.backbone]

    def key(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "backbone": self.backbone, "selector": self.label, "partition": self.partition}


class SelectionStore:
    def __init__(self, work_dir: Path) -> None:
        self.root = Path(work_dir) / "selections"

    def path(self, request: Request) -> Path:
        return self.root / request.dataset / request.backbone / slug(request.method) / f"{request.partition}.json"

    def get(self, request: Request) -> dict[str, Any] | None:
        path = self.path(request)
        if not path.exists():
            return None
        payload = read_json(path)
        if payload.get("status") != "ok":
            return None
        return payload

    def get_skip(self, request: Request) -> dict[str, Any] | None:
        path = self.path(request)
        if not path.exists():
            return None
        payload = read_json(path)
        return payload if payload.get("status") == "skipped" else None

    def put(self, request: Request, payload: dict[str, Any]) -> None:
        write_json(self.path(request), payload)

    def all_records(self) -> list[dict[str, Any]]:
        return [read_json(path) for path in sorted(self.root.rglob("*.json"))] if self.root.exists() else []


#: Selections whose outcome does not depend on the backbone budget (natural support).
BACKBONE_INDEPENDENT_METHODS = {"iv_then_boruta"}


class SelectionEngine:
    """Compute selections on demand, partition-major so encoded frames are reused."""

    def __init__(
        self,
        *,
        store: SelectionStore,
        data: DataContext,
        manifest: Manifest,
        max_tier: int = 3,
        third_ranking: list[str] | None = None,
        retry_skipped: bool = False,
    ) -> None:
        self.store = store
        self.data = data
        self.manifest = manifest
        self.max_tier = max_tier
        self._progress: list[int] = [0, 0]  # fits done / fits pending in the current run()
        self.third_ranking = third_ranking
        self.retry_skipped = retry_skipped
        self._contexts: dict[tuple[str, str], recipes.FitContext] = {}

    # ---------------------------------------------------------------- lookup
    def lookup(self, request: Request) -> dict[str, Any] | None:
        return self.store.get(request)

    def _context(self, dataset: str, partition: str) -> recipes.FitContext:
        key = (dataset, partition)
        if key not in self._contexts:
            for other in list(self._contexts):
                if other != key:
                    self._contexts[other].release()
                    del self._contexts[other]
            self._contexts[key] = recipes.FitContext(dataset=dataset, backbone="lr", partition=partition, data=self.data, third_ranking=self.third_ranking)
        return self._contexts[key]

    def release(self) -> None:
        for ctx in self._contexts.values():
            ctx.release()
        self._contexts.clear()

    # --------------------------------------------------------------- compute
    def compute(self, request: Request) -> dict[str, Any] | None:
        stored = self.store.get(request)
        if stored is not None:
            return stored
        if not self.retry_skipped:
            skipped = self.store.get_skip(request)
            if skipped is not None:
                self.manifest.skip("selection", request.key(), f"previously skipped: {skipped.get('reason')}")
                return None
        method = request.method
        if method in BACKBONE_INDEPENDENT_METHODS:
            for other in ("lr", "catboost"):
                if other == request.backbone:
                    continue
                twin = self.store.get(Request(request.dataset, other, request.label, request.partition))
                if twin is not None:
                    payload = {**twin, **request.key(), "k": request.k, "details": {**twin.get("details", {}), "shared_from_backbone": other}}
                    self.store.put(request, payload)
                    return payload
        ok, reason = recipes.supported(method, request.dataset)
        if not ok:
            return self._skip(request, reason)
        tier = recipes.COST_TIER.get(method, 3)
        if tier > self.max_tier:
            return self._skip(request, f"cost tier {tier} exceeds the requested maximum tier {self.max_tier}", persist=False)
        ctx = self._context(request.dataset, request.partition)
        ctx.backbone = request.backbone
        done, total = self._progress
        position = f" ({done + 1}/{total})" if total else ""
        target = f"{request.dataset}/{request.backbone}/{method}/{request.partition}"
        log(f"[fit] start{position} {target}: K={request.k}, cost tier {tier}")
        started = time.time()
        try:
            with heartbeat(f"fit {target}"):
                output = recipes.RECIPES[method](ctx)
        except ResourceSkip as exc:
            return self._skip(request, str(exc), persist=False)
        except FillError as exc:
            return self._skip(request, str(exc))
        except MemoryError:
            return self._skip(request, "MemoryError during fit", persist=False)
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, never fatal
            return self._skip(request, f"{type(exc).__name__}: {exc}")
        payload = {
            "status": "ok",
            **request.key(),
            "method": method,
            "k": request.k,
            "features": list(output.features),
            "ranked": list(output.ranked) if output.ranked else None,
            "n_selected": output.n_selected,
            "protocol": output.protocol,
            "fit_seconds": output.fit_seconds,
            "wall_seconds": time.time() - started,
            "details": output.details,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.store.put(request, payload)
        self.manifest.timing(f"fit:{request.dataset}/{request.backbone}/{method}/{request.partition}", output.fit_seconds, wall_seconds=payload["wall_seconds"], n_selected=output.n_selected)
        log(f"[fit] done {request.dataset}/{request.backbone}/{method}/{request.partition}: {output.n_selected} features in {output.fit_seconds:.0f}s")
        return payload

    def _skip(self, request: Request, reason: str, *, persist: bool = True) -> None:
        self.manifest.skip("selection", request.key(), reason)
        if persist:
            self.store.put(request, {"status": "skipped", **request.key(), "method": request.method, "reason": reason, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return None

    def run(self, requests: Iterable[Request]) -> None:
        """Process requests grouped by (dataset, partition) and ordered cheap-first."""

        pending = [request for request in dict.fromkeys(requests) if self.store.get(request) is None]
        if not pending:
            return
        dataset_order = {name: index for index, name in enumerate(("homecredit", "lendingclub_v2", THIRD))}
        partition_order = {name: index for index, name in enumerate(PARTITIONS)}
        pending.sort(
            key=lambda request: (
                dataset_order.get(request.dataset, 9),
                recipes.COST_TIER.get(request.method, 3) if request.dataset == THIRD else 0,
                partition_order.get(request.partition, 9),
                recipes.COST_TIER.get(request.method, 3),
                request.label,
                request.backbone,
            )
        )
        log(f"[selection] {len(pending)} selector fits pending")
        self._progress = [0, len(pending)]
        for request in pending:
            if request.method in recipes.DATA_FREE_METHODS:
                self.compute(request)
                self._progress[0] += 1
        for request in pending:
            if request.method not in recipes.DATA_FREE_METHODS:
                self.compute(request)
                self._progress[0] += 1
        self._progress = [0, 0]
        self.release()


def fold_sets(engine: SelectionEngine, dataset: str, backbone: str, label: str) -> dict[str, list[str]] | None:
    """Return the five fold selections if all are stored, else None."""

    sets: dict[str, list[str]] = {}
    for partition in PARTITIONS[:-1]:
        record = engine.lookup(Request(dataset, backbone, label, partition))
        if record is None:
            return None
        sets[partition] = list(record["features"])
    return sets


def full_dev_selection(engine: SelectionEngine, dataset: str, backbone: str, label: str) -> dict[str, Any] | None:
    return engine.lookup(Request(dataset, backbone, label, FULL_DEV))
