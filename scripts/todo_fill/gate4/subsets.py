"""Feature subsets: headline selections, base-selector rankings and the source-family cap (step 43)."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

from scripts.todo_fill import selectors as recipes
from scripts.todo_fill.common import (
    BUDGETS,
    FULL_DEV,
    HOMECREDIT,
    LENDINGCLUB,
    SEED,
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
from scripts.todo_fill.gate4.llm import Ranker
from scripts.todo_fill.gate4.records import RecordFactory
from scripts.todo_fill.selections import Request, SelectionEngine


#: ``appendix_e_subsets.csv`` labels one subset per paper cell as
#: "<Dataset>, <Backbone>: <Selector>".  These map its two prefixes onto the keys
#: this code uses; the selector part is already the paper label.
APPENDIX_E_DATASETS: dict[str, str] = {"Home Credit": HOMECREDIT, "LendingClub v2": LENDINGCLUB, "Stability 2024": THIRD}
APPENDIX_E_BACKBONES: dict[str, str] = {"LR": "lr", "CatBoost": "catboost"}


def load_appendix_e(path: Path | str | None) -> dict[tuple[str, str, str], list[str]]:
    """The paper's 12 headline subsets, keyed by (dataset, backbone, selector label).

    These are the authoritative subsets: a local refit of the same selector does not
    always land on the same features (Home Credit LR mRMR is the known case), and the
    Stability 2024 refits do not fit in RAM on this machine at all.
    """

    if path is None:
        return {}
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    import csv

    subsets: dict[tuple[str, str, str], list[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("subset_in_appendix_e", "")).strip()
            head, _, selector = label.partition(":")
            dataset_text, _, backbone_text = head.partition(",")
            dataset = APPENDIX_E_DATASETS.get(dataset_text.strip())
            backbone = APPENDIX_E_BACKBONES.get(backbone_text.strip())
            features = [item.strip() for item in str(row.get("features", "")).split(";") if item.strip()]
            if dataset and backbone and selector.strip() and features:
                subsets[(dataset, backbone, selector.strip())] = features
    return subsets


class SubsetResolver:
    def __init__(self, *, data: DataContext, engine: SelectionEngine, ranker: Ranker, records: RecordFactory, manifest: Manifest, work_dir: Path, third_refits: str = "run", appendix_e: Path | str | None = None) -> None:
        self.data = data
        self.engine = engine
        self.ranker = ranker
        self.records = records
        self.manifest = manifest
        self.work_dir = Path(work_dir)
        self.third_refits = third_refits
        self.appendix_e = load_appendix_e(appendix_e)
        self._appendix_e_reported: set[tuple[str, str, str]] = set()
        if appendix_e is not None and not self.appendix_e:
            self.manifest.note("setup", f"Appendix E subsets not read from {appendix_e}; headline cells fall back to local refits and cached truncations")
        elif self.appendix_e:
            self.manifest.note("setup", f"Appendix E headline subsets loaded from {appendix_e}: {len(self.appendix_e)} cells")

    # ------------------------------------------------------------ headline subsets
    def headline(self, dataset: str, backbone: str, label: str, step: str = "subset") -> dict[str, Any] | None:
        """Full-DEV selection of one paper cell: cached truncation, stored refit, or a new refit."""

        k = BUDGETS[backbone]
        paper = self.appendix_e.get((dataset, backbone, label))
        if paper is not None:
            universe = set(self.records.universe(dataset))
            unknown = [name for name in paper if name not in universe]
            if unknown:
                self.manifest.skip(step, {"dataset": dataset, "backbone": backbone, "selector": label}, f"Appendix E subset carries {len(unknown)} name(s) absent from this machine's candidate universe, e.g. {unknown[:3]}")
            else:
                if (dataset, backbone, label) not in self._appendix_e_reported:
                    self._appendix_e_reported.add((dataset, backbone, label))
                    self.manifest.note(step, f"{dataset}/{backbone}/{label}: using the Appendix E subset ({len(paper)} features) instead of a local refit")
                return {"features": list(paper), "ranked": list(paper), "protocol": "appendix_e_subset", "source": "appendix_e:the paper's frozen headline subset (appendix_e_subsets.csv)"}
        if label == "Pure LLM" and dataset == THIRD:
            named = self.ranker.third_named(self.records)
            if named is None:
                self.manifest.skip(step, {"dataset": dataset, "backbone": backbone, "selector": label}, "Stability 2024 named ranking unavailable (not on disk and no API key to regenerate it)")
                return None
            ranking = list(named["ranking"])
            return {"features": ranking[:k], "ranked": ranking[:k], "protocol": "regenerated_named_ranking_truncation", "source": "regenerated:Stability 2024 named ranking re-issued with the frozen snapshot/template (paper's cached ranking is not on this machine)"}
        if dataset == THIRD and self.third_refits == "skip" and method_for(label, dataset) != "llm":
            self.manifest.skip(step, {"dataset": dataset, "backbone": backbone, "selector": label}, "third-dataset selector refits disabled (--third-refits skip)")
            return None
        record = self.engine.compute(Request(dataset, backbone, label, FULL_DEV))
        if record is None:
            self.manifest.skip(step, {"dataset": dataset, "backbone": backbone, "selector": label}, "full-DEV selection unavailable on this machine (see selection skips)")
            return None
        source = "cache:artifacts/llm_cache truncation" if record["protocol"] == "cached_llm_truncation" else f"recomputed:{record['protocol']}"
        return {"features": list(record["features"]), "ranked": record.get("ranked"), "protocol": record["protocol"], "source": source, "details": record.get("details", {})}

    # ------------------------------------------------------------- base rankings
    def _store_path(self, dataset: str, backbone: str, label: str, partition: str, restricted: bool) -> Path:
        name = f"{partition}{'__depth0' if restricted else ''}.json"
        return self.work_dir / "gate4" / "base_rankings" / dataset / backbone / slug(label) / name

    def base_ranking(self, *, dataset: str, backbone: str, label: str, partition: str, columns: Sequence[str] | None = None, step: str = "43") -> dict[str, Any] | None:
        """The base selector's own complete order over its candidates for one partition."""

        restricted = columns is not None
        path = self._store_path(dataset, backbone, label, partition, restricted)
        if path.exists():
            payload = read_json(path)
            if payload.get("status") == "ok":
                return payload
            if payload.get("status") == "skipped":
                self.manifest.skip(step, {"dataset": dataset, "backbone": backbone, "selector": label, "partition": partition}, f"previously skipped: {payload.get('reason')}")
                return None
        if dataset == THIRD and not restricted and self.third_refits == "skip":
            self.manifest.skip(step, {"dataset": dataset, "backbone": backbone, "selector": label, "partition": partition}, "third-dataset full-universe selector refits disabled (--third-refits skip)")
            return None
        method = method_for(label, dataset)
        try:
            if method == "mrmr_legacy":
                payload = self._mrmr_full_order(dataset, backbone, partition)
            elif method in {"rfe_catboost", "catboost_shap"}:
                payload = self._contract_ranking(dataset, backbone, label, partition, columns)
            elif method == "iv_then_boruta":
                payload = self._iv_then_boruta_order(dataset, backbone, label, partition)
            else:
                raise FillError(f"no base-ranking recipe for {label}")
        except ResourceSkip as exc:
            self.manifest.skip(step, {"dataset": dataset, "backbone": backbone, "selector": label, "partition": partition}, str(exc))
            return None
        except (FillError, MemoryError) as exc:
            self.manifest.skip(step, {"dataset": dataset, "backbone": backbone, "selector": label, "partition": partition}, f"{type(exc).__name__}: {exc}")
            write_json(path, {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"})
            return None
        except Exception as exc:  # noqa: BLE001
            self.manifest.skip(step, {"dataset": dataset, "backbone": backbone, "selector": label, "partition": partition}, f"{type(exc).__name__}: {exc}")
            write_json(path, {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"})
            return None
        payload.update({"status": "ok", "dataset": dataset, "backbone": backbone, "selector": label, "partition": partition, "restricted_universe": list(columns) if columns is not None else None, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        write_json(path, payload)
        return payload

    def _mrmr_full_order(self, dataset: str, backbone: str, partition: str) -> dict[str, Any]:
        """Legacy RF-relevance/correlation mRMR run to exhaustion: the greedy order of every dense column."""

        from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector
        from scripts.todo_fill.frozen import collapse_model_matrix_columns

        ctx = recipes.FitContext(dataset=dataset, backbone=backbone, partition=partition, data=self.data)
        X, _ = ctx.raw()
        recipes._guard(ctx, "mrmr_legacy", len(X), int(X.shape[1] * 1.3))
        dense, y = ctx.dense()
        k = int(dense.shape[1]) - 1  # k == p short-circuits to the candidate order; p-1 keeps the greedy pass
        selector = RandomForestRelevanceMRMRSelector(k=k, method="mrmr", random_state=SEED, n_jobs=4)
        log(f"[43] {dataset}/{backbone}/mRMR(legacy, full greedy order)/{partition}: {dense.shape[0]} rows x {dense.shape[1]} dense cols")
        started = time.perf_counter()
        with heartbeat(f"mRMR full order {dataset}/{partition}"):
            selector.fit(dense, y)
        order = [str(item) for item in selector.selected_features_]
        order += [str(name) for name in dense.columns if str(name) not in set(order)]
        universe = self.records.universe(dataset)
        raw_order = collapse_model_matrix_columns(order, universe)
        ctx.release()
        return {"ranking": raw_order, "dense_order": order, "protocol": "legacy_matrix_after_dense_preprocessing (greedy run to exhaustion; prefix equals the K-budget selection)", "fit_seconds": time.perf_counter() - started}

    def _contract_ranking(self, dataset: str, backbone: str, label: str, partition: str, columns: Sequence[str] | None) -> dict[str, Any]:
        method = method_for(label, dataset)
        if columns is None:
            record = self.engine.compute(Request(dataset, backbone, label, partition))
            if record is None:
                raise FillError("selector refit unavailable (see selection skips)")
            ranked = record.get("ranked")
            if not ranked:
                raise FillError("selector record carries no ranking")
            return {"ranking": [str(item) for item in ranked], "protocol": record["protocol"], "fit_seconds": record.get("fit_seconds"), "selected_prefix": list(record["features"])}
        ctx = recipes.FitContext(dataset=dataset, backbone=backbone, partition=partition, data=self.data, columns=tuple(columns))
        log(f"[43] {dataset}/{backbone}/{method}/{partition} on a restricted universe of {len(columns)} candidates")
        with heartbeat(f"{method} restricted {dataset}/{partition}"):
            output = recipes.RECIPES[method](ctx)
        ctx.release()
        if not output.ranked:
            raise FillError("restricted selector fit returned no ranking")
        return {"ranking": [str(item) for item in output.ranked], "protocol": output.protocol, "fit_seconds": output.fit_seconds, "selected_prefix": list(output.features)}

    def _iv_then_boruta_order(self, dataset: str, backbone: str, label: str, partition: str) -> dict[str, Any]:
        record = self.engine.compute(Request(dataset, backbone, label, partition))
        if record is None:
            raise FillError("IV then Boruta refit unavailable (see selection skips)")
        confirmed = [str(item) for item in record["features"]]
        ranked = record.get("ranked")
        if ranked:
            ordered = [str(item) for item in ranked if str(item) in set(confirmed)]
            source = "Boruta confirmed support in the selector's own (IV pool) order"
        else:
            iv_order = self._iv_order(dataset, partition)
            position = {name: index for index, name in enumerate(iv_order)}
            ordered = sorted(confirmed, key=lambda name: position.get(name, math.inf))
            source = "Boruta confirmed support re-ordered by the recomputed IV ranking (stored record predates the ranking field)"
        if set(ordered) != set(confirmed):
            raise FillError("IV then Boruta ordering lost confirmed features")
        return {"ranking": ordered, "protocol": record["protocol"], "fit_seconds": record.get("fit_seconds"), "ranking_source": source, "natural_support": len(confirmed)}

    def _iv_order(self, dataset: str, partition: str) -> list[str]:
        cache = self.work_dir / "gate4" / "base_rankings" / dataset / f"iv_order__{partition}.json"
        if cache.exists():
            return list(read_json(cache)["ranking"])
        from credit_risk_fs.selectors.lightweight.iv import InformationValueSelector

        ctx = recipes.FitContext(dataset=dataset, backbone="lr", partition=partition, data=self.data)
        settings = dict(recipes.combination_settings(dataset).get("iv_woe", {}))
        settings.pop("k", None)
        numeric, y = ctx.encoded()
        selector = InformationValueSelector(k=min(300, numeric.shape[1] - 1), random_state=SEED, fit_scope="dev_fold_training_only", **recipes._filtered(InformationValueSelector, settings))
        log(f"[43] IV ranking for {dataset}/{partition}: {numeric.shape[0]} rows x {numeric.shape[1]} cols")
        selector.fit(numeric, y)
        ranking = [str(item) for item in selector.result.ranking]
        ctx.release()
        write_json(cache, {"ranking": ranking})
        return ranking


def apply_family_cap(ranking: Sequence[str], families: dict[str, str], k: int, cap: int) -> dict[str, Any]:
    """Walk the ranking, keep at most ``cap`` features per family, stop at ``k``."""

    counts: dict[str, int] = {}
    chosen: list[str] = []
    skipped: list[str] = []
    for name in ranking:
        if len(chosen) >= k:
            break
        family = families.get(name, "unknown")
        if counts.get(family, 0) >= cap:
            skipped.append(name)
            continue
        counts[family] = counts.get(family, 0) + 1
        chosen.append(name)
    return {"features": chosen, "family_counts": counts, "cap": cap, "k": k, "shortfall": k - len(chosen), "skipped_by_cap": skipped}
