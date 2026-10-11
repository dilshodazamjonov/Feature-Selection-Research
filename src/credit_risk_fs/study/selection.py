"""Fold-local selection.

Each selector is fitted on the training rows of a partition only. Rankings that do not
depend on the budget (IV, mutual-information mRMR, Boruta, IV then Boruta, CatBoost
SHAP, domain rules, Random K, the LLM rankings and the hybrid stages on a semantic
pool) are computed once per partition and cached; a subset of K features is a prefix
of the ranking. Recursive elimination and the stable-core construction depend on K and
are fitted for each budget; PCA is fitted with the backbone.

Inputs per selector: the raw candidate frame for IV, mRMR, the LLM stages, domain
rules and Random K; the one-column-per-feature numeric encoding (median-imputed
numerics, ordinal-coded categoricals) for Boruta, recursive elimination, SHAP, the
RF/correlation filter and the L1 selector.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from credit_risk_fs.study import llm
from credit_risk_fs.study.config import FULL, budget, output_dir, pool_size
from credit_risk_fs.study.utils import read_json, write_json


def _default_index(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame itself when it already has a 0..n-1 index (``reset_index`` would copy it)."""

    index = frame.index
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        return frame
    return frame.reset_index(drop=True)


class PartitionSelector:
    """Rankings and subsets for one training partition."""

    def __init__(
        self,
        cfg: dict[str, Any],
        partition: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        *,
        key_prefix: str = "",
    ) -> None:
        self.cfg = cfg
        self.partition = partition
        self.X = _default_index(X_train)
        self.y = pd.Series(np.asarray(y_train, dtype=int))
        self.candidates = [str(name) for name in self.X.columns]
        self.key_prefix = key_prefix
        self.directory = output_dir(cfg) / "rankings" / partition
        self.seed = int(cfg["seed"])
        self._encoded: pd.DataFrame | None = None

    # ------------------------------------------------------------- plumbing
    @property
    def encoded(self) -> pd.DataFrame:
        if self._encoded is None:
            from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder

            self._encoded = OriginalFeatureNumericEncoder().fit_transform(self.X)
        return self._encoded

    def release(self) -> None:
        self._encoded = None

    def cached(self, key: str, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        path = self.directory / f"{self.key_prefix}{key}.json"
        if path.exists():
            return read_json(path)
        started = time.perf_counter()
        payload = compute()
        payload["seconds"] = time.perf_counter() - started
        payload["partition"] = self.partition
        payload["n_training_rows"] = int(len(self.X))
        write_json(path, payload)
        return payload

    def _ordered(self, names: Sequence[str]) -> list[str]:
        wanted = set(map(str, names))
        return [name for name in self.candidates if name in wanted]

    # ------------------------------------------------------------- rankings
    def domain(self) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            from credit_risk_fs.selectors.domain_rules import rank_features

            ranked = rank_features(self.candidates)
            return {
                "ranking": [item["feature_name"] for item in ranked],
                "leakage_flagged": [item["feature_name"] for item in ranked if item["leakage_status"] != "LEAKAGE_SAFE_LABEL_FREE"],
            }

        return self.cached("domain", compute)

    def random(self) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            from credit_risk_fs.selectors.lightweight.controls import RandomKSelector

            selector = RandomKSelector(k=None, random_state=self.seed).fit(self.X.iloc[:1])
            return {"ranking": list(selector.result.ranking)}

        return self.cached("random", compute)

    def iv(self) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            from credit_risk_fs.selectors.lightweight.iv import InformationValueSelector

            settings = self.cfg["selectors"]["iv"]
            selector = InformationValueSelector(
                k=None,
                n_bins=int(settings["n_bins"]),
                binning_strategy=str(settings["binning_strategy"]),
                zero_count_smoothing=float(settings["zero_count_smoothing"]),
                eligibility_interval=tuple(settings["eligibility_interval"]),
                random_state=self.seed,
            ).fit(self.X, self.y)
            scores = dict(selector.iv_scores_ or {})
            return {
                "ranking": list(selector.result.ranking),
                "iv": scores,
                "max_iv": max(scores.values()) if scores else None,
                "above_upper_bound": [name for name, value in scores.items() if value > settings["eligibility_interval"][1]],
            }

        return self.cached("iv", compute)

    def _mrmr(self, X: pd.DataFrame, k: int, *, group_of: dict[str, str] | None = None, group_cap: int | None = None) -> list[str]:
        from credit_risk_fs.selectors.lightweight.mi_mrmr import MutualInformationMRMRSelector

        settings = self.cfg["selectors"]["mrmr"]
        selector = MutualInformationMRMRSelector(
            k=min(int(k), X.shape[1]),
            estimator=str(settings["estimator"]),
            sample_rows=settings.get("sample_rows"),
            n_neighbors=int(settings["n_neighbors"]),
            objective=str(settings["objective"]),
            group_of=group_of,
            group_cap=group_cap,
            random_state=self.seed,
            n_jobs=int(settings.get("n_jobs", 1)),
        ).fit(X, self.y)
        return list(selector.result.ranking)

    def mrmr(self) -> dict[str, Any]:
        k_max = int(self.cfg["selectors"]["mrmr"]["k_max"])
        return self.cached("mrmr", lambda: {"ranking": self._mrmr(self.X, k_max), "k_max": k_max})

    def _boruta(self, columns: Sequence[str]) -> dict[str, Any]:
        from credit_risk_fs.selectors.heavy.boruta_rf import BorutaRandomForestSelector

        settings = self.cfg["selectors"]["boruta"]
        frame = self.encoded.loc[:, list(columns)] if list(columns) != self.candidates else self.encoded
        selector = BorutaRandomForestSelector(
            k=None,
            selection_mode="natural_confirmed",
            forest_params=dict(settings["forest"]),
            boruta_params=dict(settings["engine"]),
            random_state=self.seed,
            n_jobs=int(settings.get("n_jobs", 1)),
        ).fit(frame, self.y)
        return {
            "confirmed": list(selector.confirmed_),
            "tentative": list(selector.tentative_),
            "rejected": list(selector.rejected_),
            "ranking": list(selector.result.ranking or []),
            "n_screened": len(list(columns)),
        }

    def boruta(self) -> dict[str, Any]:
        return self.cached("boruta", lambda: self._boruta(self.candidates))

    def iv_boruta(self) -> dict[str, Any]:
        screen = self._ordered(self.iv()["ranking"])
        payload = self.cached("iv_boruta", lambda: {**self._boruta(screen), "screen": screen})
        payload["seconds_total"] = float(payload["seconds"]) + float(self.iv()["seconds"])
        return payload

    def shap(self) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            from credit_risk_fs.selectors.heavy.catboost_shap import CatBoostShapSelector

            settings = self.cfg["selectors"]["shap"]
            selector = CatBoostShapSelector(
                k=len(self.candidates),
                explanation_sample_size=int(settings["explanation_sample_size"]),
                catboost_params=dict(settings["catboost"]),
                random_state=self.seed,
                thread_count=int(settings.get("thread_count", 1)),
            ).fit(self.encoded, self.y)
            return {"ranking": list(selector.result.ranking)}

        return self.cached("shap", compute)

    def _rfe(self, columns: Sequence[str], k: int) -> list[str]:
        from credit_risk_fs.selectors.heavy.rfe_catboost import CatBoostRFESelector

        settings = self.cfg["selectors"]["rfe"]
        frame = self.encoded.loc[:, list(columns)] if list(columns) != self.candidates else self.encoded
        selector = CatBoostRFESelector(
            k=int(k),
            step_fraction=float(settings["step_fraction"]),
            catboost_params=dict(settings["catboost"]),
            random_state=self.seed,
            thread_count=int(settings.get("thread_count", 1)),
        ).fit(frame, self.y)
        return list(selector.result.ranking)

    def rfe(self, k: int) -> dict[str, Any]:
        return self.cached(f"rfe_{int(k)}", lambda: {"ranking": self._rfe(self.candidates, k), "k": int(k)})

    def stable_core(self, k: int) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            from credit_risk_fs.selectors.stable_core_llm_fill import StableCoreLLMFillSelector

            settings = self.cfg["selectors"]["stable_core"]
            builder = StableCoreLLMFillSelector(
                description_csv_path=None,
                final_feature_budget=int(k),
                bootstrap_iterations=int(settings["bootstrap_iterations"]),
                bootstrap_fraction=float(settings["bootstrap_fraction"]),
                stability_threshold=float(settings["stability_threshold"]),
                random_state=self.seed,
                component_n_jobs=int(settings.get("n_jobs", 1)),
            )
            core, frequency = builder._bootstrap_core(self.encoded, self.y)
            return {"core": list(core), "frequency": frequency.to_dict("records"), "k": int(k)}

        return self.cached(f"stable_core_{int(k)}", compute)

    def llm_ranking(self, length: int, *, arm: str = llm.NAMED, repeat: int = 0) -> list[str]:
        names = llm.ranking(self.cfg, arm, self.partition, length, repeat=repeat)
        unknown = [name for name in names if name not in set(self.candidates)]
        if unknown:
            raise ValueError(f"cached ranking contains names outside the candidate set: {unknown[:5]}")
        return names

    def _expanded_pool(self, pool: Sequence[str]) -> pd.DataFrame:
        from credit_risk_fs.study.modeling import FittedPreprocessing

        columns = self._ordered(pool)
        preprocessing = FittedPreprocessing(self.cfg, self.X.loc[:, columns])
        return preprocessing.transform(self.X.loc[:, columns])

    def llm_mrmr(self, backbone: str, *, arm: str = llm.NAMED) -> dict[str, Any]:
        size = pool_size(self.cfg, backbone)
        tag = "" if arm == llm.NAMED else f"_{arm}"

        def compute() -> dict[str, Any]:
            pool = self.llm_ranking(size, arm=arm)[:size]
            k_max = int(self.cfg["selectors"]["mrmr"]["k_max"])
            if self.cfg.get("llm_then_mrmr_expanded", False):
                expanded = self._expanded_pool(pool)
                ranking = self._mrmr(expanded, k_max)
                return {"pool": pool, "ranking": ranking, "expanded": True, "n_expanded_columns": int(expanded.shape[1])}
            return {"pool": pool, "ranking": self._mrmr(self.X.loc[:, self._ordered(pool)], k_max), "expanded": False}

        return self.cached(f"llm_mrmr_{backbone}{tag}", compute)

    def llm_boruta(self, backbone: str, *, arm: str = llm.NAMED) -> dict[str, Any]:
        size = pool_size(self.cfg, backbone)
        tag = "" if arm == llm.NAMED else f"_{arm}"

        def compute() -> dict[str, Any]:
            pool = self.llm_ranking(size, arm=arm)[:size]
            return {**self._boruta(self._ordered(pool)), "pool": pool}

        return self.cached(f"llm_boruta_{backbone}{tag}", compute)

    def rf_corr_full(self) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector

            d = len(self.candidates)
            selector = RandomForestRelevanceMRMRSelector(
                k=d - 1,
                method="mrmr",
                random_state=self.seed,
                n_jobs=int(self.cfg["selectors"]["stable_core"].get("n_jobs", 1)),
            ).fit(self.encoded, self.y)
            ranked = list(selector.selected_features_)
            ranked += [name for name in self.candidates if name not in set(ranked)]
            return {"ranking": ranked}

        return self.cached("rf_corr_full", compute)

    def rfe_full(self) -> dict[str, Any]:
        return self.cached("rfe_full", lambda: {"ranking": self._rfe(self.candidates, 1), "k": 1})

    def l1(self, k: int) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            from credit_risk_fs.selectors.lightweight.lasso import L1LogisticSelector

            settings = self.cfg["selectors"]["l1"]
            selector = L1LogisticSelector(
                k=int(k),
                C=float(settings["C"]),
                solver=str(settings["solver"]),
                max_iter=int(settings["max_iter"]),
                class_weight=settings.get("class_weight"),
                imputation_strategy=str(settings.get("imputation_strategy", "median")),
                random_state=self.seed,
            ).fit(self.encoded, self.y)
            return {
                "selected": list(selector.selected_features_ or []),
                "natural_support_size": len(selector.result.natural_selected or ()),
                "budget_status": selector.result.budget_status,
            }

        return self.cached(f"l1_{int(k)}", compute)

    # ------------------------------------------------------------- subsets
    def subset(self, selector: str, backbone: str, k: int, *, arm: str = llm.NAMED) -> dict[str, Any]:
        """Subset of a strategy at budget ``k`` and the rankings it used (for timing)."""

        k = int(k)
        if selector == "Domain rules":
            ranking = self.domain()
            return {"features": ranking["ranking"][:k], "mode": "subset", "seconds": ranking["seconds"]}
        if selector == "Random K":
            ranking = self.random()
            return {"features": ranking["ranking"][:k], "mode": "subset", "seconds": ranking["seconds"]}
        if selector == "PCA":
            return {"features": list(self.candidates), "mode": "pca", "pca_components": k, "seconds": 0.0}
        if selector == "IV/WOE":
            ranking = self.iv()
            return {"features": ranking["ranking"][:k], "mode": "subset", "seconds": ranking["seconds"]}
        if selector == "mRMR":
            ranking = self.mrmr()
            return {"features": ranking["ranking"][:k], "mode": "subset", "seconds": ranking["seconds"]}
        if selector == "Boruta RF":
            ranking = self.boruta()
            return {
                "features": ranking["confirmed"][:k],
                "mode": "subset",
                "n_confirmed": len(ranking["confirmed"]),
                "seconds": ranking["seconds"],
            }
        if selector == "IV then Boruta":
            ranking = self.iv_boruta()
            return {
                "features": ranking["confirmed"][:k],
                "mode": "subset",
                "n_confirmed": len(ranking["confirmed"]),
                "seconds": ranking["seconds_total"],
            }
        if selector == "RFE CatBoost":
            ranking = self.rfe(k)
            return {"features": ranking["ranking"][:k], "mode": "subset", "seconds": ranking["seconds"]}
        if selector == "CatBoost SHAP":
            ranking = self.shap()
            return {"features": ranking["ranking"][:k], "mode": "subset", "seconds": ranking["seconds"]}
        if selector == "Pure LLM":
            return {"features": self.llm_ranking(k, arm=arm)[:k], "mode": "subset", "seconds": 0.0}
        if selector == "LLM then mRMR":
            ranking = self.llm_mrmr(backbone, arm=arm)
            return {
                "features": ranking["ranking"][:k],
                "mode": "expanded" if ranking.get("expanded") else "subset",
                "pool": ranking["pool"],
                "seconds": ranking["seconds"],
            }
        if selector == "LLM then Boruta":
            ranking = self.llm_boruta(backbone, arm=arm)
            return {
                "features": ranking["confirmed"][:k],
                "mode": "subset",
                "n_confirmed": len(ranking["confirmed"]),
                "seconds": ranking["seconds"],
            }
        if selector == "Stable core + LLM fill":
            core = self.stable_core(k)
            fill = self.llm_ranking(pool_size(self.cfg, backbone), arm=arm)[: pool_size(self.cfg, backbone)]
            eligible = set(self.candidates)
            features: list[str] = []
            for name in list(core["core"]) + list(fill):
                if name in eligible and name not in features:
                    features.append(name)
                if len(features) == k:
                    break
            return {"features": features, "mode": "subset", "core_size": len(core["core"]), "seconds": core["seconds"]}
        raise ValueError(f"unknown selector {selector!r}")

    def base_ranking(self, selector: str, k: int) -> list[str] | None:
        """Full ranking a classical selector fills a capped subset from (None: not applicable)."""

        if selector == "Domain rules":
            return self.domain()["ranking"]
        if selector == "Random K":
            return self.random()["ranking"]
        if selector == "IV/WOE":
            return self.iv()["ranking"]
        if selector == "Boruta RF":
            return self.boruta()["confirmed"]
        if selector == "IV then Boruta":
            return self.iv_boruta()["confirmed"]
        if selector == "RFE CatBoost":
            return self.rfe(k)["ranking"]
        if selector == "CatBoost SHAP":
            return self.shap()["ranking"]
        return None

    def capped_subset(self, selector: str, k: int, families: dict[str, str], cap: int) -> dict[str, Any]:
        """Base selector at budget ``k`` with at most ``cap`` features per source family."""

        if selector == "mRMR":
            payload = self.cached(
                f"mrmr_cap{cap}_{k}",
                lambda: {"ranking": self._mrmr(self.X, k, group_of={c: families.get(c, c) for c in self.candidates}, group_cap=cap)},
            )
            return {"features": payload["ranking"][:k], "mode": "subset"}
        ranking = self.base_ranking(selector, k)
        if ranking is None:
            raise ValueError(f"the family cap is not defined for {selector}")
        counts: dict[str, int] = {}
        features: list[str] = []
        for name in ranking:
            family = families.get(name, name)
            if counts.get(family, 0) >= cap:
                continue
            features.append(name)
            counts[family] = counts.get(family, 0) + 1
            if len(features) == k:
                break
        return {"features": features, "mode": "subset"}


def reference_features(cfg: dict[str, Any], universe: Sequence[str], candidates: Sequence[str]) -> list[str]:
    """Variables of the full-feature reference (the candidate set or the whole universe)."""

    return list(universe) if cfg.get("full_reference", "candidates") == "universe" else list(candidates)


__all__ = ["FULL", "PartitionSelector", "budget", "reference_features"]
