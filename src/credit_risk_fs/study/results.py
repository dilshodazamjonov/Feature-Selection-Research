"""Result files of one dataset, built only from the study outputs.

Writes the 25 files of ``results/<dataset folder>/`` (the schema checked by
``scripts/verify_manuscript_tables.py --schema``), publishes the accepted LLM rankings to
``results/llm_rankings/<dataset>/`` with ``results/llm_rankings/index.csv``, and rebuilds
``results/headline_subsets.csv`` from the leader-subset files present.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from credit_risk_fs.study import llm
from credit_risk_fs.study.config import (
    BACKBONES,
    BORUTA_SELECTORS,
    CLASSICAL,
    FAMILY,
    FOLDS,
    FULL,
    FULL_DEV,
    LLM_ASSISTED,
    PARTITIONS,
    SELECTORS,
    SLUGS,
    StudyError,
    budget,
    output_dir,
    pool_size,
    resolve,
    results_dir,
)
from credit_risk_fs.study.data import load_prepared, load_split
from credit_risk_fs.study.metrics import (
    PairedBootstrap,
    auc,
    delong_p_value,
    feature_psi,
    holm,
    jaccard,
    ks,
    nogueira,
    pairwise_jaccard,
)
from credit_risk_fs.study.modeling import FittedPreprocessing
from credit_risk_fs.study.records import protected_features
from credit_risk_fs.study.stages import dev_mean, leaders, load_main_records, main_scores_path, record_path
from credit_risk_fs.study.utils import load_array, log, read_json, sha256_file, sha256_text, utc_now, write_csv, write_json

HEADERS = {
    "00_run_ids.csv": "file,run_id,source_path,source_sha256,note",
    "01_cohort.csv": "cohort_cutoff,split,window_start,window_end,unique_index_units,n_rows,n_events,event_rate,share_of_rows,note",
    "02_candidate_set.csv": "screen,screen_partition,screen_rows,universe,dropped,retained,note",
    "03_ho_auc.csv": "backbone,selector,family,K,ho_auc,note",
    "04_dev_fold_auc.csv": "backbone,selector,family,K,fold1_auc,fold2_auc,fold3_auc,fold4_auc,fold5_auc,dev_mean_auc,note",
    "05_leader_delta_auc.csv": "rule,backbone,llm_leader,classical_leader,llm_dev_auc,classical_dev_auc,llm_ho_auc,classical_ho_auc,delta_auc,ci_low,ci_high,n_ho,bootstrap_resamples,bootstrap_seed,note",
    "06_leader_feature_psi.csv": "backbone,selector,mean_psi,median_psi,max_psi,max_psi_feature,note",
    "07_selection_stability.csv": "backbone,selector,K,d,nogueira,jaccard_mean,jaccard_min,jaccard_max,note",
    "08_secondary_metrics.csv": "backbone,selector,ks,brier,log_loss,capture_at_10,lift_at_10,note",
    "09_leader_subsets.csv": "backbone,selector,rank,feature,source_family,is_categorical,n_model_columns,is_protected,feature_psi,note",
    "10_boruta_attainment.csv": "backbone,selector,partition,n_confirmed,K,n_selected,below_budget,note",
    "11_subpopulation_ho_auc.csv": "subpopulation,backbone,n,llm_leader,classical_leader,llm_ho_auc,classical_ho_auc,delta_auc,ci_low,ci_high,note",
    "12_native_categorical.csv": "backbone,selector,K,ho_auc_onehot,ho_auc_native,note",
    "13_lineage_only.csv": "backbone,selector,K,ho_auc_mechanical,ho_auc_cached,delta_auc,ci_low,ci_high,overlap_with_cached,note",
    "14_l1_selector.csv": "backbone,K,ho_auc,classical_leader,classical_leader_ho_auc,shortfall,note",
    "15_diversity_controls.csv": "backbone,control,base_selector,K,cap_per_family,n_families,control_ho_auc,uncapped_ho_auc,llm_assisted_ho_auc,difference,note",
    "16_response_spread.csv": "backbone,K,draw_source,draw_id,ho_auc,is_headline_response,note",
    "17_budget_sweep.csv": "backbone,K,selector,family,dev_mean_auc,is_llm_leader,is_classical_leader,note",
    "18_response_agreement.csv": "backbone,K,draws,d,nogueira,jaccard_mean,names_in_all_draws,distinct_names,note",
    "19_obfuscation.csv": "arm,backbone,K,ho_auc_arm,ho_auc_named,delta_auc,fold1_auc,fold2_auc,fold3_auc,fold4_auc,fold5_auc,overlap_full_dev,overlap_folds_min,overlap_folds_max,overlap_folds_mean,note",
    "19b_obfuscation_ranks.csv": "arm,call,feature,rank,note",
    "19c_obfuscation_fold_jaccard.csv": "arm,K,fold_i,fold_j,jaccard,note",
    "20_rank_voting.csv": "backbone,pool_P,is_primary,reference_auc,voting_auc,delta_auc,ci_low,ci_high,holm_p,delta_ks,voted_fold_jaccard_mean,reference_fold_jaccard_mean,note",
    "21_llm_usage.csv": "call_id,partition,budget,logical_requests_served,cache_hit,input_tokens,output_tokens,model_snapshot,note",
    "22_runtime_and_memory.csv": "backbone,pipeline,stage,wall_clock_minutes,seconds,peak_rss_gb,matched_conditions,note",
}
SOURCE_STAGE = {
    "01_cohort.csv": "prepare",
    "02_candidate_set.csv": "prepare",
    "03_ho_auc.csv": "main",
    "04_dev_fold_auc.csv": "main",
    "05_leader_delta_auc.csv": "main",
    "06_leader_feature_psi.csv": "main",
    "07_selection_stability.csv": "main",
    "08_secondary_metrics.csv": "main",
    "09_leader_subsets.csv": "main",
    "10_boruta_attainment.csv": "main",
    "11_subpopulation_ho_auc.csv": "main",
    "12_native_categorical.csv": "robustness",
    "13_lineage_only.csv": "robustness",
    "14_l1_selector.csv": "robustness",
    "15_diversity_controls.csv": "robustness",
    "16_response_spread.csv": "robustness",
    "17_budget_sweep.csv": "sweep",
    "18_response_agreement.csv": "robustness",
    "19_obfuscation.csv": "robustness",
    "19b_obfuscation_ranks.csv": "robustness",
    "19c_obfuscation_fold_jaccard.csv": "robustness",
    "20_rank_voting.csv": "voting",
    "21_llm_usage.csv": "llm",
    "22_runtime_and_memory.csv": "main",
}
BACKBONE_LABEL = {"lr": "LR", "catboost": "CatBoost"}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]"),
    re.compile(r"(?i)authorization\s*:"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{10,}"),
)


INTEGER_COLUMNS = {
    "unique_index_units", "n_rows", "n_events", "screen_rows", "universe", "dropped", "retained", "K", "d",
    "n", "rank", "n_confirmed", "n_selected", "draws", "names_in_all_draws", "distinct_names",
    "overlap_full_dev", "overlap_folds_min", "overlap_folds_max", "pool_P", "budget",
    "logical_requests_served", "input_tokens", "output_tokens", "n_model_columns", "cap_per_family",
    "n_families", "n_ho", "bootstrap_resamples", "bootstrap_seed",
}


def flag(value: bool | None) -> str:
    return "" if value is None else ("TRUE" if value else "FALSE")


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Integral count columns that picked up blanks are written without a decimal point."""

    output = frame.copy()
    for column in output.columns:
        if column not in INTEGER_COLUMNS or not pd.api.types.is_float_dtype(output[column].dtype):
            continue
        values = output[column].dropna()
        if values.empty or np.all(np.equal(np.mod(values.to_numpy(), 1), 0)):
            output[column] = output[column].astype("Int64")
    return output


def _frame(name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = HEADERS[name].split(",")
    return pd.DataFrame(rows, columns=columns)


class ResultsBuilder:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.out = output_dir(cfg)
        self.prepared = load_prepared(cfg)
        self.records = load_main_records(cfg)
        self.pairs = leaders(cfg, self.records)
        self.folder = results_dir(cfg) / str(cfg["results_folder"])
        self.y_ho: np.ndarray | None = None
        self.ids_ho: np.ndarray | None = None
        self.bootstrap: PairedBootstrap | None = None
        self.settings = cfg["inference"]
        self.run_id = f"{cfg['dataset']}-study-{utc_now().replace(':', '').replace('-', '')}"
        self.files: dict[str, pd.DataFrame] = {}
        self.k = {backbone: budget(cfg, backbone) for backbone in BACKBONES}
        self.d = len(self.prepared.candidates)

    # ------------------------------------------------------------- shared inputs
    def ho_targets(self) -> tuple[np.ndarray, np.ndarray]:
        if self.y_ho is None:
            extras = pd.read_parquet(self.out / "prepare" / "ho_extras.parquet")
            self.y_ho = extras["target"].to_numpy().astype(int)
            self.ids_ho = extras["id"].astype(str).to_numpy()
        return self.y_ho, self.ids_ho

    def paired(self) -> PairedBootstrap:
        if self.bootstrap is None:
            y, ids = self.ho_targets()
            self.bootstrap = PairedBootstrap(
                y, ids, resamples=int(self.settings["bootstrap_resamples"]), seed=int(self.settings["bootstrap_seed"])
            )
        return self.bootstrap

    def main(self, backbone: str, label: str, partition: str = FULL_DEV) -> dict[str, Any]:
        return self.records.get((backbone, label, partition), {})

    def scores(self, backbone: str, label: str) -> np.ndarray:
        return load_array(main_scores_path(self.cfg, backbone, label))

    def k_of(self, backbone: str, label: str) -> int:
        if label == FULL:
            return len(self.main(backbone, FULL).get("features", []))
        return self.k[backbone]

    # ------------------------------------------------------------- 01, 02
    def cohort(self) -> None:
        payload = read_json(self.out / "prepare" / "cohort.json")
        rows = []
        for row in payload["rows"]:
            rows.append({"cohort_cutoff": payload["cohort_cutoff"], **row, "note": _cohort_note(row["split"])})
        self.files["01_cohort.csv"] = _frame("01_cohort.csv", rows)

    def candidate_set(self) -> None:
        candidates = read_json(self.out / "prepare" / "candidates.json")
        screen = pd.read_csv(self.out / "prepare" / "screen.csv")
        universe = len(candidates["universe"])
        retained = len(candidates["candidates"])
        threshold = candidates["max_missing_rate"]
        rows = [
            {
                "screen": f"missing rate <= {threshold:.2f} on fold-1 DEV training rows",
                "screen_partition": "fold1 training",
                "screen_rows": candidates["screen_rows"],
                "universe": universe,
                "dropped": universe - retained,
                "retained": retained,
                "note": "outcome-free screen computed once on the earliest training partition and held fixed across the folds",
            }
        ]
        if candidates["index_derived_excluded"]:
            rows.append(
                {
                    "screen": "functions of the ordering index",
                    "screen_partition": "",
                    "screen_rows": "",
                    "universe": universe,
                    "dropped": len(candidates["index_derived_excluded"]),
                    "retained": universe - len(candidates["index_derived_excluded"]),
                    "note": "excluded from every candidate set: " + "; ".join(candidates["index_derived_excluded"]),
                }
            )
        iv_path = self.out / "rankings" / "fold1" / "iv.json"
        if iv_path.exists():
            iv = read_json(iv_path)
            above = len(iv.get("above_upper_bound", []))
            upper = self.cfg["selectors"]["iv"]["eligibility_interval"][1]
            rows.append(
                {
                    "screen": f"information value above {upper:.2f} on fold-1 DEV training rows",
                    "screen_partition": "fold1 training",
                    "screen_rows": candidates["screen_rows"],
                    "universe": retained,
                    "dropped": above,
                    "retained": retained - above,
                    "note": f"upper bound of the IV interval; maximum IV on the earliest training partition {iv.get('max_iv')}",
                }
            )
        loader = self.cfg["loader"]
        if loader == "lendingclub_v2":
            rates = screen.set_index("feature")["missing_rate"]
            top = int(round(100 * float(rates.max())))
            bands = [
                ("missingness < 40%", rates < 0.40),
                ("missingness 40-60%", (rates >= 0.40) & (rates < 0.60)),
                (f"missingness 60-{top}%", rates >= 0.60),
            ]
            for label, mask in bands:
                rows.append(
                    {
                        "screen": label,
                        "screen_partition": "fold1 training",
                        "screen_rows": candidates["screen_rows"],
                        "universe": universe,
                        "dropped": "",
                        "retained": int(mask.sum()),
                        "note": "features by missing rate on the earliest training partition",
                    }
                )
            inventory = pd.read_csv(resolve(self.cfg, self.cfg["inventory"]), dtype=str).fillna("")
            rows.append(
                {
                    "screen": "source columns",
                    "screen_partition": "",
                    "screen_rows": "",
                    "universe": "",
                    "dropped": "",
                    "retained": _lendingclub_source_columns(inventory),
                    "note": "distinct loan-book columns referenced by the inventory formulas (names of derived features excluded)",
                }
            )
            labels = self.cfg.get("composition_labels", {})
            for feature_type, count in inventory["feature_type"].value_counts().sort_index().items():
                rows.append(
                    {
                        "screen": f"composition: {labels.get(feature_type, feature_type)}",
                        "screen_partition": "",
                        "screen_rows": "",
                        "universe": universe,
                        "dropped": "",
                        "retained": int(count),
                        "note": f"feature_type = {feature_type} in the feature inventory",
                    }
                )
        if loader == "stability_2024":
            from credit_risk_fs.data.homecredit_model_stability_2024 import load_adapter_contract

            contract = load_adapter_contract(resolve(self.cfg, self.cfg["protocol_lock"]))
            reviewed = sum(len(table.feature_rules) for table in contract.tables)
            approved = len(contract.predictor_rules)
            rows.append(
                {
                    "screen": "prediction-time availability review of raw predictors",
                    "screen_partition": "",
                    "screen_rows": "",
                    "universe": reviewed,
                    "dropped": reviewed - approved,
                    "retained": approved,
                    "note": "raw variables reviewed and approved in the protocol lock (identifiers, target and split controls are among the reviewed rows)",
                }
            )
        self.files["02_candidate_set.csv"] = _frame("02_candidate_set.csv", rows)

    # ------------------------------------------------------------- 03, 04, 08, 10
    def matrix(self) -> None:
        ho_rows, dev_rows, secondary, boruta = [], [], [], []
        for backbone in BACKBONES:
            for label in (*SELECTORS, FULL):
                record = self.main(backbone, label)
                k = self.k_of(backbone, label)
                ho_rows.append(
                    {
                        "backbone": backbone,
                        "selector": label,
                        "family": FAMILY[label],
                        "K": k,
                        "ho_auc": record.get("auc"),
                        "note": _subset_note(record),
                    }
                )
                folds = [self.main(backbone, label, fold).get("auc") for fold in FOLDS]
                dev_rows.append(
                    {
                        "backbone": backbone,
                        "selector": label,
                        "family": FAMILY[label],
                        "K": k,
                        **{f"fold{index}_auc": value for index, value in enumerate(folds, start=1)},
                        "dev_mean_auc": dev_mean(self.records, backbone, label),
                        "note": "validation AUC of the five expanding-window folds",
                    }
                )
                secondary.append(
                    {
                        "backbone": backbone,
                        "selector": label,
                        **{key: record.get(key) for key in ("ks", "brier", "log_loss", "capture_at_10", "lift_at_10")},
                        "note": "HO; Capture@10 and Lift@10 over the top ceil(10%) scores, ties broken by identifier",
                    }
                )
            for label in BORUTA_SELECTORS:
                for partition in PARTITIONS:
                    record = self.main(backbone, label, partition)
                    confirmed = record.get("n_confirmed")
                    selected = len(record.get("features", []))
                    boruta.append(
                        {
                            "backbone": backbone,
                            "selector": label,
                            "partition": partition,
                            "n_confirmed": confirmed,
                            "K": self.k[backbone],
                            "n_selected": selected,
                            "below_budget": flag(selected < self.k[backbone]),
                            "note": "confirmed features only; tentative and rejected features never fill the budget",
                        }
                    )
        self.files["03_ho_auc.csv"] = _frame("03_ho_auc.csv", ho_rows)
        self.files["04_dev_fold_auc.csv"] = _frame("04_dev_fold_auc.csv", dev_rows)
        self.files["08_secondary_metrics.csv"] = _frame("08_secondary_metrics.csv", secondary)
        self.files["10_boruta_attainment.csv"] = _frame("10_boruta_attainment.csv", boruta)

    # ------------------------------------------------------------- 05
    def leader_deltas(self) -> None:
        rows = []
        y, _ = self.ho_targets()
        computed: dict[tuple[str, str, str], dict[str, float]] = {}
        for backbone in BACKBONES:
            for rule in ("HO", "DEV", "mixed"):
                llm_leader, classical = self.pairs[backbone][rule]
                if llm_leader is None or classical is None:
                    continue
                llm_ho = self.main(backbone, llm_leader).get("auc")
                classical_ho = self.main(backbone, classical).get("auc")
                key = (backbone, llm_leader, classical)
                interval: dict[str, Any] = {}
                note = "paired HO AUC difference, LLM-assisted minus classical; 95% target-stratified paired bootstrap"
                if rule != "mixed" or key in computed:
                    if key not in computed:
                        log(f"{self.cfg['dataset']} results: bootstrap {backbone} {llm_leader} vs {classical}")
                        computed[key] = self.paired().auc_difference(self.scores(backbone, llm_leader), self.scores(backbone, classical))
                    interval = computed[key]
                    if rule == "mixed":
                        note = "same pair as another rule of this backbone; its difference and interval"
                else:
                    note = "point estimate; interval not computed (the pair re-pairs selectors already scored)"
                rows.append(
                    {
                        "rule": rule,
                        "backbone": backbone,
                        "llm_leader": llm_leader,
                        "classical_leader": classical,
                        "llm_dev_auc": dev_mean(self.records, backbone, llm_leader),
                        "classical_dev_auc": dev_mean(self.records, backbone, classical),
                        "llm_ho_auc": llm_ho,
                        "classical_ho_auc": classical_ho,
                        "delta_auc": interval.get("delta_auc", None if llm_ho is None or classical_ho is None else llm_ho - classical_ho),
                        "ci_low": interval.get("ci_low"),
                        "ci_high": interval.get("ci_high"),
                        "n_ho": len(y),
                        "bootstrap_resamples": self.settings["bootstrap_resamples"] if interval else None,
                        "bootstrap_seed": self.settings["bootstrap_seed"] if interval else None,
                        "note": note,
                    }
                )
        self.files["05_leader_delta_auc.csv"] = _frame("05_leader_delta_auc.csv", rows)

    # ------------------------------------------------------------- 06, 09
    def _psi_values(self, features: Sequence[str], record: dict[str, Any], dev: pd.DataFrame, ho: pd.DataFrame) -> dict[str, float]:
        settings = self.cfg["psi"]
        options = {"bins": int(settings["quantile_bins"]), "discrete_max_levels": int(settings["discrete_max_levels"]), "epsilon": float(settings["epsilon"])}
        values: dict[str, float] = {}
        if record.get("mode") == "expanded":
            pool = [name for name in self.prepared.candidates if name in set(record["pool"])]
            preprocessing = FittedPreprocessing(self.cfg, dev.loc[:, pool])
            missing_token = self.cfg.get("preprocessing", {}).get("cat_missing", "Missing")
            for column in features:
                source = preprocessing.source_of(column)
                if source == column:
                    values[column] = feature_psi(dev[source], ho[source], **options)
                    continue
                level = column[len(source) + 1 :]

                def indicator(frame: pd.DataFrame) -> pd.Series:
                    text = frame[source].astype("string").fillna(missing_token).astype(str)
                    return (text == level).astype(float)

                values[column] = feature_psi(indicator(dev), indicator(ho), **options)
            return values
        for name in features:
            values[name] = feature_psi(dev[name], ho[name], **options)
        return values

    def leader_subsets(self) -> None:
        psi_rows, subset_rows = [], []
        families = self.prepared.families
        categorical = set(self.prepared.categorical)
        protected = protected_features(self.cfg, self.prepared.candidates)
        wanted: list[str] = []
        for backbone in BACKBONES:
            for label in self.pairs[backbone]["HO"]:
                record = self.main(backbone, label)
                wanted += record.get("pool", []) if record.get("mode") == "expanded" else record.get("features", [])
        columns = [name for name in self.prepared.candidates if name in set(wanted)]
        dev = load_split(self.cfg, "dev", columns).X
        ho = load_split(self.cfg, "ho", columns).X
        for backbone in BACKBONES:
            for label in self.pairs[backbone]["HO"]:
                if label is None:
                    continue
                record = self.main(backbone, label)
                features = list(record.get("features", []))
                if record.get("mode") == "pca":
                    psi_rows.append({"backbone": backbone, "selector": label, "note": "principal components: feature PSI not defined"})
                    continue
                if not features:
                    psi_rows.append({"backbone": backbone, "selector": label, "note": "empty subset"})
                    continue
                values = self._psi_values(features, record, dev, ho)
                series = pd.Series(values)
                psi_rows.append(
                    {
                        "backbone": backbone,
                        "selector": label,
                        "mean_psi": float(series.mean()),
                        "median_psi": float(series.median()),
                        "max_psi": float(series.max()),
                        "max_psi_feature": str(series.idxmax()),
                        "note": "type-aware feature PSI of the full-DEV subset, DEV against HO, states learned on DEV",
                    }
                )
                expanded = record.get("mode") == "expanded"
                preprocessing = None
                if expanded:
                    pool = [name for name in self.prepared.candidates if name in set(record["pool"])]
                    preprocessing = FittedPreprocessing(self.cfg, dev.loc[:, pool])
                for rank, feature in enumerate(features, start=1):
                    source = preprocessing.source_of(feature) if preprocessing is not None else feature
                    subset_rows.append(
                        {
                            "backbone": backbone,
                            "selector": label,
                            "rank": rank,
                            "feature": feature,
                            "source_family": families.get(source, ""),
                            "is_categorical": flag(False if expanded else feature in categorical),
                            "n_model_columns": record.get("n_model_columns"),
                            "is_protected": flag(source in protected),
                            "feature_psi": values.get(feature),
                            "note": "budget slot on the expanded (one-hot) scale" if expanded else "",
                        }
                    )
        self.files["06_leader_feature_psi.csv"] = _frame("06_leader_feature_psi.csv", psi_rows)
        self.files["09_leader_subsets.csv"] = _frame("09_leader_subsets.csv", subset_rows)

    # ------------------------------------------------------------- 07
    def stability(self) -> None:
        rows = []
        single_ranking = llm.arm_partitions(self.cfg, llm.NAMED) == [FULL_DEV]
        for backbone in BACKBONES:
            for label in SELECTORS:
                if label == "PCA":
                    continue
                subsets = [self.main(backbone, label, fold).get("features", []) for fold in FOLDS]
                pairs = pairwise_jaccard(subsets)
                note = "five fold subsets; d is the screened candidate set"
                if label == "Pure LLM" and single_ranking:
                    note = "one ranking served every fold, so the fold subsets are identical by construction"
                rows.append(
                    {
                        "backbone": backbone,
                        "selector": label,
                        "K": self.k[backbone],
                        "d": self.d,
                        "nogueira": nogueira(subsets, self.d),
                        "jaccard_mean": float(np.mean(pairs)),
                        "jaccard_min": float(np.min(pairs)),
                        "jaccard_max": float(np.max(pairs)),
                        "note": note,
                    }
                )
        self.files["07_selection_stability.csv"] = _frame("07_selection_stability.csv", rows)

    # ------------------------------------------------------------- 11
    def subpopulations(self) -> None:
        rows: list[dict[str, Any]] = []
        spec = self.cfg.get("subpopulations", {})
        if spec.get("kind") in {"lendingclub", "monthly"}:
            extras = pd.read_parquet(self.out / "prepare" / "ho_extras.parquet")
            masks: list[tuple[str, np.ndarray]] = []
            if spec["kind"] == "lendingclub":
                matured = spec["matured"]
                masks.append(
                    (
                        matured["label"],
                        (extras["term"].to_numpy() == matured["term"]) & extras["issue_month"].isin(matured["months"]).to_numpy(),
                    )
                )
                for months in spec["terms"]:
                    masks.append((f"term_{int(months)}", extras["term"].to_numpy() == months))
                for month in sorted(extras["issue_month"].unique()):
                    masks.append((f"month_{month}", extras["issue_month"].to_numpy() == month))
            else:
                for month in sorted(extras["month"].unique()):
                    masks.append((str(month), extras["month"].to_numpy() == month))
            y = extras["target"].to_numpy().astype(int)
            ids = extras["id"].astype(str).to_numpy()
            for backbone in BACKBONES:
                llm_leader, classical = self.pairs[backbone]["HO"]
                score_llm = self.scores(backbone, llm_leader)
                score_classical = self.scores(backbone, classical)
                for label, mask in masks:
                    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                        continue
                    bootstrap = PairedBootstrap(y[mask], ids[mask], resamples=int(self.settings["bootstrap_resamples"]), seed=int(self.settings["bootstrap_seed"]))
                    interval = bootstrap.auc_difference(score_llm[mask], score_classical[mask])
                    rows.append(
                        {
                            "subpopulation": label,
                            "backbone": backbone,
                            "n": int(mask.sum()),
                            "llm_leader": llm_leader,
                            "classical_leader": classical,
                            "llm_ho_auc": auc(y[mask], score_llm[mask]),
                            "classical_ho_auc": auc(y[mask], score_classical[mask]),
                            "delta_auc": interval["delta_auc"],
                            "ci_low": interval["ci_low"],
                            "ci_high": interval["ci_high"],
                            "note": "row filter on the frozen HO scores of the HO-rule leaders; no refitting",
                        }
                    )
        self.files["11_subpopulation_ho_auc.csv"] = _frame("11_subpopulation_ho_auc.csv", rows)

    # ------------------------------------------------------------- 12 to 16, 18, 19
    def robustness(self) -> None:
        directory = self.out / "robustness"
        native, lineage, l1_rows, controls, spread, agreement = [], [], [], [], [], []
        for backbone in BACKBONES:
            k = self.k[backbone]
            llm_leader, classical = self.pairs[backbone]["HO"]
            if llm_leader is None or classical is None:
                continue
            if backbone == "catboost":
                for label in (llm_leader, classical):
                    path = record_path(self.cfg, "robustness", "native", SLUGS[label])
                    if path.exists():
                        payload = read_json(path)
                        native.append(
                            {
                                "backbone": backbone,
                                "selector": label,
                                "K": k,
                                "ho_auc_onehot": payload["ho_auc_onehot"],
                                "ho_auc_native": payload["ho_auc_native"],
                                "note": f"same {len(payload['variables'])} original variables, configuration and seed; {payload['n_categorical']} categorical",
                            }
                        )
            for label in dict.fromkeys(["Pure LLM", llm_leader]):
                path = record_path(self.cfg, "robustness", "lineage", f"{backbone}__{SLUGS[label]}")
                if not path.exists():
                    continue
                payload = read_json(path)
                mechanical = load_array(directory / "scores" / f"lineage__{backbone}__{SLUGS[label]}.npy")
                interval = self.paired().auc_difference(mechanical, self.scores(backbone, label))
                cached = self.main(backbone, label)
                overlap = len(set(payload["features"]) & set(cached.get("features", [])))
                lineage.append(
                    {
                        "backbone": backbone,
                        "selector": label,
                        "K": k,
                        "ho_auc_mechanical": payload["auc"],
                        "ho_auc_cached": cached.get("auc"),
                        "delta_auc": interval["delta_auc"],
                        "ci_low": interval["ci_low"],
                        "ci_high": interval["ci_high"],
                        "overlap_with_cached": f"{overlap}/{k}",
                        "note": "descriptions rebuilt from lineage; full-DEV ranking; paired bootstrap, both rankings fixed",
                    }
                )
            l1 = record_path(self.cfg, "robustness", "l1", f"{backbone}__{FULL_DEV}")
            if l1.exists():
                payload = read_json(l1)
                leader_auc = self.main(backbone, classical).get("auc")
                l1_rows.append(
                    {
                        "backbone": backbone,
                        "K": k,
                        "ho_auc": payload["auc"],
                        "classical_leader": classical,
                        "classical_leader_ho_auc": leader_auc,
                        "shortfall": None if payload["auc"] is None or leader_auc is None else leader_auc - payload["auc"],
                        "note": f"liblinear, C = {self.cfg['selectors']['l1']['C']}, balanced class weights; K largest non-zero coefficients, no padding ({payload.get('budget_status')}, {len(payload['features'])} features)",
                    }
                )
            llm_auc = self.main(backbone, llm_leader).get("auc")
            for control in ("family_cap", "depth0"):
                path = record_path(self.cfg, "robustness", control, backbone)
                if not path.exists():
                    continue
                payload = read_json(path)
                uncapped = self.main(backbone, classical).get("auc")
                below = len(payload["features"]) < k
                controls.append(
                    {
                        "backbone": backbone,
                        "control": control,
                        "base_selector": payload.get("base_selector", classical),
                        "K": k,
                        "cap_per_family": payload.get("cap_per_family", ""),
                        "n_families": payload.get("n_families", ""),
                        "control_ho_auc": payload["auc"],
                        "uncapped_ho_auc": uncapped,
                        "llm_assisted_ho_auc": llm_auc,
                        "difference": None if payload["auc"] is None or llm_auc is None else payload["auc"] - llm_auc,
                        "note": ("below budget: " if below else "") + f"{len(payload['features'])} features; fitted on full DEV and scored once on HO",
                    }
                )
            draws = []
            for path in sorted((directory / "response").glob(f"{backbone}__*.json")):
                payload = read_json(path)
                draws.append(payload)
                spread.append(
                    {
                        "backbone": backbone,
                        "K": k,
                        "draw_source": payload["draw_source"],
                        "draw_id": payload["draw_id"],
                        "ho_auc": payload["auc"],
                        "is_headline_response": "FALSE",
                        "note": "cached ranking carried through the frozen full-DEV pipeline",
                    }
                )
            headline = self.main(backbone, "Pure LLM")
            spread.append(
                {
                    "backbone": backbone,
                    "K": k,
                    "draw_source": "full_dev_ranking",
                    "draw_id": "full_dev",
                    "ho_auc": headline.get("auc"),
                    "is_headline_response": "TRUE",
                    "note": "the full-DEV ranking behind the headline tables (not one of the draws)",
                }
            )
            if draws:
                lists = [payload["features"] for payload in draws]
                shared = set(lists[0]).intersection(*map(set, lists[1:]))
                agreement.append(
                    {
                        "backbone": backbone,
                        "K": k,
                        "draws": len(lists),
                        "d": self.d,
                        "nogueira": nogueira(lists, self.d),
                        "jaccard_mean": float(np.mean(pairwise_jaccard(lists))),
                        "names_in_all_draws": len(shared),
                        "distinct_names": len(set().union(*map(set, lists))),
                        "note": "top-K names of the draws of 16_response_spread.csv",
                    }
                )
        self.files["12_native_categorical.csv"] = _frame("12_native_categorical.csv", native)
        self.files["13_lineage_only.csv"] = _frame("13_lineage_only.csv", lineage)
        self.files["14_l1_selector.csv"] = _frame("14_l1_selector.csv", l1_rows)
        self.files["15_diversity_controls.csv"] = _frame("15_diversity_controls.csv", controls)
        self.files["16_response_spread.csv"] = _frame("16_response_spread.csv", spread)
        self.files["18_response_agreement.csv"] = _frame("18_response_agreement.csv", agreement)
        self.obfuscation()

    def obfuscation(self) -> None:
        main_rows, rank_rows, jaccard_rows = [], [], []
        if self.cfg["llm"].get("obfuscation", False):
            directory = self.out / "robustness" / "obfuscation"
            arms = (llm.ARM_A, llm.ARM_B)
            for arm in arms:
                for backbone in BACKBONES:
                    k = self.k[backbone]
                    paths = [directory / f"{arm}__{backbone}__{partition}.json" for partition in PARTITIONS]
                    if not all(path.exists() for path in paths):
                        log(f"{self.cfg['dataset']} results: obfuscation arm {arm} / {backbone} is incomplete; run --stage robustness")
                        continue
                    full = read_json(directory / f"{arm}__{backbone}__{FULL_DEV}.json")
                    folds = [read_json(directory / f"{arm}__{backbone}__{fold}.json") for fold in FOLDS]
                    named_full = self.main(backbone, "Pure LLM").get("features", [])
                    overlaps = [
                        len(set(record["features"]) & set(self.main(backbone, "Pure LLM", fold).get("features", [])))
                        for record, fold in zip(folds, FOLDS)
                    ]
                    named_auc = self.main(backbone, "Pure LLM").get("auc")
                    main_rows.append(
                        {
                            "arm": arm,
                            "backbone": backbone,
                            "K": k,
                            "ho_auc_arm": full["auc"],
                            "ho_auc_named": named_auc,
                            "delta_auc": None if full["auc"] is None or named_auc is None else full["auc"] - named_auc,
                            **{f"fold{index}_auc": record["auc"] for index, record in enumerate(folds, start=1)},
                            "overlap_full_dev": len(set(full["features"]) & set(named_full)),
                            "overlap_folds_min": min(overlaps),
                            "overlap_folds_max": max(overlaps),
                            "overlap_folds_mean": float(np.mean(overlaps)),
                            "note": "A: names replaced by seeded identifiers; B: descriptions blanked",
                        }
                    )
            for arm in (llm.NAMED, *arms):
                for partition in PARTITIONS:
                    names = llm.ranking(self.cfg, arm, partition, max(llm.arm_budgets(self.cfg, arm)))
                    for rank, name in enumerate(names, start=1):
                        rank_rows.append({"arm": arm, "call": partition, "feature": name, "rank": rank, "note": ""})
                for k in sorted(set(self.k.values())):
                    lists = [llm.ranking(self.cfg, arm, fold, k)[:k] for fold in FOLDS]
                    for (i, left), (j, right) in _pairs(lists):
                        jaccard_rows.append({"arm": arm, "K": k, "fold_i": f"fold{i}", "fold_j": f"fold{j}", "jaccard": jaccard(left, right), "note": ""})
        self.files["19_obfuscation.csv"] = _frame("19_obfuscation.csv", main_rows)
        self.files["19b_obfuscation_ranks.csv"] = _frame("19b_obfuscation_ranks.csv", rank_rows)
        self.files["19c_obfuscation_fold_jaccard.csv"] = _frame("19c_obfuscation_fold_jaccard.csv", jaccard_rows)

    # ------------------------------------------------------------- 17
    def sweep(self) -> None:
        rows = []
        budgets = [int(value) for value in self.cfg["sweep"]["budgets"]]
        for backbone in BACKBONES:
            for k in budgets:
                values: dict[str, float | None] = {}
                for label in SELECTORS:
                    if k == self.k[backbone]:
                        values[label] = dev_mean(self.records, backbone, label)
                        continue
                    aucs = []
                    for fold in FOLDS:
                        path = record_path(self.cfg, "sweep", backbone, f"K{k}", SLUGS[label], fold)
                        aucs.append(read_json(path).get("auc") if path.exists() else None)
                    values[label] = None if any(value is None for value in aucs) else float(np.mean(aucs))
                best_llm = _argmax(LLM_ASSISTED, values)
                best_classical = _argmax(CLASSICAL, values)
                for label in SELECTORS:
                    rows.append(
                        {
                            "backbone": backbone,
                            "K": k,
                            "selector": label,
                            "family": FAMILY[label],
                            "dev_mean_auc": values[label],
                            "is_llm_leader": flag(label == best_llm),
                            "is_classical_leader": flag(label == best_classical),
                            "note": "mean validation AUC over the five DEV folds; no HO observation",
                        }
                    )
                present = [value for value in values.values() if value is not None]
                rows.append(
                    {
                        "backbone": backbone,
                        "K": k,
                        "selector": "all thirteen selectors",
                        "family": "pooled",
                        "dev_mean_auc": float(np.mean(present)) if len(present) == len(SELECTORS) else None,
                        "is_llm_leader": "",
                        "is_classical_leader": "",
                        "note": "mean of the thirteen selectors",
                    }
                )
        self.files["17_budget_sweep.csv"] = _frame("17_budget_sweep.csv", rows)

    # ------------------------------------------------------------- 20
    def voting(self) -> None:
        rows = []
        if self.cfg.get("voting", {}).get("enabled", False):
            pools = [int(value) for value in self.cfg["voting"]["pools"]]
            primary = int(self.cfg["voting"]["primary_pool"])
            y, _ = self.ho_targets()
            directory = self.out / "voting"
            for backbone in BACKBONES:
                needed = [directory / "scores" / f"{backbone}__reference.npy"]
                needed += [directory / "scores" / f"{backbone}__P{pool}.npy" for pool in pools]
                needed += [record_path(self.cfg, "voting", backbone, fold) for fold in FOLDS]
                if not all(path.exists() for path in needed):
                    log(f"{self.cfg['dataset']} results: voting outputs for {backbone} are incomplete; run --stage voting")
                    continue
                reference = load_array(directory / "scores" / f"{backbone}__reference.npy")
                fold_payloads = [read_json(record_path(self.cfg, "voting", backbone, fold)) for fold in FOLDS]
                reference_jaccard = float(np.mean(pairwise_jaccard([payload["reference"] for payload in fold_payloads])))
                per_pool = {}
                for pool in pools:
                    voted = load_array(directory / "scores" / f"{backbone}__P{pool}.npy")
                    per_pool[pool] = {
                        "scores": voted,
                        "p": delong_p_value(y, voted, reference),
                        "interval": self.paired().auc_difference(voted, reference),
                        "jaccard": float(np.mean(pairwise_jaccard([payload["voted"][str(pool)] for payload in fold_payloads]))),
                    }
                adjusted = holm([per_pool[pool]["p"] for pool in pools])
                for pool, holm_p in zip(pools, adjusted):
                    item = per_pool[pool]
                    rows.append(
                        {
                            "backbone": backbone,
                            "pool_P": pool,
                            "is_primary": flag(pool == primary),
                            "reference_auc": auc(y, reference),
                            "voting_auc": auc(y, item["scores"]),
                            "delta_auc": item["interval"]["delta_auc"],
                            "ci_low": item["interval"]["ci_low"],
                            "ci_high": item["interval"]["ci_high"],
                            "holm_p": holm_p,
                            "delta_ks": ks(y, item["scores"]) - ks(y, reference),
                            "voted_fold_jaccard_mean": item["jaccard"],
                            "reference_fold_jaccard_mean": reference_jaccard,
                            "note": "voters: RF/correlation filter, Boruta, RFE; normalised ranks averaged; pool reduced to K by CatBoost RFE; reference: RF/correlation filter at K; two-sided paired DeLong, Holm within the three pools",
                        }
                    )
        self.files["20_rank_voting.csv"] = _frame("20_rank_voting.csv", rows)

    # ------------------------------------------------------------- 21
    def llm_usage(self) -> None:
        served: dict[str, int] = {}
        for (backbone, label, partition), record in self.records.items():
            if label not in LLM_ASSISTED:
                continue
            length = self.k[backbone] if label == "Pure LLM" else pool_size(self.cfg, backbone)
            call_id = llm.ranking_call_id(self.cfg, llm.NAMED, partition, length)
            served[call_id] = served.get(call_id, 0) + 1
        rows = []
        for spec in llm.plan_calls(self.cfg):
            if spec.arm != llm.NAMED:
                continue
            payload = read_json(llm.ranking_path(self.cfg, spec.call_id))
            uses = served.get(spec.call_id, 0) if not spec.repeat else len(BACKBONES)
            rows.append(
                {
                    "call_id": spec.call_id,
                    "partition": spec.partition,
                    "budget": spec.budget,
                    "logical_requests_served": uses,
                    "cache_hit": "FALSE",
                    "input_tokens": payload.get("prompt_tokens"),
                    "output_tokens": payload.get("completion_tokens"),
                    "model_snapshot": payload.get("response_model") or payload.get("request_model"),
                    "note": f"accepted at attempt {payload.get('attempts')}" + (f"; repeat {spec.repeat} of the full-DEV call" if spec.repeat else ""),
                }
            )
        self.files["21_llm_usage.csv"] = _frame("21_llm_usage.csv", rows)

    # ------------------------------------------------------------- 22
    def runtime(self) -> None:
        rows = []
        peaks: list[float] = []
        for backbone in BACKBONES:
            for label in (*SELECTORS, FULL):
                records = [self.main(backbone, label, partition) for partition in PARTITIONS]
                selection = sum(float(record.get("selection_seconds", 0.0)) for record in records)
                model = sum(float(record.get("model_seconds", 0.0)) for record in records)
                peak = max(max(record.get("peak_rss_bytes", 0), record.get("selection_peak_rss_bytes", 0)) for record in records) / 1e9
                peaks.append(peak)
                rows.append(
                    {
                        "backbone": backbone,
                        "pipeline": label,
                        "stage": "pipeline",
                        "wall_clock_minutes": (selection + model) / 60.0,
                        "seconds": selection + model,
                        "peak_rss_gb": peak,
                        "matched_conditions": "FALSE",
                        "note": "selection and model fitting over the five folds and the full-DEV refit; cached rankings count once",
                    }
                )
                rows.append(
                    {
                        "backbone": backbone,
                        "pipeline": label,
                        "stage": "selection",
                        "wall_clock_minutes": selection / 60.0,
                        "seconds": selection,
                        "peak_rss_gb": "",
                        "matched_conditions": "FALSE",
                        "note": "selection stage only",
                    }
                )
            if self.cfg.get("voting", {}).get("enabled", False):
                rf = [self.out / "rankings" / partition / "rf_corr_full.json" for partition in PARTITIONS]
                fit = record_path(self.cfg, "voting", "ho", f"{backbone}__reference")
                if all(path.exists() for path in rf) and fit.exists():
                    seconds = sum(float(read_json(path)["seconds"]) for path in rf) + float(read_json(fit).get("model_seconds", 0.0))
                    rows.append(
                        {
                            "backbone": backbone,
                            "pipeline": "mRMR (RF/correlation implementation)",
                            "stage": "pipeline",
                            "wall_clock_minutes": seconds / 60.0,
                            "seconds": seconds,
                            "peak_rss_gb": "",
                            "matched_conditions": "FALSE",
                            "note": "RF/correlation filter over the five folds and full DEV, plus the full-DEV refit",
                        }
                    )
        timing_path = self.out / "robustness" / "matched_mrmr_timing.json"
        if timing_path.exists():
            timing = read_json(timing_path)
            for label, seconds in (
                (f"mRMR over {timing['universe_features']:,} features", timing["universe_seconds"]),
                (f"mRMR over {timing['pool_features']}-feature pool", timing["pool_seconds"]),
            ):
                rows.append(
                    {
                        "backbone": "",
                        "pipeline": label,
                        "stage": "selection",
                        "wall_clock_minutes": seconds / 60.0,
                        "seconds": seconds,
                        "peak_rss_gb": "",
                        "matched_conditions": "TRUE",
                        "note": f"{timing['partition']} training rows ({timing['rows']:,}), k = {timing['k']}, same machine and settings",
                    }
                )
        valid = [peak for peak in peaks if peak > 0]
        if valid:
            for name, value in (("minimum", min(valid)), ("maximum", max(valid))):
                rows.append(
                    {
                        "backbone": "",
                        "pipeline": name,
                        "stage": "peak_rss_range",
                        "wall_clock_minutes": "",
                        "seconds": "",
                        "peak_rss_gb": value,
                        "matched_conditions": "FALSE",
                        "note": "peak resident memory over the selector pipelines and the full-feature references",
                    }
                )
        self.files["22_runtime_and_memory.csv"] = _frame("22_runtime_and_memory.csv", rows)

    # ------------------------------------------------------------- writing
    def manifest(self, stage: str) -> Path:
        directory = self.out / stage
        path = self.out / "manifests" / f"{stage}.json"
        entries = {}
        if directory.exists():
            for item in sorted(directory.rglob("*")):
                if item.is_file() and item.suffix in {".json", ".csv", ".npy", ".npz", ".parquet"}:
                    entries[item.relative_to(self.out).as_posix()] = sha256_file(item)
        write_json(path, {"stage": stage, "created_at_utc": utc_now(), "files": entries})
        return path

    def write(self) -> None:
        self.cohort()
        self.candidate_set()
        self.matrix()
        self.leader_deltas()
        self.leader_subsets()
        self.stability()
        self.subpopulations()
        self.robustness()
        self.sweep()
        self.voting()
        self.llm_usage()
        self.runtime()
        manifests = {stage: self.manifest(stage) for stage in sorted(set(SOURCE_STAGE.values()))}
        root = resolve(self.cfg, ".")
        run_rows = []
        for name, stage in SOURCE_STAGE.items():
            path = manifests[stage]
            run_rows.append(
                {
                    "file": name,
                    "run_id": self.run_id,
                    "source_path": path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix(),
                    "source_sha256": sha256_file(path),
                    "note": f"written by scripts/run_study.py --stage results from the {stage} stage outputs",
                }
            )
        self.files["00_run_ids.csv"] = _frame("00_run_ids.csv", run_rows)
        for name in HEADERS:
            write_csv(self.folder / name, _tidy(self.files[name]))
        log(f"{self.cfg['dataset']}: wrote {len(HEADERS)} result files to {self.folder}")


# --------------------------------------------------------------------- helpers
def _pairs(lists: Sequence[Sequence[str]]):
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            yield (i + 1, lists[i]), (j + 1, lists[j])


def _argmax(family: Sequence[str], values: dict[str, float | None]) -> str | None:
    available = [label for label in family if values.get(label) is not None]
    if not available:
        return None
    return max(available, key=lambda label: (values[label], -SELECTORS.index(label)))


def _cohort_note(split: str) -> str:
    if split == "total":
        return "DEV and HO together"
    if split == "release":
        return "rows of the public release"
    if split.endswith("_train") or split.endswith("_valid"):
        return "expanding-window fold partition of DEV"
    return ""


def _subset_note(record: dict[str, Any]) -> str:
    if not record:
        return "not run"
    mode = record.get("mode")
    if mode == "pca":
        return f"{record.get('pca_components')} principal components of the preprocessed candidates"
    if mode == "expanded":
        return "budget counts one-hot columns of the semantic pool"
    if mode == "full":
        return f"full-feature reference over {len(record.get('features', []))} features"
    count = len(record.get("features", []))
    budget_k = record.get("K")
    return "" if budget_k is None or count == budget_k else f"below budget: {count} features"


def _lendingclub_source_columns(inventory: pd.DataFrame) -> int:
    from credit_risk_fs.feature_metadata.definitions import _FORMULA_WORDS

    derived = {
        row["feature"]
        for _, row in inventory.iterrows()
        if row["source_column_or_formula"] and row["source_column_or_formula"] != row["feature"]
    }
    columns: set[str] = set()
    for _, row in inventory.iterrows():
        formula = row["source_column_or_formula"] or row["feature"]
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula):
            if token.lower() in _FORMULA_WORDS or token in derived:
                continue
            columns.add(token)
    return len(columns)


# --------------------------------------------------------------------- publication
def publish_rankings(cfg: dict[str, Any]) -> None:
    """Copy the accepted rankings to results/llm_rankings/<dataset>/ and rebuild index.csv."""

    source_dir = output_dir(cfg) / "llm" / "rankings"
    target_dir = results_dir(cfg) / "llm_rankings" / str(cfg["dataset"])
    target_dir.mkdir(parents=True, exist_ok=True)
    keep = {
        "call_id", "dataset", "arm", "partition", "budget", "repeat", "ranking", "ranking_identifiers",
        "n_candidate_records", "candidate_names_sha256", "prompt_version", "evidence_mode", "prompt_sha256",
        "request_sha256", "request_model", "response_model", "response_id", "temperature", "attempts",
        "rejected_attempts", "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_summary",
        "selection_principles", "raw_response", "accepted_at_utc",
    }
    published: set[str] = set()
    for path in sorted(source_dir.glob("*.json")):
        payload = {key: value for key, value in read_json(path).items() if key in keep}
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise StudyError(f"{path} matches a secret pattern; it was not published")
        name = f"{cfg['dataset']}__{payload['call_id']}.json"
        (target_dir / name).write_text(text + "\n", encoding="utf-8")
        published.add(name)
    for stale in target_dir.glob("*.json"):
        if stale.name not in published:
            log(f"removing superseded ranking file {stale}")
            stale.unlink()
    _rebuild_index(cfg)


def _rebuild_index(cfg: dict[str, Any]) -> None:
    base = results_dir(cfg) / "llm_rankings"
    index_path = base / "index.csv"
    rows = []
    if index_path.exists():
        # Rows of files not yet replaced by a study run are kept while their files exist.
        previous = pd.read_csv(index_path, dtype=str, keep_default_na=False)
        for _, row in previous.iterrows():
            path = base / str(row.get("file", ""))
            if path.is_file() and "call_id" not in read_json(path):
                rows.append(row.to_dict())
    conditions = {
        llm.NAMED: "named",
        llm.ARM_A: "obfuscated names (arm A)",
        llm.ARM_B: "blanked descriptions (arm B)",
        llm.LINEAGE: "descriptions rebuilt from lineage",
    }
    for directory in sorted(path for path in base.iterdir() if path.is_dir()):
        for path in sorted(directory.glob("*.json")):
            payload = read_json(path)
            if "call_id" not in payload:
                continue
            condition = conditions.get(payload["arm"], payload["arm"])
            if payload.get("repeat"):
                condition = f"{condition}, repeat {payload['repeat']}"
            rows.append(
                {
                    "dataset": payload["dataset"],
                    "partition": payload["partition"],
                    "budget": payload["budget"],
                    "condition": condition,
                    "file": path.relative_to(base).as_posix(),
                    "sha256": sha256_file(path),
                    "model_snapshot": payload.get("response_model") or payload.get("request_model"),
                    "temperature": payload.get("temperature"),
                    "validation_passed": "TRUE",
                    "attempts": payload.get("attempts"),
                    "n_candidate_records": payload.get("n_candidate_records"),
                    "top_k_sha256": sha256_text("\n".join(payload["ranking"][:40])),
                    "source": f"outputs/study/{payload['dataset']}/llm/rankings/{payload['call_id']}.json",
                    "note": f"prompt {payload.get('prompt_version')}, evidence mode {payload.get('evidence_mode')}, prompt sha256 {payload.get('prompt_sha256')}",
                }
            )
    columns = "dataset,partition,budget,condition,file,sha256,model_snapshot,temperature,validation_passed,attempts,n_candidate_records,top_k_sha256,source,note".split(",")
    write_csv(index_path, pd.DataFrame(rows, columns=columns))


def rebuild_headline_subsets(cfg: dict[str, Any]) -> None:
    folders = cfg.get("all_results_folders", {})
    labels = cfg.get("all_dataset_labels", {})
    rows = []
    for dataset, folder in folders.items():
        path = results_dir(cfg) / folder / "09_leader_subsets.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        for _, row in frame.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "case": f"{labels.get(dataset, dataset)}, {BACKBONE_LABEL.get(row['backbone'], row['backbone'])}",
                    "backbone": row["backbone"],
                    "K": cfg["budgets"][row["backbone"]],
                    "family": FAMILY.get(row["selector"], ""),
                    "selector": row["selector"],
                    "rank": row["rank"],
                    "feature": row["feature"],
                    "source_family": row["source_family"],
                    "is_categorical": row["is_categorical"],
                    "is_protected": row["is_protected"],
                    "note": f"full-DEV subset of the HO-rule leader; {folder}/09_leader_subsets.csv",
                }
            )
    columns = "dataset,case,backbone,K,family,selector,rank,feature,source_family,is_categorical,is_protected,note".split(",")
    write_csv(results_dir(cfg) / "headline_subsets.csv", pd.DataFrame(rows, columns=columns))


def run_results(cfg: dict[str, Any]) -> None:
    ResultsBuilder(cfg).write()
    publish_rankings(cfg)
    rebuild_headline_subsets(cfg)


__all__ = ["HEADERS", "ResultsBuilder", "publish_rankings", "rebuild_headline_subsets", "run_results"]
