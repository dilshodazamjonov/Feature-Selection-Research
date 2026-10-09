#!/usr/bin/env python
"""Render and verify the manuscript tables from ``results/**`` only.

    uv run python scripts/verify_manuscript_tables.py --render   # Markdown tables from the CSVs
    uv run python scripts/verify_manuscript_tables.py --check    # compare with scripts/manuscript_oracle.py
    uv run python scripts/verify_manuscript_tables.py --schema   # structural checks of the three folders

Pure pandas; reads nothing outside ``results/`` and the oracle module; no network, no model code.
``--check`` exits non-zero on any mismatch between a CSV value and the manuscript value (rounded to the
manuscript's precision); CSV cells that are empty are reported as "no CSV source" and do not fail.
"""

from __future__ import annotations

import argparse
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import manuscript_oracle as O  # noqa: E402

RESULTS = ROOT / "results"
FOLDERS = {"homecredit": "homecredit_metrics_output", "lendingclub_v2": "lendingclub_metrics_output", "stability_2024": "stability_metrics_output"}
FILES = ["00_run_ids.csv", "01_cohort.csv", "02_candidate_set.csv", "03_ho_auc.csv", "04_dev_fold_auc.csv", "05_leader_delta_auc.csv",
         "06_leader_feature_psi.csv", "07_selection_stability.csv", "08_secondary_metrics.csv", "09_leader_subsets.csv", "10_boruta_attainment.csv",
         "11_subpopulation_ho_auc.csv", "12_native_categorical.csv", "13_lineage_only.csv", "14_l1_selector.csv", "15_diversity_controls.csv",
         "16_response_spread.csv", "17_budget_sweep.csv", "18_response_agreement.csv", "19_obfuscation.csv", "19b_obfuscation_ranks.csv",
         "19c_obfuscation_fold_jaccard.csv", "20_rank_voting.csv", "21_llm_usage.csv", "22_runtime_and_memory.csv"]


# ----------------------------------------------------------------------------- loading
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


def load(dataset: str, name: str) -> pd.DataFrame:
    """Read one results CSV as strings; missing expected columns become empty so partial files degrade to blanks."""
    path = RESULTS / FOLDERS[dataset] / name
    if not path.exists():
        return pd.DataFrame(columns=HEADERS[name].split(","))
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = HEADERS[name].split(",")
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    return df


def blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


# ----------------------------------------------------------------------------- comparison
class Check:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, table: str, where: str, column: str, csv_value, manuscript_value, status: str, note: str = "") -> None:
        self.rows.append({"table": table, "where": where, "column": column, "csv": "" if csv_value is None else str(csv_value),
                          "manuscript": "" if manuscript_value is None else str(manuscript_value), "status": status, "note": note})

    def num(self, table: str, where: str, column: str, csv_value, manuscript: str | None, *, dataset_scale: str | None = None) -> None:
        """Compare a CSV number with a manuscript string at the manuscript's precision."""
        if manuscript is None:
            return
        m = str(manuscript).replace(",", "").replace("+", "").strip()
        if blank(csv_value):
            self.add(table, where, column, "", manuscript, "blank", "no CSV source")
            return
        try:
            x = Decimal(str(csv_value).replace(",", "").strip())
            target = Decimal(m)
        except InvalidOperation:
            status = "match" if str(csv_value).strip() == str(manuscript).strip() else "mismatch"
            self.add(table, where, column, csv_value, manuscript, status)
            return
        places = -target.as_tuple().exponent if target.as_tuple().exponent < 0 else 0
        q = Decimal(1).scaleb(-places)
        up = x.quantize(q, rounding=ROUND_HALF_UP)
        if up == target:
            self.add(table, where, column, csv_value, manuscript, "match")
            return
        even = x.quantize(q, rounding=ROUND_HALF_EVEN)
        if even == target:
            self.add(table, where, column, csv_value, manuscript, "match", "half-even rounding")
            return
        self.add(table, where, column, csv_value, manuscript, "mismatch")

    def text(self, table: str, where: str, column: str, csv_value, manuscript) -> None:
        if blank(csv_value):
            self.add(table, where, column, "", manuscript, "blank", "no CSV source")
        else:
            self.add(table, where, column, csv_value, manuscript, "match" if str(csv_value).strip() == str(manuscript).strip() else "mismatch")

    def apply_known(self) -> None:
        """Downgrade mismatches that the freeze report documents as unreproducible manuscript figures."""
        known = {(t, w, c): reason for t, w, c, reason in O.KNOWN_DISCREPANCIES}
        for r in self.rows:
            key = (r["table"], r["where"], r["column"])
            if r["status"] == "mismatch" and key in known:
                r["status"] = "flagged"
                r["note"] = "known discrepancy: " + known[key]

    def summary(self) -> tuple[int, int, int, int, int]:
        n = len(self.rows)
        matched = sum(r["status"] == "match" for r in self.rows)
        blanks = sum(r["status"] == "blank" for r in self.rows)
        mism = sum(r["status"] == "mismatch" for r in self.rows)
        flagged = sum(r["status"] == "flagged" for r in self.rows)
        return n, matched, blanks, mism, flagged


def first(df: pd.DataFrame, **cond) -> dict | None:
    if df.empty:
        return None
    mask = pd.Series(True, index=df.index)
    for k, v in cond.items():
        if k not in df.columns:
            return None
        mask &= df[k].astype(str).str.strip() == str(v)
    sub = df[mask]
    return None if sub.empty else sub.iloc[0].to_dict()


def to_float(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- the checks
def check_table2(ck: Check) -> None:
    for ds, spec in O.TABLE2.items():
        c = load(ds, "01_cohort.csv")
        for split, key in (("DEV", "dev"), ("HO", "ho")):
            r = first(c, split=split)
            w = f"{ds} {split}"
            ck.num("Table 2", w, "n_rows", r and r.get("n_rows"), spec[f"{key}_n"])
            rate = to_float(r and r.get("event_rate"))
            ck.num("Table 2", w, "event_rate_pct", None if rate is None else rate * 100, spec[f"{key}_event_rate_pct"])
            ck.text("Table 2", w, "window_start", r and r.get("window_start"), spec[f"{key}_window"][0])
            ck.text("Table 2", w, "window_end", r and r.get("window_end"), spec[f"{key}_window"][1])
        s = load(ds, "02_candidate_set.csv")
        r = first(s, screen="missing rate <= 0.90 on fold-1 DEV training rows")
        ck.num("Table 2", ds, "universe", r and r.get("universe"), spec["universe"])
        ck.num("Table 2", ds, "retained", r and r.get("retained"), spec["candidate"])
    # Section 5.1 / 5.2 sentences
    hc = load("homecredit", "01_cohort.csv")
    r = first(hc, split="total")
    ck.num("Section 5.1", "homecredit covered", "n_rows", r and r.get("n_rows"), O.SECTION5["homecredit"]["applications_covered"])
    r = first(hc, split="release")
    ck.num("Section 5.1", "homecredit release", "n_rows", r and r.get("n_rows"), O.SECTION5["homecredit"]["applications_release"])
    lc = load("lendingclub_v2", "01_cohort.csv")
    spec = O.SECTION5["lendingclub_v2"]
    for split, nkey, rkey in (("DEV_term36", "dev_36m", "dev_rate_36m_pct"), ("DEV_term60", None, "dev_rate_60m_pct"),
                              ("HO_term36", "ho_36m", "ho_rate_36m_pct"), ("HO_term60", None, "ho_rate_60m_pct")):
        r = first(lc, split=split)
        if nkey:
            ck.num("Section 5.1", f"lendingclub {split}", "n_rows", r and r.get("n_rows"), spec[nkey])
        rate = to_float(r and r.get("event_rate"))
        ck.num("Section 5.1", f"lendingclub {split}", "event_rate_pct", None if rate is None else rate * 100, spec[rkey])
    s = load("lendingclub_v2", "02_candidate_set.csv")
    for screen, key in (("missingness < 40%", "missing_lt40"), ("missingness 40-60%", "missing_40_60"), ("missingness 60-82%", "missing_60_82")):
        r = first(s, screen=screen)
        ck.num("Section 5.1", f"lendingclub {screen}", "retained", r and r.get("retained"), spec[key])
    r = first(s, screen="source columns")
    ck.num("Section 5.1", "lendingclub source columns", "retained", r and r.get("retained"), spec["source_columns"])
    for comp, val in spec["composition"].items():
        r = first(s, screen=f"composition: {comp}")
        ck.num("Section 5.1", f"lendingclub composition {comp}", "retained", r and r.get("retained"), val)
    st = load("stability_2024", "02_candidate_set.csv")
    r = first(st, screen="prediction-time availability review of raw predictors")
    ck.num("Section 5.1", "stability reviewed", "universe", r and r.get("universe"), O.SECTION5["stability_2024"]["reviewed"])
    ck.num("Section 5.1", "stability approved", "retained", r and r.get("retained"), O.SECTION5["stability_2024"]["approved"])
    r = first(load("stability_2024", "01_cohort.csv"), split="HO")
    ck.num("Section 5.2", "stability HO events", "n_events", r and r.get("n_events"), O.SECTION5["stability_2024"]["ho_events"])


def check_table5(ck: Check) -> None:
    for ds, per_backbone in O.TABLE5.items():
        ho = load(ds, "03_ho_auc.csv")
        for bb, sel in per_backbone.items():
            for selector, value in sel.items():
                r = first(ho, backbone=bb, selector=selector)
                ck.num("Table 5", f"{ds} {bb} {selector}", "ho_auc", r and r.get("ho_auc"), value)
        d = load(ds, "05_leader_delta_auc.csv")
        for bb in O.BACKBONES:
            r = first(d, rule="HO", backbone=bb)
            ck.num("Table 5", f"{ds} {bb} leader delta", "delta_auc", r and r.get("delta_auc"), O.TABLE5_DELTA[(ds, bb)])


def check_table3(ck: Check) -> None:
    for row in O.TABLE3:
        ds, bb = row["dataset"], row["backbone"]
        d = first(load(ds, "05_leader_delta_auc.csv"), rule="HO", backbone=bb)
        w = f"{ds} {bb}"
        ck.text("Table 3", w, "llm_leader", d and d.get("llm_leader"), row["llm_leader"])
        ck.text("Table 3", w, "classical_leader", d and d.get("classical_leader"), row["classical_leader"])
        ck.num("Table 3", w, "llm_ho_auc", d and d.get("llm_ho_auc"), row["llm_auc"])
        ck.num("Table 3", w, "classical_ho_auc", d and d.get("classical_ho_auc"), row["classical_auc"])
        ck.num("Table 3", w, "delta_auc", d and d.get("delta_auc"), row["delta"])
        ck.num("Table 3", w, "ci_low", d and d.get("ci_low"), row["ci_low"])
        ck.num("Table 3", w, "ci_high", d and d.get("ci_high"), row["ci_high"])
        f = first(load(ds, "03_ho_auc.csv"), backbone=bb, selector=O.FULL)
        ck.num("Table 3", w, "full", f and f.get("ho_auc"), row["full"])
        psi = load(ds, "06_leader_feature_psi.csv")
        p = first(psi, backbone=bb, selector=row["llm_leader"])
        ck.num("Table 3", w, "psi_llm", p and p.get("mean_psi"), row["psi_llm"])
        p = first(psi, backbone=bb, selector=row["classical_leader"])
        ck.num("Table 3", w, "psi_classical", p and p.get("mean_psi"), row["psi_classical"])
        if d:
            ck.num("Table 3", w, "bootstrap_resamples", d.get("bootstrap_resamples"), O.BOOTSTRAP["resamples"])
            ck.num("Table 3", w, "bootstrap_seed", d.get("bootstrap_seed"), O.BOOTSTRAP["seed"])
    # Figure 3 full-precision coordinates (where the CSV carries that precision)
    for (ds, bb), co in O.FIGURE3.items():
        d = first(load(ds, "05_leader_delta_auc.csv"), rule="HO", backbone=bb)
        w = f"{ds} {bb}"
        ck.num("Figure 3", w, "classical_ho_auc", d and d.get("classical_ho_auc"), co["classical"])
        ck.num("Figure 3", w, "llm_ho_auc", d and d.get("llm_ho_auc"), co["llm"])
        f = first(load(ds, "03_ho_auc.csv"), backbone=bb, selector=O.FULL)
        ck.num("Figure 3", w, "full", f and f.get("ho_auc"), co["full"])


def check_table4(ck: Check) -> None:
    for row in O.TABLE4:
        ds, bb = row["dataset"], row["backbone"]
        d = first(load(ds, "05_leader_delta_auc.csv"), rule="DEV", backbone=bb)
        w = f"{ds} {bb}"
        ck.text("Table 4", w, "llm_leader", d and d.get("llm_leader"), row["llm_leader"])
        ck.text("Table 4", w, "classical_leader", d and d.get("classical_leader"), row["classical_leader"])
        ck.num("Table 4", w, "llm_dev_auc", d and d.get("llm_dev_auc"), row["llm_dev"])
        ck.num("Table 4", w, "classical_dev_auc", d and d.get("classical_dev_auc"), row["classical_dev"])
        ck.num("Table 4", w, "llm_ho_auc", d and d.get("llm_ho_auc"), row["llm_ho"])
        ck.num("Table 4", w, "classical_ho_auc", d and d.get("classical_ho_auc"), row["classical_ho"])
        ck.num("Table 4", w, "delta_auc", d and d.get("delta_auc"), row["delta"])
        ck.num("Table 4", w, "ci_low", d and d.get("ci_low"), row["ci_low"])
        ck.num("Table 4", w, "ci_high", d and d.get("ci_high"), row["ci_high"])
        # DEV means must also be the 04 file's means
        dev = load(ds, "04_dev_fold_auc.csv")
        for leader_key, col in (("llm_leader", "llm_dev"), ("classical_leader", "classical_dev")):
            r = first(dev, backbone=bb, selector=row[leader_key])
            ck.num("Table 4", w, f"04 dev_mean_auc {row[leader_key]}", r and r.get("dev_mean_auc"), row[col])
            if r and all(not blank(r.get(f"fold{i}_auc")) for i in range(1, 6)):
                mean = sum(float(r[f"fold{i}_auc"]) for i in range(1, 6)) / 5
                ck.num("Table 4", w, f"04 recomputed fold mean {row[leader_key]}", mean, row[col])
    d = first(load("homecredit", "05_leader_delta_auc.csv"), rule="mixed", backbone="lr")
    ck.text("Section 6.1 mixed rule", "homecredit lr", "llm_leader", d and d.get("llm_leader"), O.MIXED_RULE_HC_LR["llm_leader"])
    ck.text("Section 6.1 mixed rule", "homecredit lr", "classical_leader", d and d.get("classical_leader"), O.MIXED_RULE_HC_LR["classical_leader"])
    ck.num("Section 6.1 mixed rule", "homecredit lr", "delta_auc", d and d.get("delta_auc"), O.MIXED_RULE_HC_LR["delta"])
    if d:
        ck.add("Section 6.1 mixed rule", "homecredit lr", "interval", d.get("ci_low", "") + "|" + d.get("ci_high", ""), "not computed",
               "match" if blank(d.get("ci_low")) and blank(d.get("ci_high")) else "mismatch", "manuscript: interval not computed")


def check_table7(ck: Check) -> None:
    ob = load("homecredit", "19_obfuscation.csv")
    for row in O.TABLE7:
        r = first(ob, arm=row["arm"], backbone=row["backbone"])
        w = f"homecredit arm {row['arm']} {row['backbone']}"
        ck.num("Table 7", w, "ho_auc_arm", r and r.get("ho_auc_arm"), row["ho_auc"])
        ck.num("Table 7", w, "ho_auc_named", r and r.get("ho_auc_named"), row["named"])
        ck.num("Table 7", w, "delta_auc", r and r.get("delta_auc"), row["delta"])
        folds = [to_float(r.get(f"fold{i}_auc")) for i in range(1, 6)] if r else []
        folds = [f for f in folds if f is not None]
        ck.num("Table 7", w, "fold_min", min(folds) if folds else None, row["fold_min"])
        ck.num("Table 7", w, "fold_max", max(folds) if folds else None, row["fold_max"])
        ck.num("Table 7", w, "overlap_full_dev", r and r.get("overlap_full_dev"), row["overlap_full_dev"])
        ck.num("Table 7", w, "overlap_folds_min", r and r.get("overlap_folds_min"), row["overlap_folds_min"])
        ck.num("Table 7", w, "overlap_folds_max", r and r.get("overlap_folds_max"), row["overlap_folds_max"])
        ck.num("Table 7", w, "overlap_folds_mean", r and r.get("overlap_folds_mean"), row["overlap_folds_mean"])
    ranks = load("homecredit", "19b_obfuscation_ranks.csv")
    for arm, feats in O.TABLE7_EXT_RANKS.items():
        for feat, rank in feats.items():
            r = first(ranks, arm=arm, call="full_dev", feature=feat)
            ck.num("Table 7 note", f"homecredit {arm} full_dev", f"rank {feat}", r and r.get("rank"), str(rank))
    jac = load("homecredit", "19c_obfuscation_fold_jaccard.csv")
    for arm, (mn, mx, mean) in O.TABLE7_FOLD_JACCARD_K20.items():
        sub = jac[(jac.get("arm", pd.Series(dtype=str)) == arm) & (jac.get("K", pd.Series(dtype=str)) == "20")] if not jac.empty else jac
        vals = [to_float(v) for v in sub["jaccard"]] if not sub.empty else []
        vals = [v for v in vals if v is not None]
        ck.num("Table 7 note", f"homecredit {arm} K=20 fold Jaccard", "min", min(vals) if vals else None, mn)
        ck.num("Table 7 note", f"homecredit {arm} K=20 fold Jaccard", "max", max(vals) if vals else None, mx)
        ck.num("Table 7 note", f"homecredit {arm} K=20 fold Jaccard", "mean", sum(vals) / len(vals) if vals else None, mean)


def check_table8(ck: Check) -> None:
    for row in O.TABLE8:
        r = first(load(row["dataset"], "13_lineage_only.csv"), backbone=row["backbone"], selector=row["selector"])
        w = f"{row['dataset']} {row['backbone']} {row['selector']}"
        ck.num("Table 8", w, "ho_auc_mechanical", r and r.get("ho_auc_mechanical"), row["mechanical"])
        ck.num("Table 8", w, "ho_auc_cached", r and r.get("ho_auc_cached"), row["cached"])
        ck.num("Table 8", w, "delta_auc", r and r.get("delta_auc"), row["delta"])
        ck.num("Table 8", w, "ci_low", r and r.get("ci_low"), row["ci_low"])
        ck.num("Table 8", w, "ci_high", r and r.get("ci_high"), row["ci_high"])
        ov = r and r.get("overlap_with_cached")
        ov = None if blank(ov) else (ov if "/" in str(ov) else f"{int(float(ov))}/{row['K']}")
        ck.text("Table 8", w, "overlap_with_cached", ov, row["overlap"])


def check_table9(ck: Check) -> None:
    for row in O.TABLE9:
        r = first(load(row["dataset"], "07_selection_stability.csv"), backbone=row["backbone"], selector=row["selector"])
        w = f"{row['dataset']} {row['backbone']} {row['selector']}"
        ck.num("Table 9", w, "d", r and r.get("d"), str(O.D_OF[row["dataset"]]))
        ck.num("Table 9", w, "nogueira", r and r.get("nogueira"), row["nogueira"])
        ck.num("Table 9", w, "jaccard_mean", r and r.get("jaccard_mean"), row["jaccard"])
        ck.num("Table 9", w, "jaccard_min", r and r.get("jaccard_min"), row["min"])
        ck.num("Table 9", w, "jaccard_max", r and r.get("jaccard_max"), row["max"])


def check_table10(ck: Check) -> None:
    for row in O.TABLE10:
        ds, bb = row["dataset"], row["backbone"]
        rs = load(ds, "16_response_spread.csv")
        w = f"{ds} {bb}"
        draws = rs[(rs.get("backbone", pd.Series(dtype=str)) == bb) & (rs.get("is_headline_response", pd.Series(dtype=str)).str.upper() != "TRUE")] if not rs.empty else rs
        vals = [to_float(v) for v in draws["ho_auc"]] if not draws.empty else []
        vals = [v for v in vals if v is not None]
        ck.num("Table 10", w, "draws", len(vals) if vals else None, str(row["draws"]))
        ck.num("Table 10", w, "min", min(vals) if vals else None, row["min"])
        ck.num("Table 10", w, "max", max(vals) if vals else None, row["max"])
        ck.num("Table 10", w, "spread", (max(vals) - min(vals)) if vals else None, row["spread"])
        if row["median"]:
            s = pd.Series(vals)
            ck.num("Table 10", w, "median", float(s.median()) if vals else None, row["median"])
            ck.num("Table 10", w, "sd", float(s.std(ddof=1)) if len(vals) > 1 else None, row["sd"])
            cl = first(load(ds, "03_ho_auc.csv"), backbone=bb, selector=row["classical_leader"])
            base = to_float(cl and cl.get("ho_auc"))
            ck.num("Table 10", w, "margin_min", (min(vals) - base) if (vals and base is not None) else None, row["margin"][0])
            ck.num("Table 10", w, "margin_max", (max(vals) - base) if (vals and base is not None) else None, row["margin"][1])
            # cached value above seven of ten repeats
            head = first(rs, backbone=bb, is_headline_response="TRUE")
            hv = to_float(head and head.get("ho_auc"))
            ck.num("Section 6.6.1", w, "cached above n of ten", sum(v < hv for v in vals) if (vals and hv is not None) else None, str(O.STABILITY_TEN["cached_above_n_of_ten"]))
        head = first(rs, backbone=bb, is_headline_response="TRUE")
        ck.num("Table 10 note", w, "headline ho_auc", head and head.get("ho_auc"), O.TABLE10_HEADLINE[(ds, bb)])
    # LendingClub LR: six-response spread and margins
    rs = load("lendingclub_v2", "16_response_spread.csv")
    sub = rs[rs.get("backbone", pd.Series(dtype=str)) == "lr"] if not rs.empty else rs
    vals = [to_float(v) for v in sub["ho_auc"]] if not sub.empty else []
    vals = [v for v in vals if v is not None]
    ck.num("Table 10 note", "lendingclub_v2 lr", "six-response spread", (max(vals) - min(vals)) if len(vals) >= 6 else None, O.TABLE10_LC_LR["six_response_spread"])
    draws = [to_float(v) for v in sub[sub["is_headline_response"].str.upper() != "TRUE"]["ho_auc"]] if not sub.empty else []
    draws = [v for v in draws if v is not None]
    for key, leader in (("margin_over_iv_then_boruta", "IV then Boruta"), ("margin_over_rfe", "RFE CatBoost")):
        base = to_float((first(load("lendingclub_v2", "03_ho_auc.csv"), backbone="lr", selector=leader) or {}).get("ho_auc"))
        lo, hi = O.TABLE10_LC_LR[key]
        # The manuscript derives these margins from the printed four-decimal values (article CHANGES.md, decision 4), so round first.
        d4 = [round(v, 4) for v in draws]
        b4 = None if base is None else round(base, 4)
        ck.num("Table 10 note", "lendingclub_v2 lr", f"{key} min (from printed 4-dp values)", (min(d4) - b4) if (d4 and b4 is not None) else None, lo)
        ck.num("Table 10 note", "lendingclub_v2 lr", f"{key} max (from printed 4-dp values)", (max(d4) - b4) if (d4 and b4 is not None) else None, hi)


def check_table11(ck: Check) -> None:
    for ds, spec in O.TABLE11.items():
        sm = load(ds, "08_secondary_metrics.csv")
        if sm.empty:
            for metric in ("ks", "brier", "capture10"):
                ck.num("Table 11", f"{ds} {metric}", "value", None, spec[metric][0])
            continue
        sm = sm[sm["selector"] != O.FULL]  # Table 11 ranks selectors; the full-feature reference is not a selector
        for metric, col, best in (("ks", "ks", "max"), ("brier", "brier", "min"), ("capture10", "capture_at_10", "max")):
            s = sm[col].map(to_float)
            if s.notna().sum() == 0:
                ck.num("Table 11", f"{ds} {metric}", "value", None, spec[metric][0])
                continue
            idx = s.idxmax() if best == "max" else s.idxmin()
            r = sm.loc[idx]
            ck.num("Table 11", f"{ds} {metric}", "value", r[col], spec[metric][0])
            ck.text("Table 11", f"{ds} {metric}", "selector", r["selector"], spec[metric][1])
            ck.text("Table 11", f"{ds} {metric}", "backbone", r["backbone"], spec[metric][2])
            if metric == "capture10":
                ck.num("Table 11", f"{ds} lift10", "lift_at_10", r.get("lift_at_10"), spec["lift10"])


def check_tableB1(ck: Check) -> None:
    for row in O.TABLEB1:
        r = first(load(row["dataset"], "20_rank_voting.csv"), backbone=row["backbone"], pool_P=str(row["P"]))
        w = f"{row['dataset']} {row['backbone']} P={row['P']}"
        ck.num("Table B.1", w, "reference_auc", r and r.get("reference_auc"), row["ref"])
        ck.num("Table B.1", w, "voting_auc", r and r.get("voting_auc"), row["voting"])
        ck.num("Table B.1", w, "delta_auc", r and r.get("delta_auc"), row["delta"])
        ck.num("Table B.1", w, "ci_low", r and r.get("ci_low"), row["ci_low"])
        ck.num("Table B.1", w, "ci_high", r and r.get("ci_high"), row["ci_high"])
        ck.num("Table B.1", w, "delta_ks", r and r.get("delta_ks"), row["delta_ks"])
        hp = r and r.get("holm_p")
        if blank(hp):
            ck.add("Table B.1", w, "holm_p", "", row["holm_p"], "blank", "no CSV source")
        else:
            target = row["holm_p"]
            v = to_float(hp)
            if target.startswith("<"):
                ok = v is not None and v < float(target[1:])
            else:
                t = float(target)
                ok = v is not None and abs(v - t) <= 0.05 * t + 1e-12  # printed to two significant figures
            ck.add("Table B.1", w, "holm_p", hp, target, "match" if ok else "mismatch")


def _sweep_leaders(ds: str, bb: str) -> dict[int, tuple[float | None, float | None, float | None]]:
    sw = load(ds, "17_budget_sweep.csv")
    out: dict[int, tuple] = {}
    if sw.empty:
        return out
    sub = sw[sw["backbone"] == bb]
    for k in (10, 20, 30, 40, 50):
        rows = sub[sub["K"] == str(k)]
        if rows.empty:
            continue
        llm = [to_float(v) for f, v in zip(rows["family"], rows["dev_mean_auc"]) if f == "LLM-assisted"]
        cla = [to_float(v) for f, v in zip(rows["family"], rows["dev_mean_auc"]) if f == "classical"]
        llm = [v for v in llm if v is not None]
        cla = [v for v in cla if v is not None]
        pooled = [to_float(v) for f, v in zip(rows["family"], rows["dev_mean_auc"]) if f == "pooled"]
        pooled = [v for v in pooled if v is not None]
        out[k] = (max(llm) if llm else None, max(cla) if cla else None, pooled[0] if pooled else None)
    return out


def check_tableC1(ck: Check) -> None:
    for (ds, bb), deltas in O.TABLEC1.items():
        leaders = _sweep_leaders(ds, bb)
        fig = O.FIGUREC1B[(ds, bb)]
        for i, k in enumerate((10, 20, 30, 40, 50)):
            llm, cla, _ = leaders.get(k, (None, None, None))
            delta = (llm - cla) if (llm is not None and cla is not None) else None
            ck.num("Table C.1", f"{ds} {bb} K={k}", "leader delta", delta, deltas[i])
            if ds != "stability_2024":
                ck.num("Figure C.1b", f"{ds} {bb} K={k}", "leader delta", delta, fig[i])


def check_tableF1(ck: Check) -> None:
    sp = load("lendingclub_v2", "11_subpopulation_ho_auc.csv")
    for row in O.TABLEF1:
        for bb in O.BACKBONES:
            r = first(sp, subpopulation=row["subpopulation"], backbone=bb)
            w = f"lendingclub_v2 {row['subpopulation']} {bb}"
            ck.num("Table F.1", w, "n", r and r.get("n"), row["n"])
            ck.num("Table F.1", w, "delta_auc", r and r.get("delta_auc"), row[bb])
            if f"{bb}_ci" in row:
                ck.num("Table F.1 note", w, "ci_low", r and r.get("ci_low"), row[f"{bb}_ci"][0])
                ck.num("Table F.1 note", w, "ci_high", r and r.get("ci_high"), row[f"{bb}_ci"][1])
    months = sp[sp["subpopulation"].str.startswith("month_")] if not sp.empty else sp
    for bb in O.BACKBONES:
        sub = months[months["backbone"] == bb] if not months.empty else months
        pos = [to_float(v) for v in sub["delta_auc"]] if not sub.empty else []
        pos = [v for v in pos if v is not None]
        ck.num("Appendix F LC months", f"lendingclub_v2 {bb}", "months with positive delta", sum(v > 0 for v in pos) if pos else None, str(O.TABLEF1_MONTHS_POSITIVE))
        ck.num("Appendix F LC months", f"lendingclub_v2 {bb}", "months reported", len(pos) if pos else None, str(O.TABLEF1_MONTHS_POSITIVE))
        excl = [to_float(v) for v in sub["ci_low"]] if not sub.empty else []
        excl = [v for v in excl if v is not None]
        ck.num("Appendix F LC months", f"lendingclub_v2 {bb}", "intervals excluding zero", sum(v > 0 for v in excl) if excl else None, str(O.TABLEF1_MONTHS_POSITIVE))


def check_tableF2(ck: Check) -> None:
    for row in O.TABLEF2:
        r = first(load(row["dataset"], "15_diversity_controls.csv"), backbone=row["backbone"], control=row["rule"])
        w = f"{row['dataset']} {row['backbone']} {row['rule']}"
        ck.text("Table F.2", w, "base_selector", r and r.get("base_selector"), row["base"])
        ck.num("Table F.2", w, "control_ho_auc", r and r.get("control_ho_auc"), row["control"])
        ck.num("Table F.2", w, "uncapped_ho_auc", r and r.get("uncapped_ho_auc"), row["uncapped"])
        ck.num("Table F.2", w, "llm_assisted_ho_auc", r and r.get("llm_assisted_ho_auc"), row["llm"])
        ck.num("Table F.2", w, "difference", r and r.get("difference"), row["difference"])
        if r and row["rule"] == "family_cap":
            caps = O.FAMILY_CAPS[row["dataset"]]
            ck.num("Table 1 note", w, "cap_per_family", r.get("cap_per_family"), str(caps[row["backbone"]]))
            ck.num("Table 1 note", w, "n_families", r.get("n_families"), str(caps["F"]))


def check_tableF3(ck: Check) -> None:
    for row in O.TABLEF3:
        ls = load(row["dataset"], "09_leader_subsets.csv")
        sub = ls[(ls["backbone"] == row["backbone"]) & (ls["selector"] == row["selector"])] if not ls.empty else ls
        w = f"{row['dataset']} {row['backbone']} {row['selector']}"
        if sub.empty:
            ck.num("Table F.3", w, "n_categorical", None, str(row["n_categorical"]))
            ck.num("Table F.3", w, "columns", None, str(row["columns"]))
            continue
        ck.num("Table F.3", w, "K (rows)", len(sub), str(row["K"]))
        cat = sub["is_categorical"].str.upper()
        ck.num("Table F.3", w, "n_categorical", int((cat == "TRUE").sum()) if (cat != "").all() else None, str(row["n_categorical"]))
        cols = sub["n_model_columns"].unique()
        ck.num("Table F.3", w, "columns", cols[0] if len(cols) == 1 and not blank(cols[0]) else None, str(row["columns"]))
    # Appendix D overlaps and Section 7.5 protected ranks
    for ds, per in O.SUBSET_OVERLAPS.items():
        ls = load(ds, "09_leader_subsets.csv")
        for bb, target in per.items():
            sub = ls[ls["backbone"] == bb] if not ls.empty else ls
            sels = list(sub["selector"].unique()) if not sub.empty else []
            if len(sels) == 2:
                a = set(sub[sub["selector"] == sels[0]]["feature"])
                b = set(sub[sub["selector"] == sels[1]]["feature"])
                ck.text("Appendix D", f"{ds} {bb}", "overlap", f"{len(a & b)}/{O.K_OF[bb]}", target)
            else:
                ck.text("Appendix D", f"{ds} {bb}", "overlap", None, target)
    for ds, bb, selector, feature, rank in O.PROTECTED_RANKS:
        r = first(load(ds, "09_leader_subsets.csv"), backbone=bb, selector=selector, feature=feature)
        ck.num("Section 7.5", f"{ds} {bb} {selector}", f"rank {feature}", r and r.get("rank"), str(rank))
        if r:
            ck.text("Section 7.5", f"{ds} {bb} {selector}", f"is_protected {feature}", str(r.get("is_protected")).upper(), "TRUE")
    # EXT_SOURCE ranks 1-3 in the HC pure LLM list
    ls = load("homecredit", "09_leader_subsets.csv")
    for feat, rank in (("EXT_SOURCE_3", 1), ("EXT_SOURCE_2", 2), ("EXT_SOURCE_1", 3)):
        r = first(ls, backbone="catboost", selector="Pure LLM", feature=feat)
        ck.num("Appendix D", "homecredit catboost Pure LLM", f"rank {feat}", r and r.get("rank"), str(rank))


def check_tableF4(ck: Check) -> None:
    for row in O.TABLEF4:
        r = first(load(row["dataset"], "18_response_agreement.csv"), backbone=row["backbone"])
        w = f"{row['dataset']} {row['backbone']}"
        ck.num("Table F.4", w, "draws", r and r.get("draws"), str(row["draws"]))
        ck.num("Table F.4", w, "d", r and r.get("d"), str(O.D_OF[row["dataset"]]))
        ck.num("Table F.4", w, "nogueira", r and r.get("nogueira"), row["nogueira"])
        ck.num("Table F.4", w, "jaccard_mean", r and r.get("jaccard_mean"), row["jaccard"])
        ck.num("Table F.4", w, "names_in_all_draws", r and r.get("names_in_all_draws"), str(row["in_all"]))
        ck.num("Table F.4", w, "distinct_names", r and r.get("distinct_names"), str(row["distinct"]))


def check_appendix_f(ck: Check) -> None:
    # native categorical handling
    for ds, spec in O.NATIVE_CATEGORICAL.items():
        if ds == "max_move":
            continue
        nc = load(ds, "12_native_categorical.csv")
        for selector, value in spec.items():
            if selector in ("margin", "pure_llm_unchanged_by_construction"):
                continue
            r = first(nc, backbone="catboost", selector=selector)
            ck.num("Appendix F native categorical", f"{ds} {selector}", "ho_auc_native", r and r.get("ho_auc_native"), value)
    # L1 selector
    all_l1 = []
    for ds in O.DATASETS:
        l1 = load(ds, "14_l1_selector.csv")
        for _, r in l1.iterrows():
            v = to_float(r.get("ho_auc"))
            if v is not None:
                all_l1.append(v)
            if ds == "lendingclub_v2" and r.get("backbone") == "catboost":
                ck.num("Appendix F L1", "lendingclub_v2 catboost", "shortfall", r.get("shortfall"), O.L1_SELECTOR["shortfall_lc_catboost"])
            if ds == "stability_2024" and r.get("backbone") == "catboost":
                ck.num("Appendix F L1", "stability_2024 catboost", "shortfall", r.get("shortfall"), O.L1_SELECTOR["shortfall_st_catboost"])
    ck.num("Appendix F L1", "all datasets", "min ho_auc", min(all_l1) if all_l1 else None, O.L1_SELECTOR["ho_auc_range"][0])
    ck.num("Appendix F L1", "all datasets", "max ho_auc", max(all_l1) if all_l1 else None, O.L1_SELECTOR["ho_auc_range"][1])
    # Boruta attainment
    total, at_budget = 0, 0
    found_below = set()
    for ds in O.DATASETS:
        b = load(ds, "10_boruta_attainment.csv")
        for _, r in b.iterrows():
            if blank(r.get("n_selected")):
                continue
            total += 1
            below = str(r.get("below_budget")).upper() == "TRUE"
            at_budget += 0 if below else 1
            if below:
                found_below.add((ds, r["backbone"], r["selector"], r["partition"], int(float(r["n_selected"])), int(float(r["K"]))))
    ck.num("Appendix F Boruta", "all datasets", "refits", total if total else None, str(O.BORUTA_ATTAINMENT["refits"]))
    ck.num("Appendix F Boruta", "all datasets", "at budget", at_budget if total else None, str(O.BORUTA_ATTAINMENT["at_budget"]))
    expected_below = set(O.BORUTA_ATTAINMENT["below_budget_cells"])
    ck.add("Appendix F Boruta", "all datasets", "below-budget cells", sorted(found_below), sorted(expected_below),
           "match" if (found_below == expected_below) else ("blank" if not total else "mismatch"))
    # Stability monthly breakdown
    sp = load("stability_2024", "11_subpopulation_ho_auc.csv")
    for month, n in O.STABILITY_MONTHLY["counts"].items():
        r = first(sp, subpopulation=month, backbone="lr")
        ck.num("Appendix F ST monthly", f"stability_2024 {month}", "n", r and r.get("n"), n)
    for bb, key in (("lr", "delta_range_lr"), ("catboost", "delta_range_catboost")):
        sub = sp[(sp["backbone"] == bb) & (sp["subpopulation"].str.startswith("2020-"))] if not sp.empty else sp
        d = [to_float(v) for v in sub["delta_auc"]] if not sub.empty else []
        d = [v for v in d if v is not None]
        ck.num("Appendix F ST monthly", f"stability_2024 {bb}", "min delta", min(d) if d else None, O.STABILITY_MONTHLY[key][0])
        ck.num("Appendix F ST monthly", f"stability_2024 {bb}", "max delta", max(d) if d else None, O.STABILITY_MONTHLY[key][1])
        ck.num("Appendix F ST monthly", f"stability_2024 {bb}", "months", len(d) if d else None, "9")
    # PSI sentences
    psi06 = load("stability_2024", "06_leader_feature_psi.csv")
    r = first(psi06, backbone="lr", selector="Pure LLM")
    ck.num("Section 6.6.2", "stability_2024 lr Pure LLM", "median_psi", r and r.get("median_psi"), O.PSI_SENTENCES["st_lr_median"][0])
    r = first(psi06, backbone="lr", selector="RFE CatBoost")
    ck.num("Section 6.6.2", "stability_2024 lr RFE CatBoost", "median_psi", r and r.get("median_psi"), O.PSI_SENTENCES["st_lr_median"][1])
    r = first(psi06, backbone="catboost", selector="Pure LLM")
    ck.num("Section 6.6.2", "stability_2024 catboost Pure LLM", "median_psi", r and r.get("median_psi"), O.PSI_SENTENCES["st_cb_pure_llm_median"])
    r = first(load("homecredit", "06_leader_feature_psi.csv"), backbone="catboost", selector="Pure LLM")
    ck.num("Section 6.6.2", "homecredit catboost Pure LLM", "median_psi", r and r.get("median_psi"), O.PSI_SENTENCES["hc_cb_pure_llm_median"])
    for value, ds, bb, selector, feature in O.PSI_SENTENCES["max_features"]:
        r = first(load(ds, "06_leader_feature_psi.csv"), backbone=bb, selector=selector)
        ck.num("Section 6.6.2", f"{ds} {bb} {selector}", "max_psi", r and r.get("max_psi"), value)
        ck.text("Section 6.6.2", f"{ds} {bb} {selector}", "max_psi_feature", r and r.get("max_psi_feature"), feature)
    means = []
    for ds in O.DATASETS:
        for _, r in load(ds, "06_leader_feature_psi.csv").iterrows():
            v = to_float(r.get("mean_psi"))
            if v is not None:
                means.append(v)
    ck.num("Section 6.6.2", "all leaders", "min mean_psi", min(means) if means else None, O.PSI_SENTENCES["mean_range"][0])
    ck.num("Section 6.6.2", "all leaders", "max mean_psi", max(means) if means else None, O.PSI_SENTENCES["mean_range"][1])
    # runtime and memory, token usage
    rt = load("stability_2024", "22_runtime_and_memory.csv")
    r = first(rt, pipeline="mRMR over 1,959 features", stage="selection")
    ck.num("Section 6.7", "stability_2024 fold1", "mRMR seconds 1959", r and r.get("seconds"), O.SECTION67["mrmr_seconds_1959"])
    r = first(rt, pipeline="mRMR over 100-feature pool", stage="selection")
    ck.num("Section 6.7", "stability_2024 fold1", "mRMR seconds 100", r and r.get("seconds"), O.SECTION67["mrmr_seconds_100"])
    for ds, (lo, hi) in O.SECTION67["peak_rss_gb"].items():
        rt = load(ds, "22_runtime_and_memory.csv")
        sub = rt[rt["stage"] == "peak_rss_range"] if not rt.empty else rt
        lo_v = first(sub, pipeline="minimum")
        hi_v = first(sub, pipeline="maximum")
        ck.num("Section 6.7 peak RSS", ds, "min peak_rss_gb", lo_v and lo_v.get("peak_rss_gb"), lo)
        ck.num("Section 6.7 peak RSS", ds, "max peak_rss_gb", hi_v and hi_v.get("peak_rss_gb"), hi)
    for ds, per in O.APPENDIX_F_RUNTIME.items():
        rt = load(ds, "22_runtime_and_memory.csv")
        for bb, pipes in per.items():
            for pipe, minutes in pipes.items():
                stage = "selection" if pipe.endswith("(selection stage)") else "pipeline"
                name = pipe.replace(" (selection stage)", "")
                r = first(rt, backbone=bb, pipeline=name, stage=stage)
                ck.num("Appendix F runtime", f"{ds} {bb} {pipe}", "wall_clock_minutes", r and r.get("wall_clock_minutes"), minutes)
    tok_in = tok_out = calls = 0
    have = False
    for ds in ("homecredit", "lendingclub_v2"):
        u = load(ds, "21_llm_usage.csv")
        for _, r in u.iterrows():
            i, o = to_float(r.get("input_tokens")), to_float(r.get("output_tokens"))
            if i is not None and o is not None:
                have = True
                tok_in += i
                tok_out += o
                calls += 1
    ck.num("Section 6.7 tokens", "homecredit+lendingclub_v2", "provider calls", calls if have else None, O.SECTION67["provider_calls"])
    ck.num("Section 6.7 tokens", "homecredit+lendingclub_v2", "input_tokens", tok_in if have else None, O.SECTION67["input_tokens"])
    ck.num("Section 6.7 tokens", "homecredit+lendingclub_v2", "output_tokens", tok_out if have else None, O.SECTION67["output_tokens"])
    u = load("stability_2024", "21_llm_usage.csv")
    note = " ".join(u["note"].tolist()) if not u.empty else ""
    ck.add("Section 6.7 tokens", "stability_2024", "token usage not recorded", note, "token usage not recorded",
           "match" if "not recorded" in note else "mismatch")


def _r(x, nd):
    return None if x is None else round(float(x) + 0.0, nd)


def check_text_claims(ck: Check) -> None:
    """Prose ranges and differences, recomputed from the CSVs (inputs rounded as the manuscript prints them)."""
    T = O.TEXT_CLAIMS
    ho = {}; dev = {}; lead = {}
    for ds in O.DATASETS:
        for _, r in load(ds, "03_ho_auc.csv").iterrows():
            ho[(ds, r["backbone"], r["selector"])] = to_float(r["ho_auc"])
        for _, r in load(ds, "04_dev_fold_auc.csv").iterrows():
            dev[(ds, r["backbone"], r["selector"])] = to_float(r["dev_mean_auc"])
        for _, r in load(ds, "05_leader_delta_auc.csv").iterrows():
            lead[(ds, r["rule"], r["backbone"])] = r.to_dict()
    p4 = lambda k: _r(ho.get(k), 4)
    d4 = lambda k: _r(dev.get(k), 4)

    def rng(table, where, values, lo, hi):
        vals = [v for v in values if v is not None]
        complete = len(vals) == len(values) and vals
        ck.num(table, where, "range min", min(vals) if complete else None, lo)
        ck.num(table, where, "range max", max(vals) if complete else None, hi)

    # Section 6.1 rounded intervals
    r = lead.get(("homecredit", "HO", "catboost"))
    ck.num("Section 6.1", "homecredit catboost HO-rule CI", "ci_low (3 dp)", r and r.get("ci_low"), T["hc_cb_ho_ci_3dp"][0])
    ck.num("Section 6.1", "homecredit catboost HO-rule CI", "ci_high (3 dp)", r and r.get("ci_high"), T["hc_cb_ho_ci_3dp"][1])
    r = lead.get(("homecredit", "DEV", "lr"))
    ck.num("Section 6.1", "homecredit lr DEV-rule CI", "ci_low (3 dp)", r and r.get("ci_low"), T["hc_lr_dev_ci_3dp"][0])
    ck.num("Section 6.1", "homecredit lr DEV-rule CI", "ci_high (3 dp)", r and r.get("ci_high"), T["hc_lr_dev_ci_3dp"][1])

    # Section 6.1 / 6.2: DEV-rule leaders against HO-best, and DEV-to-HO change
    short_cl, short_llm, fall, move = [], {}, [], []
    hold = None
    for ds in O.DATASETS:
        for bb in O.BACKBONES:
            h, d = lead.get((ds, "HO", bb)), lead.get((ds, "DEV", bb))
            if not h or not d:
                short_cl.append(None); fall.append(None); move.append(None); continue
            sc = _r(to_float(h["classical_ho_auc"]), 4) - _r(to_float(d["classical_ho_auc"]), 4)
            if d["classical_leader"] != h["classical_leader"]:
                short_cl.append(sc)
            short_llm[(ds, bb)] = _r(to_float(h["llm_ho_auc"]), 4) - _r(to_float(d["llm_ho_auc"]), 4)
            c_change = p4((ds, bb, d["classical_leader"])) - d4((ds, bb, d["classical_leader"])) if d4((ds, bb, d["classical_leader"])) is not None else None
            if ds == "stability_2024" and bb == "lr":
                hold = c_change
            else:
                fall.append(None if c_change is None else -c_change)
            l_change = p4((ds, bb, d["llm_leader"])) - d4((ds, bb, d["llm_leader"])) if d4((ds, bb, d["llm_leader"])) is not None else None
            move.append(l_change)
    rng("Section 6.1", "DEV-chosen classical leader short of HO-best (5 cases)", short_cl, *O.DEV_TO_HO["dev_chosen_classical_short"])
    ck.num("Section 6.1", "DEV-chosen LLM-assisted leader below HO-best", "homecredit lr", short_llm.get(("homecredit", "lr")), O.DEV_TO_HO["dev_chosen_llm_short_hc_lr"])
    others = [v for k, v in short_llm.items() if k != ("homecredit", "lr")]
    ck.add("Section 6.1", "DEV-chosen LLM-assisted leader is HO-best", "other five cases", others, "all 0",
           "match" if others and all(abs(v) < 1e-12 for v in others) else ("blank" if not others else "mismatch"))
    rng("Section 6.2", "classical DEV-rule leaders, DEV mean to HO (5 cases)", fall, *O.DEV_TO_HO["classical_fall"])
    ck.add("Section 6.2", "classical DEV-rule leader, Stability LR", "holds level", hold, "|change| < 0.001",
           "match" if hold is not None and abs(hold) < 0.001 else ("blank" if hold is None else "mismatch"))
    rng("Section 6.2", "LLM-assisted DEV-rule leaders, DEV mean to HO", move, *O.DEV_TO_HO["llm_move"])
    rises = [v for v in move if v is not None]
    ck.num("Section 6.2", "LLM-assisted DEV-rule leaders", "cases that rise", sum(v > 0 for v in rises) if len(rises) == 6 else None, str(O.DEV_TO_HO["llm_rise_cases"]))

    # Section 6.3 within-family comparison
    six = ["IV/WOE", "mRMR", "Boruta RF", "IV then Boruta", "RFE CatBoost", "CatBoost SHAP"]
    gaps, hc_lr = [], None
    for ds in O.DATASETS:
        for bb in O.BACKBONES:
            c = [p4((ds, bb, s)) for s in six]; l = [p4((ds, bb, s)) for s in O.LLM_ASSISTED]
            if None in c or None in l:
                gaps.append(None); continue
            g = sum(l) / 4 - sum(c) / 6
            if (ds, bb) == ("homecredit", "lr"):
                hc_lr = -g
            else:
                gaps.append(g)
    rng("Section 6.3", "mean HO AUC, four LLM-assisted minus six classical (5 cases)", gaps, *O.WITHIN_FAMILY["six_classical_vs_four_llm_gap"])
    ck.num("Section 6.3", "mean HO AUC, six classical minus four LLM-assisted", "homecredit lr", hc_lr, O.WITHIN_FAMILY["hc_lr_classical_above"])
    v = None if None in (p4(("homecredit", "lr", "Stable core + LLM fill")), p4(("homecredit", "lr", "Pure LLM"))) else p4(("homecredit", "lr", "Stable core + LLM fill")) - p4(("homecredit", "lr", "Pure LLM"))
    ck.add("Section 6.3", "Pure LLM within 0.006 of the LLM-assisted leader", "homecredit lr", v, "<= 0.006", "match" if v is not None and v <= 0.006 + 1e-12 else ("blank" if v is None else "mismatch"))
    v = None if None in (p4(("lendingclub_v2", "catboost", "LLM then mRMR")), p4(("lendingclub_v2", "catboost", "Pure LLM"))) else p4(("lendingclub_v2", "catboost", "LLM then mRMR")) - p4(("lendingclub_v2", "catboost", "Pure LLM"))
    ck.num("Section 6.3", "Pure LLM trails LLM then mRMR", "lendingclub_v2 catboost", v, O.WITHIN_FAMILY["pure_llm_trails_llm_then_mrmr_lc_cb"])
    for bb, key in (("lr", "st_llm_spread_lr"), ("catboost", "st_llm_spread_catboost")):
        vals = [p4(("stability_2024", bb, s)) for s in O.LLM_ASSISTED]
        ck.num("Section 6.3", f"stability_2024 {bb} LLM-assisted spread", "max - min", (max(vals) - min(vals)) if None not in vals else None, O.WITHIN_FAMILY[key])

    # Table 6 / Appendix F family cap
    falls, gaps5 = [], []
    for ds in O.DATASETS:
        for _, r in load(ds, "15_diversity_controls.csv").iterrows():
            if r["control"] != "family_cap" or blank(r.get("control_ho_auc")):
                continue
            ctrl, unc, llm = _r(to_float(r["control_ho_auc"]), 4), _r(to_float(r["uncapped_ho_auc"]), 4), _r(to_float(r["llm_assisted_ho_auc"]), 4)
            falls.append(unc - ctrl)
            if not (ds == "homecredit" and r["backbone"] == "lr"):
                gaps5.append(llm - ctrl)
    rng("Appendix F", "family cap: classical leader falls (6 cases)", falls if len(falls) == 6 else falls + [None], *T["family_cap_fall"])
    rng("Appendix F", "family cap: capped leader below LLM-assisted leader (5 cases)", gaps5 if len(gaps5) == 5 else gaps5 + [None], *T["family_cap_gap_five"])

    # Section 6.5.2 lineage-only descriptions
    leads, bounds, hc_core = [], [], None
    for ds in O.DATASETS:
        for _, r in load(ds, "13_lineage_only.csv").iterrows():
            if blank(r.get("ho_auc_mechanical")):
                continue
            for b in ("ci_low", "ci_high"):
                if not blank(r.get(b)):
                    bounds.append(abs(to_float(r[b])))
            h = lead.get((ds, "HO", r["backbone"]))
            if not h:
                continue
            if r["selector"] == h["llm_leader"]:
                diff = _r(to_float(r["ho_auc_mechanical"]), 4) - _r(to_float(h["classical_ho_auc"]), 4)
                if (ds, r["backbone"]) == ("homecredit", "lr"):
                    hc_core = diff
                else:
                    leads.append(diff)
    rng("Section 6.5.2", "winning LLM-assisted leaders with rebuilt descriptions, lead over classical (5 cases)", leads if len(leads) == 5 else leads + [None], *T["mechanical_lead"])
    ck.num("Section 6.5.2", "homecredit lr stable core (mechanical) minus mRMR", "difference", hc_core, T["mechanical_hc_lr_vs_mrmr"])
    ck.num("Section 6.5.2", "Table 8 intervals", "widest bound", max(bounds) if len(bounds) == 16 else None, T["mechanical_widest_bound"])

    # Section 6.6.1 stability ranges
    st = {}
    for ds in O.DATASETS:
        for _, r in load(ds, "07_selection_stability.csv").iterrows():
            st[(ds, r["backbone"], r["selector"])] = r.to_dict()
    def g(ds, bb, sel, col, nd):
        r = st.get((ds, bb, sel)); return None if r is None or blank(r.get(col)) else _r(to_float(r[col]), nd)
    best = {(row["dataset"], row["backbone"]): row["selector"] for row in O.TABLE9[:6]}
    hclc = [(ds, bb) for ds in ("homecredit", "lendingclub_v2") for bb in O.BACKBONES]
    allc = [(ds, bb) for ds in O.DATASETS for bb in O.BACKBONES]
    rng("Section 6.6.1", "pure LLM Nogueira (HC, LC)", [g(ds, bb, "Pure LLM", "nogueira", 4) for ds, bb in hclc], *T["pure_llm_nogueira"])
    rng("Section 6.6.1", "pure LLM mean Jaccard (HC, LC)", [g(ds, bb, "Pure LLM", "jaccard_mean", 4) for ds, bb in hclc], *T["pure_llm_jaccard"])
    def spread(ds, bb, sel):
        a, b = g(ds, bb, sel, "jaccard_max", 3), g(ds, bb, sel, "jaccard_min", 3)
        return None if a is None or b is None else a - b
    rng("Section 6.6.1", "pure LLM fold-pair spread (HC, LC)", [spread(ds, bb, "Pure LLM") for ds, bb in hclc], *T["pure_llm_pair_spread"])
    rng("Section 6.6.1", "most reproducible classical fold-pair spread (6 cases)", [spread(ds, bb, best[(ds, bb)]) for ds, bb in allc], *T["classical_pair_spread"])
    def diff(a, b):
        return None if a is None or b is None else a - b
    rng("Section 6.6.1", "most reproducible classical minus pure LLM, Nogueira (HC, LC)",
        [diff(g(ds, bb, best[(ds, bb)], "nogueira", 4), g(ds, bb, "Pure LLM", "nogueira", 4)) for ds, bb in hclc], *T["classical_above_pure_llm"])
    sc = "Stable core + LLM fill"; lb = "LLM then Boruta"
    rng("Section 6.6.1", "stable core Nogueira (6 cases)", [g(ds, bb, sc, "nogueira", 4) for ds, bb in allc], *T["stable_core_nogueira"])
    rng("Section 6.6.1", "stable core mean Jaccard (6 cases)", [g(ds, bb, sc, "jaccard_mean", 4) for ds, bb in allc], *T["stable_core_jaccard"])
    rng("Section 6.6.1", "stable core below most reproducible classical, Nogueira (6 cases)",
        [diff(g(ds, bb, best[(ds, bb)], "nogueira", 4), g(ds, bb, sc, "nogueira", 4)) for ds, bb in allc], *T["stable_core_below_classical"])
    rng("Section 6.6.1", "stable core above pure LLM, Nogueira (HC, LC)", [diff(g(ds, bb, sc, "nogueira", 4), g(ds, bb, "Pure LLM", "nogueira", 4)) for ds, bb in hclc], *T["stable_core_above_pure_llm"])
    rng("Section 6.6.1", "LLM then Boruta Nogueira (6 cases)", [g(ds, bb, lb, "nogueira", 4) for ds, bb in allc], *T["llm_then_boruta_nogueira"])
    ck.num("Section 6.6.1", "LLM then Boruta below pure LLM, Nogueira", "homecredit lr", diff(g("homecredit", "lr", "Pure LLM", "nogueira", 4), g("homecredit", "lr", lb, "nogueira", 4)), T["hc_lr_llm_then_boruta_below_pure_llm"])
    rng("Section 6.6.1", "Stability LLM then Boruta below stable core", [diff(g("stability_2024", bb, sc, "nogueira", 4), g("stability_2024", bb, lb, "nogueira", 4)) for bb in O.BACKBONES], *T["st_llm_then_boruta_below_stable_core"])
    rng("Section 6.6.1", "Stability LLM then Boruta below most reproducible classical", [diff(g("stability_2024", bb, best[("stability_2024", bb)], "nogueira", 4), g("stability_2024", bb, lb, "nogueira", 4)) for bb in O.BACKBONES], *T["st_llm_then_boruta_below_classical"])

    # Appendix F: LendingClub event rate DEV -> HO
    c = load("lendingclub_v2", "01_cohort.csv")
    for split, val in (("DEV", T["lc_event_rate_dev_ho_pct"][0]), ("HO", T["lc_event_rate_dev_ho_pct"][1])):
        r = first(c, split=split)
        rate = to_float(r and r.get("event_rate"))
        ck.num("Appendix F", f"lendingclub_v2 {split} event rate", "percent (1 dp)", None if rate is None else rate * 100, val)

    # Appendix C: pooled development curves
    pooled = {}
    for ds in O.DATASETS:
        sw = load(ds, "17_budget_sweep.csv")
        for bb in O.BACKBONES:
            for k in (10, 20, 30, 40, 50):
                rows = sw[(sw["backbone"] == bb) & (sw["K"] == str(k))]
                sel = rows[rows["family"].isin(["classical", "LLM-assisted"])]
                pool = rows[rows["family"] == "pooled"]
                vals = [to_float(v) for v in sel["dev_mean_auc"]]
                if len(vals) == 13 and None not in vals:
                    pooled[(ds, bb, k)] = sum(vals) / 13
                elif not pool.empty and to_float(pool["dev_mean_auc"].iloc[0]) is not None:
                    pooled[(ds, bb, k)] = to_float(pool["dev_mean_auc"].iloc[0])
    def rel(bb, k, ref):
        v = [pooled.get((ds, bb, k)) for ds in O.DATASETS]; w = [pooled.get((ds, bb, ref)) for ds in O.DATASETS]
        return None if None in v + w else sum(a - b for a, b in zip(v, w)) / 3
    ck.num("Appendix C", "pooled LR DEV AUC", "K=30 minus K=20", rel("lr", 30, 20), O.APPENDIX_C_TEXT["lr_gain_k30"])
    ck.num("Appendix C", "pooled LR DEV AUC", "K=40 minus K=20", rel("lr", 40, 20), O.APPENDIX_C_TEXT["lr_gain_k40"])
    ck.num("Appendix C", "pooled CatBoost DEV AUC", "K=50 minus K=40", rel("catboost", 50, 40), O.APPENDIX_C_TEXT["cb_k50"])
    for bb, ref, coords in (("lr", 20, O.FIGUREC1A["lr"]), ("catboost", 40, O.FIGUREC1A["catboost"])):
        for k, val in coords.items():
            if k != ref:
                ck.num("Figure C.1a", f"pooled {bb}", f"K={k} relative to K={ref}", rel(bb, k, ref), val)
    cb_lr_40 = [None if None in (pooled.get((ds, "catboost", 40)), pooled.get((ds, "lr", 40))) else pooled[(ds, "catboost", 40)] - pooled[(ds, "lr", 40)] for ds in O.DATASETS]
    rng("Appendix C", "CatBoost over LR at K=40 (3 datasets)", cb_lr_40, *O.APPENDIX_C_TEXT["cb_over_lr_k40"])
    allk = [None if None in (pooled.get((ds, "catboost", k)), pooled.get((ds, "lr", k))) else pooled[(ds, "catboost", k)] - pooled[(ds, "lr", k)] for ds in O.DATASETS for k in (10, 20, 30, 40, 50)]
    rng("Appendix C", "CatBoost over LR at every budget (3 datasets x 5 budgets)", allk, *O.APPENDIX_C_TEXT["cb_over_lr_all"])
    unmatched = [None if None in (pooled.get((ds, "catboost", 40)), pooled.get((ds, "lr", 20))) else pooled[(ds, "catboost", 40)] - pooled[(ds, "lr", 20)] for ds in O.DATASETS]
    rng("Appendix C", "unmatched CatBoost K=40 minus LR K=20 (3 datasets)", unmatched, *O.APPENDIX_C_TEXT["unmatched_gap"])
    share = [None if u is None or m is None else 100 * (u - m) / u for u, m in zip(unmatched, cb_lr_40)]
    rng("Appendix C", "budget share of the unmatched difference, percent (3 datasets)", share, *O.APPENDIX_C_TEXT["budget_share_pct"])


def run_checks() -> Check:
    ck = Check()
    for fn in (check_table2, check_table5, check_table3, check_table4, check_table7, check_table8, check_table9, check_table10,
               check_table11, check_tableB1, check_tableC1, check_tableF1, check_tableF2, check_tableF3, check_tableF4, check_appendix_f,
               check_text_claims):
        fn(ck)
    ck.apply_known()
    return ck


# ----------------------------------------------------------------------------- rendering
def md(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def r4(v, places=4):
    f = to_float(v)
    return "" if f is None else f"{f:.{places}f}"


def render() -> str:
    parts = []
    # Table 2
    rows = []
    for key, label in (("DEV", "DEV"), ("HO", "HO")):
        for col in ("n_rows", "event_rate"):
            row = [f"{label} {col}"]
            for ds in O.DATASETS:
                r = first(load(ds, "01_cohort.csv"), split=key)
                v = r and r.get(col)
                row.append(f"{to_float(v) * 100:.2f}%" if (col == "event_rate" and not blank(v)) else (f"{int(float(v)):,}" if not blank(v) else ""))
            rows.append(row)
    row = ["candidate set"]
    for ds in O.DATASETS:
        r = first(load(ds, "02_candidate_set.csv"), screen="missing rate <= 0.90 on fold-1 DEV training rows")
        row.append(f"{r['retained']} of {r['universe']}" if r else "")
    rows.append(row)
    parts.append("### Table 2 (datasets)\n" + md(["", *[O.DATASET_LABEL[d] for d in O.DATASETS]], rows))
    # Table 5
    rows = []
    for selector in O.SELECTORS + [O.FULL]:
        row = [selector]
        for ds in O.DATASETS:
            for bb in O.BACKBONES:
                r = first(load(ds, "03_ho_auc.csv"), backbone=bb, selector=selector)
                row.append(r4(r and r.get("ho_auc")))
        rows.append(row)
    row = ["LLM-assisted leader - classical leader"]
    for ds in O.DATASETS:
        for bb in O.BACKBONES:
            r = first(load(ds, "05_leader_delta_auc.csv"), rule="HO", backbone=bb)
            row.append(r4(r and r.get("delta_auc"), 3))
    rows.append(row)
    parts.append("### Table 5 (HO ROC-AUC)\n" + md(["Selector", "HC LR", "HC CB", "LC LR", "LC CB", "ST LR", "ST CB"], rows))
    # Tables 3 and 4
    for rule, title in (("HO", "Table 3 (HO rule)"), ("DEV", "Table 4 (DEV rule)")):
        rows = []
        for ds in O.DATASETS:
            for bb in O.BACKBONES:
                r = first(load(ds, "05_leader_delta_auc.csv"), rule=rule, backbone=bb)
                if not r:
                    rows.append([f"{O.DATASET_LABEL[ds]}, {bb}", "", "", "", ""])
                    continue
                places = 4 if ds == "stability_2024" else 6
                if rule == "HO":
                    f = first(load(ds, "03_ho_auc.csv"), backbone=bb, selector=O.FULL)
                    psi = load(ds, "06_leader_feature_psi.csv")
                    pl = first(psi, backbone=bb, selector=r["llm_leader"])
                    pc = first(psi, backbone=bb, selector=r["classical_leader"])
                    rows.append([f"{O.DATASET_LABEL[ds]}, {bb}", f"{r['llm_leader']} ({r4(r['llm_ho_auc'])})", f"{r['classical_leader']} ({r4(r['classical_ho_auc'])})",
                                 r4(f and f.get("ho_auc")), f"{r4(r['delta_auc'], places)} [{r4(r['ci_low'], 6)}, {r4(r['ci_high'], 6)}]",
                                 f"{r4(pl and pl.get('mean_psi'))} / {r4(pc and pc.get('mean_psi'))}"])
                else:
                    rows.append([f"{O.DATASET_LABEL[ds]}, {bb}", f"{r['llm_leader']} ({r4(r['llm_dev_auc'])}; {r4(r['llm_ho_auc'])})",
                                 f"{r['classical_leader']} ({r4(r['classical_dev_auc'])}; {r4(r['classical_ho_auc'])})",
                                 f"{r4(r['delta_auc'], places)} [{r4(r['ci_low'], 6)}, {r4(r['ci_high'], 6)}]"])
        headers = ["Case", "LLM-assisted leader (AUC)", "Classical leader (AUC)", "Full", "dAUC [95% CI]", "PSI LLM / classical"] if rule == "HO" else \
                  ["Case", "LLM-assisted leader (DEV; HO)", "Classical leader (DEV; HO)", "dAUC [95% CI]"]
        parts.append(f"### {title}\n" + md(headers, rows))
    # Table 7
    rows = []
    for _, r in load("homecredit", "19_obfuscation.csv").iterrows():
        folds = [to_float(r.get(f"fold{i}_auc")) for i in range(1, 6)]
        folds = [f for f in folds if f is not None]
        rows.append([r["arm"], r["backbone"], r["K"], r4(r["ho_auc_arm"]), r4(r["ho_auc_named"]), r4(r["delta_auc"]),
                     f"{min(folds):.3f}-{max(folds):.3f}" if folds else "", f"{r['overlap_full_dev']}/{r['K']}",
                     f"{r['overlap_folds_min']}-{r['overlap_folds_max']} (mean {r['overlap_folds_mean']})"])
    parts.append("### Table 7 (obfuscation, Home Credit)\n" + md(["Arm", "Model", "K", "HO AUC", "Named", "Delta", "Fold range", "Overlap full DEV", "Overlap folds"], rows))
    # Table 8
    rows = []
    for ds in O.DATASETS:
        for _, r in load(ds, "13_lineage_only.csv").iterrows():
            if blank(r.get("ho_auc_mechanical")):
                continue
            rows.append([O.DATASET_LABEL[ds], r["backbone"], r["selector"], r["K"], r4(r["ho_auc_mechanical"]), r4(r["ho_auc_cached"]),
                         f"{r4(r['delta_auc'])} [{r4(r['ci_low'])}, {r4(r['ci_high'])}]", r["overlap_with_cached"]])
    parts.append("### Table 8 (descriptions built from lineage)\n" + md(["Case", "Model", "Pipeline", "K", "Mechanical", "Cached", "Delta [95% CI]", "Overlap"], rows))
    # Table 9
    rows = []
    for row in O.TABLE9:
        r = first(load(row["dataset"], "07_selection_stability.csv"), backbone=row["backbone"], selector=row["selector"])
        rows.append([O.DATASET_LABEL[row["dataset"]], row["backbone"], row["selector"], row["K"], r4(r and r.get("nogueira")), r4(r and r.get("jaccard_mean")),
                     r4(r and r.get("jaccard_min"), 3), r4(r and r.get("jaccard_max"), 3)])
    parts.append("### Table 9 (fold-to-fold reproducibility)\n" + md(["Case", "Model", "Selector", "K", "Nogueira", "Jaccard", "Min", "Max"], rows))
    # Table 10
    rows = []
    for row in O.TABLE10:
        rs = load(row["dataset"], "16_response_spread.csv")
        sub = rs[(rs["backbone"] == row["backbone"]) & (rs["is_headline_response"].str.upper() != "TRUE")] if not rs.empty else rs
        vals = [to_float(v) for v in sub["ho_auc"]] if not sub.empty else []
        vals = [v for v in vals if v is not None]
        if not vals:
            rows.append([O.DATASET_LABEL[row["dataset"]], row["backbone"], row["K"], "", "", "", "", ""])
            continue
        s = pd.Series(vals)
        rows.append([O.DATASET_LABEL[row["dataset"]], row["backbone"], row["K"], len(vals), f"{min(vals):.4f}-{max(vals):.4f}", f"{max(vals) - min(vals):.4f}",
                     f"{s.median():.4f}" if row["median"] else "--", f"{s.std(ddof=1):.4f}" if row["median"] else "--"])
    parts.append("### Table 10 (response sampling)\n" + md(["Case", "Model", "K", "Draws", "HO AUC range", "Spread", "Median", "s.d."], rows))
    # Table 11
    rows = []
    for ds in O.DATASETS:
        sm = load(ds, "08_secondary_metrics.csv")
        cells = [O.DATASET_LABEL[ds]]
        for col, best in (("ks", "max"), ("brier", "min"), ("capture_at_10", "max")):
            s = sm[col].map(to_float) if not sm.empty else pd.Series(dtype=float)
            if s.notna().sum() == 0:
                cells.append("")
                continue
            idx = s.idxmax() if best == "max" else s.idxmin()
            r = sm.loc[idx]
            extra = f" ({r4(r.get('lift_at_10'), 2)})" if col == "capture_at_10" and not blank(r.get("lift_at_10")) else ""
            cells.append(f"{r4(r[col])}{extra} {r['selector']}, {r['backbone']}")
        rows.append(cells)
    parts.append("### Table 11 (best secondary metric)\n" + md(["Dataset", "KS", "Brier", "Capture@10 (Lift@10)"], rows))
    # Table B.1
    rows = []
    for ds in ("homecredit", "lendingclub_v2"):
        for _, r in load(ds, "20_rank_voting.csv").iterrows():
            if blank(r.get("voting_auc")):
                continue
            rows.append([O.DATASET_LABEL[ds], r["backbone"], r["pool_P"] + (" (primary)" if str(r["is_primary"]).upper() == "TRUE" else ""), r4(r["reference_auc"]), r4(r["voting_auc"]),
                         r4(r["delta_auc"]), f"[{r4(r['ci_low'])}, {r4(r['ci_high'])}]", r["holm_p"], r4(r["delta_ks"])])
    parts.append("### Table B.1 (classical rank voting)\n" + md(["Case", "Model", "Pool P", "Ref. AUC", "Voting AUC", "dAUC", "95% CI", "Holm p", "dKS"], rows))
    # Table C.1
    rows = []
    for ds in O.DATASETS:
        for bb in O.BACKBONES:
            leaders = _sweep_leaders(ds, bb)
            cells = [f"{O.DATASET_LABEL[ds]}, {bb}"]
            for k in (10, 20, 30, 40, 50):
                llm, cla, _ = leaders.get(k, (None, None, None))
                cells.append(f"{llm - cla:+.4f}" if (llm is not None and cla is not None) else "")
            rows.append(cells)
    parts.append("### Table C.1 (DEV AUC, LLM-assisted leader minus classical leader, by budget)\n" + md(["Case", "10", "20", "30", "40", "50"], rows))
    # Table F.1
    rows = []
    sp = load("lendingclub_v2", "11_subpopulation_ho_auc.csv")
    for sub in ("matured_36m_2016-01_to_2016-05", "term_36", "term_60"):
        cells = [sub]
        for bb in O.BACKBONES:
            r = first(sp, subpopulation=sub, backbone=bb)
            if bb == "lr":
                cells.append(f"{int(float(r['n'])):,}" if r and not blank(r.get("n")) else "")
            cells.append(f"{to_float(r['delta_auc']):+.4f}" if r and not blank(r.get("delta_auc")) else "")
        rows.append(cells)
    parts.append("### Table F.1 (LendingClub sub-populations)\n" + md(["Sub-population", "n", "LR", "CatBoost"], rows))
    # Table F.2
    rows = []
    for ds in O.DATASETS:
        for _, r in load(ds, "15_diversity_controls.csv").iterrows():
            if blank(r.get("control_ho_auc")):
                continue
            rows.append([O.DATASET_LABEL[ds], r["backbone"], r["control"], r["base_selector"], r["K"], r4(r["control_ho_auc"]), r4(r["uncapped_ho_auc"]),
                         r4(r["llm_assisted_ho_auc"]), r4(r["difference"], 3)])
    parts.append("### Table F.2 (diversity controls)\n" + md(["Case", "Model", "Rule", "Base selector", "K", "Control", "Uncapped", "LLM-assisted", "Difference"], rows))
    # Table F.3
    rows = []
    for row in O.TABLEF3:
        ls = load(row["dataset"], "09_leader_subsets.csv")
        sub = ls[(ls["backbone"] == row["backbone"]) & (ls["selector"] == row["selector"])] if not ls.empty else ls
        ncat = int((sub["is_categorical"].str.upper() == "TRUE").sum()) if not sub.empty else ""
        cols = sub["n_model_columns"].unique()[0] if not sub.empty else ""
        rows.append([O.DATASET_LABEL[row["dataset"]], row["backbone"], row["selector"], row["K"], ncat, cols])
    parts.append("### Table F.3 (effective dimension)\n" + md(["Dataset", "Model", "Selector", "K", "Cat.", "Columns"], rows))
    # Table F.4
    rows = []
    for row in O.TABLEF4:
        r = first(load(row["dataset"], "18_response_agreement.csv"), backbone=row["backbone"])
        rows.append([O.DATASET_LABEL[row["dataset"]], row["backbone"], row["K"], r and r.get("draws"), r4(r and r.get("nogueira")), r4(r and r.get("jaccard_mean")),
                     r and r.get("names_in_all_draws"), r and r.get("distinct_names")])
    parts.append("### Table F.4 (agreement among cached responses)\n" + md(["Case", "Model", "K", "Draws", "Nogueira", "Jaccard", "Names in all draws", "Distinct names"], rows))
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- schema
def schema_check() -> list[str]:
    problems: list[str] = []
    headers: dict[str, dict[str, list[str]]] = {}
    for ds, folder in FOLDERS.items():
        path = RESULTS / folder
        if not path.is_dir():
            problems.append(f"{folder}: missing folder")
            continue
        present = sorted(p.name for p in path.glob("*.csv"))
        if present != sorted(FILES):
            problems.append(f"{folder}: file set differs: extra={sorted(set(present) - set(FILES))} missing={sorted(set(FILES) - set(present))}")
        for name in present:
            raw = (path / name).read_bytes()
            if b"\r" in raw:
                problems.append(f"{folder}/{name}: contains CR")
            if not raw.endswith(b"\n"):
                problems.append(f"{folder}/{name}: no final newline")
            text = raw.decode("utf-8")
            lines = text.split("\n")
            if lines[-1] == "":
                lines = lines[:-1]
            if any(set(line) <= {","} for line in lines[1:]):
                problems.append(f"{folder}/{name}: empty trailing/blank data row")
            headers.setdefault(name, {})[ds] = lines[0].split(",")
        ids = load(ds, "00_run_ids.csv")
        listed = set(ids["file"].tolist()) if not ids.empty else set()
        for name in present:
            if name != "00_run_ids.csv" and name not in listed:
                problems.append(f"{folder}: 00_run_ids.csv does not list {name}")
    for name, per in headers.items():
        hs = {tuple(h) for h in per.values()}
        if len(hs) > 1:
            problems.append(f"{name}: headers differ across folders: {per}")
    return problems


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="with --check, list every compared value")
    args = ap.parse_args(argv)
    if not (args.render or args.check or args.schema):
        ap.print_help()
        return 2
    rc = 0
    if args.schema:
        problems = schema_check()
        print("SCHEMA:", "ok" if not problems else f"{len(problems)} problem(s)")
        for p in problems:
            print("  -", p)
        rc |= 1 if problems else 0
    if args.render:
        print(render())
    if args.check:
        ck = run_checks()
        n, matched, blanks, mism, flagged = ck.summary()
        for r in ck.rows:
            if r["status"] in ("mismatch", "flagged") or (args.verbose and r["status"] != "match"):
                print(f"{r['status'].upper():8} {r['table']} | {r['where']} | {r['column']} | csv={r['csv']} manuscript={r['manuscript']} {r['note']}")
        if args.verbose:
            for r in ck.rows:
                if r["status"] == "match":
                    print(f"match    {r['table']} | {r['where']} | {r['column']} | csv={r['csv']} manuscript={r['manuscript']} {r['note']}")
        print(f"CHECK: {n} manuscript values compared, {matched} matched, {blanks} blank (no CSV source - see docs/repository_freeze.md), "
              f"{flagged} flagged (known discrepancies listed in scripts/manuscript_oracle.py), {mism} mismatched")
        if mism:
            rc |= 1
        else:
            print(f"{blanks} manuscript values have no CSV source - see docs/repository_freeze.md")
    return rc


if __name__ == "__main__":
    sys.exit(main())
