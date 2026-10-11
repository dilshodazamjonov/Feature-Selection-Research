"""Manuscript oracle: every number the submission prints, transcribed from
``~/projects/article/submission/manuscript.tex`` (and ``supplementary.tex``) on 2026-10-09.

Values are kept as the strings the manuscript prints, so that the verification script can
round a CSV value to the same number of decimals before comparing.  Each block names the
table or section it comes from.  Nothing in this module is a result; it is the comparison
target of ``scripts/verify_manuscript_tables.py``.

Dataset keys: homecredit, lendingclub_v2, stability_2024.  Backbones: lr (K=20), catboost (K=40).
Selector names are the manuscript's Table 1 names; "Full features" is the reference.
"""

from __future__ import annotations

DATASETS = ("homecredit", "lendingclub_v2", "stability_2024")
DATASET_LABEL = {"homecredit": "Home Credit", "lendingclub_v2": "LendingClub v2", "stability_2024": "Stability 2024"}
BACKBONES = ("lr", "catboost")
K_OF = {"lr": 20, "catboost": 40}

CLASSICAL = ["Domain rules", "Random K", "PCA", "IV/WOE", "mRMR", "Boruta RF", "IV then Boruta", "RFE CatBoost", "CatBoost SHAP"]
LLM_ASSISTED = ["Pure LLM", "LLM then mRMR", "LLM then Boruta", "Stable core + LLM fill"]
SELECTORS = CLASSICAL + LLM_ASSISTED
FULL = "Full features"
FAMILY = {**{s: "classical" for s in CLASSICAL}, **{s: "LLM-assisted" for s in LLM_ASSISTED}, FULL: "reference"}

# ---------------------------------------------------------------------------- Table 2 (tab:datasets)
TABLE2 = {
    "homecredit": {"dev_window": ("-600", "-241"), "dev_n": "99,092", "dev_event_rate_pct": "7.93",
                   "ho_window": ("-240", "-1"), "ho_n": "120,053", "ho_event_rate_pct": "8.90",
                   "universe": "529", "candidate": "373"},
    "lendingclub_v2": {"dev_window": ("-1795", "-1096"), "dev_n": "598,649", "dev_event_rate_pct": "19.54",
                       "ho_window": ("-1065", "-730"), "ho_n": "293,105", "ho_event_rate_pct": "23.29",
                       "universe": "675", "candidate": "675"},
    "stability_2024": {"dev_window": ("2019-01-01", "2020-02-25"), "dev_n": "1,221,743", "dev_event_rate_pct": "3.24",
                       "ho_window": ("2020-02-26", "2020-10-05"), "ho_n": "304,916", "ho_event_rate_pct": "2.74",
                       "universe": "1,959", "candidate": "1,068"},
}
# Section 5.1 / 5.2 sentences
SECTION5 = {
    "homecredit": {"applications_covered": "219,145", "applications_release": "307,511", "unused_pct": "28.7"},
    "lendingclub_v2": {"dev_36m": "445,596", "ho_36m": "232,361", "dev_rate_36m_pct": "14.5", "dev_rate_60m_pct": "34.3",
                       "ho_rate_36m_pct": "19.8", "ho_rate_60m_pct": "36.4", "max_missingness_pct": "82",
                       "missing_lt40": "566", "missing_40_60": "2", "missing_60_82": "107", "source_columns": "93",
                       "composition": {"raw": "116", "flags": "180", "ratios": "158", "missing_indicators": "88",
                                       "engineered": "78", "binned": "27", "interactions": "19", "categoricals": "9"}},
    "stability_2024": {"reviewed": "461", "approved": "434", "ho_events": "8,349", "ho_share_pct": "20.0", "decisions": "1,526,659"},
}

# ---------------------------------------------------------------------------- Table 5 (tab:full_matrix), HO ROC-AUC
TABLE5 = {
    "homecredit":     {"lr": {"Domain rules": "0.7230", "Random K": "0.6278", "PCA": "0.6838", "IV/WOE": "0.7381", "mRMR": "0.7699",
                              "Boruta RF": "0.7397", "IV then Boruta": "0.7534", "RFE CatBoost": "0.7465", "CatBoost SHAP": "0.7501",
                              "Pure LLM": "0.7436", "LLM then mRMR": "0.7432", "LLM then Boruta": "0.7337", "Stable core + LLM fill": "0.7489",
                              FULL: "0.7715"},
                       "catboost": {"Domain rules": "0.7293", "Random K": "0.6854", "PCA": "0.7091", "IV/WOE": "0.7686", "mRMR": "0.7599",
                              "Boruta RF": "0.7542", "IV then Boruta": "0.7688", "RFE CatBoost": "0.7814", "CatBoost SHAP": "0.7795",
                              "Pure LLM": "0.7935", "LLM then mRMR": "0.7758", "LLM then Boruta": "0.7657", "Stable core + LLM fill": "0.7809",
                              FULL: "0.7860"}},
    "lendingclub_v2": {"lr": {"Domain rules": "0.5525", "Random K": "0.6038", "PCA": "0.6624", "IV/WOE": "0.6821", "mRMR": "0.6859",
                              "Boruta RF": "0.6557", "IV then Boruta": "0.7098", "RFE CatBoost": "0.6989", "CatBoost SHAP": "0.6965",
                              "Pure LLM": "0.7402", "LLM then mRMR": "0.7179", "LLM then Boruta": "0.6793", "Stable core + LLM fill": "0.7048",
                              FULL: "0.7109"},
                       "catboost": {"Domain rules": "0.6128", "Random K": "0.6718", "PCA": "0.6787", "IV/WOE": "0.7027", "mRMR": "0.7020",
                              "Boruta RF": "0.6857", "IV then Boruta": "0.7201", "RFE CatBoost": "0.7190", "CatBoost SHAP": "0.7151",
                              "Pure LLM": "0.7335", "LLM then mRMR": "0.7707", "LLM then Boruta": "0.7002", "Stable core + LLM fill": "0.7277",
                              FULL: "0.7230"}},
    "stability_2024": {"lr": {"Domain rules": "0.7371", "Random K": "0.6818", "PCA": "0.7019", "IV/WOE": "0.7781", "mRMR": "0.7769",
                              "Boruta RF": "0.7611", "IV then Boruta": "0.7976", "RFE CatBoost": "0.8181", "CatBoost SHAP": "0.8092",
                              "Pure LLM": "0.8322", "LLM then mRMR": "0.8112", "LLM then Boruta": "0.8098", "Stable core + LLM fill": "0.8171",
                              FULL: "0.7659"},
                       "catboost": {"Domain rules": "0.7791", "Random K": "0.7502", "PCA": "0.7692", "IV/WOE": "0.8187", "mRMR": "0.8308",
                              "Boruta RF": "0.8095", "IV then Boruta": "0.8387", "RFE CatBoost": "0.8425", "CatBoost SHAP": "0.8509",
                              "Pure LLM": "0.8716", "LLM then mRMR": "0.8519", "LLM then Boruta": "0.8385", "Stable core + LLM fill": "0.8489",
                              FULL: "0.8198"}},
}
# Last row of Table 5: LLM-assisted leader minus classical leader, three decimals
TABLE5_DELTA = {("homecredit", "lr"): "-0.021", ("homecredit", "catboost"): "+0.012", ("lendingclub_v2", "lr"): "+0.030",
                ("lendingclub_v2", "catboost"): "+0.051", ("stability_2024", "lr"): "+0.014", ("stability_2024", "catboost"): "+0.021"}

# ---------------------------------------------------------------------------- Table 3 (tab:primary_results), HO rule
TABLE3 = [
    {"dataset": "homecredit", "backbone": "lr", "llm_leader": "Stable core + LLM fill", "llm_auc": "0.7489", "classical_leader": "mRMR", "classical_auc": "0.7699",
     "full": "0.7715", "delta": "-0.021033", "ci_low": "-0.022211", "ci_high": "-0.019780", "psi_llm": "0.0053", "psi_classical": "0.0085"},
    {"dataset": "homecredit", "backbone": "catboost", "llm_leader": "Pure LLM", "llm_auc": "0.7935", "classical_leader": "RFE CatBoost", "classical_auc": "0.7814",
     "full": "0.7860", "delta": "+0.012024", "ci_low": "+0.009313", "ci_high": "+0.014841", "psi_llm": "0.0045", "psi_classical": "0.0097"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "llm_leader": "Pure LLM", "llm_auc": "0.7402", "classical_leader": "IV then Boruta", "classical_auc": "0.7098",
     "full": "0.7109", "delta": "+0.030431", "ci_low": "+0.029359", "ci_high": "+0.031514", "psi_llm": "0.0125", "psi_classical": "0.0129"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "llm_leader": "LLM then mRMR", "llm_auc": "0.7707", "classical_leader": "IV then Boruta", "classical_auc": "0.7201",
     "full": "0.7230", "delta": "+0.050517", "ci_low": "+0.049712", "ci_high": "+0.051315", "psi_llm": "0.0104", "psi_classical": "0.0125"},
    {"dataset": "stability_2024", "backbone": "lr", "llm_leader": "Pure LLM", "llm_auc": "0.8322", "classical_leader": "RFE CatBoost", "classical_auc": "0.8181",
     "full": "0.7659", "delta": "+0.0141", "ci_low": "+0.012823", "ci_high": "+0.015339", "psi_llm": "0.0174", "psi_classical": "0.0092"},
    {"dataset": "stability_2024", "backbone": "catboost", "llm_leader": "Pure LLM", "llm_auc": "0.8716", "classical_leader": "CatBoost SHAP", "classical_auc": "0.8509",
     "full": "0.8198", "delta": "+0.0207", "ci_low": "+0.017219", "ci_high": "+0.023151", "psi_llm": "0.0109", "psi_classical": "0.0189"},
]
# Figure 3 coordinates (full precision where the figure code carries it)
FIGURE3 = {
    ("homecredit", "lr"): {"classical": "0.769890", "llm": "0.748857", "full": "0.771496", "delta": "-0.021033", "minus": "0.001178", "plus": "0.001253"},
    ("homecredit", "catboost"): {"classical": "0.781426", "llm": "0.793450", "full": "0.786032", "delta": "0.012024", "minus": "0.002711", "plus": "0.002817"},
    ("lendingclub_v2", "lr"): {"classical": "0.709803", "llm": "0.740234", "full": "0.710887", "delta": "0.030431", "minus": "0.001072", "plus": "0.001083"},
    ("lendingclub_v2", "catboost"): {"classical": "0.720147", "llm": "0.770664", "full": "0.722973", "delta": "0.050517", "minus": "0.000805", "plus": "0.000798"},
    ("stability_2024", "lr"): {"classical": "0.8181", "llm": "0.8322", "full": "0.7659", "delta": "0.0141", "minus": "0.001277", "plus": "0.001239"},
    ("stability_2024", "catboost"): {"classical": "0.8509", "llm": "0.8716", "full": "0.8198", "delta": "0.0207", "minus": "0.003481", "plus": "0.002451"},
}

# ---------------------------------------------------------------------------- Table 4 (tab:confirmatory), DEV rule
TABLE4 = [
    {"dataset": "homecredit", "backbone": "lr", "llm_leader": "LLM then mRMR", "llm_dev": "0.7472", "llm_ho": "0.7432", "classical_leader": "IV then Boruta", "classical_dev": "0.7652", "classical_ho": "0.7534",
     "delta": "-0.010179", "ci_low": "-0.012863", "ci_high": "-0.007592"},
    {"dataset": "homecredit", "backbone": "catboost", "llm_leader": "Pure LLM", "llm_dev": "0.7917", "llm_ho": "0.7935", "classical_leader": "IV then Boruta", "classical_dev": "0.7770", "classical_ho": "0.7688",
     "delta": "+0.024700", "ci_low": "+0.021656", "ci_high": "+0.027770"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "llm_leader": "Pure LLM", "llm_dev": "0.7436", "llm_ho": "0.7402", "classical_leader": "RFE CatBoost", "classical_dev": "0.7211", "classical_ho": "0.6989",
     "delta": "+0.041353", "ci_low": "+0.040619", "ci_high": "+0.042072"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "llm_leader": "LLM then mRMR", "llm_dev": "0.7811", "llm_ho": "0.7707", "classical_leader": "RFE CatBoost", "classical_dev": "0.7347", "classical_ho": "0.7190",
     "delta": "+0.051702", "ci_low": "+0.050900", "ci_high": "+0.052499"},
    {"dataset": "stability_2024", "backbone": "lr", "llm_leader": "Pure LLM", "llm_dev": "0.8302", "llm_ho": "0.8322", "classical_leader": "RFE CatBoost", "classical_dev": "0.8178", "classical_ho": "0.8181",
     "delta": "+0.0141", "ci_low": "+0.012823", "ci_high": "+0.015339"},
    {"dataset": "stability_2024", "backbone": "catboost", "llm_leader": "Pure LLM", "llm_dev": "0.8598", "llm_ho": "0.8716", "classical_leader": "RFE CatBoost", "classical_dev": "0.8635", "classical_ho": "0.8425",
     "delta": "+0.0291", "ci_low": "+0.027974", "ci_high": "+0.030174"},
]
# Section 6.1 mixed rule: Home Credit LR pairs LLM then mRMR with mRMR, point estimate only
MIXED_RULE_HC_LR = {"llm_leader": "LLM then mRMR", "classical_leader": "mRMR", "delta": "-0.027", "interval": None}
BOOTSTRAP = {"resamples": "2000", "seed": "20260721"}

# ---------------------------------------------------------------------------- Table 7 (tab:obfuscation), Home Credit only
TABLE7 = [
    {"arm": "A", "backbone": "lr", "K": 20, "ho_auc": "0.7424", "named": "0.7436", "delta": "-0.0012", "fold_min": "0.721", "fold_max": "0.741",
     "overlap_full_dev": "11", "overlap_folds_min": "12", "overlap_folds_max": "18", "overlap_folds_mean": "14.2"},
    {"arm": "A", "backbone": "catboost", "K": 40, "ho_auc": "0.7977", "named": "0.7935", "delta": "+0.0042", "fold_min": "0.751", "fold_max": "0.785",
     "overlap_full_dev": "17", "overlap_folds_min": "15", "overlap_folds_max": "28", "overlap_folds_mean": "23.2"},
    {"arm": "B", "backbone": "lr", "K": 20, "ho_auc": "0.7146", "named": "0.7436", "delta": "-0.0290", "fold_min": "0.706", "fold_max": "0.735",
     "overlap_full_dev": "14", "overlap_folds_min": "14", "overlap_folds_max": "19", "overlap_folds_mean": "16.2"},
    {"arm": "B", "backbone": "catboost", "K": 40, "ho_auc": "0.7625", "named": "0.7935", "delta": "-0.0310", "fold_min": "0.731", "fold_max": "0.753",
     "overlap_full_dev": "24", "overlap_folds_min": "26", "overlap_folds_max": "33", "overlap_folds_mean": "28.8"},
]
# Table 7 note: ranks of EXT_SOURCE_1/2/3 in the full-DEV call, and pairwise Jaccard among the five fold lists at K=20
TABLE7_EXT_RANKS = {"named": {"EXT_SOURCE_1": 1, "EXT_SOURCE_2": 2, "EXT_SOURCE_3": 3},
                    "A": {"EXT_SOURCE_1": 12, "EXT_SOURCE_2": 57, "EXT_SOURCE_3": 13},
                    "B": {"EXT_SOURCE_1": 6, "EXT_SOURCE_2": 7, "EXT_SOURCE_3": 8}}
TABLE7_FOLD_JACCARD_K20 = {"A": ("0.43", "1.00", "0.60"), "B": ("0.67", "1.00", "0.78"), "named": ("0.25", "0.67", "0.41")}  # (min, max, mean)
OBFUSCATION_SEED = "20260914"

# ---------------------------------------------------------------------------- Table 8 (tab:mechanical), descriptions built from lineage
TABLE8 = [
    {"dataset": "homecredit", "backbone": "lr", "selector": "Pure LLM", "K": 20, "mechanical": "0.7398", "cached": "0.7436", "delta": "-0.0039", "ci_low": "-0.0053", "ci_high": "-0.0017", "overlap": "12/20"},
    {"dataset": "homecredit", "backbone": "lr", "selector": "Stable core + LLM fill", "K": 20, "mechanical": "0.7471", "cached": "0.7489", "delta": "-0.0018", "ci_low": "-0.0029", "ci_high": "-0.0007", "overlap": "17/20"},
    {"dataset": "homecredit", "backbone": "catboost", "selector": "Pure LLM", "K": 40, "mechanical": "0.7953", "cached": "0.7935", "delta": "+0.0019", "ci_low": "+0.0013", "ci_high": "+0.0024", "overlap": "24/40"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "selector": "Pure LLM", "K": 20, "mechanical": "0.7444", "cached": "0.7402", "delta": "+0.0041", "ci_low": "+0.0027", "ci_high": "+0.0063", "overlap": "12/20"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "selector": "Pure LLM", "K": 40, "mechanical": "0.7271", "cached": "0.7335", "delta": "-0.0064", "ci_low": "-0.0072", "ci_high": "-0.0057", "overlap": "24/40"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "selector": "LLM then mRMR", "K": 40, "mechanical": "0.7773", "cached": "0.7707", "delta": "+0.0066", "ci_low": "+0.0056", "ci_high": "+0.0074", "overlap": "26/40"},
    {"dataset": "stability_2024", "backbone": "lr", "selector": "Pure LLM", "K": 20, "mechanical": "0.8396", "cached": "0.8322", "delta": "+0.0074", "ci_low": "+0.0034", "ci_high": "+0.0086", "overlap": "15/20"},
    {"dataset": "stability_2024", "backbone": "catboost", "selector": "Pure LLM", "K": 40, "mechanical": "0.8742", "cached": "0.8716", "delta": "+0.0026", "ci_low": "+0.0005", "ci_high": "+0.0049", "overlap": "30/40"},
]

# ---------------------------------------------------------------------------- Table 9 (tab:stability), fold-to-fold reproducibility
D_OF = {"homecredit": 373, "lendingclub_v2": 675, "stability_2024": 1068}
TABLE9 = [
    # most reproducible refitted classical selector of each case
    {"dataset": "homecredit", "backbone": "lr", "selector": "IV/WOE", "K": 20, "nogueira": "0.8838", "jaccard": "0.8053", "min": "0.667", "max": "0.905"},
    {"dataset": "homecredit", "backbone": "catboost", "selector": "IV/WOE", "K": 40, "nogueira": "0.8403", "jaccard": "0.7538", "min": "0.667", "max": "0.905"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "selector": "Boruta RF", "K": 20, "nogueira": "0.9588", "jaccard": "0.9273", "min": "0.818", "max": "1.000"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "selector": "mRMR", "K": 40, "nogueira": "0.9070", "jaccard": "0.8462", "min": "0.702", "max": "1.000"},
    {"dataset": "stability_2024", "backbone": "lr", "selector": "IV/WOE", "K": 20, "nogueira": "0.7860", "jaccard": "0.6666", "min": "0.481", "max": "0.905"},
    {"dataset": "stability_2024", "backbone": "catboost", "selector": "IV/WOE", "K": 40, "nogueira": "0.7793", "jaccard": "0.6582", "min": "0.509", "max": "0.905"},
    # LLM hybrids with a refitted statistical stage
    {"dataset": "homecredit", "backbone": "lr", "selector": "Stable core + LLM fill", "K": 20, "nogueira": "0.7411", "jaccard": "0.6110", "min": "0.481", "max": "0.818"},
    {"dataset": "homecredit", "backbone": "catboost", "selector": "Stable core + LLM fill", "K": 40, "nogueira": "0.6920", "jaccard": "0.5725", "min": "0.455", "max": "0.739"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "selector": "Stable core + LLM fill", "K": 20, "nogueira": "0.7115", "jaccard": "0.5783", "min": "0.379", "max": "0.818"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "selector": "Stable core + LLM fill", "K": 40, "nogueira": "0.7130", "jaccard": "0.5842", "min": "0.404", "max": "0.778"},
    {"dataset": "stability_2024", "backbone": "lr", "selector": "Stable core + LLM fill", "K": 20, "nogueira": "0.7554", "jaccard": "0.6259", "min": "0.429", "max": "0.905"},
    {"dataset": "stability_2024", "backbone": "catboost", "selector": "Stable core + LLM fill", "K": 40, "nogueira": "0.7091", "jaccard": "0.5727", "min": "0.356", "max": "0.778"},
    {"dataset": "homecredit", "backbone": "lr", "selector": "LLM then Boruta", "K": 20, "nogueira": "0.4294", "jaccard": "0.3031", "min": "0.212", "max": "0.429"},
    {"dataset": "homecredit", "backbone": "catboost", "selector": "LLM then Boruta", "K": 40, "nogueira": "0.4204", "jaccard": "0.3247", "min": "0.194", "max": "0.509"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "selector": "LLM then Boruta", "K": 20, "nogueira": "0.5414", "jaccard": "0.4103", "min": "0.143", "max": "0.739"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "selector": "LLM then Boruta", "K": 40, "nogueira": "0.6333", "jaccard": "0.4920", "min": "0.379", "max": "0.702"},
    {"dataset": "stability_2024", "backbone": "lr", "selector": "LLM then Boruta", "K": 20, "nogueira": "0.5414", "jaccard": "0.4081", "min": "0.212", "max": "0.905"},
    {"dataset": "stability_2024", "backbone": "catboost", "selector": "LLM then Boruta", "K": 40, "nogueira": "0.5528", "jaccard": "0.4193", "min": "0.290", "max": "0.951"},
    # pure LLM, cached per-fold lists (Home Credit and LendingClub only)
    {"dataset": "homecredit", "backbone": "lr", "selector": "Pure LLM", "K": 20, "nogueira": "0.5403", "jaccard": "0.4076", "min": "0.250", "max": "0.667"},
    {"dataset": "homecredit", "backbone": "catboost", "selector": "Pure LLM", "K": 40, "nogueira": "0.4092", "jaccard": "0.3149", "min": "0.231", "max": "0.509"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "selector": "Pure LLM", "K": 20, "nogueira": "0.5517", "jaccard": "0.4011", "min": "0.250", "max": "0.600"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "selector": "Pure LLM", "K": 40, "nogueira": "0.6280", "jaccard": "0.4828", "min": "0.404", "max": "0.538"},
]
VOTING_FOLD_JACCARD = {"voted": ("0.40", "0.49"), "reference": ("0.62", "0.72")}  # Section 7.2 / B.2

# ---------------------------------------------------------------------------- Table 10 (tab:response_spread)
TABLE10 = [
    {"dataset": "homecredit", "backbone": "lr", "K": 20, "draws": 5, "min": "0.7358", "max": "0.7410", "spread": "0.0053", "median": None, "sd": None, "margin": None},
    {"dataset": "homecredit", "backbone": "catboost", "K": 40, "draws": 5, "min": "0.7926", "max": "0.7998", "spread": "0.0072", "median": None, "sd": None, "margin": None},
    {"dataset": "lendingclub_v2", "backbone": "lr", "K": 20, "draws": 5, "min": "0.7106", "max": "0.7181", "spread": "0.0075", "median": None, "sd": None, "margin": None},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "K": 40, "draws": 5, "min": "0.7324", "max": "0.7362", "spread": "0.0038", "median": None, "sd": None, "margin": None},
    {"dataset": "stability_2024", "backbone": "lr", "K": 20, "draws": 10, "min": "0.8289", "max": "0.8374", "spread": "0.0085", "median": "0.8317", "sd": "0.0023",
     "margin": ("+0.0108", "+0.0193"), "classical_leader": "RFE CatBoost", "classical_auc": "0.8181"},
    {"dataset": "stability_2024", "backbone": "catboost", "K": 40, "draws": 10, "min": "0.8681", "max": "0.8731", "spread": "0.0050", "median": "0.8704", "sd": "0.0017",
     "margin": ("+0.0172", "+0.0222"), "classical_leader": "CatBoost SHAP", "classical_auc": "0.8509"},
]
# Table 10 note: full-DEV cached (headline) response HO AUC of pure LLM
TABLE10_HEADLINE = {("homecredit", "lr"): "0.7436", ("homecredit", "catboost"): "0.7935", ("lendingclub_v2", "lr"): "0.7402",
                    ("lendingclub_v2", "catboost"): "0.7335", ("stability_2024", "lr"): "0.8322", ("stability_2024", "catboost"): "0.8716"}
TABLE10_LC_LR = {"above_range": "0.022", "six_response_spread": "0.0296", "margin_over_iv_then_boruta": ("0.0008", "0.0083"),
                 "margin_over_rfe": ("0.0117", "0.0192"), "iv_then_boruta": "0.7098", "rfe": "0.6989"}
STABILITY_TEN = {"all_passed": True, "all_distinct": True, "cached_above_n_of_ten": 7, "margin_positive_in_all_refits": 20}

# ---------------------------------------------------------------------------- Table 11 (tab:secondary), best HO secondary metric per dataset
TABLE11 = {
    "homecredit": {"ks": ("0.4264", "Pure LLM", "catboost"), "brier": ("0.1234", "Pure LLM", "catboost"), "capture10": ("0.3630", "Pure LLM", "catboost"), "lift10": "3.63"},
    "lendingclub_v2": {"ks": ("0.3898", "LLM then mRMR", "catboost"), "brier": ("0.1824", "LLM then mRMR", "catboost"), "capture10": ("0.2751", "Pure LLM", "lr"), "lift10": "2.75"},
    "stability_2024": {"ks": ("0.5681", "Pure LLM", "catboost"), "brier": ("0.1538", "Pure LLM", "catboost"), "capture10": ("0.4417", "Pure LLM", "catboost"), "lift10": "4.42"},
}

# ---------------------------------------------------------------------------- Section 6.6.2 drift sentences
PSI_SENTENCES = {
    "mean_range": ("0.0045", "0.0189"), "no_feature_above": "0.10",
    "st_lr_median": ("0.0023", "0.0042"), "hc_cb_pure_llm_median": "0.0035", "st_cb_pure_llm_median": "0.0001",
    "max_features": [("0.098", "stability_2024", "catboost", "CatBoost SHAP", "d1__tax_registry_a__row_count"),
                     ("0.086", "stability_2024", "catboost", "Pure LLM", "d0__static__maxoutstandbalancel12m_4187113A"),
                     ("0.084", "stability_2024", "lr", "Pure LLM", "d0__static__numinstlswithoutdpd_562L")],
}

# ---------------------------------------------------------------------------- Section 6.7 scalability and cost
SECTION67 = {
    "mrmr_seconds_1959": "522", "mrmr_seconds_100": "21", "mrmr_factor": "25", "stability_fold": "fold1",
    "logical_requests": "72", "provider_calls": "30", "input_tokens": "806,757", "output_tokens": "56,495",
    "cost_usd": ("0.171", "0.413"), "prices_per_million": {"input": "0.40", "cached_input": "0.10", "output": "1.60"},
    "stability_tokens_recorded": False,
    "machine": {"logical_cores": "16", "physical_cores": "10", "ram_gib": "39.6", "estimator_threads": "4", "cpu": "Intel Core i7-13620H"},
    "peak_rss_gb": {"homecredit": ("7.4", "10.7"), "lendingclub_v2": ("19.1", "29.0"), "stability_2024": ("27.0", "29.0")},
}
# Appendix F "Unmatched wall-clock times" (minutes)
APPENDIX_F_RUNTIME = {
    "homecredit": {"lr": {"Pure LLM": "0.8", "mRMR (RF/correlation implementation)": "12.1", "Stable core + LLM fill": "24.6"}},
    "lendingclub_v2": {"catboost": {"Domain rules": "49.7", "mRMR": "65.3", "LLM then mRMR": "72.8", "Pure LLM": "73.8", "PCA": "88.0",
                                    "LLM then Boruta": "110.5", "Boruta RF": "172.6", "Stable core + LLM fill": "256.0",
                                    "Stable core + LLM fill (selection stage)": "176.6"}},
}

# ---------------------------------------------------------------------------- Table B.1 (tab:voting_cross), classical rank voting
TABLEB1 = [
    {"dataset": "homecredit", "backbone": "lr", "P": 100, "ref": "0.7448", "voting": "0.7493", "delta": "+0.0045", "ci_low": "0.0026", "ci_high": "0.0065", "holm_p": "2.0e-5", "delta_ks": "+0.0071"},
    {"dataset": "homecredit", "backbone": "lr", "P": 200, "ref": "0.7448", "voting": "0.7492", "delta": "+0.0045", "ci_low": "0.0025", "ci_high": "0.0064", "holm_p": "3.2e-5", "delta_ks": "+0.0108"},
    {"dataset": "homecredit", "backbone": "lr", "P": 300, "ref": "0.7448", "voting": "0.7467", "delta": "+0.0020", "ci_low": "0.0002", "ci_high": "0.0038", "holm_p": "0.031", "delta_ks": "+0.0090"},
    {"dataset": "homecredit", "backbone": "catboost", "P": 100, "ref": "0.7698", "voting": "0.7791", "delta": "+0.0093", "ci_low": "0.0077", "ci_high": "0.0109", "holm_p": "<1e-10", "delta_ks": "+0.0127"},
    {"dataset": "homecredit", "backbone": "catboost", "P": 200, "ref": "0.7698", "voting": "0.7786", "delta": "+0.0088", "ci_low": "0.0070", "ci_high": "0.0105", "holm_p": "<1e-10", "delta_ks": "+0.0113"},
    {"dataset": "homecredit", "backbone": "catboost", "P": 300, "ref": "0.7698", "voting": "0.7799", "delta": "+0.0100", "ci_low": "0.0082", "ci_high": "0.0119", "holm_p": "<1e-10", "delta_ks": "+0.0127"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "P": 100, "ref": "0.6915", "voting": "0.6977", "delta": "+0.0062", "ci_low": "0.0055", "ci_high": "0.0069", "holm_p": "<1e-10", "delta_ks": "+0.0086"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "P": 200, "ref": "0.6915", "voting": "0.6951", "delta": "+0.0036", "ci_low": "0.0030", "ci_high": "0.0041", "holm_p": "<1e-10", "delta_ks": "+0.0063"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "P": 300, "ref": "0.6915", "voting": "0.6990", "delta": "+0.0075", "ci_low": "0.0067", "ci_high": "0.0082", "holm_p": "<1e-10", "delta_ks": "+0.0116"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "P": 100, "ref": "0.7072", "voting": "0.7140", "delta": "+0.0068", "ci_low": "0.0062", "ci_high": "0.0074", "holm_p": "<1e-10", "delta_ks": "+0.0097"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "P": 200, "ref": "0.7072", "voting": "0.7173", "delta": "+0.0101", "ci_low": "0.0095", "ci_high": "0.0108", "holm_p": "<1e-10", "delta_ks": "+0.0156"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "P": 300, "ref": "0.7072", "voting": "0.7201", "delta": "+0.0129", "ci_low": "0.0122", "ci_high": "0.0136", "holm_p": "<1e-10", "delta_ks": "+0.0202"},
]

# ---------------------------------------------------------------------------- Table C.1 (tab:budgets) and Figure C.1
TABLEC1 = {  # LLM-assisted leader minus classical leader, DEV AUC, by K = 10, 20, 30, 40, 50
    ("homecredit", "lr"): ("-0.0140", "-0.0180", "-0.0214", "-0.0186", "-0.0178"),
    ("homecredit", "catboost"): ("+0.0233", "+0.0175", "+0.0141", "+0.0147", "+0.0211"),
    ("lendingclub_v2", "lr"): ("+0.0150", "+0.0225", "+0.0236", "+0.0243", "+0.0259"),
    ("lendingclub_v2", "catboost"): ("+0.0427", "+0.0415", "+0.0401", "+0.0464", "+0.0402"),
    ("stability_2024", "lr"): ("-0.0019", "+0.0124", "+0.0009", "+0.0190", "+0.0109"),
    ("stability_2024", "catboost"): ("-0.0012", "-0.0131", "-0.0147", "-0.0037", "-0.0080"),
}
FIGUREC1B = {  # same quantity at the figure's precision
    ("homecredit", "lr"): ("-0.013960", "-0.018022", "-0.021352", "-0.018588", "-0.017754"),
    ("homecredit", "catboost"): ("0.023343", "0.017522", "0.014054", "0.014717", "0.021106"),
    ("lendingclub_v2", "lr"): ("0.014980", "0.022480", "0.023558", "0.024256", "0.025945"),
    ("lendingclub_v2", "catboost"): ("0.042744", "0.041458", "0.040106", "0.046391", "0.040232"),
    ("stability_2024", "lr"): ("-0.0019", "0.0124", "0.0009", "0.0190", "0.0109"),
    ("stability_2024", "catboost"): ("-0.0012", "-0.0131", "-0.0147", "-0.0037", "-0.0080"),
}
FIGUREC1A = {  # mean DEV AUC of all selectors of a backbone relative to its documented budget (pooled over datasets)
    "lr": {10: "-0.014735", 20: "0.000000", 30: "0.004031", 40: "0.004990", 50: "0.004506"},
    "catboost": {10: "-0.018775", 20: "-0.008273", 30: "-0.001311", 40: "0.000000", 50: "-0.000587"},
}
APPENDIX_C_TEXT = {"lr_gain_k30": "+0.004", "lr_gain_k40": "+0.005", "cb_k50": "-0.0006",
                   "cb_over_lr_k40": ("+0.020", "+0.036"), "cb_over_lr_all": ("+0.018", "+0.036"),
                   "unmatched_gap": ("+0.026", "+0.040"), "budget_share_pct": ("10", "24")}

# ---------------------------------------------------------------------------- Table F.1 (tab:lc_maturity), LendingClub sub-populations
TABLEF1 = [
    {"subpopulation": "matured_36m_2016-01_to_2016-05", "n": "123,898", "lr": "+0.0183", "catboost": "+0.0112",
     "lr_ci": ("+0.0159", "+0.0207"), "catboost_ci": ("+0.0099", "+0.0126")},
    {"subpopulation": "term_36", "n": "232,361", "lr": "+0.0215", "catboost": "+0.0122"},
    {"subpopulation": "term_60", "n": "60,744", "lr": "+0.0141", "catboost": "+0.0101"},
]
TABLEF1_MONTHS_POSITIVE = 12  # "positive in all twelve months of 2016 under both backbones"; every interval excludes zero
LC_MATURITY_TEXT = {"robustness_row": {"matured": ("+0.018", "+0.011"), "term36": ("+0.022", "+0.012"), "term60": ("+0.014", "+0.010")}}

# ---------------------------------------------------------------------------- Table F.2 (tab:diversity)
TABLEF2 = [
    {"dataset": "homecredit", "backbone": "lr", "rule": "family_cap", "base": "mRMR", "K": 20, "control": "0.7457", "uncapped": "0.7699", "llm": "0.7489", "difference": "-0.003"},
    {"dataset": "homecredit", "backbone": "catboost", "rule": "family_cap", "base": "RFE CatBoost", "K": 40, "control": "0.7721", "uncapped": "0.7814", "llm": "0.7935", "difference": "-0.021"},
    {"dataset": "lendingclub_v2", "backbone": "lr", "rule": "family_cap", "base": "IV then Boruta", "K": 20, "control": "0.6849", "uncapped": "0.7098", "llm": "0.7402", "difference": "-0.055"},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "rule": "family_cap", "base": "IV then Boruta", "K": 40, "control": "0.7065", "uncapped": "0.7201", "llm": "0.7707", "difference": "-0.064"},
    {"dataset": "stability_2024", "backbone": "lr", "rule": "family_cap", "base": "RFE CatBoost", "K": 20, "control": "0.7494", "uncapped": "0.8181", "llm": "0.8322", "difference": "-0.083"},
    {"dataset": "stability_2024", "backbone": "catboost", "rule": "family_cap", "base": "CatBoost SHAP", "K": 40, "control": "0.8289", "uncapped": "0.8509", "llm": "0.8716", "difference": "-0.043"},
    {"dataset": "stability_2024", "backbone": "lr", "rule": "depth0", "base": "RFE CatBoost", "K": 20, "control": "0.7804", "uncapped": "0.8181", "llm": "0.8322", "difference": "-0.052"},
    {"dataset": "stability_2024", "backbone": "catboost", "rule": "depth0", "base": "CatBoost SHAP", "K": 40, "control": "0.8231", "uncapped": "0.8509", "llm": "0.8716", "difference": "-0.049"},
]
FAMILY_CAPS = {"homecredit": {"F": 6, "lr": 4, "catboost": 7}, "lendingclub_v2": {"F": 17, "lr": 2, "catboost": 3}, "stability_2024": {"F": 12, "lr": 2, "catboost": 4}}

# ---------------------------------------------------------------------------- Table F.3 (tab:dimension), effective dimension
TABLEF3 = [
    {"dataset": "homecredit", "backbone": "lr", "selector": "Stable core + LLM fill", "K": 20, "n_categorical": 0, "columns": 20},
    {"dataset": "homecredit", "backbone": "lr", "selector": "mRMR", "K": 20, "n_categorical": 3, "columns": 33},
    {"dataset": "homecredit", "backbone": "catboost", "selector": "Pure LLM", "K": 40, "n_categorical": 5, "columns": 72},
    {"dataset": "homecredit", "backbone": "catboost", "selector": "RFE CatBoost", "K": 40, "n_categorical": 3, "columns": 50},
    {"dataset": "lendingclub_v2", "backbone": "lr", "selector": "Pure LLM", "K": 20, "n_categorical": 5, "columns": 43},
    {"dataset": "lendingclub_v2", "backbone": "lr", "selector": "IV then Boruta", "K": 20, "n_categorical": 5, "columns": 48},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "selector": "LLM then mRMR", "K": 40, "n_categorical": 0, "columns": 40},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "selector": "IV then Boruta", "K": 40, "n_categorical": 7, "columns": 77},
    {"dataset": "stability_2024", "backbone": "lr", "selector": "Pure LLM", "K": 20, "n_categorical": 0, "columns": 20},
    {"dataset": "stability_2024", "backbone": "lr", "selector": "RFE CatBoost", "K": 20, "n_categorical": 1, "columns": 28},
    {"dataset": "stability_2024", "backbone": "catboost", "selector": "Pure LLM", "K": 40, "n_categorical": 0, "columns": 40},
    {"dataset": "stability_2024", "backbone": "catboost", "selector": "CatBoost SHAP", "K": 40, "n_categorical": 9, "columns": 174},
]
TABLEF3_NOTE = {"lc_llm_then_mrmr_onehot_slots": 13, "lc_llm_then_mrmr_original_variables_in_onehot": 7, "lc_llm_then_mrmr_original_variables": 34}

# ---------------------------------------------------------------------------- Table F.4 (tab:response_lists)
TABLEF4 = [
    {"dataset": "homecredit", "backbone": "lr", "K": 20, "draws": 5, "nogueira": "0.5403", "jaccard": "0.4076", "in_all": 6, "distinct": 42},
    {"dataset": "homecredit", "backbone": "catboost", "K": 40, "draws": 5, "nogueira": "0.4092", "jaccard": "0.3149", "in_all": 8, "distinct": 98},
    {"dataset": "lendingclub_v2", "backbone": "lr", "K": 20, "draws": 5, "nogueira": "0.5517", "jaccard": "0.4011", "in_all": 4, "distinct": 39},
    {"dataset": "lendingclub_v2", "backbone": "catboost", "K": 40, "draws": 5, "nogueira": "0.6280", "jaccard": "0.4828", "in_all": 14, "distinct": 71},
    {"dataset": "stability_2024", "backbone": "lr", "K": 20, "draws": 10, "nogueira": "0.8052", "jaccard": "0.6908", "in_all": 11, "distinct": 30},
    {"dataset": "stability_2024", "backbone": "catboost", "K": 40, "draws": 10, "nogueira": "0.7766", "jaccard": "0.6506", "in_all": 22, "distinct": 69},
]

# ---------------------------------------------------------------------------- Appendix F paragraphs
NATIVE_CATEGORICAL = {  # CatBoost cases re-run with native categorical handling
    "homecredit": {"Pure LLM": "0.7918", "RFE CatBoost": "0.7816", "margin": "+0.010"},
    "lendingclub_v2": {"LLM then mRMR": "0.7738", "IV then Boruta": "0.7203", "margin": "+0.054"},
    "stability_2024": {"Pure LLM": "0.8716", "CatBoost SHAP": "0.8532", "margin": "+0.018", "pure_llm_unchanged_by_construction": True},
    "max_move": "0.005",
}
L1_SELECTOR = {"ho_auc_range": ("0.674", "0.842"), "shortfall_lc_catboost": "0.005", "shortfall_st_catboost": "0.009", "shortfall_elsewhere": ("0.026", "0.096"),
               "settings": {"solver": "liblinear", "C": "0.05", "class_weight": "balanced"}}
BORUTA_ATTAINMENT = {"at_budget": 104, "refits": 108, "below_budget_cells": [
    ("stability_2024", "catboost", "LLM then Boruta", "fold1", 35, 40),
    ("homecredit", "catboost", "Boruta RF", "fold1", 26, 40),
    ("homecredit", "catboost", "IV then Boruta", "fold1", 20, 40),
    ("homecredit", "catboost", "IV then Boruta", "fold2", 39, 40)]}
STABILITY_MONTHLY = {"counts": {"2020-02": "10,952", "2020-03": "60,748", "2020-04": "8,289", "2020-05": "28,725", "2020-06": "45,962",
                                "2020-07": "28,912", "2020-08": "50,831", "2020-09": "61,905", "2020-10": "8,592"},
                     "delta_range_lr": ("+0.010", "+0.020"), "delta_range_catboost": ("+0.012", "+0.029"), "classical_lr": "RFE CatBoost", "classical_catboost": "CatBoost SHAP"}
WITHIN_FAMILY = {"six_classical_vs_four_llm_gap": ("0.010", "0.027"), "hc_lr_classical_above": "0.007",
                 "pure_llm_within_leader_hc_lr": "0.006", "pure_llm_trails_llm_then_mrmr_lc_cb": "0.037",
                 "st_llm_spread_lr": "0.022", "st_llm_spread_catboost": "0.033"}
DEV_TO_HO = {"classical_fall": ("0.008", "0.022"), "llm_move": ("-0.010", "+0.012"), "llm_rise_cases": 3,
             "dev_chosen_classical_short": ("0.001", "0.017"), "dev_chosen_llm_short_hc_lr": "0.006"}

# ---------------------------------------------------------------------------- Appendix D / supplementary (headline subsets), Section 7.5
SUBSET_OVERLAPS = {"homecredit": {"lr": "4/20", "catboost": "16/40"}, "lendingclub_v2": {"lr": "4/20", "catboost": "21/40"}, "stability_2024": {"lr": "1/20", "catboost": "4/40"}}
PROTECTED_RANKS = [  # (dataset, backbone, selector, feature, rank)
    ("homecredit", "lr", "mRMR", "CODE_GENDER", 3),
    ("homecredit", "catboost", "RFE CatBoost", "CODE_GENDER", 14),
    ("homecredit", "catboost", "Pure LLM", "CODE_GENDER", 10),
    ("homecredit", "catboost", "Pure LLM", "NAME_FAMILY_STATUS", 9),
    ("homecredit", "catboost", "RFE CatBoost", "NAME_FAMILY_STATUS", 39),
    ("stability_2024", "catboost", "CatBoost SHAP", "d1__person__sex_738L__lexical_mode", 1),
    ("stability_2024", "catboost", "CatBoost SHAP", "d1__person__sex_738L__first_by_num_group1", 6),
]
PROTECTED_FEATURES = {"homecredit": ("CODE_GENDER", "NAME_FAMILY_STATUS"), "lendingclub_v2": (), "stability_2024": ("d1__person__sex_738L__lexical_mode", "d1__person__sex_738L__first_by_num_group1")}
SOURCE_CONCENTRATION = {"st_pure_llm_all_static_0": True, "hc_cb_pure_llm_bureau": 26}
LLM_PROTOCOL = {"model_snapshot": "gpt-4.1-mini-2025-04-14", "temperature": "0", "evidence_mode": "definitions_only",
                "hc_calls": 6, "hc_names_per_call": 100, "lc_calls": 24, "lc_budgets": (20, 40, 60, 100), "st_calls": 1, "st_candidates": 1068, "st_repeats": 10,
                "max_attempts": 3, "seeds": {"classifiers_selectors": 42, "stable_core": "42-46", "bootstrap": 20260721, "obfuscation": 20260914}}
MATERIALITY_BAR = "0.010"

# ---------------------------------------------------------------------------- known discrepancies
# Each entry: (table, where, column, reason).  The verification script reports these as FLAGGED instead of MISMATCH.
# Empty: result files regenerated by scripts/run_study.py are compared value by value, and no mismatch is downgraded.
KNOWN_DISCREPANCIES: list[tuple[str, str, str, str]] = []

# ---------------------------------------------------------------------------- prose claims derived from table values
# Ranges and differences printed in the text; the manuscript derives them from the printed (rounded) table values.
TEXT_CLAIMS = {
    "hc_cb_ho_ci_3dp": ("+0.009", "+0.015"),                # Section 6.1: HC CatBoost interval reaching below the bar
    "hc_lr_dev_ci_3dp": ("-0.013", "-0.008"),               # Section 6.1: HC LR DEV-rule interval
    "lc_event_rate_dev_ho_pct": ("19.5", "23.3"),           # Appendix F: LendingClub HO event rate rises from 19.5% to 23.3%
    "family_cap_fall": ("0.009", "0.069"),                  # Table 6 row / Appendix F: classical leader falls under the cap
    "family_cap_gap_five": ("0.021", "0.083"),              # ... and stays this far below the LLM-assisted leader in 5 of 6
    "mechanical_lead": ("+0.014", "+0.057"),                # Section 6.5.2: winning leaders with rebuilt descriptions
    "mechanical_hc_lr_vs_mrmr": "-0.023",                   # Section 6.5.2: stable core (mechanical) minus mRMR on HC LR
    "mechanical_widest_bound": "+0.009",                    # Section 6.5.2: widest interval bound of Table 8
    "pure_llm_nogueira": ("0.41", "0.63"),                  # Section 6.6.1 (Home Credit and LendingClub pure LLM rows)
    "pure_llm_jaccard": ("0.31", "0.48"),
    "pure_llm_pair_spread": ("0.13", "0.42"),               # fold-pair spread (max - min) of the pure LLM lists
    "classical_pair_spread": ("0.18", "0.42"),              # ... and of the most reproducible classical selectors
    "classical_above_pure_llm": ("0.28", "0.43"),           # Nogueira, Home Credit and LendingClub
    "stable_core_nogueira": ("0.69", "0.76"),               # Section 6.6.1, six cases
    "stable_core_jaccard": ("0.57", "0.63"),
    "stable_core_below_classical": ("0.03", "0.25"),
    "stable_core_above_pure_llm": ("0.08", "0.28"),         # Home Credit and LendingClub
    "llm_then_boruta_nogueira": ("0.42", "0.63"),
    "st_llm_then_boruta_below_stable_core": ("0.16", "0.21"),
    "st_llm_then_boruta_below_classical": ("0.23", "0.24"),
    "hc_lr_llm_then_boruta_below_pure_llm": "0.11",
}
