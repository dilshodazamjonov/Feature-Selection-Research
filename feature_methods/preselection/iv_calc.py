# iv_filter.py
"""
Simple, research-oriented IVFilter class.

Features:
- Vectorized IV / WOE computation using quantile binning (pd.qcut).
- Fits with fit(X, y) and transforms with transform(X) to keep pipeline fit/transform pattern.
- Saves IV table and per-feature statistics to data/output/ for inclusion in research appendix.
- Simple leakage flags: IV > max_iv_for_leakage and perfect bin separation.
- Does NOT plot every feature by default (keeps file/CSV outputs suitable for papers).

- Main Methods to use:
    iv = IVFilter(...) — create the object, set n_bins, min_iv, etc.
    iv.fit(X_train, y_train) — compute IVs, save CSV reports, create iv_table_ and selected_features_.
    X_sel = iv.transform(X) — reduce X to selected features (fills missing columns with zeros).
    iv.fit_transform(X, y) — convenience wrapper for fit then transform.

"""

import os
from typing import Optional, Dict
import numpy as np
import pandas as pd

OUTPUT_DIR = "data/output"
IV_SUMMARY_DIR = os.path.join(OUTPUT_DIR, "iv_summaries")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IV_SUMMARY_DIR, exist_ok=True)


class IVFilter:
    def __init__(
        self,
        target_col: Optional[str] = None,
        n_bins: int = 10,
        min_iv: float = 0.02,
        max_iv_for_leakage: float = 0.5,
        min_bin_pct: Optional[float] = None,  # optional minimum bin fraction check (e.g. 0.05)
        verbose: bool = True,
        save_bin_level_stats: bool = True,    # whether to save per-bin long table (useful for appendix)
    ):
        """
        Params:
            target_col: optional name of the target 
            n_bins: quantile bins for numeric variables
            min_iv: threshold to keep a feature
            max_iv_for_leakage: flag features with IV >= this value for manual inspection
            min_bin_pct: if set, feature bins with < min_bin_pct will be noted (not merged)
            verbose: print progress
            save_bin_level_stats: save long table with bin-level statistics for all features
        """
        self.target_col = target_col
        self.n_bins = n_bins
        self.min_iv = min_iv
        self.max_iv_for_leakage = max_iv_for_leakage
        self.min_bin_pct = min_bin_pct
        self.verbose = verbose
        self.save_bin_level_stats = save_bin_level_stats

        self.iv_table_: Optional[pd.DataFrame] = None
        self.selected_features_: Optional[list] = None
        self._per_feature_stats: Dict[str, pd.DataFrame] = {}

    # -------------------------
    # Helper / core functions
    # -------------------------
    @staticmethod
    def _safe_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))

    @staticmethod
    def _compute_group_stats_from_codes(codes, y_arr):
        """Vectorized counts and proportions per group id (codes are 0..k-1 integers)."""
        codes = np.asarray(codes)
        y_arr = np.asarray(y_arr).astype(int)
        # unique order
        uniq, inv = np.unique(codes, return_inverse=True)
        counts = np.bincount(inv)
        bads = np.bincount(inv, weights=y_arr)
        goods = counts - bads

        total_bad = bads.sum()
        total_good = goods.sum()

        bad_pct = IVFilter._safe_div(bads, total_bad)
        good_pct = IVFilter._safe_div(goods, total_good)

        df = pd.DataFrame(
            {
                "group": uniq,
                "count": counts,
                "bad": bads,
                "good": goods,
                "bad_pct": bad_pct,
                "good_pct": good_pct,
            }
        ).set_index("group")
        return df

    @staticmethod
    def _woe_from_stats(stats_df):
        # compute WOE with small eps to avoid log(0)
        eps = 1e-12
        woe = np.log((stats_df["good_pct"] + eps) / (stats_df["bad_pct"] + eps))
        return woe

    @staticmethod
    def _iv_from_stats(stats_df):
        woe = IVFilter._woe_from_stats(stats_df)
        iv_bin = (stats_df["good_pct"] - stats_df["bad_pct"]) * woe
        return iv_bin.sum(), woe, iv_bin

    # -------------------------
    # Per-feature computation
    # -------------------------
    def _process_numeric_feature(self, series: pd.Series, y: pd.Series):
        # qcut into quantile bins; fallback to cut if qcut fails
        try:
            bins = pd.qcut(series, q=self.n_bins, duplicates="drop")
        except Exception:
            bins = pd.cut(series, bins=self.n_bins)

        cat = pd.Categorical(bins)
        codes = cat.codes  # -1 for NaN
        if (codes == -1).any():
            # map -1 -> max_code
            nan_mask = codes == -1
            codes = codes.copy()
            codes[nan_mask] = codes.max() + 1
        # compute stats
        stats = IVFilter._compute_group_stats_from_codes(codes, y)

        # optionally mark small bins
        if self.min_bin_pct is not None:
            total = stats["count"].sum()
            low_pct = (stats["count"] / total) < self.min_bin_pct
            if low_pct.any():
                stats["_small_bin_flag"] = low_pct.astype(int)

        iv_val, woe, iv_bin = IVFilter._iv_from_stats(stats)
        stats = stats.assign(woe=woe, iv_bin=iv_bin)
        return float(iv_val), stats

    def _process_categorical_feature(self, series: pd.Series, y: pd.Series):
        s = series.fillna("missing").astype(object)
        # treat rare categories as-is (we do not merge in this simple research version)
        cat = pd.Categorical(s)
        codes = cat.codes
        stats = IVFilter._compute_group_stats_from_codes(codes, y)
        if self.min_bin_pct is not None:
            total = stats["count"].sum()
            low_pct = (stats["count"] / total) < self.min_bin_pct
            if low_pct.any():
                stats["_small_bin_flag"] = low_pct.astype(int)
        iv_val, woe, iv_bin = IVFilter._iv_from_stats(stats)
        stats = stats.assign(woe=woe, iv_bin=iv_bin)
        # store mapping to category labels for possible reporting
        stats["_categories"] = list(pd.Categorical(s).categories)
        return float(iv_val), stats

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Compute IV for each feature in X using binary target y (0/1).
        Stores iv_table_ (feature -> IV), selected_features_ (IV >= min_iv),
        and per-feature stats in self._per_feature_stats (for reporting).
        Produces CSV reports under data/output/ suitable for inclusion in research appendix:
          - iv_table.csv (feature, IV)
          - iv_selected_features.csv
          - iv_dropped_features.csv
          - iv_feature_summaries.csv (one-line summary per feature: num_bins, max_woe, mean_woe, max_bad_pct, etc.)
          - iv_bin_level_stats.csv (long table with per-bin statistics for all features) [optional]
          - iv_leakage_flags.csv (leakage heuristics)
        """
        ivs = {}
        stats_store = {}
        y_arr = np.asarray(y)

        for col in X.columns:
            series = X[col]
            if pd.api.types.is_numeric_dtype(series):
                iv_val, stats = self._process_numeric_feature(series, y_arr)
            else:
                iv_val, stats = self._process_categorical_feature(series, y_arr)

            ivs[col] = iv_val
            stats_store[col] = stats
            if self.verbose:
                print(f"[IV] {col}: {iv_val:.6f}")

        # -------------------------
        # Save IV table and selected/dropped lists
        # -------------------------
        iv_df = pd.DataFrame.from_dict(ivs, orient="index", columns=["IV"])
        iv_df = iv_df.sort_values("IV", ascending=False)
        self.iv_table_ = iv_df
        self.selected_features_ = iv_df[iv_df["IV"] >= self.min_iv].index.tolist()
        self._per_feature_stats = stats_store

        # save IV table
        iv_out = os.path.join(IV_SUMMARY_DIR, "iv_table.csv")
        iv_df.to_csv(iv_out)
        if self.verbose:
            print(f"[IV] saved IV table -> {iv_out}")

        # save selected features list
        selected_out = os.path.join(IV_SUMMARY_DIR, "iv_selected_features.csv")
        pd.Series(self.selected_features_, name="selected_feature").to_frame().to_csv(selected_out, index=False)
        if self.verbose:
            print(f"[IV] saved selected features -> {selected_out}")

        # save dropped features list (IV < min_iv)
        dropped = iv_df[iv_df["IV"] < self.min_iv]
        dropped_out = os.path.join(IV_SUMMARY_DIR, "iv_dropped_features.csv")
        dropped.to_csv(dropped_out)
        if self.verbose:
            print(f"[IV] saved dropped features -> {dropped_out}")

        # -------------------------
        # Produce per-feature summary (one line per feature) for paper appendix
        # -------------------------
        summaries = []
        for col, stats in stats_store.items():
            num_bins = int(len(stats))
            max_woe = float(stats["woe"].max()) if "woe" in stats.columns else np.nan
            min_woe = float(stats["woe"].min()) if "woe" in stats.columns else np.nan
            mean_woe = float(stats["woe"].mean()) if "woe" in stats.columns else np.nan
            max_bad_pct = float(stats["bad_pct"].max())
            mean_bad_pct = float(stats["bad_pct"].mean())
            max_iv_bin = float(stats["iv_bin"].max()) if "iv_bin" in stats.columns else np.nan
            perfect_sep = bool((stats["bad_pct"] == 0).any() or (stats["good_pct"] == 0).any())
            small_bin_flag = int(stats["_small_bin_flag"].any()) if "_small_bin_flag" in stats.columns else 0

            summaries.append({
                "feature": col,
                "IV": float(ivs[col]),
                "num_bins": num_bins,
                "max_woe": max_woe,
                "min_woe": min_woe,
                "mean_woe": mean_woe,
                "max_bad_pct": max_bad_pct,
                "mean_bad_pct": mean_bad_pct,
                "max_iv_bin": max_iv_bin,
                "perfect_bin_sep": int(perfect_sep),
                "small_bin_exists": small_bin_flag
            })

        summary_df = pd.DataFrame(summaries).sort_values("IV", ascending=False)
        summary_out = os.path.join(IV_SUMMARY_DIR, "iv_feature_summaries.csv")
        summary_df.to_csv(summary_out, index=False)
        if self.verbose:
            print(f"[IV] saved feature summaries -> {summary_out}")

        # -------------------------
        # Optionally save bin-level long table (feature, bin_id, count, bad, good, bad_pct, good_pct, woe, iv_bin)
        # Useful for appendix / reproducibility. Controlled by self.save_bin_level_stats.
        # -------------------------
        if self.save_bin_level_stats:
            long_rows = []
            for col, stats in stats_store.items():
                stats_copy = stats.copy()
                # ensure woe and iv_bin exist
                if "woe" not in stats_copy.columns:
                    stats_copy = stats_copy.assign(woe=IVFilter._woe_from_stats(stats_copy))
                if "iv_bin" not in stats_copy.columns:
                    iv_bin_vals = (stats_copy["good_pct"] - stats_copy["bad_pct"]) * stats_copy["woe"]
                    stats_copy = stats_copy.assign(iv_bin=iv_bin_vals)
                # add feature column
                stats_copy = stats_copy.reset_index().rename(columns={"group": "bin_id"})
                stats_copy["feature"] = col
                long_rows.append(stats_copy[["feature", "bin_id", "count", "bad", "good", "bad_pct", "good_pct", "woe", "iv_bin"]])
            if long_rows:
                long_df = pd.concat(long_rows, ignore_index=True)
                long_out = os.path.join(IV_SUMMARY_DIR, "iv_bin_level_stats.csv")
                long_df.to_csv(long_out, index=False)
                if self.verbose:
                    print(f"[IV] saved bin-level stats -> {long_out}")

        # -------------------------
        # generate simple leakage flags and save
        # -------------------------
        leakage = {}
        for col, iv_val in ivs.items():
            flags = []
            if iv_val >= self.max_iv_for_leakage:
                flags.append(f"high_iv>={self.max_iv_for_leakage}")
            stats = stats_store[col]
            # perfect separation: any bin with bad_pct==0 or good_pct==0
            if (stats["bad_pct"] == 0).any() or (stats["good_pct"] == 0).any():
                flags.append("perfect_bin_sep")
            leakage[col] = ";".join(flags) if flags else ""

        leakage_df = pd.Series(leakage, name="leakage_flags").to_frame()
        leakage_out = os.path.join(IV_SUMMARY_DIR, "iv_leakage_flags.csv")
        leakage_df.to_csv(leakage_out)
        if self.verbose:
            print(f"[IV] saved leakage flags -> {leakage_out}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce X to selected features (columns).
        If X misses some selected columns, they are created with zeros.
        """
        if self.selected_features_ is None:
            raise RuntimeError("IVFilter is not fitted. Call fit(X, y) first.")
        # Keep order from iv_table_
        sel = [c for c in self.iv_table_.index.tolist() if c in self.selected_features_]
        return X.reindex(columns=sel, fill_value=0)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)







        