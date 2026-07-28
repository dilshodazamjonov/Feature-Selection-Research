"""Evidence-driven figures for the Prompt 6 package.

Every figure states its research question, source table, direction convention,
and uncertainty in its own caption sidecar.  No decorative plots are produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CONFIGURATION_ORDER = ["reference", "voting_k100", "voting_k200", "voting_k300"]
CONFIGURATION_LABEL = {
    "reference": "reference\n(rf_corr_mrmr)",
    "voting_k100": "voting K=100",
    "voting_k200": "voting K=200\n(primary)",
    "voting_k300": "voting K=300",
}
CELL_LABEL = {
    ("homecredit", "lr"): "Home Credit / LR",
    ("homecredit", "catboost"): "Home Credit / CatBoost",
    ("lendingclub_v2", "lr"): "LendingClub v2 / LR",
    ("lendingclub_v2", "catboost"): "LendingClub v2 / CatBoost",
}
_PALETTE = ["#1b6ca8", "#c1663a", "#3f7d5a", "#7a5195"]


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def voting_budget_auc(
    run_metrics: pd.DataFrame, path: Path, *, source_table: str
) -> dict[str, Any]:
    """Q: how does locked OOT AUC move across candidate-pool budgets?"""

    oot = run_metrics.loc[run_metrics["split"] == "OOT"].copy()
    cells = [key for key in CELL_LABEL if not oot.loc[(oot["dataset"] == key[0]) & (oot["model"] == key[1])].empty]
    figure, axes = plt.subplots(1, len(cells), figsize=(4.1 * len(cells), 4.0), sharey=False)
    axes = [axes] if len(cells) == 1 else list(axes)
    for index, (cell, axis) in enumerate(zip(cells, axes, strict=True)):
        dataset, model = cell
        subset = (
            oot.loc[(oot["dataset"] == dataset) & (oot["model"] == model)]
            .set_index("configuration")
            .reindex(CONFIGURATION_ORDER)
            .dropna(subset=["auc"])
        )
        axis.plot(
            range(len(subset)),
            subset["auc"].to_numpy(dtype=float),
            marker="o",
            color=_PALETTE[index % len(_PALETTE)],
        )
        axis.set_xticks(range(len(subset)))
        axis.set_xticklabels(
            [CONFIGURATION_LABEL[name] for name in subset.index], fontsize=7.5
        )
        axis.set_title(CELL_LABEL[cell], fontsize=9.5)
        axis.set_ylabel("locked OOT ROC AUC" if index == 0 else "")
        axis.grid(alpha=0.25, linewidth=0.6)
    figure.suptitle(
        "Voting candidate-pool budget versus locked OOT AUC (recomputed from saved predictions)",
        fontsize=10.5,
    )
    _save(figure, path)
    return {
        "figure": path.name,
        "research_question": (
            "Within each locked dataset-model cell, how does recomputed OOT ROC AUC "
            "differ across the reference selector and the three voting candidate-pool budgets?"
        ),
        "source_table": source_table,
        "direction_convention": "higher AUC is higher ranking discrimination in this locked comparison",
        "uncertainty_shown": "none; paired uncertainty is reported in the forest plot",
        "caption": (
            "Recomputed locked-OOT ROC AUC by configuration for each dataset-model cell. "
            "Points are single locked evaluations, not repeated measurements, so no "
            "interval is drawn here and no universal ranking of selectors is implied."
        ),
    }


def paired_auc_delta_forest(
    inference: pd.DataFrame, path: Path, *, source_table: str
) -> dict[str, Any]:
    """Q: what is the paired OOT AUC delta and its interval per comparison?"""

    frame = inference.sort_values(
        ["family", "candidate_pool_budget"], kind="mergesort"
    ).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(8.2, 0.46 * len(frame) + 2.1))
    positions = range(len(frame))
    for position, row in zip(positions, frame.itertuples(index=False), strict=True):
        lower = row.auc_delta_ci95_lower
        upper = row.auc_delta_ci95_upper
        colour = _PALETTE[0] if row.designation == "primary" else "#6b7280"
        if pd.notna(lower) and pd.notna(upper):
            axis.plot([lower, upper], [position, position], color=colour, linewidth=1.8)
        axis.plot(
            [row.auc_delta_comparator_minus_reference],
            [position],
            marker="D" if row.designation == "primary" else "o",
            color=colour,
            markersize=6 if row.designation == "primary" else 5,
        )
    axis.axvline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    axis.set_yticks(list(positions))
    axis.set_yticklabels(
        [
            f"{row.family}  K={int(row.candidate_pool_budget)}"
            f"{'  (primary)' if row.designation == 'primary' else ''}"
            f"   Holm p={row.holm_adjusted_p_value:.3g}"
            for row in frame.itertuples(index=False)
        ],
        fontsize=8,
    )
    axis.invert_yaxis()
    axis.set_xlabel(
        "paired OOT AUC delta: voting minus reference\n"
        "(bars are 95% percentile stratified paired bootstrap intervals)",
        fontsize=9,
    )
    axis.set_title(
        "Predeclared paired AUC comparisons with Holm-adjusted p-values", fontsize=10.5
    )
    axis.grid(axis="x", alpha=0.25, linewidth=0.6)
    _save(figure, path)
    return {
        "figure": path.name,
        "research_question": (
            "For each predeclared comparison, what is the paired locked-OOT AUC "
            "difference between the voting configuration and its same-cell "
            "rf_corr_mrmr reference, and how uncertain is it?"
        ),
        "source_table": source_table,
        "direction_convention": "comparator minus reference; positive means the voting run ranked higher in this locked comparison",
        "uncertainty_shown": "95% percentile stratified paired bootstrap interval (2,000 attempts, seed 20260721)",
        "caption": (
            "Paired AUC deltas on identical OOT applicant rows. Diamonds mark the "
            "primary K=200 comparisons; circles mark K=100/K=300 sensitivity "
            "comparisons. Holm adjustment is applied within each dataset-model "
            "family of three tests. A non-significant result is not evidence of "
            "equivalence, and statistical significance is not business materiality."
        ),
    }


def drift_stability_comparison(
    evidence: pd.DataFrame, path: Path, *, source_table: str, references: list[float]
) -> dict[str, Any]:
    """Q: how do descriptive drift and selection stability differ by configuration?"""

    frame = evidence.copy()
    frame["cell"] = [
        CELL_LABEL.get((dataset, model), f"{dataset}/{model}")
        for dataset, model in zip(frame["dataset"], frame["model"], strict=True)
    ]
    panels = [
        ("score_psi", "score PSI (DEV OOF bins applied to OOT)", references),
        ("feature_psi_type_aware_mean", "mean selected-feature PSI (type-aware)", references),
        ("fold_jaccard_mean", "mean pairwise fold Jaccard", []),
        ("fold_kuncheva_mean", "mean pairwise fold Kuncheva", []),
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2))
    cells = list(dict.fromkeys(frame["cell"]))
    width = 0.8 / max(len(CONFIGURATION_ORDER), 1)
    for axis, (column, title, guides) in zip(axes, panels, strict=True):
        for offset, configuration in enumerate(CONFIGURATION_ORDER):
            subset = frame.loc[frame["configuration"] == configuration]
            values = [
                float(subset.loc[subset["cell"] == cell, column].iloc[0])
                if not subset.loc[subset["cell"] == cell, column].empty
                and pd.notna(subset.loc[subset["cell"] == cell, column].iloc[0])
                else float("nan")
                for cell in cells
            ]
            axis.bar(
                [index + offset * width - 0.4 + width / 2 for index in range(len(cells))],
                values,
                width=width,
                label=CONFIGURATION_LABEL[configuration].replace("\n", " "),
                color=_PALETTE[offset % len(_PALETTE)],
            )
        for guide in guides:
            axis.axhline(
                float(guide), color="#6b7280", linestyle=":", linewidth=0.9
            )
        axis.set_xticks(range(len(cells)))
        axis.set_xticklabels(cells, rotation=25, ha="right", fontsize=7.5)
        axis.set_title(title, fontsize=9)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].legend(fontsize=7, loc="upper left")
    figure.suptitle(
        "Descriptive drift and selection stability by configuration (not discrimination measures)",
        fontsize=10.5,
    )
    _save(figure, path)
    return {
        "figure": path.name,
        "research_question": (
            "How do descriptive score drift, selected-feature drift, and fold-level "
            "selection stability differ across configurations within each cell?"
        ),
        "source_table": source_table,
        "direction_convention": (
            "lower PSI is a smaller descriptive DEV-to-OOT distribution difference; "
            "higher Jaccard/Kuncheva is greater fold-to-fold selection agreement"
        ),
        "uncertainty_shown": "none; all four panels are descriptive point summaries",
        "caption": (
            "Dotted guides mark the descriptive PSI reference values 0.10 and 0.25; "
            "the frozen protocol does not define PSI categories. Jaccard and "
            "Kuncheva measure selection agreement across folds and are not "
            "predictive-performance measures, so a higher value here does not imply "
            "higher AUC."
        ),
    }


def runtime_comparison(
    runtime: pd.DataFrame, path: Path, *, source_table: str
) -> dict[str, Any]:
    """Q: how does total wall-clock runtime vary by configuration?"""

    frame = runtime.copy()
    frame["cell"] = [
        CELL_LABEL.get((dataset, model), f"{dataset}/{model}")
        for dataset, model in zip(frame["dataset"], frame["model"], strict=True)
    ]
    cells = list(dict.fromkeys(frame["cell"]))
    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    width = 0.8 / max(len(CONFIGURATION_ORDER), 1)
    for offset, configuration in enumerate(CONFIGURATION_ORDER):
        subset = frame.loc[frame["configuration"] == configuration]
        values = [
            float(subset.loc[subset["cell"] == cell, "total_wall_clock_seconds"].iloc[0] / 60.0)
            if not subset.loc[subset["cell"] == cell, "total_wall_clock_seconds"].empty
            and pd.notna(subset.loc[subset["cell"] == cell, "total_wall_clock_seconds"].iloc[0])
            else float("nan")
            for cell in cells
        ]
        axis.bar(
            [index + offset * width - 0.4 + width / 2 for index in range(len(cells))],
            values,
            width=width,
            label=CONFIGURATION_LABEL[configuration].replace("\n", " "),
            color=_PALETTE[offset % len(_PALETTE)],
        )
    axis.set_xticks(range(len(cells)))
    axis.set_xticklabels(cells, fontsize=8.5)
    axis.set_ylabel("total measured wall-clock runtime (minutes)")
    axis.set_title(
        "Total run wall-clock time by configuration (observational, not a controlled benchmark)",
        fontsize=10,
    )
    axis.legend(fontsize=7.5)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    _save(figure, path)
    return {
        "figure": path.name,
        "research_question": (
            "How much total wall-clock time did each completed configuration consume "
            "under the single frozen execution policy?"
        ),
        "source_table": source_table,
        "direction_convention": "lower is less measured wall-clock time for this one execution",
        "uncertainty_shown": "none; each run was executed once",
        "caption": (
            "Total measured wall-clock seconds per run, converted to minutes. Stage "
            "level timers were not populated by these runs, runs were executed once "
            "each on shared hardware, and several were interrupted and resumed, so "
            "these totals are observational evidence of cost rather than a "
            "controlled runtime benchmark."
        ),
    }


__all__ = [
    "drift_stability_comparison",
    "paired_auc_delta_forest",
    "runtime_comparison",
    "voting_budget_auc",
]
