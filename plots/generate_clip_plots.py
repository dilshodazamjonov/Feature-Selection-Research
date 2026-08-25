"""Generate publication-ready plots for the completed CLIP Stability experiment.

The script reads only authenticated experiment outputs. It does not refit CLIP,
rerun feature selection, touch OOT predictions, or alter experiment artifacts.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "prompt_16_homecredit_model_stability_2024"
    / "clip_experiment_v1"
)
RESULT_ROOT = (
    ROOT
    / "results"
    / "prompt_16_homecredit_model_stability_2024"
    / "clip_experiment_v1"
)

INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE8"
BLUE = "#2F6B9A"
BLUE_LIGHT = "#A9CBE3"
BLUE_OPEN = "#EAF2F8"
GOLD = "#C98A16"
GOLD_LIGHT = "#F2D89B"
GOLD_OPEN = "#FFF7E6"
ORANGE = "#D97745"
GREY = "#8B96A5"
GREY_LIGHT = "#EEF1F5"
WHITE = "#FFFFFF"

DIRECTION_ORDER = [
    "stability_to_stability",
    "homecredit_to_stability",
    "lendingclub_to_stability",
]
DIRECTION_LABEL = {
    "stability_to_stability": "Stability",
    "homecredit_to_stability": "HomeCredit transfer",
    "lendingclub_to_stability": "LendingClub transfer",
}
SHORT_DIRECTION = {
    "stability_to_stability": "Stability",
    "homecredit_to_stability": "HomeCredit",
    "lendingclub_to_stability": "LendingClub",
}
MODEL_ORDER = ["lr", "catboost"]
MODEL_LABEL = {"lr": "Logistic regression", "catboost": "CatBoost"}
MODEL_SHORT = {"lr": "LR", "catboost": "CatBoost"}
MODEL_COLOR = {"lr": GOLD, "catboost": BLUE}
MODEL_HATCH = {"lr": "//", "catboost": ""}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.bbox": "tight",
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C0CC")
    ax.spines["bottom"].set_color("#B8C0CC")
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def add_figure_header(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    source: str,
    title_y: float = 0.975,
    subtitle_y: float = 0.935,
) -> None:
    fig.suptitle(title, x=0.06, y=title_y, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.06, subtitle_y, subtitle, ha="left", va="top", color=MUTED, fontsize=10.5)
    fig.text(0.06, 0.018, source, ha="left", va="bottom", color=MUTED, fontsize=8.5)


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = PLOTS / filename
    fig.savefig(path, dpi=240, facecolor=WHITE)
    plt.close(fig)


def load_final_results() -> pd.DataFrame:
    data = pd.read_csv(RESULT_ROOT / "analysis" / "final_clip_results.csv")
    data["direction"] = pd.Categorical(
        data["direction"], categories=DIRECTION_ORDER, ordered=True
    )
    data["classifier"] = pd.Categorical(
        data["classifier"], categories=MODEL_ORDER, ordered=True
    )
    return data.sort_values(["direction", "classifier"]).reset_index(drop=True)


def plot_oot_performance(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    add_figure_header(
        fig,
        "CLIP-ranked downstream performance on untouched OOT data",
        (
            "Home Credit Model Stability 2024 OOT cohort: 304,916 applications, "
            "8,349 bad outcomes; every panel starts at zero."
        ),
        "Source: authenticated final_clip_results.csv; one-time OOT evaluation after the pre-OOT freeze gate.",
    )

    panels = [
        ("oot_auc", "ROC-AUC", (0, 0.84), "{:.3f}"),
        ("oot_ks", "Kolmogorov-Smirnov statistic", (0, 0.46), "{:.3f}"),
        ("oot_lift_at_10pct", "Lift at top 10%", (0, 4.15), "{:.2f}x"),
        (
            "oot_capture_at_10pct",
            "Bad-event capture at top 10%",
            (0, 0.42),
            "{:.3f}",
        ),
    ]
    x = np.arange(len(DIRECTION_ORDER), dtype=float)
    width = 0.34

    for ax, (metric, title, ylim, formatter) in zip(axes.flat, panels):
        for offset, model in zip((-width / 2, width / 2), MODEL_ORDER):
            subset = (
                data[data["classifier"] == model]
                .set_index("direction")
                .reindex(DIRECTION_ORDER)
            )
            values = subset[metric].to_numpy(dtype=float)
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=MODEL_COLOR[model],
                edgecolor=INK,
                linewidth=0.75,
                hatch=MODEL_HATCH[model],
                label=MODEL_LABEL[model],
                zorder=3,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (ylim[1] * 0.018),
                    formatter.format(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
        ax.set_title(title, loc="left", pad=10)
        ax.set_ylim(*ylim)
        ax.set_xticks(x, [DIRECTION_LABEL[d] for d in DIRECTION_ORDER])
        ax.tick_params(axis="x", labelrotation=0)
        clean_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.10, hspace=0.38, wspace=0.22)
    save_figure(fig, "01_oot_performance_metrics.png")


def plot_dev_to_oot_auc(data: pd.DataFrame) -> None:
    ordered = data.sort_values(["direction", "classifier"]).reset_index(drop=True)
    labels = [
        f"{DIRECTION_LABEL[row.direction]} - {MODEL_SHORT[str(row.classifier)]}"
        for row in ordered.itertuples()
    ]
    y = np.arange(len(ordered))[::-1].astype(float)

    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    add_figure_header(
        fig,
        "DEV-to-OOT ROC-AUC generalization",
        (
            "Fold mean +/- one fold SD, pooled DEV OOF AUC, and untouched OOT AUC. "
            "Focused AUC scale; chance (0.50) lies outside the view."
        ),
        "Source: authenticated final_clip_results.csv; DEV OOF n=1,018,631 and OOT n=304,916 for every cell.",
    )

    for idx, row in enumerate(ordered.itertuples()):
        yi = y[idx]
        model = str(row.classifier)
        color = MODEL_COLOR[model]
        mean_auc = float(row.dev_fold_auc_mean)
        sd_auc = float(row.dev_fold_auc_sd)
        pooled_auc = float(row.dev_pooled_oof_auc)
        oot_auc = float(row.oot_auc)

        ax.plot(
            [pooled_auc, oot_auc],
            [yi, yi],
            color=color,
            linewidth=2.2,
            alpha=0.82,
            zorder=2,
        )
        ax.errorbar(
            mean_auc,
            yi,
            xerr=sd_auc,
            fmt="o",
            markersize=7,
            markerfacecolor=WHITE,
            markeredgecolor=color,
            markeredgewidth=1.7,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            zorder=4,
        )
        ax.scatter(
            pooled_auc,
            yi,
            marker="D",
            s=55,
            facecolor=WHITE,
            edgecolor=INK,
            linewidth=1.2,
            zorder=5,
        )
        ax.scatter(
            oot_auc,
            yi,
            marker="o",
            s=78,
            facecolor=color,
            edgecolor=INK,
            linewidth=0.9,
            zorder=6,
        )
        delta = oot_auc - pooled_auc
        ax.text(
            oot_auc + 0.003,
            yi,
            f"{oot_auc:.3f}  ({delta:+.3f})",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_yticks(y, labels)
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.59, 0.795)
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    clean_axis(ax, grid_axis="x")
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=GREY,
            markerfacecolor=WHITE,
            markeredgecolor=GREY,
            linewidth=1.5,
            label="Fold mean +/- SD",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=WHITE,
            markeredgecolor=INK,
            label="Pooled DEV OOF",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GREY,
            markeredgecolor=INK,
            label="Untouched OOT",
        ),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, ncol=3)
    fig.subplots_adjust(left=0.27, right=0.95, top=0.84, bottom=0.12)
    save_figure(fig, "02_dev_to_oot_auc_generalization.png")


def load_seed_metrics() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    seed_root = OUTPUT_ROOT / "representation" / "stability" / "seeds"
    for seed in (11, 22, 33, 44, 55):
        with (seed_root / f"seed_{seed}" / "representation_metrics.json").open(
            "r", encoding="utf-8"
        ) as handle:
            payload = json.load(handle)
        metrics = payload["validation_retrieval"]
        rows.append(
            {
                "seed": seed,
                "epoch": int(payload["checkpoint_epoch"]),
                "r1": np.mean(
                    [
                        metrics["text_to_statistical_recall_at_1"],
                        metrics["statistical_to_text_recall_at_1"],
                    ]
                ),
                "r5": np.mean(
                    [
                        metrics["text_to_statistical_recall_at_5"],
                        metrics["statistical_to_text_recall_at_5"],
                    ]
                ),
                "r10": np.mean(
                    [
                        metrics["text_to_statistical_recall_at_10"],
                        metrics["statistical_to_text_recall_at_10"],
                    ]
                ),
                "mrr": metrics["mean_reciprocal_rank"],
                "positive": metrics["positive_pair_cosine_mean"],
                "negative": metrics["allowed_negative_cosine_mean"],
                "margin": metrics["positive_minus_negative_margin"],
            }
        )
    return pd.DataFrame(rows)


def plot_seed_retrieval(seed_data: pd.DataFrame) -> None:
    fig, (ax_heat, ax_margin) = plt.subplots(
        1,
        2,
        figsize=(15.5, 7.8),
        gridspec_kw={"width_ratios": [1.12, 1.0]},
    )
    add_figure_header(
        fig,
        "Five-seed CLIP validation retrieval and alignment",
        (
            "Selected checkpoint for each Stability seed; bidirectional retrieval is averaged across "
            "text-to-statistical and statistical-to-text directions."
        ),
        "Source: five authenticated representation_metrics.json files; validation split contains 395 paired features per seed.",
    )

    matrix = seed_data[["r1", "r5", "r10", "mrr"]].to_numpy(dtype=float)
    image = ax_heat.imshow(matrix, cmap="Blues", vmin=0.0, vmax=0.60, aspect="auto")
    ax_heat.set_title("Bidirectional retrieval quality", loc="left", pad=12)
    ax_heat.set_xticks(np.arange(4), ["Recall@1", "Recall@5", "Recall@10", "MRR"])
    ax_heat.set_yticks(
        np.arange(len(seed_data)),
        [
            f"Seed {int(row.seed)}  (epoch {int(row.epoch)})"
            for row in seed_data.itertuples()
        ],
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax_heat.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=WHITE if value > 0.36 else INK,
                fontweight="bold",
                fontsize=10,
            )
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.035)
    colorbar.set_label("Retrieval metric")
    colorbar.outline.set_edgecolor("#B8C0CC")

    y = np.arange(len(seed_data))[::-1]
    for yi, row in zip(y, seed_data.itertuples()):
        ax_margin.plot(
            [row.negative, row.positive],
            [yi, yi],
            color=BLUE_LIGHT,
            linewidth=6,
            solid_capstyle="round",
            zorder=2,
        )
        ax_margin.scatter(
            row.negative,
            yi,
            s=75,
            facecolor=WHITE,
            edgecolor=GOLD,
            linewidth=2,
            zorder=4,
        )
        ax_margin.scatter(
            row.positive,
            yi,
            s=82,
            facecolor=BLUE,
            edgecolor=INK,
            linewidth=0.8,
            zorder=5,
        )
        ax_margin.text(
            row.positive + 0.015,
            yi,
            f"margin {row.margin:.3f}",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax_margin.set_title("Positive vs allowed-negative cosine", loc="left", pad=12)
    ax_margin.set_yticks(y, [f"Seed {int(seed)}" for seed in seed_data["seed"]])
    ax_margin.set_xlabel("Mean cosine similarity")
    ax_margin.set_xlim(0.0, 0.82)
    ax_margin.set_ylim(-0.7, len(seed_data) - 0.3)
    clean_axis(ax_margin, grid_axis="x")
    ax_margin.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=WHITE,
                markeredgecolor=GOLD,
                markeredgewidth=2,
                label="Allowed negative",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=BLUE,
                markeredgecolor=INK,
                label="Positive pair",
            ),
        ],
        loc="lower right",
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.11, wspace=0.30)
    save_figure(fig, "03_five_seed_retrieval_stability.png")


def load_rankings() -> dict[str, pd.DataFrame]:
    rank_root = OUTPUT_ROOT / "rankings"
    result: dict[str, pd.DataFrame] = {}
    for direction in DIRECTION_ORDER:
        frame = pd.read_csv(
            rank_root / f"{direction}.csv",
            usecols=["rank", "feature_id", "feature_name", "clip_score"],
        )
        result[direction] = frame.sort_values("rank").reset_index(drop=True)
    return result


def pair_rank_frame(
    rankings: dict[str, pd.DataFrame], left: str, right: str
) -> pd.DataFrame:
    return rankings[left][["feature_id", "rank", "clip_score"]].merge(
        rankings[right][["feature_id", "rank", "clip_score"]],
        on="feature_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )


def plot_ranking_agreement(rankings: dict[str, pd.DataFrame]) -> None:
    pairs = list(combinations(DIRECTION_ORDER, 2))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.2))
    add_figure_header(
        fig,
        "Feature-ranking agreement across CLIP source representations",
        (
            "All 1,959 Stability features are compared at identical grain. Blue points are shared top-100 "
            "features; rank 1 is best."
        ),
        "Source: three authenticated CLIP ranking CSVs. Spearman is Pearson correlation of exact rank vectors; top-K overlap is derived without targets or OOT data.",
    )

    for ax, (left, right) in zip(axes.flat[:3], pairs):
        paired = pair_rank_frame(rankings, left, right)
        rho = float(np.corrcoef(paired["rank_left"], paired["rank_right"])[0, 1])
        shared = (paired["rank_left"] <= 100) & (paired["rank_right"] <= 100)
        shared_count = int(shared.sum())

        ax.scatter(
            paired.loc[~shared, "rank_left"],
            paired.loc[~shared, "rank_right"],
            s=9,
            color=GREY,
            alpha=0.22,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        ax.scatter(
            paired.loc[shared, "rank_left"],
            paired.loc[shared, "rank_right"],
            s=19,
            color=BLUE,
            alpha=0.85,
            edgecolors=INK,
            linewidths=0.25,
            zorder=4,
        )
        ax.plot([1, 1959], [1, 1959], linestyle="--", color=INK, linewidth=1.0, alpha=0.7)
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 2000)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{SHORT_DIRECTION[left]} rank")
        ax.set_ylabel(f"{SHORT_DIRECTION[right]} rank")
        ax.set_title(
            f"{SHORT_DIRECTION[left]} vs {SHORT_DIRECTION[right]}",
            loc="left",
            pad=10,
        )
        ax.text(
            0.04,
            0.94,
            f"Spearman rho = {rho:.3f}\nShared top 100 = {shared_count}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE, edgecolor=GRID),
        )
        clean_axis(ax, grid_axis="both")

    ax_overlap = axes.flat[3]
    k_values = np.array([10, 20, 40, 60, 100, 250, 500, 1000], dtype=int)
    pair_styles = [
        (BLUE, "o", "-"),
        (GOLD, "s", "--"),
        (ORANGE, "^", "-."),
    ]
    for (left, right), (color, marker, linestyle) in zip(pairs, pair_styles):
        jaccards = []
        for k in k_values:
            left_set = set(rankings[left].nsmallest(k, "rank")["feature_id"])
            right_set = set(rankings[right].nsmallest(k, "rank")["feature_id"])
            jaccards.append(len(left_set & right_set) / len(left_set | right_set))
        ax_overlap.plot(
            k_values,
            jaccards,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=6,
            label=f"{SHORT_DIRECTION[left]} vs {SHORT_DIRECTION[right]}",
        )
    random_jaccard = k_values / (2 * 1959 - k_values)
    ax_overlap.plot(
        k_values,
        random_jaccard,
        color=INK,
        linestyle=":",
        linewidth=1.6,
        label="Random-order expectation",
    )
    ax_overlap.set_xscale("log")
    ax_overlap.set_xticks(k_values, [str(k) for k in k_values])
    ax_overlap.set_ylim(0, 1.0)
    ax_overlap.set_xlabel("Top-K cutoff")
    ax_overlap.set_ylabel("Feature-set Jaccard")
    ax_overlap.set_title("Top-K agreement by cutoff", loc="left", pad=10)
    clean_axis(ax_overlap, grid_axis="y")
    ax_overlap.legend(loc="upper left", frameon=False, fontsize=8.7)

    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.08, hspace=0.34, wspace=0.25)
    save_figure(fig, "04_cross_source_ranking_agreement.png")


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.16,
        y + height - 0.25,
        title,
        ha="left",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        color=INK,
        zorder=3,
    )
    ax.text(
        x + 0.16,
        y + height - 0.66,
        body,
        ha="left",
        va="top",
        fontsize=8.9,
        color=MUTED,
        linespacing=1.38,
        zorder=3,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = BLUE,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=color,
            connectionstyle="arc3,rad=0.0",
            zorder=1,
        )
    )


def plot_methodology() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle(
        "CLIP methodology and leakage-control boundary",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.925,
        "Representation learning and ranking are target-free; model fitting stays inside DEV; OOT is opened only after the hash-frozen gate passes.",
        ha="left",
        va="top",
        color=MUTED,
        fontsize=10.5,
    )
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Lane labels and quiet lane backgrounds.
    lane_specs = [
        (5.35, 2.05, "TARGET-FREE REPRESENTATION", BLUE_OPEN, BLUE),
        (2.95, 1.95, "DEV-ONLY MODEL DEVELOPMENT", GREY_LIGHT, INK),
        (0.55, 1.85, "LOCKED OOT EVALUATION", GOLD_OPEN, GOLD),
    ]
    for y, height, label, face, edge in lane_specs:
        ax.add_patch(
            FancyBboxPatch(
                (0.15, y),
                15.65,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=face,
                edgecolor="none",
                alpha=0.78,
                zorder=0,
            )
        )
        ax.text(
            0.35,
            y + height - 0.18,
            label,
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color=edge,
        )

    rounded_box(
        ax,
        0.85,
        5.65,
        3.0,
        1.30,
        "1  Authenticate descriptors",
        "1,959 Stability features\ntext + 13 statistical values\nTarget: no   OOT: no",
        WHITE,
        BLUE,
    )
    rounded_box(
        ax,
        4.25,
        5.65,
        3.0,
        1.30,
        "2  Train five CLIP seeds",
        "32-D normalized projections\nsymmetric contrastive loss\nseeds 11, 22, 33, 44, 55",
        WHITE,
        BLUE,
    )
    rounded_box(
        ax,
        7.65,
        5.65,
        3.0,
        1.30,
        "3  Build consensus rankings",
        "Orthogonal Procrustes alignment\nseed 11 reference\n3 lists, exact ranks 1-1,959",
        WHITE,
        BLUE,
    )
    arrow(ax, (3.85, 6.30), (4.23, 6.30))
    arrow(ax, (7.25, 6.30), (7.63, 6.30))

    rounded_box(
        ax,
        2.00,
        3.25,
        4.15,
        1.25,
        "4  Five expanding temporal folds",
        "Fold TRAIN: mRMR + preprocessing + model + threshold\nFold validation: score only   |   OOF rows: 1,018,631",
        WHITE,
        INK,
    )
    rounded_box(
        ax,
        7.00,
        3.25,
        3.85,
        1.25,
        "5  Refit and freeze on full DEV",
        "6 cells: LR K=20; CatBoost K=40\nselectors, models, thresholds and references hashed",
        WHITE,
        INK,
    )
    arrow(ax, (9.15, 5.64), (5.00, 4.51), color=INK)
    arrow(ax, (6.15, 3.88), (6.98, 3.88), color=INK)

    rounded_box(
        ax,
        3.05,
        0.88,
        3.70,
        1.20,
        "6  Pre-OOT freeze gate",
        "PASS: 50 frozen-file checks\nconfiguration + artifacts locked before OOT",
        WHITE,
        GOLD,
    )
    rounded_box(
        ax,
        9.20,
        0.88,
        4.20,
        1.20,
        "7  One-time untouched OOT scoring",
        "304,916 rows; 2020-02-26 to 2020-10-05\nno tuning; metrics recomputed; prediction hashes verified",
        WHITE,
        GOLD,
    )
    arrow(ax, (8.93, 3.24), (5.60, 2.09), color=GOLD)
    arrow(ax, (6.75, 1.48), (9.18, 1.48), color=GOLD)

    # The visible temporal barrier is the key methodological boundary.
    ax.plot([8.0, 8.0], [0.55, 2.35], color=GOLD, linewidth=2.0, linestyle="--")
    ax.text(
        8.0,
        2.42,
        "OOT remains sealed until gate PASS",
        ha="center",
        va="bottom",
        color=GOLD,
        fontsize=9.2,
        fontweight="bold",
    )

    guardrails = [
        "PASS  target excluded from CLIP",
        "PASS  OOT excluded before freeze",
        "PASS  fold-local fitting",
        "PASS  287 declared hash checks",
    ]
    for idx, text in enumerate(guardrails):
        ax.text(
            0.75 + idx * 3.85,
            0.28,
            text,
            ha="left",
            va="center",
            fontsize=8.7,
            color=INK,
            fontweight="bold",
        )

    fig.text(
        0.055,
        0.018,
        "Source: authenticated experiment, downstream, ranking, pre-OOT, and final-integrity manifests. Diagram describes the implemented contract; it does not claim immunity to unknown upstream defects.",
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.04, right=0.985, top=0.88, bottom=0.065)
    save_figure(fig, "05_methodology_leakage_boundaries.png")


def main() -> None:
    configure_style()
    PLOTS.mkdir(parents=True, exist_ok=True)
    final_results = load_final_results()
    plot_oot_performance(final_results)
    plot_dev_to_oot_auc(final_results)
    plot_seed_retrieval(load_seed_metrics())
    plot_ranking_agreement(load_rankings())
    plot_methodology()
    print("Generated 5 CLIP figures in", PLOTS)


if __name__ == "__main__":
    main()
