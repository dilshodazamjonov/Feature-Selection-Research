from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a monthly LendingClub EDA chart for observation count and bad rate."
    )
    parser.add_argument(
        "--input-file",
        default="data/lendingclub/processed/application_train.csv",
        help="Processed LendingClub modeling file.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory where the CSV summary and SVG chart will be written.",
    )
    return parser


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _nice_count_limit(value: float) -> int:
    if value <= 0:
        return 1
    magnitude = 10 ** int(math.floor(math.log10(value)))
    scaled = value / magnitude
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return int(nice * magnitude)


def build_monthly_summary(input_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file, usecols=["issue_d", "TARGET"], low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    monthly = (
        df.dropna(subset=["issue_d"])
        .groupby(pd.Grouper(key="issue_d", freq="MS"))
        .agg(observation_count=("TARGET", "size"), bad_rate=("TARGET", "mean"))
        .reset_index()
        .sort_values("issue_d")
    )
    monthly["month"] = monthly["issue_d"].dt.strftime("%Y-%m")
    monthly["bad_rate_pct"] = monthly["bad_rate"] * 100.0
    return monthly[["issue_d", "month", "observation_count", "bad_rate", "bad_rate_pct"]]


def render_svg(monthly: pd.DataFrame, output_path: Path) -> None:
    width = 1600
    height = 900
    margin_left = 95
    margin_right = 95
    margin_top = 80
    margin_bottom = 130
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    count_max = _nice_count_limit(monthly["observation_count"].max() * 1.08)
    rate_max = max(0.30, math.ceil(monthly["bad_rate"].max() * 100 / 5) * 0.05)

    def x_pos(index: int) -> float:
        if len(monthly) == 1:
            return margin_left + plot_width / 2
        return margin_left + (index / (len(monthly) - 1)) * plot_width

    def y_count(value: float) -> float:
        return margin_top + plot_height - (value / count_max) * plot_height

    def y_rate(value: float) -> float:
        return margin_top + plot_height - (value / rate_max) * plot_height

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #1f2937; }',
        '.title { font-size: 28px; font-weight: bold; }',
        '.subtitle { font-size: 15px; fill: #4b5563; }',
        '.axis { font-size: 12px; }',
        '.grid { stroke: #d1d5db; stroke-width: 1; }',
        '.bar { fill: #8ecae6; opacity: 0.85; }',
        '.line { fill: none; stroke: #c2410c; stroke-width: 3; }',
        '.point { fill: #c2410c; }',
        '.legend { font-size: 13px; }',
        '</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />',
        f'<text class="title" x="{margin_left}" y="40">LendingClub Monthly Observations And Bad Rate</text>',
        (
            f'<text class="subtitle" x="{margin_left}" y="64">'
            f'Processed accepted loans, {monthly["month"].iloc[0]} to {monthly["month"].iloc[-1]}</text>'
        ),
    ]

    for tick in range(0, 6):
        count_value = count_max * tick / 5
        y = y_count(count_value)
        svg.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" />')
        svg.append(
            f'<text class="axis" x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end">{int(count_value):,}</text>'
        )

    for tick in range(0, 6):
        rate_value = rate_max * tick / 5
        y = y_rate(rate_value)
        svg.append(
            f'<text class="axis" x="{width - margin_right + 12}" y="{y + 4:.2f}" text-anchor="start">{_format_pct(rate_value)}</text>'
        )

    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5" />'
    )
    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5" />'
    )
    svg.append(
        f'<line x1="{width - margin_right}" y1="{margin_top}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5" />'
    )

    bar_width = max(3.0, min(10.0, plot_width / max(len(monthly) * 1.6, 1)))
    line_points: list[str] = []

    for idx, row in monthly.reset_index(drop=True).iterrows():
        x = x_pos(idx)
        bar_top = y_count(float(row["observation_count"]))
        bar_height = margin_top + plot_height - bar_top
        svg.append(
            f'<rect class="bar" x="{x - bar_width / 2:.2f}" y="{bar_top:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" />'
        )
        line_points.append(f"{x:.2f},{y_rate(float(row['bad_rate'])):.2f}")
        if idx % 12 == 0 or idx == len(monthly) - 1:
            label = escape(str(row["month"]))
            svg.append(
                f'<text class="axis" x="{x:.2f}" y="{height - margin_bottom + 22}" text-anchor="middle" transform="rotate(45 {x:.2f},{height - margin_bottom + 22})">{label}</text>'
            )

    svg.append(f'<polyline class="line" points="{" ".join(line_points)}" />')
    for point in line_points[:: max(1, len(line_points) // 24)]:
        x, y = point.split(",")
        svg.append(f'<circle class="point" cx="{x}" cy="{y}" r="3.5" />')

    legend_x = margin_left
    legend_y = height - 55
    svg.append(f'<rect x="{legend_x}" y="{legend_y - 12}" width="18" height="12" class="bar" />')
    svg.append(f'<text class="legend" x="{legend_x + 28}" y="{legend_y - 2}">Observation count (left axis)</text>')
    svg.append(
        f'<line x1="{legend_x + 250}" y1="{legend_y - 6}" x2="{legend_x + 285}" y2="{legend_y - 6}" class="line" />'
    )
    svg.append(f'<circle class="point" cx="{legend_x + 268}" cy="{legend_y - 6}" r="3.5" />')
    svg.append(f'<text class="legend" x="{legend_x + 295}" y="{legend_y - 2}">Bad rate (right axis)</text>')

    svg.append(
        f'<text class="axis" x="{margin_left}" y="{height - 18}">Source: data/lendingclub/processed/application_train.csv</text>'
    )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_file = PROJECT_ROOT / args.input_file
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly = build_monthly_summary(input_file)
    csv_path = output_dir / "lendingclub_monthly_bad_rate_observation_count.csv"
    svg_path = output_dir / "lendingclub_monthly_bad_rate_observation_count.svg"
    monthly.to_csv(csv_path, index=False)
    render_svg(monthly, svg_path)

    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote SVG chart: {svg_path}")
    print(
        "Monthly range: "
        f"{monthly['month'].iloc[0]} to {monthly['month'].iloc[-1]} | "
        f"Rows: {len(monthly):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
