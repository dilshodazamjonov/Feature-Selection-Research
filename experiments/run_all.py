from __future__ import annotations

import argparse

from experiments.hybrid_comparison import main as hybrid_main
from experiments.llm_vs_statistical import main as llm_vs_statistical_main
from experiments.statistical_baselines import main as statistical_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run statistical comparison, LLM vs statistical, and hybrid comparison in sequence.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--model-selector",
        "--model",
        dest="model",
        default=None,
        help="Optional override for the model to use across all three runs.",
    )
    return parser


def _build_stage_args(config_path: str, model_name: str | None) -> list[str]:
    args = ["--config", config_path]
    if model_name:
        args.extend(["--model-selector", model_name])
    return args


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stage_args = _build_stage_args(args.config, args.model)

    print("Starting Part 1: Statistical baselines")
    statistical_main(stage_args)

    print("Starting Part 2: LLM vs statistical")
    llm_vs_statistical_main(stage_args)

    print("Starting Part 3: Hybrid comparison")
    hybrid_main(stage_args)

    print("All experiment pipelines completed.")


if __name__ == "__main__":
    main()
