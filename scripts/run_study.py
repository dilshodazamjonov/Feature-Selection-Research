"""Run the study for one dataset, stage by stage.

    uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage prepare
    uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage llm
    uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage main
    uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage robustness
    uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage voting
    uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage sweep
    uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage results

``--stage all`` runs the seven stages in order. Every stage resumes: finished units of
work are read back from ``outputs/study/<dataset>/`` instead of being recomputed. The
``llm`` stage is the only one that calls the OpenAI API (``OPENAI_API_KEY`` in ``.env``);
``--offline`` makes it fail on any ranking that is not already cached.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

STAGES = ("prepare", "llm", "main", "robustness", "voting", "sweep", "results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="configs/study/<dataset>.yaml")
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument("--offline", action="store_true", help="never call the API; every ranking must be cached")
    parser.add_argument("--output-root", default=None, help="override output_root (default outputs/study)")
    parser.add_argument("--results-root", default=None, help="override results_root (default results)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from credit_risk_fs.study import stages
    from credit_risk_fs.study.config import load_config
    from credit_risk_fs.study.results import run_results
    from credit_risk_fs.study.utils import log

    cfg = load_config(args.config)
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.results_root:
        cfg["results_root"] = args.results_root
    runners = {
        "prepare": stages.run_prepare,
        "llm": lambda config: stages.run_llm(config, allow_request=not args.offline),
        "main": stages.run_main,
        "robustness": stages.run_robustness,
        "voting": stages.run_voting,
        "sweep": stages.run_sweep,
        "results": run_results,
    }
    for stage in STAGES if args.stage == "all" else (args.stage,):
        log(f"{cfg['dataset']}: stage {stage} started")
        runners[stage](cfg)
        log(f"{cfg['dataset']}: stage {stage} finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
