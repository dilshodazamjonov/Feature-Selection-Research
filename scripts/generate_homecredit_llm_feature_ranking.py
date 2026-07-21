"""Deprecated command wrapper for the deterministic Home Credit rule ranker.

Despite this historical filename, the implementation never calls an LLM.
Use ``scripts/generate_homecredit_domain_rule_ranking.py`` instead.
"""

from __future__ import annotations

import warnings

if __package__:
    from scripts import generate_homecredit_domain_rule_ranking as _canonical
else:  # Support direct ``python scripts/<name>.py`` execution.
    import generate_homecredit_domain_rule_ranking as _canonical


INPUT_PATH = _canonical.INPUT_PATH
OUTPUT_DIR = _canonical.OUTPUT_DIR
OUTPUT_PATH = _canonical.OUTPUT_PATH
SUMMARY_PATH = _canonical.SUMMARY_PATH
build_ranking = _canonical.build_ranking
validate_input = _canonical.validate_input
validate_output = _canonical.validate_output


def main() -> int:
    warnings.warn(
        "generate_homecredit_llm_feature_ranking.py is deprecated: it is a "
        "deterministic domain-rule ranker and makes no LLM call. Use "
        "generate_homecredit_domain_rule_ranking.py.",
        FutureWarning,
        stacklevel=2,
    )
    return _canonical.main()


if __name__ == "__main__":
    raise SystemExit(main())
