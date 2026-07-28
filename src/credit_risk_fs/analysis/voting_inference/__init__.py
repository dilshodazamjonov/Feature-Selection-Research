"""Prompt 6 voting inference and evidence packaging.

Every module here treats completed run directories and the frozen legacy bundle
as immutable inputs.  Nothing in this package fits a model, runs a selector,
regenerates a voting ranking, or edits a frozen definition.
"""

from credit_risk_fs.analysis.voting_inference.config import (
    AnalysisConfig,
    load_analysis_config,
)

__all__ = ["AnalysisConfig", "load_analysis_config"]
