"""Lightweight classical feature-selection controls (Prompt 7).

Every selector here implements the repository's existing ``FeatureSelector``
protocol, so the fold runner consumes them unchanged, and additionally records a
:class:`~credit_risk_fs.selectors.lightweight.contract.SelectionResult` carrying
method identity, ranking, budget feasibility, fit boundary, and tie policy.
"""

from __future__ import annotations

from credit_risk_fs.selectors.lightweight.contract import (
    CONTRACT_VERSION,
    DEV_FOLD_TRAINING_ONLY,
    LONG_FRAME_COLUMNS,
    SELECTION_MODES,
    ControlledSelectorFailure,
    LightweightSelector,
    SelectionResult,
    SelectorContractError,
)
from credit_risk_fs.selectors.lightweight.controls import (
    FullCandidateFeaturesSelector,
    RandomKSelector,
)
from credit_risk_fs.selectors.lightweight.iv import InformationValueSelector
from credit_risk_fs.selectors.lightweight.lasso import L1LogisticSelector
from credit_risk_fs.selectors.lightweight.mi_mrmr import MutualInformationMRMRSelector
from credit_risk_fs.selectors.lightweight.registry import (
    LIGHTWEIGHT_METHODS,
    MethodDescriptor,
    get_method_descriptor,
    lightweight_method_ids,
    registry_snapshot,
    resolve_method_id,
    validate_method_selection_mode,
)

__all__ = [
    "CONTRACT_VERSION",
    "DEV_FOLD_TRAINING_ONLY",
    "LIGHTWEIGHT_METHODS",
    "LONG_FRAME_COLUMNS",
    "SELECTION_MODES",
    "ControlledSelectorFailure",
    "FullCandidateFeaturesSelector",
    "InformationValueSelector",
    "L1LogisticSelector",
    "LightweightSelector",
    "MethodDescriptor",
    "MutualInformationMRMRSelector",
    "RandomKSelector",
    "SelectionResult",
    "SelectorContractError",
    "get_method_descriptor",
    "lightweight_method_ids",
    "registry_snapshot",
    "resolve_method_id",
    "validate_method_selection_mode",
]
