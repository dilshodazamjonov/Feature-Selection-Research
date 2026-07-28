"""Heavy selectors registered through the shared Prompt 7 selector contract.

These methods fit real estimators -- sometimes many times -- so they carry a
``heavy`` cost class, stage logging, and resource-observation metadata. They are
*not* a second framework: every class here subclasses
:class:`credit_risk_fs.selectors.lightweight.contract.LightweightSelector` and is
declared in the same ``MethodDescriptor`` registry as the light methods.

The pre-existing :class:`~credit_risk_fs.selectors.rfe.RFESelector` and
:class:`~credit_risk_fs.selectors.boruta.BorutaSelector` are untouched. The frozen
voting protocol continues to resolve its ``boruta`` voter to the historical
implementation; nothing here is permitted in that protocol.
"""

from __future__ import annotations

from credit_risk_fs.selectors.heavy.boruta_rf import BorutaRandomForestSelector
from credit_risk_fs.selectors.heavy.catboost_shap import CatBoostShapSelector
from credit_risk_fs.selectors.heavy.rfe_catboost import CatBoostRFESelector

__all__ = [
    "BorutaRandomForestSelector",
    "CatBoostRFESelector",
    "CatBoostShapSelector",
]
