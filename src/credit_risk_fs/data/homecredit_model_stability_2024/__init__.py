"""Frozen Home Credit Model Stability 2024 relational adapter.

Importing this package is deliberately inert. Data access is possible only
through an explicitly constructed adapter invocation.
"""

from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
    ADAPTER_VERSION,
    AdapterContract,
    FeatureRule,
    PartitionSpec,
    ProtocolContractError,
    TableSpec,
    load_adapter_contract,
)

__all__ = [
    "ADAPTER_VERSION",
    "AdapterContract",
    "FeatureRule",
    "PartitionSpec",
    "ProtocolContractError",
    "TableSpec",
    "load_adapter_contract",
]
