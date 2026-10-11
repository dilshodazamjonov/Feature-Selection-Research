"""Outcome-independent feature records for the LLM ranking prompt.

Every record is built from documentation and lineage only; no data row, statistic or
outcome enters it. Fields used in the prompt: ``name``, ``description``,
``source_family``, ``lineage`` (the lineage formula) and ``semantic_group``. The
lineage parts (``raw_table``, ``raw_variable``, ``depth``, ``operation``, ``window``,
``type_rule``) are kept so that a description can be rebuilt from lineage alone.

* Home Credit: base definitions are the entries of the data dictionary distributed
  with the competition data; engineered aggregates add a clause stating the
  aggregation operation over the applicant's records.
* LendingClub: definitions of the source columns and template descriptions of derived
  features from ``data/lendingclub_v2/metadata/feature_inventory.csv``; the semantic
  group is the source family and ``source_column_or_formula`` is the lineage formula.
* Stability 2024: the approved definitions of the raw predictors recorded in the
  protocol lock, with a clause for the aggregation operation and window.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from credit_risk_fs.feature_metadata.builder import infer_semantic_group

HOMECREDIT_SOURCE_BY_PREFIX = {
    "BURO": "bureau",
    "PREV": "previous_application",
    "POS": "POS_CASH_balance",
    "INSTAL": "installments_payments",
    "CC": "credit_card_balance",
}
HOMECREDIT_AGGREGATIONS = {
    "MIN": "minimum",
    "MAX": "maximum",
    "MEAN": "mean",
    "SUM": "total",
    "VAR": "variance",
}
STABILITY_OPERATION_TEXT = {
    "identity_after_family_prefix": "none",
    "signed_days_relative_to_base_date_decision": "signed days relative to the decision date",
    "row_count": "count of related rows",
    "count_non_missing": "count of non-missing values",
    "missing_count": "count of missing values",
    "mean": "mean",
    "min": "minimum",
    "max": "maximum",
    "sum": "total",
    "sample_variance_ddof_1": "sample variance",
    "first_by_num_group1": "first recorded value",
    "last_by_num_group1": "last recorded value",
    "lexical_mode": "most frequent value",
    "nunique": "number of distinct values",
    "false_count": "count of false values",
    "true_count": "count of true values",
    "any": "whether any value is true",
    "all": "whether all values are true",
}
STABILITY_TYPE_RULE = {
    "P": "P (days past due)",
    "A": "A (amount)",
    "M": "M (masked category)",
    "D": "D (date)",
    "T": "T (unspecified transform)",
    "L": "L (unspecified transform)",
}
LENDINGCLUB_SOURCE_TABLE = "lendingclub_application"
_FORMULA_WORDS = {
    "is_missing", "log1p", "sqrt", "max", "min", "mean", "fixed_bins", "group", "domain_cap",
    "derived_from_safe_fields", "abs", "log", "exp", "clip", "sum", "avg", "median", "mode",
    "var", "std", "count", "where", "if", "else", "and", "or", "not", "nan", "none", "true",
    "false", "region", "numeric_months",
}
_WINDOW = re.compile(r"(\d+)(m|d|y)(?![a-z])", re.IGNORECASE)


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).split())


def _record(
    *,
    name: str,
    description: str,
    source_family: str,
    lineage: str,
    raw_table: str,
    raw_variable: str,
    depth: str,
    operation: str,
    window: str,
    type_rule: str,
    is_categorical: bool,
    semantic_group: str | None = None,
) -> dict[str, Any]:
    group = semantic_group or infer_semantic_group(name, description=description, table=raw_table)
    return {
        "name": str(name),
        "description": _clean(description),
        "source_family": str(source_family),
        "lineage": _clean(lineage),
        "semantic_group": str(group),
        "raw_table": str(raw_table),
        "raw_variable": str(raw_variable),
        "depth": str(depth),
        "operation": str(operation),
        "window": str(window),
        "type_rule": str(type_rule),
        "is_categorical": bool(is_categorical),
    }


def mechanical_description(record: Mapping[str, Any]) -> str:
    """Description rebuilt from lineage alone, with no statement of credit meaning."""

    return (
        f"table {record['raw_table']}; raw variable {record['raw_variable']}; depth {record['depth']}; "
        f"operation {record['operation']}; window {record['window']}; type {record['type_rule']}"
    )


def with_mechanical_descriptions(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replace every description by its lineage-only rebuild; all other fields unchanged."""

    rebuilt = []
    for record in records:
        item = dict(record)
        item["description"] = mechanical_description(record)
        rebuilt.append(item)
    return rebuilt


# ------------------------------------------------------------------ Home Credit
def homecredit_lineage(feature_name: str) -> tuple[str, str, str, str]:
    """Return ``(source_table, original_variable, aggregation, depth)`` from the name."""

    parts = feature_name.split("_")
    prefix = parts[0] if parts else ""
    if prefix in HOMECREDIT_SOURCE_BY_PREFIX and len(parts) >= 3 and parts[-1] in HOMECREDIT_AGGREGATIONS:
        return HOMECREDIT_SOURCE_BY_PREFIX[prefix], "_".join(parts[1:-1]), parts[-1].lower(), "1"
    return "application_train", feature_name, "identity", "0"


def _homecredit_dictionary(description_csv_path: str | Path) -> list[dict[str, str]]:
    path = Path(description_csv_path)
    try:
        frame = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin1")
    columns = {str(column).casefold(): str(column) for column in frame.columns}
    for required in ("table", "row", "description"):
        if required not in columns:
            raise ValueError(f"Home Credit data dictionary lacks the {required!r} column: {path}")
    entries = []
    for _, row in frame.iterrows():
        variable = _clean(row[columns["row"]])
        if variable:
            entries.append(
                {
                    "variable": variable,
                    "table": _clean(row[columns["table"]]),
                    "description": _clean(row[columns["description"]]),
                }
            )
    return entries


def _homecredit_definition(entries: Sequence[Mapping[str, str]], variable: str, table: str) -> str:
    matches = [entry for entry in entries if entry["variable"] == variable]
    if matches:
        token = table.casefold().replace("_", "")
        preferred = [entry for entry in matches if token in entry["table"].casefold().replace("_", "")]
        chosen = (preferred or matches)[0]
        if chosen["description"]:
            return chosen["description"]
    return variable.replace("_", " ").casefold()


def homecredit_definition_records(
    names: Sequence[str],
    *,
    description_csv_path: str | Path,
    categorical: set[str] | None = None,
) -> list[dict[str, Any]]:
    entries = _homecredit_dictionary(description_csv_path)
    categorical = set(categorical or ())
    records = []
    for name in names:
        table, variable, aggregation, depth = homecredit_lineage(str(name))
        definition = _homecredit_definition(entries, variable, table)
        if aggregation == "identity":
            description = definition
            lineage = f"{table}.{variable}"
            operation, window = "none", "none"
        else:
            word = HOMECREDIT_AGGREGATIONS[aggregation.upper()]
            description = f"{definition}; {word} over all of the applicant's {table} records"
            lineage = f"{aggregation}({table}.{variable}) over all {table} records of the applicant"
            operation, window = word, "all records"
        records.append(
            _record(
                name=str(name),
                description=description,
                source_family=table,
                lineage=lineage,
                raw_table=table,
                raw_variable=variable,
                depth=depth,
                operation=operation,
                window=window,
                type_rule="categorical" if name in categorical else "numeric",
                is_categorical=name in categorical,
            )
        )
    return records


# ------------------------------------------------------------------ LendingClub
def _lendingclub_inventory(inventory_path: str | Path) -> dict[str, dict[str, str]]:
    with Path(inventory_path).open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["feature"]: row for row in csv.DictReader(handle)}


def _lendingclub_source_columns(name: str, formula: str, feature_type: str) -> str:
    if not formula or formula == name or feature_type == "raw":
        return formula or name
    tokens = [
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
        if token.lower() not in _FORMULA_WORDS
    ]
    ordered = [token for token in dict.fromkeys(tokens) if token != name]
    return ", ".join(ordered) if ordered else name


def lendingclub_definition_records(
    names: Sequence[str],
    *,
    inventory_path: str | Path,
) -> list[dict[str, Any]]:
    inventory = _lendingclub_inventory(inventory_path)
    records = []
    for name in names:
        row = inventory.get(str(name))
        if row is None:
            raise ValueError(f"{name} is missing from the LendingClub feature inventory {inventory_path}")
        formula = _clean(row.get("source_column_or_formula")) or str(name)
        feature_type = _clean(row.get("feature_type")) or "raw"
        group = _clean(row.get("semantic_group")) or "unknown"
        dtype = _clean(row.get("dtype")).lower()
        categorical = dtype in {"str", "object", "category", "string"}
        records.append(
            _record(
                name=str(name),
                description=_clean(row.get("description")),
                source_family=group,
                lineage=formula,
                raw_table=LENDINGCLUB_SOURCE_TABLE,
                raw_variable=_lendingclub_source_columns(str(name), formula, feature_type),
                depth="0",
                operation=feature_type,
                window="none",
                type_rule="categorical" if categorical else "numeric",
                is_categorical=categorical,
                semantic_group=group,
            )
        )
    return records


def lendingclub_families(names: Sequence[str], *, inventory_path: str | Path) -> dict[str, str]:
    inventory = _lendingclub_inventory(inventory_path)
    return {str(name): _clean(inventory.get(str(name), {}).get("semantic_group")) or "unknown" for name in names}


# ------------------------------------------------------------------ Stability 2024
def load_stability_lineage(matrix_root: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((Path(matrix_root) / "lineage.json").read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"lineage registry is missing in {matrix_root}/lineage.json")
    return {str(row["output_feature"]): dict(row) for row in features}


def stability_definition_records(
    names: Sequence[str],
    *,
    matrix_root: str | Path,
    protocol_lock: str | Path,
) -> list[dict[str, Any]]:
    from credit_risk_fs.data.homecredit_model_stability_2024 import load_adapter_contract

    lineage = load_stability_lineage(matrix_root)
    contract = load_adapter_contract(protocol_lock)
    rules = {
        (table.family, rule.feature_name): rule
        for table in contract.tables
        for rule in table.feature_rules
    }
    depth_of = {table.family: str(table.depth) for table in contract.tables}
    records = []
    for name in names:
        row = lineage.get(str(name))
        if row is None:
            raise ValueError(f"{name} is missing from the Stability 2024 lineage registry")
        family = str(row["source_family"])
        source = row.get("source_feature")
        aggregation = str(row["aggregation"])
        logical_type = str(row.get("logical_type", ""))
        depth = depth_of.get(family, "1")
        operation = STABILITY_OPERATION_TEXT.get(aggregation, aggregation)
        if source is None:
            definition = f"Number of related rows in the {family} source table for the case"
            raw_variable = "row_count"
            lineage_formula = f"row_count({family}) by case"
            description = definition
            window = "none"
            type_rule = "count"
        else:
            rule = rules.get((family, str(source)))
            if rule is None or not _clean(rule.description):
                raise ValueError(f"approved definition is missing for {name}")
            definition = _clean(rule.description)
            raw_variable = str(source)
            if aggregation == "identity_after_family_prefix":
                description = definition
                lineage_formula = f"{family}.{source}"
            elif aggregation == "signed_days_relative_to_base_date_decision":
                description = f"{definition}; expressed as signed days relative to the decision date"
                lineage_formula = f"days({family}.{source} - decision date)"
            else:
                description = f"{definition}; {operation} over the case's related {family} rows"
                lineage_formula = f"{aggregation}({family}.{source}) by case"
            stem, _, code = str(source).rpartition("_")
            found = _WINDOW.findall(stem.lower())
            window = "".join(found[-1]) if found else "none"
            letter = code[-1] if code else ""
            type_rule = STABILITY_TYPE_RULE.get(letter, f"{letter} (unspecified)") if letter.isalpha() else "unspecified"
        records.append(
            _record(
                name=str(name),
                description=description,
                source_family=family,
                lineage=lineage_formula,
                raw_table=family,
                raw_variable=raw_variable,
                depth=depth,
                operation=operation,
                window=window,
                type_rule=type_rule,
                is_categorical=logical_type in {"categorical", "string"},
            )
        )
    return records


# ------------------------------------------------------------------ generic
def generic_definition_records(
    names: Sequence[str],
    *,
    description_csv_path: str | Path | None,
) -> list[dict[str, Any]]:
    """Records for a dataset described only by a name/description CSV."""

    descriptions: dict[str, str] = {}
    tables: dict[str, str] = {}
    if description_csv_path is not None and Path(description_csv_path).exists():
        try:
            frame = pd.read_csv(description_csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            frame = pd.read_csv(description_csv_path, encoding="latin1")
        columns = {str(column).casefold(): str(column) for column in frame.columns}
        name_column = next((columns[key] for key in ("row", "feature", "column", "name") if key in columns), None)
        description_column = next((columns[key] for key in ("description", "desc") if key in columns), None)
        table_column = next((columns[key] for key in ("table", "source") if key in columns), None)
        if name_column and description_column:
            for _, row in frame.iterrows():
                key = _clean(row[name_column])
                if key:
                    descriptions[key] = _clean(row[description_column])
                    if table_column:
                        tables[key] = _clean(row[table_column])
    records = []
    for name in names:
        table, variable, aggregation, depth = homecredit_lineage(str(name))
        definition = descriptions.get(str(name)) or descriptions.get(variable) or variable.replace("_", " ").casefold()
        source = tables.get(str(name)) or table
        if aggregation == "identity":
            description, lineage, operation, window = definition, f"{source}.{variable}", "none", "none"
        else:
            word = HOMECREDIT_AGGREGATIONS[aggregation.upper()]
            description = f"{definition}; {word} over all of the applicant's {source} records"
            lineage = f"{aggregation}({source}.{variable}) over all {source} records of the applicant"
            operation, window = word, "all records"
        records.append(
            _record(
                name=str(name),
                description=description,
                source_family=source,
                lineage=lineage,
                raw_table=source,
                raw_variable=variable,
                depth=depth,
                operation=operation,
                window=window,
                type_rule="unspecified",
                is_categorical=False,
            )
        )
    return records


__all__ = [
    "generic_definition_records",
    "homecredit_definition_records",
    "homecredit_lineage",
    "lendingclub_definition_records",
    "lendingclub_families",
    "load_stability_lineage",
    "mechanical_description",
    "stability_definition_records",
    "with_mechanical_descriptions",
]
