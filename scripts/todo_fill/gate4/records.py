"""Outcome-independent candidate records for the target-free ranking prompt (Appendix A template).

Three record builders per dataset share one schema (the ``TARGET_FREE_METADATA_FIELDS`` of
``LLMSelector``): ``named`` (the cached run's definitions), ``mechanical`` (step 44: the
description rebuilt from lineage alone by a per-dataset template), and the two obfuscation
arms of step 47 (``A``: names replaced by seeded identifiers and scrubbed from every text
field; ``B``: definitions blanked, every name/lineage field kept).

* Home Credit uses the frozen B6 builder (``scripts/b6_obfuscation.py``): Kaggle dictionary
  text plus semantic group and lineage clause.
* LendingClub v2 uses ``data/lendingclub_v2/metadata/feature_inventory.csv``: the author's
  description, the semantic group as source family, the lineage formula as original feature
  and the feature type as aggregation.
* The third dataset uses the frozen Prompt-16 renderer over the adapter lineage and the
  approved feature definitions, restricted to the availability-filtered candidate set.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.todo_fill import llm_cache
from scripts.todo_fill.common import (
    FULL_DEV,
    HOMECREDIT,
    LENDINGCLUB,
    REPO_ROOT,
    THIRD,
    FillError,
    Manifest,
    log,
    read_json,
    require_ram,
    write_json,
)
from scripts.todo_fill.data import THIRD_LOCK, DataContext
from scripts.todo_fill.gate4.config import THIRD_DEPTH0_FAMILIES, THIRD_EXPECTED_CANDIDATES

LC_INVENTORY = REPO_ROOT / "data/lendingclub_v2/metadata/feature_inventory.csv"
LC_SOURCE_TABLE = "lendingclub_application"
_FORMULA_FUNCTIONS = {
    "is_missing", "log1p", "sqrt", "max", "min", "mean", "fixed_bins", "group", "domain_cap",
    "derived_from_safe_fields", "abs", "log", "exp", "clip", "sum", "avg", "median", "mode", "var",
    "std", "count", "where", "if", "else", "and", "or", "not", "nan", "none", "true", "false",
}
#: Third-dataset official suffix letters -> the dataset's type rule.
THIRD_TYPE_RULE = {
    "P": "P (days past due transform)",
    "A": "A (amount transform)",
    "M": "M (masked category)",
    "D": "D (date transform)",
    "T": "T (unspecified transform)",
    "L": "L (unspecified transform)",
}
THIRD_OPERATION = {
    "identity_after_family_prefix": "none",
    "count_non_missing": "count",
    "missing_count": "count of missing",
    "row_count": "count",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "sum": "sum",
    "sample_variance_ddof_1": "std (sample variance)",
    "first_by_num_group1": "first",
    "last_by_num_group1": "last",
    "lexical_mode": "mode",
    "nunique": "nunique",
    "signed_days_relative_to_base_date_decision": "none (signed days relative to decision date)",
    "false_count": "count of false",
    "true_count": "count of true",
    "any": "any",
    "all": "all",
}
_WINDOW = re.compile(r"(\d+)(m|d|y)(?![a-z])", re.IGNORECASE)


def _b6():
    import scripts.b6_obfuscation as helpers
    import scripts.run_b6_obfuscation_ablation as runner

    return helpers, runner


class RecordFactory:
    """Universe, dtype, family and record access for the three datasets."""

    def __init__(self, data: DataContext, work_dir: Path, manifest: Manifest, candidate_source: str = "cache") -> None:
        self.data = data
        self.work_dir = Path(work_dir)
        self.manifest = manifest
        self.candidate_source = candidate_source
        self._universe: dict[str, list[str]] = {}
        self._dtypes: dict[str, dict[str, str]] = {}
        self._families: dict[str, dict[str, str]] = {}
        self._lc_inventory: dict[str, dict[str, str]] | None = None
        self._third_lineage: dict[str, dict[str, Any]] | None = None
        self._third_contract: Any = None
        self._mapping: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------ universe
    def universe(self, dataset: str) -> list[str]:
        if dataset not in self._universe:
            if dataset == THIRD:
                self._universe[dataset] = list(self.data.third().predictors)
            else:
                self._universe[dataset] = list(json.loads((self.work_dir / "frames" / dataset / "universe.json").read_text(encoding="utf-8")))
        return self._universe[dataset]

    def dtype_strings(self, dataset: str) -> dict[str, str]:
        if dataset not in self._dtypes:
            if dataset == THIRD:
                metadata = read_json(self.work_dir / "hcms2024_matrix" / "metadata.json")
                arrow = {str(row["name"]): str(row["arrow_type"]) for row in metadata["columns"]}
                self._dtypes[dataset] = {name: arrow.get(name, "double") for name in self.universe(dataset)}
            else:
                self._dtypes[dataset] = json.loads((self.work_dir / "frames" / dataset / "dev_dtypes.json").read_text(encoding="utf-8"))
        return self._dtypes[dataset]

    def is_categorical(self, dataset: str, name: str) -> bool:
        dtype = self.dtype_strings(dataset).get(name, "")
        if dataset == THIRD:
            return dtype in {"string", "large_string"}
        return dtype in {"str", "object", "category", "string"}

    def families(self, dataset: str) -> dict[str, str]:
        """Source family of every universe feature (HC: source table; LC: semantic group; third: table)."""

        if dataset not in self._families:
            if dataset == HOMECREDIT:
                helpers, _ = _b6()
                self._families[dataset] = {name: helpers._source_and_lineage(name)[0] for name in self.universe(dataset)}
            elif dataset == LENDINGCLUB:
                inventory = self.lc_inventory()
                self._families[dataset] = {name: inventory.get(name, {}).get("semantic_group") or "unknown" for name in self.universe(dataset)}
            else:
                lineage = self.third_lineage()
                self._families[dataset] = {name: str(lineage[name]["source_family"]) for name in self.universe(dataset)}
        return self._families[dataset]

    def n_families(self, dataset: str) -> int:
        return len(set(self.families(dataset).values()))

    def third_depth0_universe(self) -> list[str]:
        lineage = self.third_lineage()
        return [name for name in self.universe(THIRD) if lineage[name]["source_family"] in THIRD_DEPTH0_FAMILIES]

    # ------------------------------------------------------------ metadata
    def lc_inventory(self) -> dict[str, dict[str, str]]:
        if self._lc_inventory is None:
            with LC_INVENTORY.open("r", encoding="utf-8-sig", newline="") as handle:
                self._lc_inventory = {row["feature"]: row for row in csv.DictReader(handle)}
        return self._lc_inventory

    def third_lineage(self) -> dict[str, dict[str, Any]]:
        if self._third_lineage is None:
            payload = read_json(self.work_dir / "hcms2024_matrix" / "lineage.json")
            self._third_lineage = {str(row["output_feature"]): row for row in payload["features"]}
        return self._third_lineage

    def third_contract(self):
        if self._third_contract is None:
            from credit_risk_fs.data.homecredit_model_stability_2024 import load_adapter_contract

            self._third_contract = load_adapter_contract(THIRD_LOCK)
        return self._third_contract

    # ------------------------------------------------------- candidate sets
    def candidate_set(self, dataset: str, partition: str, budget: int | None = None) -> list[str]:
        """Candidate names in universe order: the cached run's list (default) or the full universe."""

        universe = self.universe(dataset)
        if self.candidate_source == "universe" and dataset != THIRD:
            return list(universe)
        if dataset == THIRD:
            return self.third_candidates()
        if dataset == HOMECREDIT:
            cached = set(llm_cache.ranking(HOMECREDIT, partition).candidate_features)
        else:
            files = [item for item in llm_cache.canonical_files(LENDINGCLUB) if item.partition == partition]
            if budget is not None and any(item.feature_budget == budget for item in files):
                files = [item for item in files if item.feature_budget == budget]
            if not files:
                raise FillError(f"no cached LendingClub ranking for {partition}")
            sets = [set(item.candidate_features) for item in files]
            cached = sets[0]
            if any(other != cached for other in sets[1:]):
                cached = set().union(*sets)
                self.manifest.note("records", f"lendingclub_v2/{partition}: cached candidate lists differ across budget files; using their union ({len(cached)} names)")
        names = [name for name in universe if name in cached]
        if len(names) != len(cached):
            self.manifest.note("records", f"{dataset}/{partition}: {len(cached) - len(names)} cached candidates are outside the loaded universe and were dropped")
        return names

    def third_candidates(self) -> list[str]:
        """Frozen Prompt-16 availability filter: fold-1 DEV-training missing rate <= 0.90."""

        cache = self.work_dir / "gate4" / "third_candidates.json"
        if cache.exists():
            return list(read_json(cache)["retained"])
        third = self.data.third()
        universe = self.universe(THIRD)
        require_ram(6.0, "third dataset availability filter (fold-1 training slice)")
        log("[records] computing the third-dataset availability filter on the fold-1 training slice")
        date_min, date_max, expected = third.date_range("fold1")
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        from datetime import date

        missing = np.zeros(len(universe), dtype=np.int64)
        rows = 0
        for path in third._part_paths():
            boundary = pq.read_table(path, columns=["date_decision"])["date_decision"]
            lower, upper = pd.Timestamp(date_min), pd.Timestamp(date_max)
            values = pd.to_datetime(boundary.to_pandas())
            mask = ((values >= lower) & (values <= upper)).to_numpy()
            if not mask.any():
                continue
            indices = np.flatnonzero(mask)
            table = pq.read_table(path, columns=universe).take(indices)
            rows += table.num_rows
            for position, name in enumerate(universe):
                missing[position] += int(pc.sum(pc.is_null(table.column(name), nan_is_null=True)).as_py() or 0)
            del table
        if rows != expected:
            raise FillError(f"availability filter saw {rows} fold-1 training rows; protocol lock expects {expected}")
        rate = missing / rows
        retained = [name for name, value in zip(universe, rate) if value <= 0.90]
        payload = {"rule": "fold-1 DEV-training missing rate <= 0.90", "rows": rows, "retained": retained, "dropped": [name for name in universe if name not in set(retained)], "missing_rate": {name: float(value) for name, value in zip(universe, rate)}}
        write_json(cache, payload)
        if len(retained) != THIRD_EXPECTED_CANDIDATES:
            self.manifest.note("records", f"third-dataset availability filter retained {len(retained)} candidates; TODO.md states {THIRD_EXPECTED_CANDIDATES}")
        return retained

    # ---------------------------------------------------------------- mapping
    def mapping(self, dataset: str) -> dict[str, str]:
        if dataset not in self._mapping:
            helpers, _ = _b6()
            universe = self.universe(dataset)
            self._mapping[dataset] = helpers.build_global_feature_mapping(universe, expected_count=len(universe))
        return self._mapping[dataset]

    # ---------------------------------------------------------------- records
    def _pandas_dtype(self, dataset: str, name: str):
        text = self.dtype_strings(dataset).get(name, "float64")
        try:
            return pd.Series([], dtype=text).dtype
        except (TypeError, ValueError):
            return np.dtype("O") if self.is_categorical(dataset, name) else np.dtype("float64")

    def named_records(self, dataset: str, names: Sequence[str]) -> list[dict[str, Any]]:
        names = list(names)
        helpers, runner = _b6()
        if dataset == HOMECREDIT:
            dtypes = {name: self._pandas_dtype(dataset, name) for name in names}
            return helpers.build_homecredit_definition_records(names, dtypes=dtypes, description_csv_path=REPO_ROOT / runner.DESCRIPTION_PATH, expected_count=len(names))
        if dataset == LENDINGCLUB:
            inventory = self.lc_inventory()
            records = []
            for name in names:
                row = inventory.get(name)
                if row is None:
                    raise FillError(f"{name} is missing from the LendingClub v2 feature inventory")
                records.append(helpers._render_definition_record(self._lc_fields(name, row, " ".join(str(row.get("description", "")).split()))))
            return records
        from credit_risk_fs.experiments.prompt_16_llm_supplement import render_target_free_feature_descriptions

        metadata = read_json(self.work_dir / "hcms2024_matrix" / "metadata.json")
        lineage = read_json(self.work_dir / "hcms2024_matrix" / "lineage.json")
        return render_target_free_feature_descriptions(predictors=names, lineage_payload=lineage, metadata_payload=metadata, contract=self.third_contract())

    def _lc_fields(self, name: str, row: dict[str, str], definition: str) -> dict[str, Any]:
        return {
            "name": name,
            "source_family": row.get("semantic_group") or "unknown",
            "source_table": LC_SOURCE_TABLE,
            "original_feature": (row.get("source_column_or_formula") or name).strip(),
            "depth": "0",
            "aggregation": row.get("feature_type") or "raw",
            "dtype": self.dtype_strings(LENDINGCLUB).get(name, "float64"),
            "logical_type": "categorical" if self.is_categorical(LENDINGCLUB, name) else "numeric",
            "approved_definition": definition,
        }

    def _lc_source_columns(self, name: str, row: dict[str, str]) -> list[str]:
        formula = (row.get("source_column_or_formula") or name).strip()
        if formula == name or (row.get("feature_type") or "") == "raw":
            return [name]
        tokens = [token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula) if token.lower() not in _FORMULA_FUNCTIONS]
        ordered = [token for token in dict.fromkeys(tokens) if token != name]
        return ordered or [name]

    def hc_flag_columns(self) -> set[str]:
        cache = self.work_dir / "gate4" / "hc_flag_columns.json"
        if cache.exists():
            return set(read_json(cache))
        import pyarrow.parquet as pq

        universe = self.universe(HOMECREDIT)
        numeric = [name for name in universe if not self.is_categorical(HOMECREDIT, name)]
        table = pq.read_table(self.work_dir / "frames" / HOMECREDIT / "dev_X.parquet", columns=numeric)
        flags: list[str] = []
        for name in numeric:
            values = pd.Series(table.column(name).to_pandas()).dropna().unique()
            if len(values) and set(np.asarray(values, dtype=float).tolist()) <= {0.0, 1.0}:
                flags.append(name)
        write_json(cache, flags)
        return set(flags)

    def mechanical_records(self, dataset: str, names: Sequence[str]) -> list[dict[str, Any]]:
        """Descriptions rebuilt by code from lineage alone; every other field as in ``named_records``."""

        helpers, _ = _b6()
        named = self.named_records(dataset, names)
        records = []
        if dataset == HOMECREDIT:
            flags = self.hc_flag_columns()
            for record in named:
                table, original, aggregation, _ = helpers._source_and_lineage(record["name"])
                kind = "categorical" if record["logical_type"] == "categorical" else ("flag" if record["name"] in flags else "numeric")
                definition = f"table {table}; raw variable {original}; operation {'none' if aggregation == 'identity' else aggregation}; type {kind}"
                records.append(self._with_definition(record, definition))
            return records
        if dataset == LENDINGCLUB:
            inventory = self.lc_inventory()
            for record in named:
                row = inventory[record["name"]]
                columns = ", ".join(self._lc_source_columns(record["name"], row))
                definition = f"{row.get('feature_type') or 'raw'}; source columns {columns}; type {record['logical_type']}"
                records.append(self._with_definition(record, definition))
            return records
        lineage = self.third_lineage()
        for record in named:
            row = lineage[record["name"]]
            source = row.get("source_feature")
            aggregation = str(row["aggregation"])
            operation = THIRD_OPERATION.get(aggregation, aggregation)
            raw_name = str(source) if source else "row_count"
            window = "none"
            suffix_rule = "count"
            if source:
                stem, _, code = str(source).rpartition("_")
                letter = code[-1] if code else ""
                suffix_rule = THIRD_TYPE_RULE.get(letter, f"{letter} (unspecified)") if letter.isalpha() else "unspecified"
                found = _WINDOW.findall(stem.lower())
                if found:
                    window = "".join(found[-1])
            definition = f"table {record['source_family']}; depth {record['depth']}; raw variable {raw_name}; operation {operation}; window {window}; type {suffix_rule}"
            records.append(self._with_definition(record, definition))
        return records

    @staticmethod
    def _with_definition(record: dict[str, Any], definition: str) -> dict[str, Any]:
        helpers, _ = _b6()
        fields = {key: value for key, value in record.items() if key not in {"rendered_description", "description_sha256"}}
        fields["approved_definition"] = definition
        return helpers._render_definition_record(fields)

    def arm_a(self, dataset: str, named: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        """Names replaced by seeded identifiers and scrubbed from every text field."""

        helpers, _ = _b6()
        mapping = self.mapping(dataset)
        records = helpers.obfuscate_definition_records(list(named), mapping)
        ids = [mapping[str(record["name"])] for record in named]
        return records, ids

    def arm_b(self, named: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Names and lineage fields kept, free-text definition removed."""

        return [self._with_definition(record, "") for record in named]

    def reverse_mapping(self, dataset: str) -> dict[str, str]:
        helpers, _ = _b6()
        return helpers.reverse_feature_mapping(self.mapping(dataset))
