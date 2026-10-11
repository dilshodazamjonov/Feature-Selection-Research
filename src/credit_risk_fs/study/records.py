"""Definition-only feature records, source families and the record ablations.

Every record carries the feature name, its description, source family, lineage formula
and semantic group (see :mod:`credit_risk_fs.feature_metadata.definitions`); no data
row or statistic enters a record.

Ablations of the records:

* name obfuscation (arm A): every name is replaced by an identifier ``F001``,
  ``F002``, ... drawn from a seeded permutation of the universe, a fresh permutation
  for each partition, and every literal occurrence of a universe name is replaced in
  every other field; the identifier-to-name table never enters a prompt;
* blanked definitions (arm B): names kept, descriptions blanked;
* lineage-only descriptions: every description rebuilt from lineage by a fixed
  template (raw table, raw variable, depth, operation, window, type rule).
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

import numpy as np

from credit_risk_fs.study.config import PARTITIONS, StudyError, resolve


def candidate_records(cfg: dict[str, Any], names: Sequence[str], categorical: Sequence[str]) -> list[dict[str, Any]]:
    from credit_risk_fs.feature_metadata import definitions

    loader = cfg["loader"]
    if loader == "homecredit":
        return definitions.homecredit_definition_records(
            names,
            description_csv_path=resolve(cfg, cfg["description_csv"]),
            categorical=set(categorical),
        )
    if loader == "lendingclub_v2":
        return definitions.lendingclub_definition_records(names, inventory_path=resolve(cfg, cfg["inventory"]))
    if loader == "stability_2024":
        return definitions.stability_definition_records(
            names,
            matrix_root=resolve(cfg, cfg["matrix_root"]),
            protocol_lock=resolve(cfg, cfg["protocol_lock"]),
        )
    raise StudyError(f"unknown loader {loader!r}")


def source_families(cfg: dict[str, Any], names: Sequence[str]) -> dict[str, str]:
    from credit_risk_fs.feature_metadata import definitions

    loader = cfg["loader"]
    if loader == "homecredit":
        return {str(name): definitions.homecredit_lineage(str(name))[0] for name in names}
    if loader == "lendingclub_v2":
        return definitions.lendingclub_families(names, inventory_path=resolve(cfg, cfg["inventory"]))
    if loader == "stability_2024":
        lineage = definitions.load_stability_lineage(resolve(cfg, cfg["matrix_root"]))
        return {str(name): str(lineage[str(name)]["source_family"]) for name in names}
    raise StudyError(f"unknown loader {loader!r}")


def identifier_mapping(universe: Sequence[str], partition: str, seed: int) -> dict[str, str]:
    """Seeded name -> identifier table, drawn afresh for each partition."""

    names = sorted(str(name) for name in universe)
    index = PARTITIONS.index(partition)
    permutation = np.random.default_rng([int(seed), index]).permutation(len(names))
    width = max(3, len(str(len(names))))
    return {names[int(position)]: f"F{rank:0{width}d}" for rank, position in enumerate(permutation, start=1)}


def obfuscated_records(
    records: Sequence[dict[str, Any]],
    universe: Sequence[str],
    partition: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Arm A records and the identifier -> name table (kept out of every prompt)."""

    mapping = identifier_mapping(universe, partition, seed)
    pattern = re.compile("|".join(re.escape(name) for name in sorted(mapping, key=lambda item: (-len(item), item))))

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            return pattern.sub(lambda match: mapping[match.group(0)], value)
        return value

    scrubbed = []
    for record in records:
        item = {key: scrub(value) for key, value in record.items()}
        item["name"] = mapping[str(record["name"])]
        scrubbed.append(item)
    serialised = json.dumps(
        [{key: record[key] for key in ("name", "description", "source_family", "lineage", "semantic_group")} for record in scrubbed],
        ensure_ascii=False,
    )
    leaks = [name for name in mapping if name in serialised]
    if leaks:
        raise StudyError(f"obfuscated records still contain {len(leaks)} original name literal(s)")
    return scrubbed, {identifier: name for name, identifier in mapping.items()}


def blanked_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Arm B records: names kept, descriptions blanked."""

    return [{**dict(record), "description": ""} for record in records]


def lineage_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    from credit_risk_fs.feature_metadata.definitions import with_mechanical_descriptions

    return with_mechanical_descriptions(records)


def protected_features(cfg: dict[str, Any], names: Sequence[str]) -> set[str]:
    exact = set(cfg.get("protected", {}).get("names", []))
    patterns = list(cfg.get("protected", {}).get("patterns", []))
    return {str(name) for name in names if name in exact or any(token in str(name) for token in patterns)}


__all__ = [
    "blanked_records",
    "candidate_records",
    "identifier_mapping",
    "lineage_records",
    "obfuscated_records",
    "protected_features",
    "source_families",
]
