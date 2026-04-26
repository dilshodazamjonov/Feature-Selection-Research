from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_RANDOM_SEED = 42
DEFAULT_RESULTS_DIR = "results"
DEFAULT_FEATURE_BUDGETS = {"lr": 20, "catboost": 40}
DEFAULT_EXCLUDED_FEATURE_COLUMNS = [
    "TARGET",
    "recent_decision",
    "PREV_recent_decision_MAX",
    "DAYS_DECISION",
    "application_time_proxy",
]
DEFAULT_CONFIG: dict[str, Any] = {
    "model_selector": "lr",
    "data_dir": "data/inputs",
    "description_path": "data/HomeCredit_columns_description.csv",
    "results_dir": DEFAULT_RESULTS_DIR,
    "random_seed": DEFAULT_RANDOM_SEED,
    "feature_budgets": DEFAULT_FEATURE_BUDGETS,
    "n_splits": 5,
    "dev_start_day": -600,
    "oot_start_day": -240,
    "oot_end_day": 0,
    "cv_gap_groups": 1,
    "excluded_feature_columns": DEFAULT_EXCLUDED_FEATURE_COLUMNS,
    "llm": {
        "enabled": True,
        "shared_ranking_enabled": True,
        "ranking_budget": 40,
        "model": "gpt-4.1-mini",
        "max_features": 40,
        "cache_dir": "results/llm_selector_cache",
    },
    "model_params": {
        "lr": {},
        "rf": {},
        "catboost": {},
    },
    "statistical_comparison": {
        "output_dir": "results/statistical_comparison",
        "selectors": ["mrmr", "boruta", "pca"],
    },
    "llm_vs_statistical": {
        "output_dir": "results/llm_vs_statistical",
        "stat_selectors": ["mrmr", "boruta"],
    },
    "hybrid_comparison": {
        "output_dir": "results/hybrid_comparison",
        "stat_selector": "mrmr",
        "improvement_tolerance": 1e-4,
    },
    "single_experiment": {
        "output_dir": "results/single_experiment",
        "selector": "llm",
    },
}


def extract_config_path(argv: list[str] | None = None) -> str:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    known_args, _ = bootstrap.parse_known_args(argv)
    return known_args.config


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    output: list[str] = []
    for char in line:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        output.append(char)
    return "".join(output).rstrip()


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    lower = text.lower()
    if lower in {"null", "none"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if text.startswith(("'", '"')) and text.endswith(("'", '"')):
        return ast.literal_eval(text)
    try:
        if any(token in text for token in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_yaml_block(lines: list[tuple[int, str]], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        return {}, start

    _, first_line = lines[start]
    is_list = first_line.startswith("- ")
    if is_list:
        items: list[Any] = []
        index = start
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent < indent:
                break
            if line_indent != indent or not content.startswith("- "):
                break
            item_text = content[2:].strip()
            index += 1
            if item_text:
                items.append(_parse_scalar(item_text))
            else:
                nested, index = _parse_yaml_block(lines, index, indent + 2)
                items.append(nested)
        return items, index

    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent:
            break
        if ":" not in content:
            raise ValueError(f"Invalid config line: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        value_text = raw_value.strip()
        index += 1
        if value_text:
            mapping[key] = _parse_scalar(value_text)
        else:
            nested, index = _parse_yaml_block(lines, index, indent + 2)
            mapping[key] = nested
    return mapping, index


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    normalized_lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        normalized_lines.append((indent, line.strip()))
    if not normalized_lines:
        return {}
    parsed, _ = _parse_yaml_block(normalized_lines, 0, normalized_lines[0][0])
    if not isinstance(parsed, dict):
        raise ValueError("Top-level config must be a mapping.")
    return parsed


def load_project_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)

    raw_text = config_path.read_text(encoding="utf-8")
    parsed = _parse_simple_yaml(raw_text)
    return _merge_dicts(DEFAULT_CONFIG, parsed)


def canonical_config_json(config: dict[str, Any]) -> str:
    """Serialize a config deterministically for audit hashes."""
    return json.dumps(config, sort_keys=True, ensure_ascii=True, default=str)


def compute_config_hash(config: dict[str, Any]) -> str:
    """Return a SHA-256 hash of the exact effective config payload."""
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def apply_random_seed_to_kwargs(value: Any, random_seed: int) -> Any:
    """
    Recursively apply a run seed to selector/model kwargs that expose a
    ``random_state`` field.

    Existing ``random_state`` values are overwritten intentionally so one run
    seed controls the whole experiment and reruns are defensible.
    """
    if isinstance(value, dict):
        seeded = {
            key: apply_random_seed_to_kwargs(item, random_seed)
            for key, item in value.items()
        }
        if "random_state" in seeded:
            seeded["random_state"] = random_seed
        return seeded
    if isinstance(value, list):
        return [apply_random_seed_to_kwargs(item, random_seed) for item in value]
    return value


def resolve_feature_budget(config: dict[str, Any], model_name: str) -> int:
    """Return the configured selected-feature budget for a model family."""
    budgets = config.get("feature_budgets", DEFAULT_FEATURE_BUDGETS)
    if not isinstance(budgets, dict):
        return int(DEFAULT_FEATURE_BUDGETS.get(model_name.lower(), 40))
    return int(budgets.get(model_name.lower(), DEFAULT_FEATURE_BUDGETS.get(model_name.lower(), 40)))


def apply_feature_budget_to_selector_kwargs(
    selector_name: str,
    selector_kwargs: dict[str, Any],
    feature_budget: int,
) -> dict[str, Any]:
    """Apply model-specific feature budgets to supported selectors."""
    name = selector_name.lower()
    updated = copy.deepcopy(selector_kwargs)

    if name == "mrmr":
        updated["k"] = feature_budget
    elif name in {"boruta", "boruta_rfe"}:
        rfe_kwargs = dict(updated.get("rfe_kwargs", {}))
        rfe_kwargs["n_features"] = feature_budget
        updated["rfe_kwargs"] = rfe_kwargs
        updated["n_features"] = feature_budget
    elif name == "pca":
        updated["n_components"] = feature_budget
    elif name == "llm":
        updated["feature_budget"] = feature_budget

    return updated


def build_parser_defaults(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = config.get(section_name, {})
    llm = config.get("llm", {})
    return {
        "config_path": DEFAULT_CONFIG_PATH,
        "model": config.get("model_selector", "lr"),
        "data_dir": config.get("data_dir", DEFAULT_CONFIG["data_dir"]),
        "description_path": config.get("description_path", DEFAULT_CONFIG["description_path"]),
        "results_dir": config.get("results_dir", DEFAULT_CONFIG["results_dir"]),
        "random_seed": config.get("random_seed", DEFAULT_CONFIG["random_seed"]),
        "feature_budgets": config.get("feature_budgets", DEFAULT_CONFIG["feature_budgets"]),
        "n_splits": config.get("n_splits", DEFAULT_CONFIG["n_splits"]),
        "dev_start_day": config.get("dev_start_day", DEFAULT_CONFIG["dev_start_day"]),
        "oot_start_day": config.get("oot_start_day", DEFAULT_CONFIG["oot_start_day"]),
        "oot_end_day": config.get("oot_end_day", DEFAULT_CONFIG["oot_end_day"]),
        "cv_gap_groups": config.get("cv_gap_groups", DEFAULT_CONFIG["cv_gap_groups"]),
        "llm_model": llm.get("model", DEFAULT_CONFIG["llm"]["model"]),
        "llm_max_features": llm.get("max_features", DEFAULT_CONFIG["llm"]["max_features"]),
        "llm_shared_ranking_enabled": llm.get(
            "shared_ranking_enabled",
            DEFAULT_CONFIG["llm"]["shared_ranking_enabled"],
        ),
        "llm_ranking_budget": llm.get("ranking_budget", DEFAULT_CONFIG["llm"]["ranking_budget"]),
        "llm_cache_dir": llm.get("cache_dir", DEFAULT_CONFIG["llm"]["cache_dir"]),
        "output_dir": section.get("output_dir"),
        "selectors": section.get("selectors"),
        "stat_selectors": section.get("stat_selectors"),
        "stat_selector": section.get("stat_selector"),
        "selector": section.get("selector"),
        "improvement_tolerance": section.get("improvement_tolerance"),
    }


def resolve_model_kwargs(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    model_params = config.get("model_params", {})
    selected = model_params.get(model_name.lower(), {})
    model_kwargs = copy.deepcopy(selected) if isinstance(selected, dict) else {}
    random_seed = int(config.get("random_seed", DEFAULT_RANDOM_SEED))

    if model_name.lower() in {"lr", "rf", "catboost"}:
        model_kwargs["random_state"] = random_seed

    return apply_random_seed_to_kwargs(model_kwargs, random_seed)
