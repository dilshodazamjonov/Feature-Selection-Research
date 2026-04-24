from __future__ import annotations

import argparse
import ast
import copy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_CONFIG: dict[str, Any] = {
    "model_selector": "lr",
    "data_dir": "data/inputs",
    "description_path": "data/HomeCredit_columns_description.csv",
    "n_splits": 5,
    "dev_start_day": -600,
    "oot_start_day": -240,
    "oot_end_day": 0,
    "cv_gap_groups": 1,
    "llm": {
        "model": "gpt-4.1-mini",
        "max_features": 50,
        "cache_dir": "outputs/llm_selector_cache",
    },
    "model_params": {
        "lr": {},
        "rf": {},
        "catboost": {},
    },
    "statistical_comparison": {
        "output_dir": "outputs/statistical_comparison",
        "selectors": ["mrmr", "boruta", "pca"],
    },
    "llm_vs_statistical": {
        "output_dir": "outputs/llm_vs_statistical",
        "stat_selectors": ["mrmr", "boruta"],
    },
    "hybrid_comparison": {
        "output_dir": "outputs/hybrid_comparison",
        "stat_selector": "mrmr",
        "improvement_tolerance": 1e-4,
    },
    "single_experiment": {
        "output_dir": "outputs/single_experiment",
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


def build_parser_defaults(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = config.get(section_name, {})
    llm = config.get("llm", {})
    return {
        "config_path": DEFAULT_CONFIG_PATH,
        "model": config.get("model_selector", "lr"),
        "data_dir": config.get("data_dir", DEFAULT_CONFIG["data_dir"]),
        "description_path": config.get("description_path", DEFAULT_CONFIG["description_path"]),
        "n_splits": config.get("n_splits", DEFAULT_CONFIG["n_splits"]),
        "dev_start_day": config.get("dev_start_day", DEFAULT_CONFIG["dev_start_day"]),
        "oot_start_day": config.get("oot_start_day", DEFAULT_CONFIG["oot_start_day"]),
        "oot_end_day": config.get("oot_end_day", DEFAULT_CONFIG["oot_end_day"]),
        "cv_gap_groups": config.get("cv_gap_groups", DEFAULT_CONFIG["cv_gap_groups"]),
        "llm_model": llm.get("model", DEFAULT_CONFIG["llm"]["model"]),
        "llm_max_features": llm.get("max_features", DEFAULT_CONFIG["llm"]["max_features"]),
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
    return copy.deepcopy(selected) if isinstance(selected, dict) else {}
