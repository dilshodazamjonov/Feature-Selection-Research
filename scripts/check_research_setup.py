from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import dotenv
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.utils import get_model_bundle, get_selector
from experiments.config import DEFAULT_CONFIG_PATH, load_project_config
from experiments.matrix import HYBRID_SELECTORS, MODELS, STAT_SELECTORS, validate_matrix


REQUIRED_APPLICATION_COLUMNS = {"SK_ID_CURR", "TARGET"}
REQUIRED_DATA_FILES = [
    "application_train.csv",
    "previous_application.csv",
    "bureau.csv",
]
TEMPORAL_SOURCE_COLUMNS = {
    "previous_application.csv": {"SK_ID_CURR", "DAYS_DECISION"},
    "bureau.csv": {"SK_ID_CURR", "DAYS_CREDIT"},
}
REQUIRED_DEPENDENCIES = ["pandas", "numpy", "sklearn", "catboost", "openai"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the research setup before training.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    return parser


def _read_header(path: Path) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns)


def _check_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()


def main(argv: list[str] | None = None) -> int:
    dotenv.load_dotenv()
    args = build_parser().parse_args(argv)

    checks: list[tuple[str, bool, str]] = []

    config_path = Path(args.config)
    config_exists = config_path.exists()
    checks.append(("config exists", config_exists, str(config_path)))
    config = load_project_config(config_path)

    data_dir = Path(config["data_dir"])
    checks.append(("data path exists", data_dir.exists(), str(data_dir)))

    missing_files = [filename for filename in REQUIRED_DATA_FILES if not (data_dir / filename).exists()]
    checks.append(("required CSV files exist", not missing_files, f"missing={missing_files}"))

    app_train_path = data_dir / "application_train.csv"
    app_train_ok = app_train_path.exists()
    app_columns: set[str] = set()
    if app_train_ok:
        app_columns = _read_header(app_train_path)
        missing = sorted(REQUIRED_APPLICATION_COLUMNS - app_columns)
        checks.append(("required columns exist", not missing, f"missing={missing}"))
    else:
        checks.append(("required columns exist", False, f"missing file={app_train_path}"))

    configured_time_col = str(config.get("time_col", "recent_decision"))
    source_matches = []
    for filename, required_cols in TEMPORAL_SOURCE_COLUMNS.items():
        file_path = data_dir / filename
        if not file_path.exists():
            continue
        columns = _read_header(file_path)
        if required_cols.issubset(columns):
            source_matches.append(filename)
    temporal_ok = configured_time_col in app_columns or bool(source_matches)
    checks.append(
        ("temporal split column/source exists", temporal_ok, f"sources={source_matches}"),
    )

    try:
        validate_matrix()
        matrix_ok = True
        matrix_message = f"models={MODELS}, stat={STAT_SELECTORS}, hybrid={HYBRID_SELECTORS}"
    except Exception as exc:
        matrix_ok = False
        matrix_message = repr(exc)
    checks.append(("experiment matrix is valid", matrix_ok, matrix_message))

    model_names = set(MODELS) | {str(config.get("model_selector", "lr"))}
    model_ok = True
    model_message = ""
    for model_name in sorted(model_names):
        try:
            get_model_bundle(model_name, model_kwargs={})
        except Exception as exc:
            model_ok = False
            model_message = f"{model_name}: {exc}"
            break
    checks.append(("model selector is valid", model_ok, model_message or ",".join(sorted(model_names))))

    selector_names = set(STAT_SELECTORS) | set(HYBRID_SELECTORS) | {"llm"}
    selector_ok = True
    selector_message = ""
    for selector_name in sorted(selector_names):
        try:
            get_selector(selector_name)
        except Exception as exc:
            selector_ok = False
            selector_message = f"{selector_name}: {exc}"
            break
    checks.append(
        ("feature selectors are valid",
         selector_ok,
         selector_message or ",".join(sorted(selector_names))),
    )

    budgets = config.get("feature_budgets", {})
    llm_config = config.get("llm", {})
    budgets_ok = (
        isinstance(budgets, dict)
        and int(budgets.get("lr", 0)) > 0
        and int(budgets.get("catboost", 0)) > 0
        and int(llm_config.get("ranking_budget", 0)) >= int(budgets.get("catboost", 0))
        and bool(llm_config.get("shared_ranking_enabled", False))
    )
    checks.append(
        (
            "feature budgets are valid",
            budgets_ok,
            f"lr={budgets.get('lr')}, catboost={budgets.get('catboost')}, "
            f"llm_ranking_budget={llm_config.get('ranking_budget')}, "
            f"shared={llm_config.get('shared_ranking_enabled')}",
        )
    )

    description_path = Path(config["description_path"])
    checks.append(("description file exists", description_path.exists(), str(description_path)))

    results_dir = Path(config.get("results_dir", "results"))
    try:
        _check_writable(results_dir)
        writable_ok = True
        writable_message = str(results_dir)
    except Exception as exc:
        writable_ok = False
        writable_message = repr(exc)
    checks.append(("output folders are writable", writable_ok, writable_message))

    llm_enabled = bool(config.get("llm", {}).get("enabled", True))
    api_key_ok = (not llm_enabled) or bool(os.getenv("OPENAI_API_KEY"))
    checks.append(
        ("OPENAI_API_KEY exists if LLM is enabled",
         api_key_ok,
         "enabled" if llm_enabled else "disabled"),
    )

    missing_deps = [
        dependency
        for dependency in REQUIRED_DEPENDENCIES
        if importlib.util.find_spec(dependency) is None
    ]
    checks.append(
        (
            "required dependencies importable",
            not missing_deps,
            f"missing={missing_deps}",
        )
    )

    all_ok = True
    for label, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"{status}: {label} ({detail})")
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
