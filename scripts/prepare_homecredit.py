from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _schema_snapshot(raw_dir: Path) -> dict[str, object]:
    snapshot = {"dataset": "homecredit", "files": []}
    for csv_path in sorted(raw_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        snapshot["files"].append({"name": csv_path.name, "columns": header})
    return snapshot


def main() -> int:
    legacy_dir = PROJECT_ROOT / "data" / "inputs"
    raw_dir = PROJECT_ROOT / "data" / "homecredit" / "raw"
    metadata_dir = PROJECT_ROOT / "data" / "homecredit" / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if legacy_dir.exists():
        for file_path in legacy_dir.glob("*.csv"):
            destination = raw_dir / file_path.name
            if not destination.exists():
                file_path.replace(destination)

    legacy_description = PROJECT_ROOT / "data" / "HomeCredit_columns_description.csv"
    description_path = metadata_dir / "columns_description.csv"
    if legacy_description.exists() and not description_path.exists():
        legacy_description.replace(description_path)

    (metadata_dir / "raw_schema_snapshot.json").write_text(
        json.dumps(_schema_snapshot(raw_dir), indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "leakage_columns.yaml").write_text(
        "excluded_feature_columns:\n"
        "  - TARGET\n"
        "  - recent_decision\n"
        "  - PREV_recent_decision_MAX\n"
        "  - DAYS_DECISION\n"
        "  - application_time_proxy\n",
        encoding="utf-8",
    )

    print(f"Prepared Home Credit raw data: {raw_dir}")
    print(f"Description file: {description_path}")
    print(f"Schema snapshot: {metadata_dir / 'raw_schema_snapshot.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
