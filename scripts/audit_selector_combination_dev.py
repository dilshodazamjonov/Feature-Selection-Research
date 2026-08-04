#!/usr/bin/env python
"""Generate the Prompt 13 review package from persisted artifacts only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credit_risk_fs.experiments.prompt_13_dev_audit import (  # noqa: E402
    AUDIT_DIR,
    finalize_manifest_and_lock,
    write_review_package,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--package", action="store_true", help="Build portable HTML, manifest, and review lock.")
    args = parser.parse_args(argv)
    root = Path(args.repository_root).resolve()
    package = write_review_package(root)
    output = root / AUDIT_DIR
    result: dict[str, object] = {
        "status": "review_files_written",
        "output": str(output),
        "raw_dataset_paths_resolved": False,
        "workers_started": 0,
    }
    if args.package:
        plugin_root = Path(
            r"C:\Users\DILSHOD\.codex\plugins\cache\openai-curated-remote\data-analytics\0.2.8-13ceeea1f599"
        )
        command = [
            "node",
            str(plugin_root / "skills/build-report/scripts/deliver_portable_artifact.mjs"),
            "--input",
            str(output / "artifact.json"),
            "--output",
            str(output / "report.html"),
        ]
        completed = subprocess.run(command, cwd=plugin_root, check=True, capture_output=True, text=True)
        receipt_text = completed.stdout.strip()
        try:
            receipt = json.loads(receipt_text)
        except json.JSONDecodeError:
            receipt = {"receipt": receipt_text}
        finalized = finalize_manifest_and_lock(root, package, receipt)
        result.update(
            {
                "status": "ready_for_manual_oot",
                "report_receipt": receipt,
                "review_lock": str(finalized["lock_path"]),
                "manual_oot_command_not_executed": package["manual_command"],
            }
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
