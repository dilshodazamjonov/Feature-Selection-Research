"""Create deterministic repository inventory snapshots for cleanup auditing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


CHUNK_SIZE = 4 * 1024 * 1024


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def utc_mtime(path: Path) -> str:
    timestamp = path.stat().st_mtime
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def collect_paths(root: Path, excluded_roots: set[str]) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    directories: list[Path] = []
    for current_root, directory_names, file_names in os.walk(root):
        current = Path(current_root)
        if current == root:
            directory_names[:] = [
                name
                for name in directory_names
                if name != ".git" and name not in excluded_roots
            ]
        directory_names.sort()
        file_names.sort()
        directories.extend(current / name for name in directory_names)
        files.extend(current / name for name in file_names)
    return files, directories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot-name", default="pre_cleanup")
    parser.add_argument("--exclude-root", action="append", default=[])
    parser.add_argument("--recorded-branch")
    parser.add_argument("--recorded-commit")
    parser.add_argument("--recorded-status")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_roots = set(args.exclude_root)

    tracked = {
        line
        for line in run_git(root, "ls-files", "-z").split("\0")
        if line
    }
    files, directories = collect_paths(root, excluded_roots)

    with ThreadPoolExecutor(max_workers=4) as executor:
        hashes = dict(zip(files, executor.map(sha256_file, files)))

    rows: list[dict[str, object]] = []
    total_size = 0
    for directory in directories:
        relative = relative_posix(directory, root)
        rows.append(
            {
                "path": relative + "/",
                "type": "directory",
                "size_bytes": 0,
                "sha256": "",
                "git_status": "tracked" if relative in tracked else "untracked",
                "modified_time_utc": utc_mtime(directory),
            }
        )
    for file_path in files:
        relative = relative_posix(file_path, root)
        size = file_path.stat().st_size
        total_size += size
        rows.append(
            {
                "path": relative,
                "type": file_path.suffix.lower().lstrip(".") or "file",
                "size_bytes": size,
                "sha256": hashes[file_path],
                "git_status": "tracked" if relative in tracked else "untracked",
                "modified_time_utc": utc_mtime(file_path),
            }
        )
    rows.sort(key=lambda row: str(row["path"]).casefold())

    inventory_path = output_dir / f"{args.snapshot_name}_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    branch = args.recorded_branch or run_git(root, "branch", "--show-current").strip()
    commit = args.recorded_commit or run_git(root, "rev-parse", "HEAD").strip()
    status = args.recorded_status
    if status is None:
        status = run_git(root, "status", "--short", "--branch").rstrip()

    status_path = output_dir / f"{args.snapshot_name}_git_status.txt"
    status_path.write_text(
        f"branch: {branch}\ncommit: {commit}\nstatus:\n{status}\n",
        encoding="utf-8",
    )

    summary = {
        "branch": branch,
        "commit": commit,
        "directory_count": len(directories),
        "excluded_top_level_paths": sorted(excluded_roots | {".git"}),
        "file_count": len(files),
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "repository_root": str(root),
        "size_bytes": total_size,
        "status": status.splitlines(),
    }
    summary_path = output_dir / f"{args.snapshot_name}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
