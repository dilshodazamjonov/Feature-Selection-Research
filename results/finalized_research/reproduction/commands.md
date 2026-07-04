# Reproduction commands

Run from the repository root with the locked environment.

Validate central registries, active paths, final-package hashes, and pending
analysis inputs:

```powershell
.\.venv\Scripts\python.exe cleanup/tools/validate_repository_state.py --root .
```

Run the active unit and contract tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Compile active source:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Rebuild only the report package from immutable saved artifacts:

```powershell
.\.venv\Scripts\python.exe scripts/build_final_research_package_v2.py
```

Do not rerun training, feature selection, prediction generation, embedding
generation, checkpoint creation, or dataset splitting. Those operations would
create new scientific identities rather than reproduce the report.
