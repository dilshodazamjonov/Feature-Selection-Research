# Environment

- Operating system used for final cleanup: Windows
- Python: 3.13
- Dependency lock: `uv.lock`
- Project metadata: `pyproject.toml`
- Test configuration: `pytest.ini`

Create or synchronize an environment with the checked-in lock:

```powershell
uv sync --locked
```

Environment variables belong in an untracked `.env`; never commit credentials.
Saved artifacts, registries, and manifests—not mutable package caches—are the
scientific source of truth.
