# Credit-risk feature-selection research

The repository contains completed Home Credit and LendingClub v2 experiments
covering statistical, LLM-assisted, corrected contrastive, and directional
transfer feature-selection pipelines.

Start with:

- `results/finalized_research/README.md` for the canonical research index
- `results/finalized_research/STATUS.md` for completed and pending work
- `results/final_research_package_v2/final_research_report.md` for the report
- `results/research_summary/results_access_guide.md` for registry access

The scientific outputs are immutable saved artifacts. Do not rerun training,
feature selection, predictions, embeddings, checkpoints, or data splits to
reproduce the report.

Validate the current repository state:

```powershell
.\.venv\Scripts\python.exe cleanup/tools/validate_repository_state.py --root .
.\.venv\Scripts\python.exe -m pytest tests -q
```

See `results/finalized_research/reproduction/` for the complete validation and
saved-artifact report-build boundary.
