# CLIP Stability figures

These five figures are generated only from the completed CLIP Stability experiment's authenticated result tables, ranking files, seed metrics, and manifests. They do not use baseline-method results and do not refit or tune any model.

## Figure index

1. `01_oot_performance_metrics.png` compares the six preregistered CLIP-ranked downstream cells on four untouched-OOT metrics: ROC-AUC, KS, lift at 10%, and bad-event capture at 10%. All four panels use zero baselines.
2. `02_dev_to_oot_auc_generalization.png` compares temporal-fold mean ROC-AUC (with one fold SD), pooled DEV OOF ROC-AUC, and untouched OOT ROC-AUC. Its focused axis is explicitly disclosed; chance AUC 0.50 is outside the view.
3. `03_five_seed_retrieval_stability.png` shows selected-checkpoint bidirectional validation retrieval and positive-versus-allowed-negative cosine separation for all five Stability CLIP seeds.
4. `04_cross_source_ranking_agreement.png` compares the exact ranks of all 1,959 Stability features across the three source representations and reports target-free top-K feature-set overlap.
5. `05_methodology_leakage_boundaries.png` documents the implemented target-free, DEV-only, pre-OOT freeze, and one-time OOT evaluation boundaries.

An embedding-cluster plot is intentionally omitted because standalone learned 32-dimensional consensus vectors were not persisted in the completed result bundle. Reconstructing them after the fact would create a new derived artifact that could be mistaken for a saved experimental result. Figure 4 uses the authenticated final rankings directly instead.

## Reproduction

Run from the repository root:

```powershell
& .\.venv\Scripts\python.exe .\plots\generate_clip_plots.py
```

The script reads:

- `results/prompt_16_homecredit_model_stability_2024/clip_experiment_v1/analysis/final_clip_results.csv`
- `outputs/prompt_16_homecredit_model_stability_2024/clip_experiment_v1/representation/stability/seeds/seed_*/representation_metrics.json`
- `outputs/prompt_16_homecredit_model_stability_2024/clip_experiment_v1/rankings/*.csv`
- completed experiment manifests for the methodology figure

The figures are descriptive evidence. They do not add confidence intervals, significance tests, or superiority claims that were not part of the locked experiment.
