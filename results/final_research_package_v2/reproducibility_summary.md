# Reproducibility summary

## Scope

This v2 package is a report-only derivative. It reads immutable saved metrics, predictions, fold selections, manifests, and embedding matrices. It does not execute prepare, train, project, evaluate, register, model fitting, mRMR, prediction generation, or embedding generation.

## Metric authentication

For all ten required pipeline/model rows, OOT AUC and KS were independently recomputed from saved row-level predictions and matched the run summary and central reusable-metrics registry within `1e-12`. All OOT files contain 120,053 rows. Task 1 files have no stable borrower IDs and no saved pooled DEV OOF predictions; those fields remain empty. Reverse-transfer DEV OOF and OOT metrics use authenticated stable IDs and prediction manifests.

Task 1 PSI uses final DEV-fit in-sample scores as reference and OOT scores as comparison through the shared `calculate_psi` implementation. It is comparable within the direct LLM-only versus combined pairs. Reverse PSI uses pooled DEV OOF probabilities, DEV-OOF quantile bins, and OOT comparison probabilities. These scopes are not compared numerically across tasks.

Pairwise Jaccard was recomputed from the five saved fold-selected feature sets and matched each saved scalar. Task 1 Nogueira/Kuncheva values were excluded because their files used universe size 529 rather than the authenticated candidate pools of 60 for LR and 100 for CatBoost.

## Forward directional visualization

- Source representation: corrected Home Credit-trained CLIP, selected corrected seed-55 checkpoint.
- Projected dataset: LendingClub v2.
- Embedding source: `results/corrected_homecredit_clip/training/lendingclub_v2_joint_embeddings.parquet`.
- Valid features: 576; semantic groups: 17.
- Semantic labels: embedded row metadata authenticated against the external-pair count.
- Coordinates: newly calculated from saved embeddings; no embedding generation.

## Reverse directional visualization

- Source representation: five-seed corrected LendingClub v2-trained CLIP consensus.
- Projected dataset: Home Credit.
- Embedding source: `results/corrected_lendingclub_to_homecredit_transfer/reverse_projection/homecredit_reverse_embeddings.parquet`.
- Valid features: 436; excluded reconciliation records: 95.
- Alignment: orthogonal Procrustes to seed 11; normalized mean consensus.
- Semantic labels: `results/corrected_homecredit_clip/feature_universe/feature_universe_reconciliation.csv`, joined one-to-one by feature ID.
- Coordinates: newly calculated from saved embeddings.

## Common diagnostic procedure

Both directions use umap-learn 0.5.12, cosine metric, 15 neighbours, minimum distance 0.1, two components, and random seed 42. Original-space kNN purity uses the validated Task 3 operational procedure at k=10. Labels are shuffled 200 times with NumPy seed 20260625 while embeddings and neighbour identities remain fixed. Trustworthiness uses k=10 and cosine distance.

The source hashes are recorded in `final_package_manifest.json` and summarized in `artifact_inventory.md`. Output hashes cover every generated file except the package manifest itself. The manifest intentionally has no self-hash.

## Reproduction command

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\build_final_research_package_v2.py
```

The command rebuilds only `results/final_research_package_v2/`. It does not alter the v1 package or any scientific source artifact.
