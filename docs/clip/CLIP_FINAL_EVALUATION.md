# CLIP Final Evaluation

This stage evaluates the frozen CLIP-style selectors `clip` and `clip_then_mrmr` with the existing credit-risk modeling protocol.

Primary evidence is OOT performance. DEV CV is diagnostic because the CLIP representation was prepared once at DEV level and was not rebuilt fold-locally.

Frozen components:

- selected Prompt 5 checkpoint
- Home Credit training-split anchor
- statistical preprocessor
- text embeddings
- fusion rule
- negative policy
- screening-pool size

Active datasets are `homecredit` and `lendingclub_v2`. Legacy `lendingclub` is rejected.

The evaluation writes outputs under `results/clip/final_evaluation/`, including run manifests, OOT predictions, selected features, metric summaries, score PSI, semantic coverage, redundancy, seed sensitivity, and paired bootstrap AUC comparisons where baseline predictions exist.

Score PSI compares DEV model scores against OOT model scores only. Thresholds are:

- `<0.10`: low
- `0.10` to `<0.25`: moderate
- `>=0.25`: high

No final claim should state that CLIP is superior from DEV CV alone. OOT metrics and paired prediction comparisons are the relevant downstream evidence.
