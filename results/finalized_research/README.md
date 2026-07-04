# Finalized research index

This repository studies credit-risk feature selection across Home Credit and
LendingClub v2, including statistical, LLM-assisted, corrected contrastive
feature-selection, and directional transfer pipelines.

## Canonical evidence

- Home Credit corrected CLIP: `results/corrected_homecredit_clip/`
- LendingClub v2 to Home Credit reverse transfer:
  `results/corrected_lendingclub_to_homecredit_transfer/`
- Final report package: `results/final_research_package_v2/`
- Central registry: `results/research_summary/`
- Valid baselines: registry-approved runs under `results/homecredit/` and
  `results/lendingclub_v2/`
- Canonical report index: `results/finalized_research/reports/current/`

Large predictions, checkpoints, model files, and embeddings remain in their
authenticated scientific directories. They are indexed, not copied, by
`canonical_artifact_manifest.json` and `canonical_artifact_inventory.csv`.

## Completion state

Corrected Home Credit CLIP, forward projection, reverse transfer, baseline
comparisons, and the v2 final report package are complete. Paired five-fold
significance, complete feature-level drift analysis, and LLM cost/scalability
analysis remain pending; their retained inputs are indexed under
`pending_analyses/`.

Old-policy CLIP training/evaluation outputs based on the faulty
pre-`identity_equivalence_v2` negative policy were removed from active
evidence. Their registry records and hashes are retained as compact tombstones
under `audit/history/`. Failed runs, incomplete outputs, smoke artifacts, and
the superseded v1 report package were also removed after dependency checks.

## Reproduction boundary

Use `reproduction/commands.md` to validate registries and rebuild the report
from immutable saved artifacts. Do not rerun training, feature selection,
predictions, embeddings, checkpoints, or data splits as part of report
reproduction.
