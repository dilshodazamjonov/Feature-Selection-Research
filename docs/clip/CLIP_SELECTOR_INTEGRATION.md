# CLIP Selector Integration

This integration adds frozen CLIP-style feature scoring as a selector family for the research pipeline. It is not the final experiment matrix and it does not train LR, CatBoost, LLM, or contrastive models.

## Selectors

Registered selector names:

- `clip`
- `clip_then_mrmr`

`clip` ranks candidate features by frozen learned similarity and keeps the configured model budget.

`clip_then_mrmr` first screens candidates by frozen learned similarity, then runs the existing mRMR selector only on the DEV training rows passed by the pipeline.

Existing selector names and LLM workflows remain unchanged.

## Binding

Configuration:

```text
configs/clip/selector.yaml
```

Bound checkpoint:

```text
results/clip/training/seeds/seed_55/best_checkpoint.pt
```

Checkpoint hash:

```text
3f21fc12060036f117aedf9a610856c72fa9a0ce6a1540403e572de4423d7385
```

Anchor hash:

```text
e5446c5141cbdf8fde8677022b75cb1905c6e4e42fb48378727ecaf31eca604d
```

Statistical view:

```text
missingness_only
```

The LendingClub v2 score path is external-validation only and uses the unchanged Home Credit learned anchor. Legacy `lendingclub` is rejected.

## Outputs

Smoke-only selector integration outputs are written under:

```text
results/clip/selector_integration/
```

Expected files:

- `integration_manifest.json`
- `checkpoint_binding.json`
- `homecredit_clip_scores.csv`
- `lendingclub_v2_clip_scores.csv`
- `homecredit_clip_selection_smoke.csv`
- `homecredit_clip_then_mrmr_smoke.csv`
- `lendingclub_v2_clip_selection_smoke.csv`
- `lendingclub_v2_clip_then_mrmr_smoke.csv`
- `selector_registry_audit.json`
- `integration_audit.json`

The score cache key includes dataset, feature name, checkpoint hash, anchor hash, text/statistical projection hashes, preprocessor hash, fusion rule, statistical view scope, cache version, and selector code version. Stale checkpoint or anchor hashes fail validation.

## Commands

Dry-run validation:

```powershell
uv run python scripts/validate_clip_selector_integration.py --dry-run
```

Smoke artifact generation:

```powershell
uv run python scripts/validate_clip_selector_integration.py --smoke-test
```

Full matrix command remains intentionally separate and must not be run without explicit approval:

```powershell
python scripts/run_matrix.py --dataset lendingclub_v2
```

## Current Limitation

The Prompt 5 encoder uses one DEV-only statistical field, `missing_rate_dev`, so this should be treated as a CLIP-style architectural proof of concept. It should not be described as a broad learned statistical-quality model until richer approved statistical views are added.

Because the current CLIP checkpoint was built from Home Credit DEV-level feature evidence rather than fold-local retraining inside each temporal CV fold, downstream development CV should be treated as diagnostic. OOT evaluation remains the primary final evidence. Strict nested selector-CV evidence would require a fold-local CLIP retraining design.
