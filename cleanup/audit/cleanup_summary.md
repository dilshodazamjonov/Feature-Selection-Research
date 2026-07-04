# Final repository cleanup summary

Canonical entry point: `results/finalized_research/README.md`

## Impact

- Pre-cleanup files: 42,488
- Post-cleanup files at report generation: 40,273
- Original paths removed: 2,279
- New audit/canonical paths added: 64
- Pre-cleanup size: 14,418,127,790 bytes
- Post-cleanup size at report generation: 14,155,316,150 bytes
- Net size reduction: 262,811,640 bytes
- Original directories removed: 760
- Active source modules removed: 13
- Scripts removed: 31
- Configs removed: 11
- Tests removed: 22

## Deleted directory roots

| Path | Category | Bytes | Reason | Replacement |
|---|---|---:|---|---|
| `.pytest_cache` | TEMPORARY_OR_CACHE | 698 | generated cache or runtime output | `` |
| `.uv-cache` | TEMPORARY_OR_CACHE | 3,545 | generated cache or runtime output | `` |
| `.uv-tools` | TEMPORARY_OR_CACHE | 1 | generated cache or runtime output | `` |
| `__pycache__` | TEMPORARY_OR_CACHE | 52,797 | generated cache or runtime output | `` |
| `catboost_info` | TEMPORARY_OR_CACHE | 652,605 | generated cache or runtime output | `` |
| `results/clip/statistical_baseline/dry_run` | DRY_RUN_OR_SMOKE_OUTPUT | 73,624 | non-scientific dry-run output | `` |
| `results/clip/text_baseline/dry_run` | DRY_RUN_OR_SMOKE_OUTPUT | 8,411 | non-scientific dry-run output; full target-free text baseline retained | `results/clip/text_baseline` |
| `results/clip_pairing_repair/smoke` | DRY_RUN_OR_SMOKE_OUTPUT | 5,489 | checkpoint marked NOT_FOR_SCIENTIFIC_USE | `results/clip_pairing_repair` |
| `results/clip_v2/contrastive_data` | INVALID_SCIENTIFIC_OUTPUT | 749,519 | old-policy contrastive pairs superseded by identity_equivalence_v2 | `results/corrected_homecredit_clip/contrastive_data` |
| `results/clip_v2/dry_run` | DRY_RUN_OR_SMOKE_OUTPUT | 631,014 | obsolete CLIP-v2 dry-run output | `` |
| `results/clip_v2/final_analysis` | INVALID_SCIENTIFIC_OUTPUT | 1,117,992 | analysis derives from invalid old-policy CLIP outputs | `results/final_research_package_v2` |
| `results/clip_v2/final_evaluation` | INVALID_SCIENTIFIC_OUTPUT | 187,232,613 | predictions and metrics derive from invalid old-policy checkpoints | `results/corrected_homecredit_clip/combined_pipeline` |
| `results/clip_v2/selector_integration` | INVALID_SCIENTIFIC_OUTPUT | 1,006,314 | selector cache derives from invalid old-policy checkpoints | `results/corrected_homecredit_clip/combined_pipeline` |
| `results/clip_v2/text_baseline` | DUPLICATE_CONTENT | 3,403,422 | byte-identical duplicate of retained target-free text baseline | `results/clip/text_baseline` |
| `results/clip_v2/training` | INVALID_SCIENTIFIC_OUTPUT | 1,933,061 | checkpoints trained with faulty pre-identity_equivalence_v2 policy | `results/corrected_homecredit_clip/training` |
| `results/corrected_lendingclub_to_homecredit_transfer/downstream/logistic_regression.incomplete_oof_overlap_20260629` | FAILED_OR_INCOMPLETE_OUTPUT | 94,969,157 | incomplete overlapping-OOF attempt; authenticated replacement exists | `results/corrected_lendingclub_to_homecredit_transfer/downstream/logistic_regression` |
| `results/corrected_lendingclub_to_homecredit_transfer_failed_20260629_140834` | FAILED_OR_INCOMPLETE_OUTPUT | 411 | failed reverse-transfer attempt; successful canonical run exists | `results/corrected_lendingclub_to_homecredit_transfer` |
| `results/corrected_lendingclub_to_homecredit_transfer_implementation_backup` | MIGRATION_BACKUP | 38,771 | implementation is preserved in tracked source and safety tag | `src/credit_risk_fs/pipelines/reverse_transfer.py` |
| `results/final_research_package` | SUPERSEDED_VALID_OUTPUT | 524,239 | superseded by authenticated final_research_package_v2 | `results/final_research_package_v2` |
| `results/research_summary/migrations/final_pre_prompt4_stability_repair_v1/backups` | MIGRATION_BACKUP | 16,844 | completed migration payload; compact manifests retained | `results/research_summary/migrations/final_pre_prompt4_stability_repair_v1` |
| `results/research_summary/migrations/final_pre_prompt4_stability_repair_v1/staged` | MIGRATION_BACKUP | 35,220 | completed migration payload; compact manifests retained | `results/research_summary/migrations/final_pre_prompt4_stability_repair_v1` |
| `results/research_summary/migrations/final_pre_prompt4_stability_repair_v2/backups` | MIGRATION_BACKUP | 26,224 | completed migration payload; compact manifests retained | `results/research_summary/migrations/final_pre_prompt4_stability_repair_v2` |
| `results/research_summary/migrations/final_pre_prompt4_stability_repair_v2/staged` | MIGRATION_BACKUP | 35,220 | completed migration payload; compact manifests retained | `results/research_summary/migrations/final_pre_prompt4_stability_repair_v2` |
| `results/research_summary/migrations/registry_artifact_identity_migration_v1_20260629T133634Z` | MIGRATION_BACKUP | 487,569 | failed preliminary migration contains backup payload only | `results/research_summary` |
| `results/research_summary/migrations/registry_artifact_identity_migration_v1_20260629T133751Z` | MIGRATION_BACKUP | 487,569 | failed preliminary migration contains backup payload only | `results/research_summary` |
| `results/research_summary/migrations/registry_artifact_identity_migration_v1_20260629T134550Z/backups` | MIGRATION_BACKUP | 487,569 | completed migration payload; compact manifests retained | `results/research_summary/migrations/registry_artifact_identity_migration_v1_20260629T134550Z` |
| `results/research_summary/migrations/registry_artifact_identity_migration_v1_20260629T134725Z/backups` | MIGRATION_BACKUP | 487,569 | completed migration payload; compact manifests retained | `results/research_summary/migrations/registry_artifact_identity_migration_v1_20260629T134725Z` |
| `results/research_summary/migrations/registry_artifact_identity_migration_v1_20260629T135128Z/backups` | MIGRATION_BACKUP | 515,168 | completed migration payload; compact manifests retained | `results/research_summary/migrations/registry_artifact_identity_migration_v1_20260629T135128Z` |
| `results/research_summary/migrations/registry_artifact_identity_migration_v1_20260704T060945Z/backups` | MIGRATION_BACKUP | 313,762 | completed migration payload; compact manifests retained | `results/research_summary/migrations/registry_artifact_identity_migration_v1_20260704T060945Z` |
| `Models` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `Notebooks` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `Notebooks/archive` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `Notebooks/archive/homecredit` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `Notebooks/archive/lendingclub` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `Notebooks/homecredit` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `Notebooks/homecredit/notebooks` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `Notebooks/lendingclub` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `Notebooks/lendingclub/notebooks` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `Preprocessing` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `artifacts/prompts` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `artifacts/prompts/llm_screening` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `data/homecredit/interim` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `data/homecredit/processed` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `data/inputs` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `data/lendingclub/interim` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `evaluation` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `experiments` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `feature_selection` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `pipelines` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `report_input_bundle` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `results/clip/statistical_baseline` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `results/clip_v2/audit` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `src/credit_risk_fs/repairs` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/data` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/evaluation` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/experiments` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/feature_engineering` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/feature_metadata` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/fixtures` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/fixtures/homecredit` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/fixtures/lendingclub` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/preprocessing` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/regression` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `tests/selectors` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `training` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |
| `utils` | EMPTY_DIRECTORY | 0 | empty after canonical cleanup |  |

## Retained ambiguous directories

| Path | Reason |
|---|---|
| `.pytest_tmp` | Windows ACL blocks read, permission repair, move, and deletion |
| `pytest_tmp` | Windows ACL blocks read, permission repair, move, and deletion |
| `results/lendingclub` | legacy dataset outputs have unclear provenance and no deletion authorization |
| `reports/archive` | compact historical reports may contain unique interpretation |
| `results/final_project_audit` | compact audit evidence retained conservatively |

## Deleted source files

- `src/credit_risk_fs/clip/evaluation_aggregation.py`
- `src/credit_risk_fs/clip/score_cache.py`
- `src/credit_risk_fs/clip/selector_adapter.py`
- `src/credit_risk_fs/clip/selector_validation.py`
- `src/credit_risk_fs/clip/statistical_baseline.py`
- `src/credit_risk_fs/clip/statistical_fields.py`
- `src/credit_risk_fs/clip/statistical_schema.py`
- `src/credit_risk_fs/clip/statistical_validation.py`
- `src/credit_risk_fs/clip/v1_freeze.py`
- `src/credit_risk_fs/repairs/__init__.py`
- `src/credit_risk_fs/repairs/reverse_transfer_stability.py`
- `src/credit_risk_fs/selectors/clip_screening.py`
- `src/credit_risk_fs/selectors/clip_then_mrmr.py`

## Deleted scripts

- `scripts/audit_clip_v2.py`
- `scripts/build_clip_contrastive_data.py`
- `scripts/build_clip_final_analysis.py`
- `scripts/build_clip_final_report_figures.py`
- `scripts/build_clip_statistical_baseline.py`
- `scripts/build_clip_v2_contrastive_data.py`
- `scripts/build_clip_v2_final_analysis.py`
- `scripts/build_review_support_artifacts.py`
- `scripts/check_research_setup.py`
- `scripts/finalize_corrected_homecredit_clip.py`
- `scripts/freeze_clip_v1.py`
- `scripts/migrate_registry_artifact_integrity.py`
- `scripts/plan_clip_v2_ablation.py`
- `scripts/rebuild_clip_evaluation_aggregates.py`
- `scripts/rebuild_clip_v2_evaluation_aggregates.py`
- `scripts/repair_reverse_transfer_stability.py`
- `scripts/run_all_experiments.py`
- `scripts/run_clip_final_evaluation.py`
- `scripts/run_clip_v2_final_evaluation.py`
- `scripts/run_clip_v2_pipeline.py`
- `scripts/run_corrected_reverse_transfer_all.ps1`
- `scripts/run_hybrid_comparison.py`
- `scripts/run_llm_vs_statistical.py`
- `scripts/run_single_experiment.py`
- `scripts/run_statistical_comparison.py`
- `scripts/strengthen_final_report_artifacts.py`
- `scripts/train_clip_encoder.py`
- `scripts/train_clip_v2_encoder.py`
- `scripts/validate_clip_pairing_repair.py`
- `scripts/validate_clip_selector_integration.py`
- `scripts/validate_clip_v2_selector_integration.py`

## Deleted configs

- `configs/clip/contrastive_data.yaml`
- `configs/clip/selector.yaml`
- `configs/clip/statistical_baseline.yaml`
- `configs/clip/training.yaml`
- `configs/clip_v2/analysis.yaml`
- `configs/clip_v2/contrastive_data.yaml`
- `configs/clip_v2/evaluation.yaml`
- `configs/clip_v2/selector.yaml`
- `configs/clip_v2/training.yaml`
- `configs/experiments/clip_homecredit_matrix.yaml`
- `configs/experiments/clip_lendingclub_v2_matrix.yaml`

## Deleted tests

- `tests/clip/conftest.py`
- `tests/clip/test_clip_baseline_comparison.py`
- `tests/clip/test_clip_evaluation_aggregation.py`
- `tests/clip/test_clip_final_analysis.py`
- `tests/clip/test_clip_final_evaluation_config.py`
- `tests/clip/test_clip_final_evaluation_recovery.py`
- `tests/clip/test_clip_final_report_figures.py`
- `tests/clip/test_clip_metric_consistency.py`
- `tests/clip/test_clip_prediction_artifacts.py`
- `tests/clip/test_clip_selector_behavior.py`
- `tests/clip/test_clip_selector_cache.py`
- `tests/clip/test_clip_selector_integration.py`
- `tests/clip/test_clip_selector_registry.py`
- `tests/clip/test_clip_significance.py`
- `tests/clip/test_clip_statistical_baseline.py`
- `tests/clip/test_clip_statistical_external_validation.py`
- `tests/clip/test_clip_statistical_fields.py`
- `tests/clip/test_clip_v1_freeze_and_v2_boundaries.py`
- `tests/clip/test_clip_v2_audit.py`
- `tests/clip/test_clip_v2_execution_pipeline.py`
- `tests/clip/test_clip_v2_pipeline_orchestrator.py`
- `tests/clip/test_clip_v2_semantic_map.py`

## Retained reproduction scripts

- `scripts/aggregate_results.py`
- `scripts/analyze_corrected_homecredit_clip.py`
- `scripts/analyze_feature_level_drift.py`
- `scripts/analyze_stability_significance.py`
- `scripts/build_clip_text_baseline.py`
- `scripts/build_clip_training_manifest.py`
- `scripts/build_clip_v2_statistical_view.py`
- `scripts/build_final_research_package_v2.py`
- `scripts/run_corrected_homecredit_clip_pipelines.py`
- `scripts/run_corrected_lendingclub_to_homecredit_transfer.py`
- `scripts/run_reverse_transfer_pipeline.py`

## Validation

- Full tests: 375 passed, 0 failed, 0 skipped in 172.053 seconds.
- Corrected CLIP and reverse-transfer focused tests: 297 passed.
- Active source compileall: passed.
- `git diff --check`: passed.
- Registry bundle, summary hashes, active paths, final package, canonical artifact hashes, stage manifests, and pending inputs: passed.
