# Prompt 10 configuration freeze report

Status: **PROMPT_10_CONFIGURATION_AND_PIPELINE_IMPLEMENTATION_PASS**

Prompt 9 commits `0972dfd` and `087d09a` and their Prompt 8 ancestor `b7baffc`
authenticated. All six pilot artifacts passed their embedded authentication,
identity, terminal-state, selector-evidence, and DEV-only checks.

The final configuration is `configs/experiments/full_baseline_v1.yaml`, schema
`full_baseline_config_v1`, SHA-256
`f03647c376fe834f9bb1c3d6834ed42732ef3e7e1047eeff352af49b31ed607f`.
It freezes 36 cells, seed 42, LR-20, CatBoost-40, nine canonical method IDs,
five time folds with gap one, the established final model parameters, all selector
parameters, CPU/thread limits, memory/disk controls, and per-family wall limits.
Selectors fit a training-only, one-column-per-original-feature numeric encoding
before the final-model preprocessor; this preserves the authenticated 529/675
candidate identities and prevents fold-specific one-hot columns from becoming
selector candidates.

Home Credit Boruta's 26 confirmed features are preserved as an honest
`infeasible_natural_support` result. Neither its 25 tentative nor 478 rejected
features are promoted. Full-baseline folds may therefore produce fewer than the
requested model budget without being mislabelled or treated as a runner failure.

The implementation adds one entry point, `scripts/run_full_baseline.py`, on the
existing registered-run, checkpoint, resource-supervisor, artifact, and logging
architecture. It does not fork or edit the frozen voting protocol.

No real full-baseline worker was invoked during Prompt 10. Status, pilot audit,
constructor checks, mocked orchestration, and bounded synthetic registered-run
tests were used instead. No real research OOT result was accessed or evaluated.
