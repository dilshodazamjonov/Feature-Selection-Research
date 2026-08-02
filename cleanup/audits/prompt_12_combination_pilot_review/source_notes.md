# Prompt 12 review source notes

## Evidence boundary

The review used only authenticated pilot state and artifact files under results/selector_combinations_v1, the frozen configuration and protocol registries, sanitized research logs, process and lock metadata, and Git metadata. No file under data was opened, inspected, hashed, copied, or otherwise accessed. The safe status command reported raw_dataset_paths_resolved=false and workers_started=0.

Predictive metric values were not extracted, ranked, compared, or used for the retention recommendation. Retention is controlled only by authentication, declared stage-support behavior, and runtime-resource evidence.

## Controlling sources

- configs/experiments/selector_combination_research_v1.yaml
- configs/protocols/selector_combinations_v1/combination_protocol_lock.json
- configs/protocols/selector_combinations_v1/combination_method_registry.json
- configs/protocols/selector_combinations_v1/combination_budget_applicability.csv
- results/selector_combinations_v1/pilot/selections
- results/selector_combinations_v1/pilot/evaluations
- results/selector_combinations_v1/incomplete/attempt_history/selections
- logs/events.jsonl
- logs/runs.log

## Transformations

The 18 selection checkpoint JSON files were projected to selection identity, requested and realized support, feasibility state, fit seconds, wall limit, peak process-tree RSS, minimum system-available RAM, warning count, and terminal state. The 24 evaluation checkpoint JSON files were projected to identity, terminal state, exit code, stop code, final stage, resource peaks, and wall limits. Latest-success-session supervisor finalization events supplied evaluation elapsed seconds. No prediction or metric column was used.

The active tree digest sorts all 168 files by path relative to the active pilot root and hashes UTF-8 lines in the form relative_path|size_bytes|sha256 followed by LF.

## Table and visual choice

One horizontal bar chart compares the four method-level peak selector-fit RSS values against the 24 GiB warning threshold. It answers the only single-measure category comparison where visual shape materially helps. Exact support, timing, RAM-floor, warning, and retention evidence remains in full-width, explicitly sorted audit tables because those decisions require keyed multi-field lookup.

## Technical report structure

The portable report maps the technical specification as follows:

- Technical summary: recommendation and approval boundary.
- Key findings: authentication, stage-support, runtime, and retention tables.
- Scope and definitions: evidence boundary and feasibility semantics.
- Methodology: artifact authentication and aggregation rules.
- Limitations and robustness: single-fold evidence, later-fold scaling, natural-support mismatch, and no predictive tuning.
- Recommended next step: explicit digest-and-scope approval before any lock.
- Further questions: user decision on the proposed all-method scope.

## Limitations

The pilot covers DEV fold 1 only. Runtime and RAM margins establish pilot operability, not a guarantee that every later expanding fold will have identical cost or natural support. Existing sequential supervision, wall limits, warning thresholds, and abort floors remain required for DEV. The review does not authorize a pilot rerun, DEV execution, OOT access, or creation of pilot_approval_lock.json.

The portable artifact passed schema validation, packaging, and structural verification. Browser interaction and source-dialog verification are not claimed: the packaged validator could not find a compatible preinstalled Chromium headless-shell, and no browser was downloaded.
