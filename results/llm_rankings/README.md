# LLM rankings

Accepted language-model responses behind the LLM-assisted selectors, one JSON file per call, published by
`scripts/run_study.py --stage results` from the study cache (`outputs/study/<dataset>/llm/rankings/`). **The cached
response is the reproducible artefact**: a fresh call with the same snapshot (`gpt-4.1-mini-2025-04-14`), prompt and
temperature 0 can return a different ordering. The prompt, the feature records and the validation rules are in
[`docs/llm_prompt_protocol.md`](../../docs/llm_prompt_protocol.md).

## Files

`<dataset>/<dataset>__<arm>__<partition>__b<budget>[__r<repeat>].json`

| Part | Values |
|---|---|
| `arm` | `named` (definition-only records), `A` (names replaced by seeded identifiers, Home Credit), `B` (descriptions blanked, Home Credit), `lineage` (descriptions rebuilt from lineage) |
| `partition` | `fold1` to `fold5` (training partitions of the five development folds) or `full_dev` |
| `budget` | number of names requested |
| `repeat` | `r01` to `r10`: repetitions of the Stability 2024 full-DEV call with the same prompt |

Each file holds `ranking` (decoded feature names in priority order; for arm A also `ranking_identifiers`), the request
settings (`request_model`, `temperature`, `prompt_version`, `evidence_mode`), `prompt_sha256` and `request_sha256`,
`n_candidate_records`, the provider `response_id` and `response_model`, `attempts` and the reasons any earlier attempt
was rejected, token counts, the model's `reasoning_summary` and `selection_principles`, and the raw response.

## index.csv

One row per file: `dataset, partition, budget, condition, file, sha256, model_snapshot, temperature,
validation_passed, attempts, n_candidate_records, top_k_sha256, source, note`. `top_k_sha256` is the SHA-256 of the
newline-joined first 40 names (every budget up to 40 is a prefix); `source` is the cache file the copy was made from.

When a dataset's `results` stage runs, it replaces every file of that dataset's folder and rebuilds the index; rows of
files written before the study runner (earlier caches without a `call_id` field) are kept only while those files exist.

Every file is scanned for API-key, `Authorization` and `Bearer` patterns before it is written; a match stops the
publication.
