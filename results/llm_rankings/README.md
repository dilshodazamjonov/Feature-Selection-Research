# Cached LLM rankings

These are the cached language-model responses behind the LLM-assisted selectors of the manuscript
(*Row-Independent Feature Pre-Screening with Language Models for High-Dimensional Credit Scoring*), copied
byte-for-byte from the local caches. **The cached response is the reproducible artefact**: a fresh call with the
same snapshot (`gpt-4.1-mini-2025-04-14`), prompt and temperature 0 can return a different ordering, as the
manuscript states (Section 4.4) and as the re-called `namedrecall` files below show.

Information contract of every call: `definitions_only` (feature names, definitions, lineage, aggregation operation
and window; no data rows, no statistics, no outcomes). The prompt template is Appendix A of the manuscript.

## Files

`<dataset>__<partition>__k<budget>__<condition>.json`, where `partition` is `fold1`..`fold5` or `full_dev`,
`budget` the number of names requested, and `condition`:

| condition | meaning | files |
|---|---|---|
| `named` | the accepted cached response of the frozen run (Home Credit: 6 calls of 100 names; LendingClub v2: 24 calls, 6 partitions x budgets 20/40/60/100) | `homecredit/`, `lendingclub_v2/` |
| `armA` | obfuscation arm A (names replaced by identifiers `F001`..`F529`, seed 20260914, definitions kept); the file stores identifiers, decoded through the identifier table in `homecredit_metrics_output/19b_obfuscation_ranks.csv` sources | `homecredit/` |
| `armB` | obfuscation arm B (names kept, definitions blanked) | `homecredit/` |
| `namedrecall` | the named re-call made in the same obfuscation run (2026-09-20); **not** the cached ranking of the headline tables; it is the reference against which the fold-overlap column of Table 7 reproduces | `homecredit/` |
| `lineage` | lineage-only (mechanical description) calls of 2026-09-22 (Table 8); see the index notes for which Table 8 overlaps they reproduce | all three |
| `repeat01`..`repeat10` | the ten repeated Stability 2024 full-DEV calls, as lists of 100 names plus the provider response ids (raw JSON not available locally) | `stability_2024/` |

The accepted Stability 2024 full-DEV ranking (one call of 100 names over the 1,068 candidates) is **not** in this
folder: the raw cached file is on the authors' Windows machine only. Its top-40 (and top-20) are the Stability
rows of `results/headline_subsets.csv` and `results/stability_metrics_output/09_leader_subsets.csv`.

## index.csv

One row per call: `dataset, partition, budget, condition, file, sha256, model_snapshot, temperature,
validation_passed, attempts, n_candidate_records, top_k_sha256, source, note`.

- `n_candidate_records` is the number of feature records in the request payload (`candidate_features` /
  `feature_metadata_rows` of the cache file). It is **not** the Table 2 candidate set: Home Credit calls carried
  374 / 369 / 365 / 374 / 372 / 391 records (fold1..fold5, full_dev) and LendingClub v2 calls 155 / 154 / 155 / 157 / 160 / 161.
  See `docs/repository_freeze.md`, finding 2.
- `validation_passed` is recomputed here from the stored response: TRUE when every returned name is a candidate
  record, names are distinct and the count equals the requested budget. Several LendingClub v2 responses carry
  fewer or more names than the budget and a few names outside the candidate records (all at ranks above 40);
  the note column states each case.
- `top_k_sha256` is the SHA-256 of the newline-joined first 40 names (the K=20 subset is its prefix).
- `temperature` is recorded in the file where the file stores it; otherwise "not recorded in the file".
- `source` is the local path the file was copied from (`REPO:` = this repository, git-ignored; `ARTICLE:` = the
  manuscript repository). Local paths are not in a fresh clone.

Not included (not manuscript artefacts): 21 older cache files in `artifacts/llm_cache` (LendingClub v1 and
prefiltered runs, and three extra LendingClub v2 budget-20 calls), and the regenerated Stability 2024 control
ranking of the Gate-4 fills.

Every file was scanned for `sk-`, `api_key`, `Authorization` and `Bearer` before being committed; none was found.
