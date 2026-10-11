# LLM prompt protocol

This file reproduces, verbatim from `src/credit_risk_fs/selectors/llm_screening.py`, the two prompts that the code
sends to the language model, and states which calls used which. The manuscript's Appendix A prints a third text
that neither code path sends ([last section](#relation-to-appendix-a-of-the-manuscript)).

Both paths call `gpt-4.1-mini` through Chat Completions at temperature 0 with a JSON response format. Temperature 0
does not make the model deterministic; the accepted, cached response is the reproducible artefact.

## Which calls used which path

| Path | Code | Calls |
|---|---|---|
| Target-free (`prompt_version` `stability_expert_v3`) | `LLMSelector.rank_target_free` renders `build_target_free_prompt`. Each record is a rendered definition restricted to `TARGET_FREE_METADATA_FIELDS` and validated against the candidate universe; the response must match a strict JSON schema with exactly the requested number of names; up to three attempts, then an error; no fallback. | The Stability 2024 ranking (`src/credit_risk_fs/experiments/prompt_16_llm_supplement.py`), the obfuscation arms (`scripts/run_b6_obfuscation_ablation.py`), the lineage-only and the repeated calls (`scripts/todo_fill/gate4/llm.py`) |
| Legacy (`LLMSelector.fit`) | `_build_prompt`. Candidates pass a missingness filter (0.95) and an IV/WOE prefilter fitted on the fold's training labels (`iv_filter_kwargs`: `min_iv` 0.01, `max_iv_for_leakage` 0.5, `encode` true; registry defaults in `src/credit_risk_fs/selectors/registry.py`). Records are built from the WOE-encoded training frame (`src/credit_risk_fs/feature_metadata/builder.py`). Up to three attempts on unparseable JSON, then a deterministic fallback to the first candidates (`_fallback_payload`). Names outside the candidates are dropped and a ranking shorter than the requested size is padded with the highest-IV unranked candidates. | The 30 cached Home Credit and LendingClub rankings (`results/llm_rankings/`, condition `named`). None of the 30 used the fallback; ten LendingClub responses were shorter than 100 names and were padded. |

## Target-free prompt (verbatim)

System message:

```text
You are a senior retail credit-risk feature-screening expert specializing in interpretable, stable variable selection.
```

User message. `{ranking_budget}` is the requested number of names and `{features_text}` holds one rendered
definition per candidate:

```text
You are a senior retail credit-risk feature-screening expert.

Context:
- You are reviewing outcome-independent feature definitions and adapter lineage only, the way a strong credit-risk expert would review a variable pack before modeling.
- Your job is to create a broad, stability-aware candidate ranking for downstream model development.
- A downstream statistical selector may perform fold-local supervised stability fitting and final redundancy pruning.

Task:
Produce a broad, stability-aware candidate ranking of exactly {ranking_budget} features for a binary loan-default model.
Do not try to make the final statistical selection yourself. Downstream consumers will apply the frozen model-specific budgets.

Priority criteria:
1. Stable out-of-time generalization.
2. Interpretable credit-risk meaning grounded only in the supplied definitions and lineage.
3. Durable repayment behavior, indebtedness, leverage, exposure, utilization, capacity, delinquency, and customer-history signals.
4. Avoid leakage-like artifacts, policy/process contamination, and brittle operational proxies identifiable from definitions alone.
5. Control redundancy among near-duplicate aggregates, especially within the same source family, unless variants clearly add distinct information.
6. Prefer representatives that a credit-risk expert could defend in a stable scorecard review.

Rules:
1. Use only the feature names provided below and copy every selected name verbatim.
2. You are seeing no rows, targets, split statistics, validation information, OOT information, performance, drift, missingness rates, IV, correlation, mutual information, SHAP, or model importance.
3. Prefer broad semantic coverage rather than over-concentrating on one narrow family.
4. Return exactly {ranking_budget} distinct features in priority order, best first.
5. Every candidate not placed in the {ranking_budget}-item selected_features array is intentionally not selected.
6. Do not return more or fewer than {ranking_budget} names. Do not invent, normalize, replace, or duplicate a selected name.

Features:
{features_text}

Return ONLY valid JSON:
{
  "selected_features": ["feature_1", "feature_2"],
  "reasoning_summary": "One concise high-level reason.",
  "selection_principles": ["stability", "coverage", "redundancy control"],
  "feature_reasons": {}
}

Keep the response compact. Do not include per-feature explanations unless absolutely necessary.
```

Appended to the user message on attempts 2 and 3:

```text
CRITICAL RETRY INSTRUCTION: the previous response failed the frozen JSON or candidate-coverage contract. Fill exactly the requested number of slots in selected_features with distinct names copied verbatim from the candidate list. Candidates outside those slots are intentionally not selected. Do not return extra names and do not invent, normalize, replace, or duplicate a selected name.
```

## Legacy prompt (verbatim)

The system message is the same as above. User message, with `{ranking_budget}` the requested number of names (100
for every cached Home Credit and LendingClub call):

```text
You are a senior retail credit-risk feature-screening expert.

Context:
- You are reviewing metadata only, the way a strong credit-risk expert would review a variable pack before modeling.
- Your job is to create a broad, stability-aware candidate ranking for downstream model development.
- A downstream statistical selector will make the final redundancy pruning and final budget decision.

Task:
Produce a broad, stability-aware candidate ranking of up to {ranking_budget} features for a binary loan-default model.
Do not try to make the final statistical selection yourself. The downstream selector will refine this list for redundancy and final budget.

Priority criteria:
1. Stable out-of-time generalization.
2. Low missingness and broad coverage.
3. Interpretable credit-risk meaning.
4. Durable repayment behavior, indebtedness, leverage, exposure, utilization, capacity, delinquency, and customer history signals.
5. Avoid leakage-like artifacts, policy/process contamination, brittle operational proxies, and unstable sparse tail-dominated variables.
6. Control redundancy among near-duplicate aggregates, especially within the same semantic group, unless variants clearly add distinct information.
7. Prefer representatives that a credit-risk expert could defend in a stable scorecard review.

Rules:
1. Use only the feature names provided below.
2. You are seeing training-slice metadata only, not validation or OOT data.
3. Prefer broad semantic coverage rather than over-concentrating on one narrow family.
4. When two features look similar, prefer the less-missing, more interpretable, more operationally durable one.
5. Avoid selecting many duplicate aggregates from one business concept unless their summaries suggest meaningfully different information.
6. Return features in priority order, best first.
7. Do not invent new feature names.

Features:
{features_text}

Return ONLY valid JSON:
{
  "selected_features": ["feature_1", "feature_2"],
  "reasoning_summary": "One concise high-level reason.",
  "selection_principles": ["stability", "coverage", "redundancy control"],
  "feature_reasons": {}
}

Keep the response compact. Do not include per-feature explanations unless absolutely necessary.
```

Each line of `{features_text}` has the form

```text
- {name} | semantic_group={semantic_group} | table={table} | dtype={dtype} | missing_rate={missing_rate:.1%} | non_null_count={non_null_count} | desc={description}
```

followed, for a numeric record, by `| mean= | min= | max= | std= | var= | p05= | p25= | p50= | p75= | p95=` and
otherwise by `| unique_count=`. Because the records are built after the IV/WOE prefilter has replaced each value by
the weight of evidence of its bin, these summaries describe the WOE-encoded column, which is derived from the
training labels.

Appended to the user message on attempts 2 and 3:

```text
CRITICAL RETRY INSTRUCTION: the previous response was not parseable JSON. Return compact valid JSON only. Include selected_features, a short reasoning_summary, selection_principles, and set feature_reasons to {}.
```

## Relation to Appendix A of the manuscript

Appendix A prints a "unified prompt template" with an evidence-mode contract (`definitions_only` or
`training_metadata`), a `status` key in its output contract and an error branch. That text is not the prompt of
either code path above and does not occur in any commit of this repository. The target-free prompt above is the
one the code sent for Stability 2024 and the robustness calls, and the legacy prompt the one it sent for the cached
Home Credit and LendingClub rankings. README Section 15 lists what this means for the reported results.

## Cached responses

- Tracked copies: `results/llm_rankings/`. `index.csv` gives each file's partition, budget and condition, and in
  `source` the name under which it was cached. The accepted Stability 2024 ranking is not included; the ten
  repeated Stability 2024 calls are included as name lists.
- The legacy path looks a response up as `<scope>_<fold_key>_<sha256 of the cache-key payload>.json` in its cache
  directory (`artifacts/llm_cache/` for the registry routes). Each payload records `request_sha256`,
  `prompt_sha256`, the metadata signature, the response id and the token counts.
- Fresh calls need `OPENAI_API_KEY` (see `.env.example`), return different rankings and are not needed to read
  the results. Never commit `.env`.
