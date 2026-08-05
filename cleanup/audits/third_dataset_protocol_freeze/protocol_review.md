# Third-dataset protocol Stage 1 review

Status: **review package only; no canonical lock; no pilot/DEV/OOT authorization**

## Answer first

The local snapshot is structurally suitable for a locked third-benchmark protocol. It contains 18 included training Parquet files and 14 excluded depth-2 Parquet files. All included relations have zero orphan `case_id` rows, base IDs are unique, the target is exactly binary {0,1}, and every one of the 434 depth-0/1 raw predictors has an official definition. No unresolved leakage row or unsupported dtype remains.

The proposed OOT split is the latest whole-date tail closest to 20%: DEV is through 2020-02-25 and locked OOT starts 2020-02-26. This is the prompt-authorized fallback because no existing dataset-specific third-benchmark boundary exists. The canonical five-fold expanding splitter with one unique-date gap is then applied inside DEV.

## Identity and structural scope

- Official dataset: Home Credit - Credit Risk Model Stability 2024
- Local identity: `homecredit_model_stability_2024`
- Frozen included-input digest: `8adb1db82c9dafb662657db08fd7d1dcf2eb4794d5ff7925e9ca4dd25f73fad2`
- Base: 1,526,659 rows; target 0 = 1,478,665; target 1 = 47,994; dates 2019-01-01..2020-10-05
- Included: base plus depth 0 and depth 1 only. Depth 2 is inventoried and excluded.
- Included table families: base, static, static_cb, applprev, credit_bureau_a, credit_bureau_b, debitcard, deposit, other, person, tax_registry_a, tax_registry_b, tax_registry_c.
- Relational findings: zero orphans in every included family; depth-0 families are one-to-one; depth-1 families follow their observed one-to-one/one-to-many profile. `other` is currently one-to-one but remains governed by the depth-1 aggregation contract.
- `static_cb` covers 1,500,476/1,526,659 base cases; missing relation rows remain missing after the base-left-join.

## Exact temporal proposal

- DEV: 1,221,743 rows (1,182,098/39,645), 2019-01-01..2020-02-25.
- OOT: 304,916 rows (296,567/8,349), 2020-02-26..2020-10-05, 19.972764055% of base.
- OOT membership: `date_decision >= 2020-02-26`; whole dates remain intact; ordered case-ID hashes authenticate DEV, OOT, and every fold.

| Fold | Train dates | Train rows (0/1) | Validation dates | Validation rows (0/1) |
|---:|---|---:|---|---:|
| 1 | 2019-01-01..2019-03-28 | 200,661 (195,314/5,347) | 2019-03-30..2019-06-19 | 204,567 (198,636/5,931) |
| 2 | 2019-01-01..2019-06-17 | 402,103 (390,901/11,202) | 2019-06-20..2019-08-24 | 203,798 (198,250/5,548) |
| 3 | 2019-01-01..2019-08-22 | 604,598 (587,862/16,736) | 2019-08-25..2019-10-27 | 205,980 (198,821/7,159) |
| 4 | 2019-01-01..2019-10-25 | 810,904 (787,058/23,846) | 2019-10-28..2019-12-20 | 201,466 (194,186/7,280) |
| 5 | 2019-01-01..2019-12-18 | 1,012,061 (981,057/31,004) | 2019-12-21..2020-02-25 | 202,820 (194,521/8,299) |

## Adapter and leakage decisions

The adapter is specification-only. It anchors one row per base `case_id`, concatenates multipart families in numeric part order, requires depth-0 uniqueness, and aggregates every depth-1 family in fixed `num_group1` order. Numeric, logical-date, boolean, and categorical aggregation lists are fixed in `proposed_adapter_protocol.json`; no result can choose them later. Every output is prefixed by depth and family.

The review has 461 source-family schema rows: 434 included predictors, 27 excluded identifiers/target/split controls, and 0 unresolved. `target`, every `case_id`/`num_group*`, `date_decision`, `MONTH`, and `WEEK_NUM` are excluded. Fold-local canonical preprocessing is preserved; OOT is transformed only with full-DEV-fitted objects. No domain-crafted feature engineering is allowed beyond the fixed relational reduction.

## Scientific matrix and gates

The proposed matrix contains the nine frozen baselines followed by the four approved combinations in their approved order. IV→Boruta keeps pools 100/200/300 (200 primary); LR requests K=20 and CatBoost K=40; seed 42 is universal. This yields 15 variants per model and 30 evaluation configurations.

- Bounded fold-1 pilot: 27 selector-fit calls and 30 evaluations; OOT inaccessible.
- Five-fold DEV: 135 selector-fit calls and 150 held-out fold evaluations, summarized as 30 configurations.
- Locked OOT: 27 full-DEV selector refits and 30 one-time evaluation cells, only after separate pilot and DEV authentication/review gates.
- Natural support: no padding; any realized support below K is labeled `infeasible_natural_support` and compared with the realized count visible.

Source-table-family coverage may be reported descriptively, but formal semantic coverage is deferred because no pre-existing semantic-group map covers this schema. Semantic selector/voter extensions, LLM-assisted methods, corrected contrastive methods, directional transfer, and the existing two-dataset cross-dataset voting lock are also deferred because no pre-Prompt-14 plan specifies their use on this third dataset. This is a visible scope decision, not a performance-based removal.

## Blockers and approval boundary

There is no unresolved Stage 1 identity, schema, split-feasibility, leakage, or matrix blocker. The adapter is intentionally not implemented, the canonical protocol lock is intentionally absent, and all execution gates remain closed. Stage 2 may create a lock only after the user quotes the exact review digest and explicitly approves the full listed scope.

The canonical review digest is recorded in `review_digest.json`.
