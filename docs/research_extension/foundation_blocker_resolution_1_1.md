# Foundation blocker resolution 1.1

Status: **PASS** on 2026-07-21.

Prompt 1.1 resolved every forward-science blocker without running an experiment:

1. A streamed sidecar recovers the genuine LendingClub raw `id` as canonical `loan_id` and authenticates it against exact raw and processed hashes.
2. DEV/OOT identities are unique, non-missing, disjoint, and stored with ordered ID and ID+target hashes.
3. Equal LendingClub timestamps use `recent_decision ASC, loan_id ASC` under a machine-stable serialization.
4. `cross_dataset_rank_voting_v1` replaces the unverifiable historical claim as an explicitly new prospective design with two fold-local voters (`rf_corr_mrmr`, `boruta`) and fold-local RFE downstream.
5. Corrected CLIP is excluded; its missing historical provenance is a limitation, not a forward blocker.
6. Paired DeLong, paired stratified bootstrap, 95% percentile intervals, four comparison families, and within-family Holm correction are implemented and tested.
7. The 31 historical CLIP integrations require a declared, hash-valid `clip_complete_v1` profile. The current incomplete bundle skips honestly as `required_external_evidence_unavailable`; a declared but bad bundle fails.

The detailed scientific contract and exact hashes are in [foundation_protocol_freeze.md](foundation_protocol_freeze.md), [row_alignment_contract_v1.json](../../configs/protocols/row_alignment_contract_v1.json), [cross_dataset_rank_voting_v1.yaml](../../configs/protocols/cross_dataset_rank_voting_v1.yaml), and [lendingclub_identity_evidence.json](../../cleanup/audits/foundation_protocol_freeze/lendingclub_identity_evidence.json).

No action beyond preserving this freeze is required before Prompt 2. Prompt 2 resource hardening has not started.
