# Research status

| Task | Status | Canonical evidence | Limitations | Remaining work |
|---|---|---|---|---|
| Home Credit corrected CLIP and combined pipelines | Complete | `results/corrected_homecredit_clip/` | Saved-artifact reproduction only; experiments must not be rerun | None |
| Home Credit to LendingClub v2 directional projection | Complete | `results/corrected_homecredit_clip/training/lendingclub_v2_joint_embeddings.parquet` | External projection, not target-domain refitting | None |
| LendingClub v2 to Home Credit reverse transfer | Complete | `results/corrected_lendingclub_to_homecredit_transfer/` | Frozen directional transfer contract | None |
| Final report package | Complete | `results/final_research_package_v2/` | Reflects completed analyses available at package build time | Rebuild only from authenticated saved artifacts when registry metadata changes |
| Paired five-fold significance | Pending | Baseline and corrected-run `cv_results.csv`, fold manifests, selected features, and paired-fold tables | Not every final comparison has a completed paired test | Complete and report paired tests |
| Feature-level PSI/drift | Pending | Home Credit and LendingClub v2 selected-feature PSI and drift evidence | Coverage is incomplete for all feature families | Complete full feature-level analysis |
| LLM cost and scalability | Pending | `artifacts/llm_cache/`, LLM summaries, runtime files, prompts/responses | Provider billing authentication may require external pricing records | Compute token, cache, runtime, and scale scenarios |
