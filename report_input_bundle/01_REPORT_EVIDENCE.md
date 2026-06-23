# Report Evidence Bundle for Credit-Risk Feature Selection

This document is a compact evidence handoff for a later professor-facing report. It is not the final academic report. It consolidates audited project artifacts, recomputed prediction metrics, methodology boundaries, limitations, and plot interpretation guidance. Exact tabular values are in `02_REPORT_TABLES.xlsx`; plots are bundled in `03_REPORT_PLOTS.pdf`.

## 1. Study Overview

The objective was to evaluate feature-selection strategies for credit-risk classification under temporally separated development (DEV) and out-of-time (OOT) evaluation. The central research question was: **Did replacing the missingness-only CLIP-v1 statistical view with the compact 13-dimensional target-free CLIP-v2 statistical view improve feature selection and downstream out-of-time credit-risk performance?**

The study covers Home Credit and LendingClub v2. Home Credit served as the representation-training dataset for both CLIP versions and also as a downstream evaluation dataset. LendingClub v2 served as external application/validation for CLIP representation and as a separate downstream DEV/OOT task; it must not be described as a second CLIP representation-training dataset. Downstream models were Logistic Regression and CatBoost. Feature-selection methods were mRMR, LLM, LLM -> mRMR, CLIP-v1, CLIP-v1 -> mRMR, CLIP-v2, and CLIP-v2 -> mRMR. LR used 20 features and CatBoost used 40 features.

The motivation was to test whether semantic-statistical contrastive feature representations can produce useful compact feature sets in tabular credit risk. CLIP-v1 used frozen semantic feature text and a one-dimensional missingness statistical view. CLIP-v2 kept the semantic-statistical concept but replaced the statistical side with a compact target-free descriptor vector. Neither CLIP version directly predicts default; both rank/select features before conventional supervised downstream models are fit on DEV data.

## 2. Dataset and Validation Evidence

Home Credit: DEV rows 99092; OOT rows 120053; CLIP-v2 candidate features 529. LendingClub v2: DEV rows 598649; OOT rows 293105; CLIP-v2 candidate features 675.

For both datasets, the validation design used temporal DEV/OOT separation. OOT was not used for preprocessing, feature selection, checkpoint selection, anchor selection, or CLIP representation training. Post-outcome fields, target/label fields, ID fields, split/fold/prediction fields, and OOT/PSI fields were excluded from model inputs. LendingClub v2 remained transform-only for CLIP representation application: its descriptors were transformed with the unchanged Home Credit-fitted CLIP-v2 statistical scaler, and the learned anchor was the unchanged Home Credit anchor.

Final OOT prediction counts match saved artifacts: Home Credit files contain 120,053 OOT rows per CLIP run; LendingClub v2 files contain 293,105 OOT rows per CLIP run. This bundle recomputed AUC, KS, Lift@10, and row count for 16 CLIP prediction files. Score PSI is preserved from final aggregate/run tables; DEV score vectors were not persisted, so score PSI is not independently recomputable from prediction files alone.

## 3. Methodology

mRMR is a supervised DEV-only selector balancing relevance and redundancy. LLM ranks features from metadata/description evidence without labels or OOT outcomes. LLM -> mRMR first screens with LLM and then refines with DEV-only mRMR.

CLIP-v1 used frozen semantic feature text plus `missing_rate_dev`. It used Home Credit contrastive training, LendingClub v2 external application, and a frozen stable-core ranking anchor. Its selectors were `clip` and `clip_then_mrmr`. CLIP-v1 -> mRMR applies mRMR after CLIP-v1 screening.

CLIP-v2 used the same general semantic-statistical contrastive concept but replaced the one-dimensional statistical input with: missing_rate, unique_ratio, concentration_share, signed_log_mean, log_standard_deviation, clipped_skewness, normalized_entropy, is_numeric, is_categorical, is_binary, numeric_stats_valid, skewness_valid, entropy_valid. The statistical scaler was fitted only on Home Credit training-split feature vectors. LendingClub v2 was transform-only. Its selectors were `clip_v2` and `clip_v2_then_mrmr`.

Logistic Regression and CatBoost were downstream supervised models trained after feature selection. AUC measures ranking discrimination, KS measures maximum cumulative separation, Lift@10 measures bad-outcome concentration in the highest-risk decile, and score PSI describes score-distribution drift. Low PSI does not establish predictive quality or production safety. Semantic coverage counts selected semantic groups; redundancy captures repeated or near-duplicate feature families.

## 4. Contrastive-Training Evidence

CLIP-v1 used 384-dimensional frozen text embeddings and one statistical field, `missing_rate_dev`. The selected seed was 55, chosen by `lowest Home Credit validation loss; LendingClub v2 not inspected`. Its checkpoint hash was `3f21fc12060036f117aedf9a610856c72fa9a0ce6a1540403e572de4423d7385` and anchor hash was `e5446c5141cbdf8fde8677022b75cb1905c6e4e42fb48378727ecaf31eca604d`. The statistical preprocessor hash was `aa789679336fda5f656af23eb17528b49a8ab58b349377e9ec18967304fa5dba`.

CLIP-v2 used the compact 13-dimensional target-free statistical vector. It trained seeds 11, 22, 33, 44, 55 and selected seed 55 by `lowest Home Credit validation loss; LendingClub v2 not inspected`. Its checkpoint hash was `87907f848b6a24edad608a93744cb827b90195f7cb0560cd2699f9c3aa3c2de9`. The learned anchor hash was `9d460966cd7890ace01d0e42cde7cfffed802af04907f185309e100738749279`. The statistical preprocessor hash was `98265cde0bc0271a339ee7a1fe6bbb816f58953c45c482a567c7162ec50131c9` and the statistical anchor hash was `5e25adde60d6ef1376271060927f31fb82523df717ad762bcfdfcb3f7f8a8b9a`. Collapse audit statuses are `pass`.

Both versions used Home Credit for contrastive training and kept LendingClub v2 external. Neither version trained a default-prediction classifier inside the contrastive model.

## 5. Main Downstream Results

Exact results are in `Main_Results`. Core values:

- homecredit / lr:
  - `mrmr`: AUC 0.746, KS nan, Lift@10 3.096, score PSI 0.006, selected 20.
  - `llm`: AUC 0.740, KS nan, Lift@10 3.032, score PSI 0.007, selected 20.
  - `llm_then_mrmr`: AUC 0.738, KS nan, Lift@10 2.951, score PSI 0.004, selected 20.
  - `clip`: AUC 0.623, KS 0.185, Lift@10 1.800, score PSI 0.124, selected 20.
  - `clip_then_mrmr`: AUC 0.663, KS 0.242, Lift@10 2.206, score PSI 0.004, selected 20.
  - `clip_v2`: AUC 0.684, KS 0.270, Lift@10 2.537, score PSI 0.012, selected 20.
  - `clip_v2_then_mrmr`: AUC 0.740, KS 0.354, Lift@10 3.015, score PSI 0.012, selected 20.
- homecredit / catboost:
  - `mrmr`: AUC 0.767, KS nan, Lift@10 3.404, score PSI 0.010, selected 40.
  - `llm`: AUC 0.757, KS nan, Lift@10 3.290, score PSI 0.005, selected 40.
  - `llm_then_mrmr`: AUC 0.763, KS nan, Lift@10 3.372, score PSI 0.008, selected 40.
  - `clip`: AUC 0.662, KS 0.238, Lift@10 2.213, score PSI 0.018, selected 40.
  - `clip_then_mrmr`: AUC 0.703, KS 0.297, Lift@10 2.557, score PSI 0.021, selected 40.
  - `clip_v2`: AUC 0.745, KS 0.364, Lift@10 3.167, score PSI 0.004, selected 40.
  - `clip_v2_then_mrmr`: AUC 0.762, KS 0.391, Lift@10 3.338, score PSI 0.025, selected 40.
- lendingclub_v2 / lr:
  - `mrmr`: AUC 0.689, KS nan, Lift@10 2.095, score PSI 0.005, selected 20.
  - `llm`: AUC 0.693, KS nan, Lift@10 2.109, score PSI 0.007, selected 20.
  - `llm_then_mrmr`: AUC 0.691, KS nan, Lift@10 2.106, score PSI 0.008, selected 20.
  - `clip`: AUC 0.655, KS 0.223, Lift@10 1.938, score PSI 0.007, selected 20.
  - `clip_then_mrmr`: AUC 0.684, KS 0.268, Lift@10 2.084, score PSI 0.010, selected 20.
  - `clip_v2`: AUC 0.640, KS 0.205, Lift@10 1.756, score PSI 0.006, selected 20.
  - `clip_v2_then_mrmr`: AUC 0.690, KS 0.272, Lift@10 2.104, score PSI 0.006, selected 20.
- lendingclub_v2 / catboost:
  - `mrmr`: AUC 0.701, KS nan, Lift@10 2.167, score PSI 0.006, selected 40.
  - `llm`: AUC 0.714, KS nan, Lift@10 2.233, score PSI 0.006, selected 40.
  - `llm_then_mrmr`: AUC 0.704, KS nan, Lift@10 2.184, score PSI 0.006, selected 40.
  - `clip`: AUC 0.664, KS 0.233, Lift@10 1.989, score PSI 0.005, selected 40.
  - `clip_then_mrmr`: AUC 0.700, KS 0.287, Lift@10 2.150, score PSI 0.006, selected 40.
  - `clip_v2`: AUC 0.689, KS 0.274, Lift@10 2.049, score PSI 0.002, selected 40.
  - `clip_v2_then_mrmr`: AUC 0.707, KS 0.297, Lift@10 2.193, score PSI 0.006, selected 40.

Direct CLIP-v2 improved substantially over direct CLIP-v1 on Home Credit LR, Home Credit CatBoost, and LendingClub v2 CatBoost, but not on LendingClub v2 LR. CLIP-v2 -> mRMR improved over CLIP-v1 -> mRMR in all four dataset/model panels by AUC point estimate. The hybrid CLIP-v2 -> mRMR result is the most favorable CLIP-v2 configuration.

## 6. CLIP-v1 Versus CLIP-v2

Direct CLIP-v2 outperformed direct CLIP-v1 in 3 of 4 AUC panels. CLIP-v2 -> mRMR outperformed CLIP-v1 -> mRMR in 4 of 4 AUC panels. Evidence is **moderate** that the richer statistical view helped hybrid CLIP screening because every hybrid AUC point estimate moved upward. Evidence is weaker for direct CLIP-v2 because LendingClub v2 LR declined relative to direct CLIP-v1. V2-v1 uncertainty intervals were not computed, so do not imply statistical significance for v2-v1 deltas.

## 7. Comparison With Baselines

CLIP-v2 remained weaker than the strongest baseline in several important panels. On Home Credit, mRMR and LLM baselines generally stayed ahead of direct CLIP-v2. On LendingClub v2, CLIP-v2 -> mRMR was competitive with, and sometimes above, mRMR or LLM -> mRMR, but LLM remained strong for CatBoost. Direct CLIP-v2 should not be called best. The defensible conclusion is that CLIP-v2 improved the CLIP family and made CLIP + mRMR more viable, but did not replace the strongest baseline selectors.

## 8. External Validation

Frozen from Home Credit: CLIP representation training, selected checkpoint, anchor policy, and fitted CLIP-v2 statistical scaler. Recalculated from LendingClub v2 DEV: dataset-specific descriptors transformed under the Home Credit-fitted scaler and downstream feature selection/model fitting inside LendingClub v2 DEV. Not refitted: CLIP representation, checkpoint choice, anchor, and scaler. LendingClub v2 must be described as external application, not a second development dataset.

External OOT evidence is mixed but useful. CLIP-v2 transferred better after mRMR refinement than as direct screening. LendingClub v2 CatBoost direct CLIP-v2 improved over CLIP-v1 direct, and hybrid CLIP-v2 improved over hybrid CLIP-v1. LendingClub v2 LR direct CLIP-v2 declined versus direct CLIP-v1, while hybrid CLIP-v2 improved strongly.

## 9. Stability and Drift

Score PSI values are in `Score_PSI`. Project bands are low below 0.10, moderate from 0.10 to below 0.25, and high at 0.25 or above. Most CLIP-v2 score PSI values are low. CLIP-v1 Home Credit LR direct CLIP had moderate PSI. Some cases show improved AUC with low PSI, while others show lower AUC despite low PSI. **Low PSI does not establish predictive quality or production safety.**

## 10. Semantic Coverage and Redundancy

Semantic evidence is in `Semantic_Coverage`, `Redundancy`, and `Semantic_Map_Data`. CLIP-v1 direct selectors often concentrated in repeated families, especially on Home Credit. CLIP-v2 reduced repeated-family redundancy in final selected sets. CLIP-v2 also changed semantic group counts and largest-group shares. Broader semantic coverage is descriptive and does not necessarily improve prediction.

## 11. Feature-Selection Semantic Map

The semantic-map plot uses frozen text-embedding PCA. All eligible features are background points and selected features are highlighted for Home Credit and LendingClub v2, comparing CLIP-v1 and CLIP-v2 under CatBoost budgets. It reports selected count, semantic group count, v1/v2 Jaccard overlap, and largest-group share. **The semantic map is descriptive. It does not establish predictive superiority, causal importance, or statistically optimal feature clusters.**

## 12. Seed Robustness

CLIP-v1 and CLIP-v2 both selected seed 55 by lowest Home Credit validation loss; LendingClub v2 was not inspected for selection. Seed-level validation loss and MRR are in `Seed_Robustness`. Collapse diagnostics passed. Seed stability alone does not prove predictive superiority, and downstream multi-seed evaluation was not run.

## 13. Statistical Uncertainty

Paired-bootstrap uncertainty is available for CLIP-v1 versus selected baselines and is in `Statistical_Uncertainty`. CLIP-v1 deltas versus mRMR/LLM baselines were often negative with confidence intervals away from zero. For CLIP-v2 versus CLIP-v1, final artifacts provide point differences but not paired-bootstrap confidence intervals. Therefore, v2-v1 conclusions must distinguish point-estimate improvement from formal uncertainty.

## 14. Scientific Interpretation

### Facts

All eight CLIP-v2 runs completed valid. Final audit verdict: `PASS - CLIP-v2 is scientifically defensible and ready to archive`. Metric recomputation passed for 16 CLIP prediction files and 48 prediction-derived metric comparisons. Selected-feature counts matched expected LR/CatBoost budgets for all CLIP-v1 and CLIP-v2 runs.

### Interpretation

The compact target-free vector improved the CLIP family relative to CLIP-v1, especially when paired with mRMR refinement. Direct CLIP-v2 remained mixed. mRMR remained important because it consistently improved CLIP-v2 selected sets and helped external transfer. CLIP-v2 transferred externally in the hybrid setting but should not be claimed as a replacement for the best baselines.

### Recommendations

Future work should test richer target-safe descriptors, anchor ablations, more datasets, downstream multi-seed evaluation, and paired bootstrap for CLIP-v2 versus CLIP-v1. A stronger claim would require CLIP-v2 to beat or match mRMR/LLM baselines across more datasets and models with uncertainty support.

## 15. Limitations

Supported limitations include two credit datasets, one external dataset, Home Credit-only representation training, target-free statistical representation, no IV/WoE/univariate AUC/bad-rate/calibration descriptors, stable-core anchor dependence, metadata quality dependence, generic text encoder, fixed feature budgets, limited downstream model families, descriptive PCA plots, no fairness analysis, no probability calibration analysis, no economic-loss or approval-policy simulation, and multiple comparisons without v2-v1 uncertainty intervals.

## 16. Reproducibility Evidence

Current Git commit: `a51048478b842ef27f259015e70f0f4c5d5a5180`. CLIP-v1 freeze tag: `clip-v1-frozen`; CLIP-v2 readiness tag observed: `clip-v2-pipeline-ready`. Freeze manifest: `results/clip_versions/v1/freeze_manifest.json`. CLIP-v1 checkpoint hash: `3f21fc12060036f117aedf9a610856c72fa9a0ce6a1540403e572de4423d7385`. CLIP-v2 checkpoint hash: `87907f848b6a24edad608a93744cb827b90195f7cb0560cd2699f9c3aa3c2de9`. CLIP-v1 anchor hash: `e5446c5141cbdf8fde8677022b75cb1905c6e4e42fb48378727ecaf31eca604d`. CLIP-v2 learned anchor hash: `9d460966cd7890ace01d0e42cde7cfffed802af04907f185309e100738749279`. CLIP-v2 statistical preprocessor hash: `98265cde0bc0271a339ee7a1fe6bbb816f58953c45c482a567c7162ec50131c9`. Python: `Python 3.13.5`. Current verification run: full tests previously passed with 206 passed and 108 warnings; CLIP tests passed with 135 passed and 1 warning. Prediction hashes and run IDs are preserved in the workbook.

## 17. Recommended Final-Report Conclusion

Recommended conclusion: **mixed**.

Suggested conclusion paragraph: The experiment provides mixed but scientifically useful evidence. Replacing the missingness-only CLIP-v1 statistical input with a compact 13-dimensional target-free CLIP-v2 descriptor improved the CLIP family in most point-estimate comparisons and consistently improved the CLIP -> mRMR hybrid over the CLIP-v1 hybrid. However, direct CLIP-v2 was not uniformly better, v2-v1 uncertainty intervals were not computed, and the strongest conventional/LLM baselines remained competitive or superior in several key panels. The defensible contribution is therefore not that CLIP-v2 is the best feature selector, but that richer target-free statistical descriptors materially improve semantic-statistical contrastive feature screening and create a stronger candidate pool for supervised refinement.

## 18. Plot Index

See the `Plot_Index` sheet. The PDF includes 11 pages: all final CLIP-v2 plot-manifest figures, including the feature semantic map, plus CLIP-v1 reference plots needed for baseline, uncertainty, drift, semantic coverage, and seed robustness context.

## Additional Report-Writer Evidence Notes

### How to Frame the Research Contribution

The safest way to frame the contribution is methodological rather than purely performance-maximizing. The study does not show that CLIP-v2 is the universally best selector. It shows that a richer target-free statistical view materially changes and often improves CLIP-style feature screening relative to the missingness-only CLIP-v1 view. This distinction matters for the final report. A professor-facing report can argue that the experiment contributes evidence about how much information is gained by adding compact distributional descriptors to semantic feature text, but it should not claim that the resulting selector dominates standard or LLM-assisted baselines.

The most defensible narrative is that CLIP-v1 was an intentionally narrow first version. It combined semantic feature text with one statistical descriptor, `missing_rate_dev`. That made the contrastive representation heavily dependent on text semantics and sparse missingness behavior. CLIP-v2 retained the target-free constraint but expanded the statistical side to include missingness, uniqueness, concentration, signed log mean, log standard deviation, clipped skewness, normalized entropy, type flags, and validity flags. The v2 design therefore tests whether feature-distribution shape, not just missingness, helps align semantic and statistical representations.

The evidence supports a mixed conclusion. CLIP-v2 improved over CLIP-v1 in most point-estimate comparisons, especially after mRMR refinement. This means the richer representation improved the CLIP family. But direct CLIP-v2 still failed to beat key baselines in several panels. The conclusion should therefore emphasize improvement over the prior CLIP design, not superiority over all methods.

### How to Avoid Overclaiming

Do not say that CLIP-v2 predicts default. The contrastive model ranks features. Default prediction is performed only by Logistic Regression or CatBoost after features are selected. This is important because the CLIP training objective is not a supervised credit-risk loss. It is a representation-learning objective over feature text and statistical descriptors.

Do not say that LendingClub v2 was used to train the representation. It was not. LendingClub v2 was external application/validation. The CLIP-v2 statistical scaler, learned anchor, and checkpoint were selected from Home Credit only. LendingClub v2 descriptors were transformed using the unchanged Home Credit-fitted scaler. Downstream LendingClub v2 models were still trained on LendingClub v2 DEV because the downstream task requires dataset-specific supervised fitting, but that is separate from CLIP representation training.

Do not describe the semantic map as proof of better feature clusters. It is a visualization of frozen text-embedding PCA coordinates with selected features highlighted. It can show that CLIP-v2 and CLIP-v1 select different regions of the semantic space and different semantic groups. It cannot prove predictive quality, causality, or optimal clustering.

Do not interpret low score PSI as production safety. Score PSI only describes distributional shift in model scores between DEV and OOT. It says little about calibration, fairness, approval policy, expected loss, or stability under future macroeconomic changes. The final report can say that most CLIP-v2 score PSI values were low by project bands, but it must also say that this does not establish deployment readiness.

### Dataset-Specific Interpretation

Home Credit provides the strongest evidence that CLIP-v2 improved the CLIP family. Direct CLIP-v2 improved over direct CLIP-v1 for both LR and CatBoost. Hybrid CLIP-v2 -> mRMR also improved over CLIP-v1 -> mRMR. This is consistent with the fact that Home Credit was the representation-training dataset. However, Home Credit also shows that strong non-CLIP baselines remain difficult to beat. The final report should not treat Home Credit improvements over CLIP-v1 as evidence that CLIP-v2 beats all methods.

LendingClub v2 is the more important test of external behavior. The results are mixed but informative. CatBoost direct CLIP-v2 improved over direct CLIP-v1, and the hybrid selector improved as well. Logistic Regression direct CLIP-v2 declined relative to direct CLIP-v1, while Logistic Regression CLIP-v2 -> mRMR improved strongly over CLIP-v1 -> mRMR. This pattern suggests that the richer statistical view may be useful as a screening stage but still benefits from supervised redundancy/relevance refinement before downstream modeling.

Because LendingClub v2 is external, improvements there should be valued more cautiously but also more scientifically. A gain on LendingClub v2 is evidence of transfer, but the study has only one external dataset. The report should say that transfer evidence exists, especially for hybrid CLIP-v2, but cannot establish broad cross-domain generalization.

### Model-Specific Interpretation

Logistic Regression is more sensitive to feature selection because it has less built-in nonlinearity and interaction handling than CatBoost. The Home Credit LR result shows a large improvement from CLIP-v1 to CLIP-v2, and the hybrid LR result improves further. LendingClub v2 LR direct CLIP-v2 is the main counterexample. This means a final report should avoid saying that CLIP-v2 is uniformly better for linear models. It is better to say that CLIP-v2 can materially improve linear-model feature pools but still needs refinement and dataset-specific validation.

CatBoost is generally more robust to feature quality and nonlinear relationships. CatBoost results show that CLIP-v2 improved over CLIP-v1 on both datasets, especially through the hybrid selector. However, strong baselines such as LLM and mRMR remained competitive. For CatBoost, CLIP-v2 is best interpreted as a useful representation-driven screening strategy rather than a replacement for established selectors.

### Direct Versus Hybrid Selectors

The direct selectors answer whether the representation alone is sufficient to rank the final feature budget. The hybrid selectors answer whether the representation is useful as a candidate pool for a supervised refinement stage. The evidence is stronger for the second question. CLIP-v2 -> mRMR improved over CLIP-v1 -> mRMR in all four AUC panels. This is the cleanest evidence that the richer target-free representation adds value.

The final report should explicitly distinguish direct CLIP-v2 from CLIP-v2 -> mRMR. If they are averaged together rhetorically, the conclusion becomes confusing. Direct CLIP-v2 is mixed. Hybrid CLIP-v2 is consistently better than hybrid CLIP-v1 by point estimate. Baseline comparison remains mixed because even hybrid CLIP-v2 does not uniformly exceed the strongest non-CLIP selectors.

### What the Uncertainty Evidence Supports

The project contains paired-bootstrap uncertainty for CLIP-v1 versus selected baselines. Those intervals generally show CLIP-v1 underperforming strong baselines. The project does not contain paired-bootstrap confidence intervals for CLIP-v2 versus CLIP-v1. Therefore, the final report should not use language such as statistically significant improvement for CLIP-v2 over CLIP-v1 unless additional uncertainty analysis is performed later.

The appropriate evidence language is: "by point estimate," "moderate evidence," "consistent across hybrid panels," and "uncertainty intervals not computed for v2-v1." Practical importance can still be discussed if deltas are large and consistent, but it must be separated from statistical significance.

### How to Use the Excel Workbook

Use `Main_Results` for exact OOT AUC, KS, Lift@10, score PSI, selected feature count, run ID, prediction hash, and feature-set hash. Use `V1_vs_V2` for matched CLIP-v1 and CLIP-v2 comparisons. Use `Baseline_Comparison` to compare CLIP variants to mRMR, LLM, and LLM -> mRMR. Use `Dataset_Design` and `Method_Design` for methods and validation text. Use `Metric_Verification` for the statement that prediction-derived metrics were recomputed. Use `SelectedFeature_Checks` for budget validation. Use `Semantic_Map_Data` only for the semantic-map discussion.

The workbook preserves full numeric values; the displayed decimals may be rounded. Hash fields are text and should not be interpreted numerically. The final report should cite exact values from the workbook rather than reading numbers from plot labels.

### How to Use the PDF

The PDF is a delivery container for plots, not a separate source of exact numeric truth. Each page has a caption, source artifact, interpretation, and limitation. The CLIP-v2 plot manifest figures appear first, followed by CLIP-v1 reference plots. The feature semantic map is included and should be used in either a main figure or appendix depending on final report space. If only a few figures can appear in the main report, prioritize the CLIP-v2 OOT AUC comparison, CLIP-v2 AUC deltas, score PSI comparison, and semantic map. Baseline and seed robustness figures can be appendix material unless the professor expects full methodological traceability in the main body.

### Recommended Language for the Final Report

Preferred wording: "CLIP-v2 improved the CLIP-based feature-selection family relative to CLIP-v1, especially when followed by mRMR refinement." Avoid wording such as "CLIP-v2 solved feature selection," "CLIP-v2 is the best method," or "CLIP-v2 generalizes broadly." A good final sentence would be: "The richer target-free statistical view appears useful as a representation-driven screening signal, but supervised refinement and conventional baselines remain essential."

The final report can present the result as a constructive mixed outcome. A negative or mixed performance result is still scientifically useful because it identifies which part of the proposed method works. Here, the useful part is the enriched target-free statistical representation. The insufficient part is direct CLIP-only ranking as a standalone replacement for mRMR or LLM-assisted methods.

### Checklist for the Later Academic Report

The later report should open with the scientific comparison, not with implementation chronology. The reader should understand that the study compares a missingness-only semantic-statistical contrastive selector against a richer target-free version, while also benchmarking both against mRMR and LLM-based selectors. The report should then move from design, to validation boundaries, to downstream evidence, to interpretation.

When writing the dataset section, use `Dataset_Design` and avoid adding raw-data details that are not in the bundle. The important design facts are row counts, feature counts, target role, temporal DEV/OOT separation, leakage boundary, and the external status of LendingClub v2. The report should mention target rates only as descriptive context, not as a claim about population prevalence outside the study windows.

When writing the methods section, keep the target-free versus target-aware boundary clear. CLIP and LLM screening are target-free in the sense that they do not consume labels or OOT outcomes for ranking. mRMR is target-aware and must stay inside DEV/fold boundaries. This distinction explains why CLIP -> mRMR can improve over direct CLIP: the first stage supplies a representation-driven candidate pool, while the second stage uses supervised DEV information to select a compact final subset.

When writing the results section, do not use only average AUC. The four dataset/model panels matter because the pattern is not uniform. Report direct CLIP-v2 and hybrid CLIP-v2 separately. Then compare each against CLIP-v1 and against baselines. The final text should state that CLIP-v2 -> mRMR is the strongest CLIP-v2 variant, but should also state where non-CLIP baselines remain stronger.

When writing the external-validation section, make LendingClub v2 the scientific stress test. The report should explain that external validation is not merely a second performance table. It tests whether the Home Credit-trained representation and anchor remain useful when applied to LendingClub v2 without representation refitting. The mixed LendingClub v2 LR/CatBoost pattern is therefore central to the conclusion.

When writing limitations, keep them tied to evidence. Do not add speculative limitations that the artifacts do not support. Strong limitations include dataset count, single external dataset, missing v2-v1 uncertainty intervals, target-free descriptor limitations, metadata dependence, anchor dependence, lack of calibration/fairness/economic-loss analysis, and descriptive-only PCA maps. These limitations do not invalidate the study; they define the correct scope of the claim.

When writing the conclusion, use the mixed verdict directly. A clean conclusion should say that CLIP-v2 improved over CLIP-v1, especially with mRMR refinement, but did not establish a universal replacement for mRMR or LLM-assisted selectors. This is stronger than an overpositive conclusion because it matches the audited evidence and makes the contribution precise.

## Metric Verification Summary

Recomputed prediction-derived metrics passed with tolerance 1e-09. Checked metric rows: 64. Conflicts found: 0. Score PSI was copied from audited aggregate/run tables because DEV score vectors are not persisted.
