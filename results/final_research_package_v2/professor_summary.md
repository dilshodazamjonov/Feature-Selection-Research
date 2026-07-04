# Professor summary

This revision completes the requested comparison and directional representation analysis using only authenticated saved artifacts. It adds the missing standalone LLM → mRMR comparator, retains corrected CLIP → mRMR and LLM → corrected CLIP → mRMR, and separates Home Credit → LendingClub v2 from LendingClub v2 → Home Credit. No training, feature selection, prediction, or embedding generation was rerun.

## 1. LLM + corrected CLIP + mRMR

The direct question is: did adding corrected CLIP after LLM help?

For Logistic Regression, the answer is **not predictively**. LLM → mRMR had OOT AUC 0.738098 and KS 0.353691. The combined pipeline had AUC 0.736996 and KS 0.354224. AUC fell by 0.001102; KS rose by only 0.000533. Score PSI increased from 0.003798 to 0.004898, so score stability was slightly worse. Feature-selection Jaccard improved substantially, from 0.404344 to 0.729185. LR therefore gained selection reproducibility but not prediction or score stability.

For CatBoost, the answer is **a stability benefit without clear predictive benefit**. AUC moved from 0.762954 to 0.763820, an increase of only 0.000866. KS decreased from 0.392924 to 0.392046. The tiny, untested AUC change does not establish meaningful improvement. PSI fell from 0.008286 to 0.004094, and Jaccard rose from 0.296557 to 0.795195.

The result is mixed rather than uniformly positive. Corrected CLIP made selected feature sets more reproducible, especially for CatBoost, but did not consistently improve discrimination. The full Home Credit mRMR baseline remained descriptively higher in OOT AUC for both models.

The stability comparison required a provenance correction. The saved Task 1 Nogueira and Kuncheva values identify 529 as the selectable universe, although the actual LLM and combined candidate manifests contain 60 LR and 100 CatBoost features. Those universe-dependent values are not used. Mean pairwise Jaccard was recomputed directly from the five saved fold feature sets and matched the stored Jaccard values. Task 1 PSI is directly comparable within each model pair because both methods use the same DEV/OOT windows and historical calculation, but it is not compared numerically with reverse-transfer PSI, whose reference is pooled DEV OOF predictions.

## 2. Reverse transfer

The LendingClub v2-trained corrected CLIP representation was projected frozen to Home Credit using five aligned seed spaces. It produced 436 eligible Home Credit embeddings and fixed 60/100 candidate pools before DEV-only mRMR. Reverse LR had OOT AUC 0.573203; reverse CatBoost had 0.676603. CatBoost recovered more signal and had low PSI, but both models were weaker than Home Credit-trained pipelines. Reverse predictive transfer was therefore validly evaluated but not competitive.

The reverse evaluation has the strongest row provenance in the study: 82,647 pooled DEV OOF predictions and 120,053 OOT predictions retain stable `SK_ID_CURR` values and reconcile to prediction and metric manifests. LR OOT performance was weak, while CatBoost retained moderate signal but remained roughly 0.090 AUC below the Home Credit mRMR baseline. Low score PSI does not overcome that discrimination gap.

## 3. Directional embedding evidence

The forward UMAP represents 576 LendingClub v2 features projected through the Home Credit-trained corrected representation. Original-space kNN purity was 0.528125, versus shuffled mean 0.101023 and 97.5th percentile 0.110430; trustworthiness was 0.979712.

The reverse UMAP represents 436 Home Credit features projected through the five-seed LendingClub-trained consensus. Purity was 0.753899, versus shuffled mean 0.140146 and 97.5th percentile 0.153905; trustworthiness was 0.987602. Both directions show non-random representation structure. Their taxonomies differ, so purity values and UMAP locations must not be ranked directly.

The scientific distinction is decisive: representation success is supported in both directions; predictive success is not. Stability improved after corrected CLIP mainly for CatBoost. Bidirectional robustness remains unsupported because the forward direction has no corrected downstream model and reverse performance is not competitive.

All directional coordinates were newly calculated from the already-saved 32-dimensional matrices using the same cosine UMAP settings: 15 neighbours, minimum distance 0.1, two components, and seed 42. The original embeddings were not regenerated. UMAP axes and cluster locations have no direct substantive interpretation, and the two layouts are not geometrically comparable. The original-space kNN diagnostics, rather than visual position, support the representation conclusion.
