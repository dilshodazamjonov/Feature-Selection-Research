# Final CLIP Limitations

- non-fold-local CLIP preparation: rebuild representation fold-locally in future
- DEV-CV diagnostic limitation: treat OOT as primary
- limited statistical view: The current contrastive encoder aligns semantic feature metadata with a limited DEV statistical view, primarily reflecting missingness behavior. It is an architectural and screening experiment rather than a comprehensive statistical feature-quality representation.
- Home Credit-only contrastive training: train on more datasets after leakage review
- LendingClub v2 external-only application: add more external datasets
- limited seed count: increase seeds for representation and downstream evaluation
- unavailable paired baseline predictions where applicable: persist all baseline predictions
- independent PSI-recomputation limitation: persist DEV score vectors
- fixed feature budgets: evaluate budget sensitivity
- limited dataset count: add datasets
- no fairness analysis: run fairness audit before deployment claims
- no operational cost analysis: measure cost if operationalized
- no causal interpretation: avoid causal language
- no production-readiness claim: separate production validation
