# Reproducibility summary

The analysis consumes only authenticated persisted artifacts under the Phase-1 lock commit `fd98d3c6d445e042b69dd24b0d6e8355157548dd`. It recomputes saved-prediction metrics at absolute tolerance 1e-10, uses paired DeLong and the registered 2,000-draw target-stratified bootstrap, and applies Holm within all 36 complete families. No raw research data or experiment runner is accessed.
