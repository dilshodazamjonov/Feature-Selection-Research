# Third-benchmark preregistration and analysis rules

Status: proposed in Stage 1; not canonical and not authorized for execution.

## Role and evidence hierarchy

Home Credit - Credit Risk Model Stability 2024 is the third robustness/replication benchmark. It is not fully independent institutional evidence because it shares Home Credit lineage with the earlier Home Credit dataset. Locked OOT evidence has priority over DEV evidence. DEV is for feasibility, diagnostics, and pre-authorized gating only; no method will be called “best” from DEV alone.

Predictive performance, feature-selection stability, calibration, drift, and resource cost will be reported in separate sections before any combined interpretation. Natural-support runs will always display requested K and realized support and will not be described as matched-K evidence when they differ. Failed, timed-out, resource-infeasible, or inapplicable configurations stay visible.

## Primary hypotheses

1. Each approved combination is compared on locked OOT with every component comparator registered before execution, within the same model, split, ordered case IDs, budget policy, and seed.
2. Each canonical standalone selector is compared with matched `full_features` and `random_k` controls on locked OOT.
3. A method’s predictive evidence and stability evidence are distinct: improved AUC does not imply improved stability, and stability does not substitute for AUC.
4. Replication across this benchmark strengthens robustness evidence but does not erase shared Home Credit lineage.

## Exact evidence-language rule

For each preregistered paired OOT comparison, let ΔAUC be comparator method minus reference. “Strong” predictive evidence requires ΔAUC > 0, a two-sided paired DeLong Holm-adjusted p < 0.05 within its frozen family, and a 95% paired stratified-bootstrap ΔAUC interval wholly above zero. “Moderate” requires ΔAUC > 0 and exactly one of those two inferential criteria. “Weak” requires ΔAUC > 0 and neither inferential criterion. “Not supported” applies when ΔAUC <= 0, paired identity/target alignment fails, or the required inference is unavailable or invalid. These labels apply only to the named comparison; they never establish a global “best” method.

Stability, calibration, drift, and resource results receive no significance label unless an already-frozen test exists. They are reported descriptively and separately, including Nogueira, all-pairwise Jaccard, eligible Kuncheva, PSI, log loss, Brier score, runtime, and memory. Any combined narrative must state the direction of each domain and any conflict.

## Deviations

After a canonical Stage 2 lock, every scope or implementation deviation requires a versioned amendment written and authenticated before the affected result is inspected. Silent deletion, replacement, padding, outcome-driven tuning, and post-OOT adaptation are forbidden.
