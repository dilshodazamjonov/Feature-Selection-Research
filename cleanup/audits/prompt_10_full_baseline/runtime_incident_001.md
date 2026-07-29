# Full-baseline runtime incident 001

Status: **root cause fixed; failed attempt retained; real rerun not started by the agent**

## Observed failure

Cell `fbv1-001-homecredit-lr-full-features-s42` completed all five DEV folds and
then failed while writing stability artifacts:

```text
ValueError: selected set is larger than the candidate universe
```

This was not a resource stop. Peak process RAM stayed well below the policy limit.
The manifest ended `failed/worker_crash`; the checkpoint authenticated only
`initialized` and `data_validated`. There is no OOT prediction, metric, or other
OOT artifact in the failed directory.

## Root cause

`full_features` was instantiated after the fold-local final-model preprocessor.
It therefore selected every one-hot/numeric model column: 646, 651, 651, 654, and
655 columns across folds, with a 656-column union. Stability was intentionally
given the authenticated original Home Credit universe of 529 features. Comparing
an encoded selected set to an original-feature denominator triggered the correct
fail-closed validation.

Changing the denominator to the encoded union would conceal the boundary error
and make the candidate universe depend on fold-specific observed categories. That
was rejected.

## Correction

`OriginalFeatureSelectorAdapter` now applies the authenticated Prompt 9 boundary
to every full-baseline selector:

1. take only the current DEV training fold's original candidate columns;
2. fit `OriginalFeatureNumericEncoder` on that training boundary;
3. fit the frozen selector on its one-column-per-original-feature numeric view;
4. return original feature names; and
5. fit the final-model preprocessor only after raw-column projection.

This keeps the selector universe at 529/675, retains training-only fitting, and
allows the final model to use its normal fold-local categorical preprocessing.
An integration regression test deliberately varies categorical levels between
folds and verifies that `full_features` records the three original candidates,
not the larger fold-specific encoded matrices.

## Failed-attempt evidence before archival

| Artifact | SHA-256 |
|---|---|
| `config.json` | `534efaef879192e1dd372630e9e51a7d04acd7a2b1da42040515137b80113cdf` |
| `manifest.json` | `3fc25557c53a2c9143cb42ce94576f215a415c8a2b12f06f5ac27cf61eb96905` |
| `checkpoint.json` | `4d97f7ce7b794c6c79fcf3493aee2c609c018ecc049a57d5138106f7ec77ceb0` |
| `resource_usage.json` | `6a749d327a38867fefaa53eaf78d9c47c75fa0a0dc7c1d9f281a648595805292` |

Failed configuration SHA-256:
`a5ebd3776f1670d8327bb089f50e2a8d7e1a9eeede949c37f4455972b84b73a1`.

Corrected frozen configuration SHA-256:
`f03647c376fe834f9bb1c3d6834ed42732ef3e7e1047eeff352af49b31ed607f`.

The recoverable archive is:
`results/full_baseline_v1/incomplete/superseded/fbv1-001-homecredit-lr-full-features-s42-config-a5ebd377-failed/`
(15 files). The fixed run ID is now free to restart from cell 001.
