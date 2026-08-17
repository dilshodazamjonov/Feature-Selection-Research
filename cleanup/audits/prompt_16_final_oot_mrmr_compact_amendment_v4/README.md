# Prompt 16 compact mRMR amendment v4

This execution-only amendment replaces the failed eager int64 mRMR code
dictionary with atomic, SHA-256-authenticated int8 NPY batches and separately
sealed MI vectors. It preserves all 1,959 ordered features and all 1,221,743
full-DEV rows. The canonical discretizer, missing code -1, sklearn MI estimator,
greedy arithmetic, name tie-break, ranking, seeds, preprocessing, and model
settings are unchanged.

Representative fixtures and a full-DEV replay of fit_007 matched the sealed
predecessor exactly. The locked OOT slice was not opened during validation.
The successor resumes at fit_008 and reuses the authenticated fit_007 MI work.
