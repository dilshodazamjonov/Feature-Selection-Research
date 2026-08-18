# Prompt 16 sparse final-model preprocessing amendment v6

This amendment replaces only the final-model encoded matrix representation:
the fitted numeric and categorical semantics are unchanged, while the numeric
block and direct sparse one-hot block are assembled as canonical float32 CSR.
DEV and OOT scores are produced in bounded 50,000-row batches, and each newly
sealed OOT evaluation exits its worker process before the next cell begins.

The exact 1,221,743-row cell-3 DEV fit passed under the restored 24/4/6/8 GiB
policy without loading locked OOT. The production tree remains unchanged at
OOT 03/34. The predecessor authorization is preserved and explicitly
superseded by the new hash-authenticated authorization in this directory.
