# Prompt 16 final OOT execution blocker — 2026-08-16

The authorized final OOT controller stopped safely after the initial attempt
and five permitted automatic resource-recovery restarts all failed at the same
`locked_oot_data_loading` boundary. No evaluation cell, prediction, metric,
selection, analysis artifact, worker-success marker, or overall success marker
was promoted.

The frozen scientific registry remains unchanged. The identical command can
safely inspect and resume the authenticated partial state, but should not be
rerun under unchanged machine/runtime conditions: five of six attempts crossed
the 24 GiB process-tree hard cap, and the remaining attempt reached the 4 GiB
system-available hard floor before the OOT load completed.

This package records an execution blocker, not a scientific result and not a
completed Prompt-16 experiment.
