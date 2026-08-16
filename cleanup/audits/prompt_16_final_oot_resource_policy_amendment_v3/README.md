# Prompt 16 final OOT resource-policy amendment v3

This execution-only amendment resolves the authenticated `fit_008` pause after
zero OOT model cells, predictions, or metrics were promoted. The old supervisor
suspended an opaque selector at 22.2 GiB RSS and then required available RAM to
rise from 7.5 GiB to 8 GiB; suspension retained the arrays and made that wait
self-sustaining on the observed 39.6 GiB laptop.

The amended envelope uses more of the installed RAM while retaining a 2 GiB
hard Windows floor, a 32 GiB process-tree cap, cooperative 3/4 GiB pause/resume
boundaries, and all disk and process-tree monitoring. Opaque numerical fits are
not suspended at the soft threshold; they remain subject to both hard limits.
The sealed selector checkpoints and disk-backed encoding are authenticated for
reuse, so execution continues at `fit_008` without changing scientific methods.
