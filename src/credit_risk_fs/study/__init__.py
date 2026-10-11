"""End-to-end study runner: one protocol, three datasets, one configuration per dataset.

Stages (``scripts/run_study.py --stage ...``):

``prepare``     load the dataset, split DEV/HO, form the five expanding-window folds,
                apply the missingness screen on the earliest training partition and
                build the definition-only feature records.
``llm``         request (or reuse from the cache) every LLM ranking of the call plan.
``main``        the thirteen strategies and the full-feature reference, both backbones,
                five folds and the full-DEV refit scored once on HO.
``robustness``  native categorical handling, the L1 selector, the diversity controls,
                response sampling, name obfuscation, descriptions rebuilt from lineage
                and the matched mRMR timing.
``voting``      the pre-declared classical rank-voting family.
``sweep``       DEV-only feature-budget sweep.
``results``     the 25 result files of the dataset, the published LLM rankings and the
                headline subsets.
"""

__all__: list[str] = []
