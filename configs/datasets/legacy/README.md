# Legacy Dataset Configs

`lendingclub.yaml` is kept here for reproduction of the LendingClub v1 line, whose prepared frame
(`scripts/prepare_lendingclub.py`) is the input of LendingClub v2.

The Home Credit and LendingClub v2 dataset configurations are:

- `configs/datasets/homecredit.yaml`
- `configs/datasets/lendingclub_v2.yaml`

Stability 2024 is configured by `src/credit_risk_fs/data/homecredit_model_stability_2024/` and its protocol lock
under `configs/protocols/homecredit_model_stability_2024_v1/`. The study configurations of all three datasets are in
`configs/study/`.
