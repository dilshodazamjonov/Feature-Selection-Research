# LLM Prompt Protocol

The LLM selector is not framed as a replacement for statistical feature selection. It is a first-stage metadata screener.

## Prompt Role

The prompt asks the model to:
- review metadata only
- produce a broad candidate ranking
- emphasize stability, low missingness, interpretability, and semantic coverage
- avoid leakage-like or brittle operational proxies

## Reproducibility

For each LLM-assisted run, the repository stores:
- prompt text
- prompt hash
- metadata signature
- selected features payload
- raw response payload
- ranking summary CSV

## Interpretation Constraint

The paper claim should remain:
- LLM metadata screening is helpful as an initial screen

The paper claim should not become:
- LLM alone replaces statistical feature selection
