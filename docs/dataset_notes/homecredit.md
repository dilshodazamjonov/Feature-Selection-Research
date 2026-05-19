# Home Credit Notes

Home Credit is the main dataset used for method development.

## Layout

Raw files are expected under:
- `data/homecredit/raw`

Metadata lives under:
- `data/homecredit/metadata`

## Temporal Handling

The framework derives a time proxy using application and historical tables. The resulting temporal columns are used for DEV/OOT splitting and then excluded from model features.

## Role in Paper

Home Credit is the main dataset for:
- method comparison
- selector behavior analysis
- stability and drift analysis
