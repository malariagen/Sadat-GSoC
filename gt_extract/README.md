# gt-extract

Extract per-contig **GT (genotype) arrays** from remote `.zarr.zip` archives hosted over HTTPS without downloading the full archive.

## Install

```bash
pip install -e .
```

## CLI Usage

```bash
# Basic run
gt-extract --input-tsv selected_samples.tsv --output-dir data/gt_extracted

# Limit to 2 samples, 8 workers
gt-extract --input-tsv selected_samples.tsv --limit-samples 2 --workers 8

# Filter contigs
gt-extract --input-tsv selected_samples.tsv --contig-include "^3[LR]$"

# See all options
gt-extract --help
```

Or via `python -m`:

```bash
python -m gt_extract --help
```

## Python API

```python
from gt_extract import Config, run_pipeline, format_run_summary

cfg = Config(
    input_tsv="selected_samples.tsv",
    output_dir="data/gt_extracted",
    limit_samples=2,
    workers=4,
)
summary = run_pipeline(cfg)
print(format_run_summary(summary))
```

## Input Format

**`selected_samples.tsv`** — two tab-separated columns, no header:

```
https://example.com/sample1.gatk.zarr.zip	gambiae
https://example.com/sample2.gatk.zarr.zip	coluzzii
```

## Output

```
data/gt_extracted/
  sample1.zarr/
    3L/calldata/GT/...
    3R/calldata/GT/...
    _SUCCESS.json
  sample2.zarr/
    ...
```

Each sample directory contains per-contig Zarr v2 GT arrays and a `_SUCCESS.json` marker for resume/skip support.
