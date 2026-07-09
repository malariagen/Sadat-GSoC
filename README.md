# fastq-classifier

Reference-free FASTQ feature extraction and classification components.

## Fetch first N paired reads

The fetcher reads an ENA run report TSV. The report must include
`run_accession` and `fastq_ftp`.

Runtime tools:

- `curl`, available on `PATH` or passed with `--curl`
- `seqkit`, available on `PATH` or passed with `--seqkit`

SeqKit is an external binary, not a Python dependency. Install it from the
[SeqKit download page](https://bioinf.shenwei.me/seqkit/download/) or with a
package manager such as Bioconda.

### CLI

Install from the zip:

```powershell
python -m pip install fastq_classifier.zip
```

On Windows, put SeqKit in a local `tools` folder:

```powershell
mkdir tools
curl.exe -L -o tools\seqkit.tar.gz https://github.com/shenwei356/seqkit/releases/download/v2.13.0/seqkit_windows_amd64.exe.tar.gz
tar.exe -xzf tools\seqkit.tar.gz -C tools
.\tools\seqkit.exe version
```

Run the fetch command and pass the SeqKit path directly:

```powershell
fastq-classifier fetch-first-n `
  --ena-report filereport_read_run_PRJNA1169887.tsv `
  --out-dir fetched_reads `
  --read-pairs 25000 `
  --jobs 4 `
  --seqkit .\tools\seqkit.exe
```

Outputs are written under `fetched_reads`:

- `invalid_rows.tsv`: input rows that could not be downloaded
- `fetch_results.tsv`: completed, skipped, and failed downloads
- `runs/<run_accession>/*.fastq.gz`: one R1 and one R2 subset per run

### Python API

```python
from fastq_classifier import fetch_first_n

report = fetch_first_n(
    "filereport_read_run_PRJNA1169887.tsv",
    "fetched_reads",
    read_pairs=25_000,
    jobs=4,
)

print(f"downloads: {len(report.downloads)}")
print(f"invalid rows: {len(report.invalid_rows)}")

for run in report.downloads:
    print(run.run_accession, run.status, run.written_read_pairs, run.error)
```

`fetch_first_n` returns:

- `DownloadReport.downloads`: one `DownloadedRun` for each accepted row
- `DownloadReport.invalid_rows`: input rows that could not be used

Setup errors are raised as `InputError` or `FetchError`.

Result statuses:

- `completed`: R1 and R2 files were downloaded and checked
- `skipped`: both output files already existed and had matching record counts
- `failed`: the download failed; inspect `run.error`

Invalid rows are input problems, not download failures:

```python
for row in report.invalid_rows:
    print(row.row_number, row.reason)
```

Pass explicit tool paths when `curl` or `seqkit` are not on `PATH`:

```python
report = fetch_first_n(
    "filereport_read_run_PRJNA1169887.tsv",
    "fetched_reads",
    read_pairs=25_000,
    jobs=4,
    curl=(r"C:\Windows\System32\curl.exe",),
    seqkit=(r"C:\tools\seqkit.exe",),
)
```

The `curl` and `seqkit` arguments are command prefixes. They can include extra
wrapper arguments when needed, for example `("conda", "run", "-n", "bio", "seqkit")`.

## Extract exact k-mer features

The feature extractor reads `fetch_results.tsv` from the fetch step and runs
KMC on each completed or skipped paired-end FASTQ subset.

Runtime tool:

- `kmc`, available on `PATH` or passed with `--kmc`

KMC is an external binary, not a Python dependency. Install it from the
[KMC release page](https://github.com/refresh-bio/KMC/releases) or with a
package manager such as Bioconda.

### CLI

```powershell
fastq-classifier extract-kmers `
  --fetch-results fetched_reads\fetch_results.tsv `
  --out-dir features_k13 `
  --k 13 `
  --jobs 4 `
  --kmc .\tools\kmc.exe
```

Outputs are written under `features_k13`:

- `invalid_rows.tsv`: fetch-result rows that could not be counted
- `feature_results.tsv`: completed, skipped, and failed KMC runs
- `runs/<run_accession>/*.kmc_pre` and `*.kmc_suf`: exact canonical k-mer count
  databases
- `runs/<run_accession>/*.stats.json`: KMC count summary

### Python API

```python
from fastq_classifier import extract_kmer_features

extraction = extract_kmer_features(
    "fetched_reads/fetch_results.tsv",
    "features_k13",
    k=13,
    jobs=4,
)

for database in extraction.databases:
    unique = None if database.stats is None else database.stats.unique_kmers
    print(database.run_accession, database.status, unique, database.error)
```

## Build a sparse feature matrix

The matrix builder reads `feature_results.tsv` from the KMC extraction step and
uses `kmc_dump` to convert each KMC database into sparse matrix entries.

Runtime tool:

- `kmc_dump`, available on `PATH` or passed with `--kmc-dump`

### CLI

```powershell
fastq-classifier build-matrix `
  --feature-results features_k13\feature_results.tsv `
  --out-dir matrix_k13 `
  --kmc-dump .\tools\kmc_dump.exe
```

Outputs are written under `matrix_k13`:

- `samples.tsv`: matrix rows and the original feature-result metadata
- `features.tsv`: matrix columns and their k-mers
- `matrix.npz`: sparse SciPy CSR count matrix
- `invalid_rows.tsv`: feature-result rows that could not become matrix samples

### Python API

```python
from fastq_classifier import build_kmer_matrix

matrix = build_kmer_matrix(
    "features_k13/feature_results.tsv",
    "matrix_k13",
)

print(matrix.sample_count, matrix.feature_count, matrix.entry_count)
print(matrix.matrix_path)
```

## Evaluate a classifier

The classifier reads a matrix directory from `build-matrix`. Labels are read
from a column in `samples.tsv`. Each class needs at least two samples, and the
train/test split must leave at least one sample from each class on both sides.

### CLI

```powershell
fastq-classifier evaluate-classifier `
  --matrix-dir matrix_k13 `
  --out-dir evaluation_k13 `
  --label-column scientific_name `
  --test-size 0.25 `
  --seed 1
```

Outputs are written under `evaluation_k13`:

- `metrics.tsv`: sample counts and test-set metrics
- `predictions.tsv`: test-set labels and predictions
- `confusion_matrix.tsv`: confusion matrix counts
- `confusion_matrix.png`: confusion matrix plot

### Python API

```python
from fastq_classifier import (
    evaluate_classifier,
)

evaluation = evaluate_classifier(
    "matrix_k13",
    "evaluation_k13",
    "scientific_name",
)

print(evaluation.accuracy, evaluation.balanced_accuracy)
print(evaluation.confusion_matrix_path)
```
