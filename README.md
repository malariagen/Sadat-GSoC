# fastq-classifier

`fastq-classifier` assigns paired-end whole-genome sequencing reads to one of five major
*Anopheles* groups from a short FASTQ prefix, before read alignment or variant calling. The
predicted group can guide the choice of reference genome for later analysis.

The model uses five labels:

| Label | Taxonomic group |
|---|---|
| `gambiae_complex` | *An. gambiae*, *An. coluzzii*, and *An. arabiensis* pooled |
| `darlingi` | *An. darlingi* |
| `minimus` | *An. minimus* |
| `stephensi` | *An. stephensi* |
| `funestus` | *An. funestus* |

The classifier treats *An. gambiae*, *An. coluzzii*, and *An. arabiensis* as a single class. Samples
from other taxa also receive one of the five labels because there is no rejection class.

## Pipeline overview

There are three components in the pipeline:

1. Dataset preparation (`download`, `count-kmers`, and `build-matrix`) produces the count matrix
   used for training and prediction. With an ENA run report, `download` retrieves the requested read
   pairs and writes a FASTQ manifest. If the paired FASTQ files are already on disk, write the
   manifest yourself and begin with `count-kmers`. `count-kmers` creates one KMC database per run,
   and `build-matrix` combines the databases into a directory containing the count matrix, k-mer
   vocabulary, run order, and read-pair count.

2. Classifier training (`assign-folds` and `train`) fits a classifier from labeled development
   samples. `assign-folds` uses the sample metadata to keep related samples in the same fold.
   `train` evaluates the candidate values of `C`, fits the selected model on all development
   samples, and writes the model, out-of-fold predictions, and development metrics to a classifier
   directory.

3. Prediction (`predict`) applies a fitted classifier to new samples. Prepare the new samples as a
   count matrix using the same number of read pairs and the same k-mer vocabulary used for
   training. `predict` writes the run accession, predicted label, and five class probabilities to a
   TSV file.

## Method

By default, `download` takes the first 25,000 read pairs from each sequencing run, and `count-kmers`
counts canonical 8-mers. A k-mer and its reverse complement are treated as the same feature, giving
32,896 features when `k=8`. Use `download --read-pairs` to change the sampling depth; for local
FASTQ files, record the chosen value in the manifest. `count-kmers --k` accepts k-mer lengths from
4 to 8.

Before training, the pipeline converts the counts to counts per million, applies `log1p`, and
normalizes each sample to unit L2 length. Training selects the value of `C` across four grouped
development folds and fits a class-balanced multinomial logistic regression. For prediction, keep
the read-pair count and k-mer vocabulary the same as in training.

## Installation

The package requires Python 3.11 or later and KMC 3, with both `kmc` and `kmc_tools` available on
`PATH`.

Install the package from the repository root:

```console
python -m pip install .
```

The command installs the `fastq-classifier` program and the `fastq_classifier` Python package.

## Pipeline stages

### 1. Download paired FASTQ prefixes

To download reads, provide a tab-separated ENA run report with `run_accession` and `fastq_ftp`
columns. Each `fastq_ftp` value must contain two `.fastq.gz` URLs separated by a semicolon, with
read 1 first and read 2 second.

```text
run_accession	fastq_ftp
ERR000001	ftp.sra.ebi.ac.uk/.../ERR000001_1.fastq.gz;ftp.sra.ebi.ac.uk/.../ERR000001_2.fastq.gz
```

By default, `download` saves the first 25,000 read pairs from each run and works on four runs at a
time. Before writing the compressed FASTQ files, it checks the record structure and confirms that
the read identifiers agree between mates.

#### CLI

```console
fastq-classifier download ena_runs.tsv work/fastq --read-pairs 25000 --jobs 4
```

#### Python

```python
from fastq_classifier import download_read_pairs

fastq_manifest = download_read_pairs(
    "ena_runs.tsv",
    "work/fastq",
    read_pairs=25_000,
    jobs=4,
)
```

The output directory looks like this:

```text
work/fastq/
  fastq_manifest.tsv
  ERR000001/
    ERR000001_1.fastq.gz
    ERR000001_2.fastq.gz
```

`fastq_manifest.tsv` contains one row per run:

| Column | Contents |
|---|---|
| `run_accession` | ENA run accession |
| `read1_path` | Absolute path to the read 1 FASTQ file |
| `read2_path` | Absolute path to the read 2 FASTQ file |
| `read_pairs` | Number of read pairs, equal to the number of records in each mate |

### 2. Count canonical k-mers

If you already have paired FASTQ files, start here by listing them in a tab-separated manifest with
`run_accession`, `read1_path`, `read2_path`, and `read_pairs` columns. Both paths must be absolute.
The `read_pairs` value gives the number of records in each mate, and it must be the same for every
run in the manifest.

`count-kmers` runs KMC once for each run, using both mates. The default k-mer length is 8, although
you can set `k` to any integer from 4 through 8. Four KMC processes run at a time by default, each
using one thread and 2 GB of memory.

#### CLI

```console
fastq-classifier count-kmers work/fastq/fastq_manifest.tsv work/kmc --k 8 --jobs 4
```

#### Python

```python
from fastq_classifier import count_kmers

kmc_manifest = count_kmers(
    "work/fastq/fastq_manifest.tsv",
    "work/kmc",
    k=8,
    jobs=4,
)
```

Each run has its own KMC database and two metadata files:

```text
work/kmc/
  kmc_manifest.tsv
  ERR000001/
    ERR000001.kmc_pre
    ERR000001.kmc_suf
    run.json
    stats.json
```

`run.json` records the FASTQ paths, k-mer length, read-pair count, KMC version, and KMC settings.
`stats.json` contains the count summary reported by KMC. `kmc_manifest.tsv` keeps the run order from
the FASTQ manifest and gives the database path and count statistics for each run.

### 3. Build a count matrix

`build-matrix` converts each database in `kmc_manifest.tsv` into one row of a dense `uint32` NumPy
array. The array is stored in `.npy` format and can be memory-mapped instead of loaded in full. All
databases in the manifest must have the same k-mer length, read-pair count, and KMC version.

#### CLI

```console
fastq-classifier build-matrix work/kmc/kmc_manifest.tsv work/matrix --jobs 4
```

#### Python

```python
from fastq_classifier import build_count_matrix

counts_path = build_count_matrix(
    "work/kmc/kmc_manifest.tsv",
    "work/matrix",
    jobs=4,
)
```

The matrix directory contains:

```text
work/matrix/
  counts.npy
  kmers.txt
  matrix.json
  runs.tsv
```

| File | Contents |
|---|---|
| `counts.npy` | `uint32` count matrix with one row per sequencing run |
| `kmers.txt` | Canonical vocabulary in matrix-column order |
| `matrix.json` | Read-pair count shared by the matrix rows |
| `runs.tsv` | Row index and run accession for each matrix row |

### 4. Assign development folds

`assign-folds` prepares the grouped cross-validation folds used for classifier training. If you
already have a fitted classifier, build a count matrix for the new samples and skip to stage 6.

`assign-folds` takes a count-matrix directory and a tab-separated development-sample table with
the following columns:

| Column | Contents |
|---|---|
| `specimen_id` | Unique specimen identifier |
| `run_accession` | Run accession present in the count matrix |
| `label` | One of the five classifier labels |
| `country` | Grouping field for `gambiae_complex` |
| `source` | Grouping field for `darlingi` |
| `location` | Grouping field for `minimus` and `funestus` |
| `year` | Additional grouping field for `funestus` |
| `study_id` | Grouping field for `stephensi` |

All columns are required, but only the grouping fields used for a sample's label need values; the
remaining grouping fields may be empty. Use a development-only table without a `split` column.

`assign-folds` matches the table to the matrix by run accession, keeps each blocking group in one
fold, and balances the number of samples per label across the folds. Each label needs at least four
distinct blocking groups to populate all four folds.

#### CLI

```console
fastq-classifier assign-folds work/matrix development_samples.tsv work/folds
```

#### Python

```python
from fastq_classifier import assign_development_folds

folds_path = assign_development_folds(
    "work/matrix",
    "development_samples.tsv",
    "work/folds",
)
```

`folds.tsv` follows the matrix row order and records the specimen, label, blocking group, and
assigned fold for each row.

### 5. Train the classifier

Training uses the count-matrix directory together with `folds.tsv`, which must have one entry for
every development matrix row in the same order.

`train` fits grouped out-of-fold models for `C` values of 0.01, 0.1, 1, and 10. It selects the value
with the highest balanced accuracy, using higher macro-F1, lower log loss, and then the smaller
value of `C` to break ties. The selected model is then refitted using all development rows.

#### CLI

```console
fastq-classifier train work/matrix work/folds/folds.tsv work/classifier
```

#### Python

```python
from fastq_classifier import train_classifier

model_path = train_classifier(
    "work/matrix",
    "work/folds/folds.tsv",
    "work/classifier",
)
```

The classifier directory contains:

```text
work/classifier/
  model.npz
  model.json
  kmers.txt
  development_metrics.json
  oof_predictions.tsv
  domain_metrics.tsv
```

| File | Contents |
|---|---|
| `model.npz` | Logistic-regression coefficients and intercepts |
| `model.json` | Classes, features, normalization, selected `C`, and software versions |
| `kmers.txt` | K-mer vocabulary expected by the model |
| `development_metrics.json` | Metrics for each candidate value of `C` |
| `oof_predictions.tsv` | Out-of-fold label and probability estimates for each development row |
| `domain_metrics.tsv` | Accuracy, mean true-class probability, and log loss by blocking group |

### 6. Classify a count matrix

`predict` reads a fitted classifier directory and a count-matrix directory. It checks the matrix
against the read-pair count and k-mer vocabulary stored with the model before classifying any rows.

#### CLI

```console
fastq-classifier predict work/classifier unseen/matrix predictions.tsv --batch-size 64
```

#### Python

```python
from fastq_classifier import classify_count_matrix

predictions_path = classify_count_matrix(
    "work/classifier",
    "unseen/matrix",
    "predictions.tsv",
    batch_size=64,
)
```

`--batch-size` sets the number of matrix rows processed at once and defaults to 64. `predict` writes
`row_index`, `run_accession`, `predicted_label`, and one probability for each label to the output
TSV. The five probabilities in each row sum to one even when a sample belongs to another taxon,
because the classifier has no rejection class.

## Classify an in-memory count array

If the k-mer counts are already in memory, pass them to `predict_kmer_counts` as a two-dimensional
`uint32` array. Each row represents one sample, and the columns must follow the vocabulary in the
classifier's `kmers.txt`:

```python
from fastq_classifier import predict_kmer_counts

labels, probabilities = predict_kmer_counts(
    "work/classifier",
    count_rows,
    batch_size=64,
)
```

The function returns one label per input row and a `float64` probability array whose columns follow
the class order in `model.json`. When preparing `count_rows`, use the `features.read_pairs` value
from the same file to match the sampling depth used for training.

## Reuse existing output

`download` and `count-kmers` can resume in an existing output directory. Before reusing a completed
run, each command checks its files and settings. If the saved run was created from different inputs
or settings, use a new output directory.

`assign-folds` replaces `folds.tsv` in its output directory. `build-matrix` and `train` require new
output directories, and `predict` requires a new output file.

## Development results and limitations

Across four grouped development folds containing 4,401 samples, the classifier reached 0.9746
balanced accuracy and 0.9772 macro-F1. The `minimus` and `funestus` samples were all classified
correctly in these folds, although both groups had fewer samples and narrower geographic coverage
than the Gambiae complex.


The [project report](PROJECT_REPORT.md) describes the study design, development results, and
limitations.
