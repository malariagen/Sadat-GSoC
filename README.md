# fastq-classifier

`fastq-classifier` uses a fixed number of read pairs from the beginning of the two FASTQ files for a
paired-end whole-genome sequencing sample. The k-mer counts from these read pairs are used to
classify the sample as one of five *Anopheles* groups before read alignment or variant calling, and
the predicted group can then guide the choice of reference genome for alignment.

For every sample, the classifier reports one of the following five labels:

| Label | Taxonomic group |
|---|---|
| `gambiae_complex` | *An. gambiae*, *An. coluzzii*, and *An. arabiensis* pooled |
| `darlingi` | *An. darlingi* |
| `minimus` | *An. minimus* |
| `stephensi` | *An. stephensi* |
| `funestus` | *An. funestus* |

`gambiae_complex` contains *An. gambiae*, *An. coluzzii*, and *An. arabiensis*, which were pooled
when the classifier was trained and are therefore not reported separately. The classifier has no
rejection class for samples from other taxa; their k-mer counts are still evaluated against the five
classes and the label with the highest probability is returned.

## Pipeline overview

There are three components in the pipeline:

1. Dataset preparation uses `download`, `count-kmers`, and `build-matrix` to build the count matrix
   required for training or prediction. When the reads are obtained from ENA, `download` reads the
   ENA run report and downloads the requested number of read pairs for each sequencing run; the
   FASTQ paths and read-pair count are then recorded in `fastq_manifest.tsv`. If the paired FASTQ
   files are already available locally, the same manifest can be prepared directly with
   `run_accession`, `read1_path`, `read2_path`, and `read_pairs`. `count-kmers` runs KMC on the two
   FASTQ files for each sequencing run and records the KMC database path in `kmc_manifest.tsv`. The
   databases listed in `kmc_manifest.tsv` are used by `build-matrix` to write `counts.npy`,
   `kmers.txt`, `runs.tsv`, and `matrix.json`.

2. Classifier training uses `assign-folds` and `train`. Before the classifier is fitted,
   `assign-folds` divides the development samples into four folds, using `country` for
   `gambiae_complex`, `source` for `darlingi`, `location` for `minimus`, `location` and `year` for
   `funestus`, and `study_id` for `stephensi`. All samples belonging to one blocking group are kept
   in the same fold, which avoids using that blocking group for both fitting and validation in the
   same cross-validation split. `train` fits one class-balanced multinomial logistic regression for
   each candidate value of `C` and each held-out fold. After `C` has been selected, `train` fits the
   classifier again on all development samples and writes `model.npz`, `model.json`, `kmers.txt`,
   `development_metrics.json`, `oof_predictions.tsv`, and `domain_metrics.tsv`.

3. `predict` uses a fitted classifier to assign one of the five labels to each row of a count matrix.
   The read-pair count for the classifier is recorded in `model.json`, while the read-pair count for
   the matrix is recorded in `matrix.json`, and the canonical k-mer vocabulary is given by
   `kmers.txt` in the two directories. The read-pair count and k-mer vocabulary of the matrix are
   compared with those of the classifier before classification, and `predict` stops if either is
   different. Otherwise, the count rows are normalized and classified in batches, and the output
   TSV gives `row_index`, `run_accession`, `predicted_label`, and the probability for each of the five
   labels.

## Method

By default, `download` takes the first 25,000 read pairs from each sequencing run, and `count-kmers`
counts canonical 8-mers in these reads. A k-mer and its reverse complement are represented by the
same feature, which gives 32,896 features when `k=8`. The number of read pairs can be changed with
`download --read-pairs`; for FASTQ files supplied locally, this number is entered in the `read_pairs`
column of `fastq_manifest.tsv`. For the k-mer length, `count-kmers --k` accepts values from 4 to 8.

Before the classifier is fitted, the k-mer counts for each sample are converted to counts per
million, transformed with `log1p`, and normalized to unit L2 length. For each candidate value of
`C`, a class-balanced multinomial logistic regression is fitted on three of the grouped development
folds and evaluated on the remaining fold, with each fold used once for validation. After `C` has
been selected from these results, one classifier is fitted on all development samples. The same
normalization is applied to a count matrix during prediction.

## Installation

`fastq-classifier` requires Python 3.11 or later. K-mer counting and count-matrix construction use
KMC 3, so the `kmc` and `kmc_tools` programs must both be available on `PATH`.

From the repository root, install the package with:

```console
python -m pip install .
```

After installation, run `fastq-classifier` from the command line or import the Python API from
`fastq_classifier`.

## Pipeline stages

### 1. Download paired FASTQ prefixes

`download` reads a tab-separated ENA run report containing the `run_accession` and `fastq_ftp`
columns. For each sequencing run, `fastq_ftp` must contain two `.fastq.gz` URLs separated by a
semicolon, with the read 1 URL first and the read 2 URL second.

```text
run_accession	fastq_ftp
ERR000001	ftp.sra.ebi.ac.uk/.../ERR000001_1.fastq.gz;ftp.sra.ebi.ac.uk/.../ERR000001_2.fastq.gz
```

By default, `download` takes the first 25,000 read pairs from each sequencing run, and four runs are
downloaded in parallel. During the download, each FASTQ record is checked before it is written to
the compressed FASTQ file. The two compressed FASTQ files are then read together, and for every
read pair the read identifier from read 1 is compared with the read identifier from read 2.

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

For each sequencing run, `download` places the two compressed FASTQ files in a directory named after
the run accession, while `fastq_manifest.tsv` is written in the download directory:

```text
work/fastq/
  fastq_manifest.tsv
  ERR000001/
    ERR000001_1.fastq.gz
    ERR000001_2.fastq.gz
```

For each run, `fastq_manifest.tsv` records the run accession, the absolute path of each FASTQ file,
and the number of read pairs:

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

`count-kmers` creates one directory for each sequencing run, in which the KMC database is stored
together with `run.json` and `stats.json`:

```text
work/kmc/
  kmc_manifest.tsv
  ERR000001/
    ERR000001.kmc_pre
    ERR000001.kmc_suf
    run.json
    stats.json
```

`run.json` records the FASTQ paths, k-mer length, read-pair count, KMC version, and the settings used
to create the database. The count summary returned by KMC is stored in `stats.json`. After all runs
have been counted, their database paths and count statistics are written to `kmc_manifest.tsv` in
the same order as in `fastq_manifest.tsv`.

### 3. Build a count matrix

`build-matrix` reads the KMC databases in the order listed in `kmc_manifest.tsv`, and for every
database the k-mer counts from that sequencing run are written to one row of a dense `uint32` NumPy
array. The array is stored in `.npy` format, so it can be memory-mapped without loading the complete
array into memory.
Before the matrix is built, the k-mer length, read-pair count, and KMC version must be the same for
every database in the manifest.

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

When a classifier is being trained, `assign-folds` divides the development samples into four grouped
folds, and these folds are used by `train` when the candidate values of `C` are evaluated. Fold
assignment is not required when a fitted classifier is used for prediction. In that case, the new
samples are prepared as a count matrix and passed to `predict` in stage 6.

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

All columns shown in the table must be present, and every row must have `specimen_id`,
`run_accession`, and `label`. Among the grouping fields, only the field or fields used for that row's
label require values; the grouping fields for the other labels may be left empty. The table must
contain development samples only and must not include a `split` column.

`assign-folds` first matches the development-sample table to the count matrix by run accession.
Samples from the same blocking group remain in one fold, which prevents a blocking group from being
used for both fitting and validation in the same cross-validation split. For each label, the larger
blocking groups are assigned first, and each group is placed in the fold that currently has the
fewest samples of that label. At least four distinct blocking groups are required for every label,
because each of the four folds must contain that label.

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

`folds.tsv` follows the matrix row order and records `row_index`, `specimen_id`, `run_accession`,
`label`, `blocking_group`, and the assigned `fold` for each row.

### 5. Train the classifier

`train` reads the count-matrix directory together with `folds.tsv`. Each count-matrix row must have
one row in `folds.tsv` with the same `row_index` and `run_accession`, and the fold rows must appear in
the matrix order.

`train` evaluates `C` values of 0.01, 0.1, 1, and 10. For each value of `C`, a class-balanced
multinomial logistic regression is fitted on three folds and used to predict the fourth; this is
repeated until each fold has been held out once. Balanced accuracy is calculated from the combined
out-of-fold predictions and is used to select `C`. If balanced accuracy is equal, higher macro-F1 is
preferred, followed by lower log loss and then the smaller value of `C`. The selected value of `C`
is then used to fit one classifier on all development rows.

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

`predict` reads the fitted classifier and count-matrix directories. Before it classifies the count
rows, the read-pair count in `matrix.json` is compared with the read-pair count in `model.json`, and
the `kmers.txt` files from the two directories are also compared. If the read-pair count or k-mer
vocabulary differs, prediction stops without writing any rows.

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

`predict` reads 64 count-matrix rows at a time by default, although another batch size can be set
with `--batch-size`. For each row, the output TSV records `row_index`, `run_accession`,
`predicted_label`, and the probability assigned to each of the five labels. These five probabilities
sum to one. A sample from another taxon still receives probabilities for the five classifier labels,
because the classifier has no rejection class.

## Classify an in-memory count array

If you already have the k-mer counts in a NumPy array, pass the array to `predict_kmer_counts`. The
array must be two-dimensional and have dtype `uint32`, with one sample in each row and the columns
in the order given by the classifier's `kmers.txt`:

```python
from fastq_classifier import predict_kmer_counts

labels, probabilities = predict_kmer_counts(
    "work/classifier",
    count_rows,
    batch_size=64,
)
```

`predict_kmer_counts` returns one label for each input row and a `float64` array containing the five
probabilities, and the columns in this array follow the order of `classes` in `model.json`. When
generating `count_rows`, use the number of read pairs recorded as `features.read_pairs` in
`model.json` for every sample.

## Reuse existing output

`download` and `count-kmers` can be run again with an existing output directory. A downloaded FASTQ
pair is reused after the two FASTQ files and their read-pair count have been checked. Before a KMC
database is reused, `count-kmers` checks the database files and statistics and compares `run.json`
with the current FASTQ paths, k-mer length, read-pair count, KMC version, and KMC command settings.
If a completed run does not match, the command stops; use a new output directory for changed FASTQ
files or settings.

`assign-folds` replaces `folds.tsv` in its output directory. `build-matrix` and `train` require new
output directories, and `predict` requires a new output file.

## Development results and limitations

Across four grouped development folds containing 4,401 samples, the classifier reached 0.9746
balanced accuracy and 0.9772 macro-F1. The `minimus` and `funestus` samples were all classified
correctly in these folds, although both groups had fewer samples and narrower geographic coverage
than the Gambiae complex.


The [project report](PROJECT_REPORT.md) describes the study design, development results, and
limitations.
