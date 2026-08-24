# Building a machine-learning taxon classifier for genomic classification in malaria mosquitoes

`fastq-classifier` assigns paired-end whole-genome sequencing reads to one of five major
*Anopheles* groups. It works directly from FASTQ files, before read alignment and variant calling.
The predicted group can then guide the choice of reference genome and other genomic resources.

The classifier uses five labels:

| Label | Taxonomic interpretation |
|---|---|
| `gambiae_complex` | *An. gambiae*, *An. coluzzii*, and *An. arabiensis* pooled |
| `darlingi` | *An. darlingi* |
| `minimus` | *An. minimus* |
| `stephensi` | *An. stephensi* |
| `funestus` | *An. funestus* |

Species-level classification within the Gambiae complex is not supported.

## Method

For each sequencing run, the software downloads the first 25,000 read pairs and counts all
canonical 8-mers with KMC. The resulting 32,896 counts are converted to counts per million,
transformed with `log1p`, and normalized to unit L2 length. A class-balanced multinomial logistic
regression assigns the sample to one of the five groups.

## Installation

The software requires Python 3.11 or later and KMC 3. The `kmc` and `kmc_tools` executables must be
available on `PATH`.

```powershell
python -m pip install .
```

## Prepare a count matrix

The download command accepts a tab-separated ENA run report containing `run_accession` and
`fastq_ftp` columns. Each `fastq_ftp` value must contain the read 1 and read 2 URLs, in that order,
separated by a semicolon.

```text
run_accession	fastq_ftp
ERR000001	ftp.sra.ebi.ac.uk/.../ERR000001_1.fastq.gz;ftp.sra.ebi.ac.uk/.../ERR000001_2.fastq.gz
```

Run the three preparation commands in sequence:

```powershell
fastq-classifier download ena_runs.tsv work\fastq --jobs 4

fastq-classifier count-kmers `
  work\fastq\fastq_manifest.tsv `
  work\kmc `
  --jobs 4

fastq-classifier build-matrix `
  work\kmc\kmc_manifest.tsv `
  work\matrix `
  --jobs 4
```

The matrix directory contains the count matrix, the fixed k-mer vocabulary, and the run order.

## Train the classifier

Training requires a tab-separated file with these columns:

| Column | Purpose |
|---|---|
| `specimen_id` | Unique specimen identifier |
| `run_accession` | Run represented by the corresponding matrix row |
| `label` | One of the five classifier labels |
| `country` | Grouping field for the Gambiae complex |
| `source` | Grouping field for Darlingi |
| `location` | Grouping field for Minimus and Funestus |
| `year` | Additional grouping field for Funestus |
| `study_id` | Grouping field for Stephensi |

The file must contain every column. Grouping fields that do not apply to a sample may be empty.
Assign the grouped folds and train the classifier:

```powershell
fastq-classifier assign-folds `
  work\matrix `
  development_samples.tsv `
  work\folds

fastq-classifier train `
  work\matrix `
  work\folds\folds.tsv `
  work\classifier
```

## Classify new samples

Prepare a separate count matrix for the new runs, then apply the classifier directory produced by
the training command:

```powershell
fastq-classifier predict `
  work\classifier `
  new_samples\matrix `
  predictions.tsv
```

`predictions.tsv` contains the run accession, the predicted label, and the probability assigned to
each supported group. These probabilities compare only the five trained classes. The software does
not identify samples from unsupported taxa.

## Development results

Across four grouped development folds containing 4,401 samples, the five-class classifier reached
0.9746 balanced accuracy and 0.9772 macro-F1. Minimus and Funestus had no errors in these folds,
although both had fewer samples and narrower geographic coverage than the Gambiae complex. The
reserved five-class test set has not been evaluated.

The [project report](PROJECT_REPORT.md) describes the study design, results, and limitations.
