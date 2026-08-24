# Building a machine-learning taxon classifier for genomic classification in malaria mosquitoes

- Contributor: Sadat Bashir
- Mentors: Jon Brenas, Chris Clarkson, Tristan Dennis
- Project: `fastq-classifier`

## Abstract

Many *Anopheles* species look similar and cannot be identified reliably from morphology alone.
Researchers need to know the taxonomic group before choosing the reference genome and related
genomic resources for a sample. I developed a classifier that assigns paired-end whole-genome
sequencing FASTQs to five major groups: Gambiae complex, Darlingi, Minimus, Stephensi, and
Funestus. It uses exact canonical 8-mer counts from a fixed number of read pairs and does not
require read alignment or variant calling.

The study began by comparing k-mer lengths on 987 mosquitoes. Later analyses evaluated multiclass
models while keeping samples from the same country or collection group in the same fold. The final
five-class classifier uses the first 25,000 read pairs from each sample. Across four development
folds containing 4,401 specimens, it reached 0.9746 balanced accuracy and 0.9772 macro-F1.

## Background and project objective

Accurate identification of *Anopheles* mosquitoes is important for malaria surveillance and
control. Closely related species may be difficult to separate by morphology, so genomic data are
often needed to identify them. A wrong assignment can distort epidemiological analyses, lead to
poorly targeted control decisions, and cause reads to be aligned against an unsuitable reference
genome. The resulting errors can then affect variant calls and population-genetic analyses.

Current genomic identification methods commonly use genotype calls produced after read alignment
and variant calling. They can classify samples accurately, but they require the user to choose a
reference genome before the taxon has been established. They also add a substantial processing
step, and changes to reference panels or taxonomic definitions may require the methods to be
revised. Researchers often receive raw FASTQ reads before alignments or variant calls are
available. A classifier that operates on these reads can provide an initial taxonomic assignment
without first running a complete variant-calling workflow.

The objective was to build a machine-learning classifier that works directly from raw sequencing
reads and identifies the major taxonomic group to which a sample belongs. The resulting assignment
can guide the choice of reference genome and other genomic resources. The classifier does not
separate species within the Gambiae complex. I examined that finer classification experimentally,
including the identification of *An. coluzzii*, but the available data did not support its use in
the final classifier.

The classifier uses five output labels:

| Label | Taxonomic interpretation |
|---|---|
| `gambiae_complex` | *An. gambiae*, *An. coluzzii*, and *An. arabiensis* pooled |
| `darlingi` | *An. darlingi* |
| `minimus` | *An. minimus* |
| `stephensi` | *An. stephensi* |
| `funestus` | *An. funestus* |

The available samples from species within the Gambiae complex were too unevenly distributed among
countries to train a species-level classifier that could be expected to generalize across
geographic regions. I therefore pooled these species into a single class.

## Initial k-mer study

The first analysis used 987 samples from Ag1000G and partner collections: 395 *coluzzii*, 395
*gambiae*, and 197 *funestus*. These samples came from 21 sample sets in 15 countries. I fetched
the first 100,000 paired reads for each sample from the European Nucleotide Archive and counted
canonical k-mers with KMC for `k=5-8`. Logistic regression was fitted with balanced class weights,
and balanced accuracy was used as the main selection metric.

The three-class analysis used both stratified five-fold cross-validation and
leave-one-sample-set-out validation. The binary *gambiae*/*coluzzii* analysis used grouped splits.

| k | Three-class, five-fold | Three-class, leave-one-set-out | Binary, grouped |
|---:|---:|---:|---:|
| 5 | 0.968 | 0.913 | 0.887 |
| 6 | 0.989 | 0.960 | 0.935 |
| 7 | 0.992 | 0.979 | 0.966 |
| 8 | 0.997 | 0.987 | 0.971 |

At `k=8`, the five-fold three-class analysis made four errors among 987 samples. Three *coluzzii*
samples were classified as *gambiae*, one *gambiae* sample was classified as *coluzzii*, and all
*funestus* samples retained the correct label. As a check for collection-specific batch signal, I
also trained a classifier to predict the sample set from the same k-mer counts. Its accuracy fell
from 0.841 at `k=5` to 0.773 at `k=8`, while taxon accuracy increased.

The binary analysis was extended to `k=5-9` at 10,000, 25,000, and 100,000 read pairs. With
`k=9` and 100,000 pairs, balanced accuracy was 1.000 under a random split. It was also 1.000 when
each country was held out in turn and the scores were averaged. When each sampling group was held
out in turn, mean balanced accuracy was 0.996. The later multiclass work retained `k=8` because it
used fewer features than `k=9` and maintained high grouped accuracy with 25,000 read pairs.

## Four-class study

The next analysis pooled *gambiae*, *coluzzii*, and *arabiensis* into the Gambiae complex, then
added Stephensi, Darlingi, and Minimus. The complete canonical 8-mer vocabulary contains 32,896
features. Counts were taken from the first 25,000 read pairs for each sample.

A random split can place related samples, or samples from the same collection batch, in both the
training and validation sets. This can inflate performance because the validation data resemble
samples already seen during training. I therefore kept each sampling group entirely within one
side of a split, following the recommendations of Roberts et al. (2017). The grouping variable
depended on the taxon: country for the Gambiae complex, study for Stephensi, source batch for
Darlingi, and collection location for Minimus.

The four-class cohort contained 5,709 labeled samples. Of these, 3,028 were assigned to training,
1,196 to validation, and 1,485 to the test pool. The Gambiae-complex training set contained 500
samples from each of *gambiae*, *coluzzii*, and *arabiensis*. Another 368 samples from taxa outside
the four classes and 307 samples without taxon labels were excluded from model fitting and kept for
later analyses of prediction confidence.

I compared logistic regression with linear and radial-basis-function support vector machines on
the blocked validation set. Logistic regression performed best. It made 6 errors among 1,196
validation samples, with 0.9983 balanced accuracy and 0.9931 macro-F1. Before extending the model
to five classes, I evaluated the three methods once on a designated subset of 299 test samples.
Logistic regression made no errors; each support vector machine made one. The remaining 1,186
four-class test samples were reserved.

## Five-class classifier

The five-class development set combined the earlier training and validation samples and added 177
Funestus specimens. It contained 4,401 samples in total.

| Label | Samples | Fold grouping |
|---|---:|---|
| Gambiae complex | 2,384 | Country |
| Darlingi | 863 | Source batch |
| Minimus | 255 | Collection location |
| Stephensi | 722 | Study |
| Funestus | 177 | Location and year |
| Total | 4,401 | |

Four development folds were constructed. All samples from a given country, study, source batch,
collection location, or location-year group were assigned to the same fold. The folds differed in
size because the sampling groups differed in size.

For each sample, KMC counted all canonical 8-mers in the first 25,000 read pairs. Counts were
converted to counts per million, transformed with `log1p`, and normalized to unit L2 length. A
class-balanced multinomial logistic regression was then fitted to all 32,896 features. There was
no feature selection, dimensionality reduction, probability calibration, or outlier detector in
this model.

Logistic regression had already performed best in the four-class comparison, so the five-class
training compared only the inverse regularization strengths `C = {0.01, 0.1, 1, 10}`. For each
value of `C`, I generated predictions for the held-out fold and compared balanced accuracy.
Macro-F1 and log loss were secondary selection criteria.

| C | Balanced accuracy | Macro-F1 | Log loss |
|---:|---:|---:|---:|
| 0.01 | 0.9469 | 0.9248 | 1.4724 |
| 0.1 | 0.9456 | 0.9274 | 1.0544 |
| 1 | 0.9495 | 0.9348 | 0.4688 |
| 10 | 0.9746 | 0.9772 | 0.1975 |

`C=10` was selected, and the final classifier was fitted on all 4,401 development samples.

## Five-class results

For each fold, I trained the model on the other three folds and predicted the samples in the
held-out fold. Each development sample therefore received one prediction from a model that had
not seen any sample from the same sampling group. Across these predictions, accuracy was 0.9711,
balanced accuracy was 0.9746, and macro-F1 was 0.9772. Log loss was 0.1975, the multiclass Brier
score was 0.0779, and the 10-bin expected calibration error was 0.0824.

| Class | Precision | Recall | F1 | Samples |
|---|---:|---:|---:|---:|
| Gambiae complex | 0.9735 | 0.9845 | 0.9789 | 2,384 |
| Darlingi | 0.9317 | 0.9328 | 0.9323 | 863 |
| Minimus | 1.0000 | 1.0000 | 1.0000 | 255 |
| Stephensi | 0.9942 | 0.9557 | 0.9746 | 722 |
| Funestus | 1.0000 | 1.0000 | 1.0000 | 177 |

The model made 127 errors. Most were confusions between Gambiae complex and Darlingi: 36
Gambiae-complex samples were classified as Darlingi, and 55 Darlingi samples were classified as
Gambiae complex. Among the other errors, 9 Stephensi samples were classified as Gambiae complex,
23 Stephensi samples as Darlingi, 3 Darlingi samples as Stephensi, and 1 Gambiae-complex sample as
Stephensi. Minimus and Funestus had no errors in these folds. Both classes had fewer samples and
narrower geographic coverage, so their performance on new collections remains uncertain.

These results measure performance across the four grouped development folds. The reserved
five-class test set was not used. The 299-sample interim test belonged to the earlier four-class
study.

## Generalization studies

### Unfamiliar taxa

I tested an Isolation Forest using 30 samples from three taxa absent from four-class training:
*longipalpis*, *parensis*, and *vaneedeni*. It flagged 29 of the 30 unfamiliar samples as outside
the training distribution and retained 1,172 of 1,196 known validation samples. Each unfamiliar
taxon contributed only 10 samples. That sample size was too small to set a reliable threshold for
flagging unsupported taxa and to estimate how well it would work. I therefore excluded the
Isolation Forest from the five-class classifier.

### Darlingi geography

The Darlingi sample-count study varied the number of training samples per country from 4 to 64 and
held out one country at a time. With 64 samples per training country, none of the 47 Belize samples
were identified. Training on all 813 non-Belize samples increased the number correctly classified
to 24 of 47. Broader geographic coverage was therefore more important than adding samples from
countries already represented in training.

### Species within the Gambiae complex

An experimental support vector machine separated *arabiensis*, *coluzzii*, and *gambiae* with
0.816 validation accuracy, 0.748 balanced accuracy, and 0.785 macro-F1. Recall was 0.532 for
*arabiensis*, 0.713 for *coluzzii*, and 1.000 for *gambiae*. Of the 500 *arabiensis* training
samples, 498 came from Tanzania. The result could therefore depend on geography as well as species,
and the classifier was not included in the software.

## Software

The project produced a Python package and command-line program that retrieves paired FASTQ reads
from the European Nucleotide Archive, checks that the read pairs are complete, counts canonical
k-mers with KMC, constructs the input matrix, assigns grouped development folds, trains the
five-class classifier, and predicts new samples. The matrix always contains the same 32,896
canonical 8-mers in a fixed order, including k-mers that are absent from a sample. Its columns
therefore do not depend on the samples or labels used for training.

Users can run the training and prediction workflows from the command line. Prediction output
contains one taxonomic assignment per sequencing run and the probability assigned to each of the
five supported groups.

## Limitations and remaining work

The reserved five-class test set should be evaluated once, with no further model selection
afterward.

All Darlingi samples came from one sequencing project, and all Minimus samples came from one
project. Their development folds separated source batches or collection locations within those
projects, but performance on an independent project remains unknown. The experiment with
unfamiliar taxa also needs more taxa and broader sampling before the classifier can reliably
reject samples outside its five supported groups.

The available data support classification of the pooled Gambiae complex. Species-level prediction
within that complex will require *arabiensis* samples from a wider geographic range and an
evaluation designed so that predictions cannot rely on country-specific signal.

## Acknowledgements

I thank Jon Brenas, Tristan Dennis, Anastasia Hernandez Koutoucheva, and Kate Rowlands for their
support and advice on the study design, evaluation, implementation, and interpretation of the
results. I also thank the researchers and sequencing projects that made the mosquito reads and
metadata available through Ag1000G, partner collections, and the European Nucleotide Archive.

## References

1. Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial,
   hierarchical, or phylogenetic structure. *Ecography* 40, 913-929. DOI:
   10.1111/ecog.02881.
2. Kokot, M., Dlugosz, M. and Deorowicz, S. (2017). KMC 3: counting and manipulating k-mer
   statistics. *Bioinformatics* 33, 2759-2761. DOI: 10.1093/bioinformatics/btx304.
3. MalariaGEN. Anopheles gambiae 1000 Genomes Project.
