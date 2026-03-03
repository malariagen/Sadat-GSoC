# Anopheles Taxon Classification Pipeline

> **Note:** These represent initial experiments conducted as part of developing a proposal for the GSoC 2025 "Building a machine-learning taxon classifier to inform genomic classification in malaria mosquitoes" project.

This repository will contain the pipeline for downloading, processing, and analyzing Anopheles mosquito genotype data. It will include both the core execution modules and accompanying notebook-based "reports" for visualization and presentation. 

## Pipeline Stages

### Stage 0: Sample Selection
This stage selects the specific mosquito samples for downstream processing.
* Reads source data containing sample URLs and labels.
* Selects samples based on specified species (e.g., `gambiae`, `coluzzii`) and requested counts.
* Implements selection via random sampling or by taking the first available records.
* Outputs a mapping file containing the target URLs and species labels.

### Stage 1: Genotype Extraction
This stage extracts Genotype (GT) arrays from remote Zarr stores to local storage.
* Reads target URLs from the sample mapping file.
* Extracts GT arrays from remote compressed files using HTTP range requests.
* Discovers and maintains the original contig hierarchy during extraction.
* Saves extracted arrays to local per-sample storage directories.

### Stage 2: Feature Extraction
> **Note:** The features currently calculated in this stage are naive and experimental. Specifically, the Runs of Homozygosity (ROH) metrics represent simple contiguous homozygous call counts rather than true ROH. All features require further review and input before the pipeline is finalized.

This stage processes the local data stores to calculate tabular features for downstream analysis.
* Uses JIT-compiled kernels to iterate over the extracted data for high performance.
* Calculates per-contig metrics including missing rates, heterozygosity, homozygosity (reference and alternate), and approximated pseudo-ROH counts.
* Outputs a wide-format dataset containing the compiled metrics.
* Generates an accompanying metadata file capturing run configurations and statistics.

### Stage 3: PCA Analysis (Extra)
*(In Revision)* Code to perform Principal Component Analysis (PCA) on the extracted features was drafted last year. It is currently being restructured and cleaned up before being integrated into this pipeline for presentation.

### Stage 4: Species Classification
*(In Revision)* The core machine learning classification module to predict Anopheles species and identify outliers was also completed last year. This module is also undergoing code cleanup and formatting before being added to this repository.