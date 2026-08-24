"""Mosquito species classification from paired-end FASTQ reads."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastq_classifier.classifier import classify_count_matrix, predict_kmer_counts
    from fastq_classifier.download import download_read_pairs
    from fastq_classifier.folds import assign_development_folds
    from fastq_classifier.kmc import count_kmers
    from fastq_classifier.matrix import build_count_matrix
    from fastq_classifier.species import SPECIES_LABELS
    from fastq_classifier.training import train_classifier

__all__ = [
    "SPECIES_LABELS",
    "assign_development_folds",
    "build_count_matrix",
    "classify_count_matrix",
    "count_kmers",
    "download_read_pairs",
    "predict_kmer_counts",
    "train_classifier",
]

_MODULE_BY_EXPORT = {
    "SPECIES_LABELS": "fastq_classifier.species",
    "assign_development_folds": "fastq_classifier.folds",
    "build_count_matrix": "fastq_classifier.matrix",
    "classify_count_matrix": "fastq_classifier.classifier",
    "count_kmers": "fastq_classifier.kmc",
    "download_read_pairs": "fastq_classifier.download",
    "predict_kmer_counts": "fastq_classifier.classifier",
    "train_classifier": "fastq_classifier.training",
}


def __getattr__(name: str) -> object:
    try:
        module_name = _MODULE_BY_EXPORT[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    exported_symbol = getattr(import_module(module_name), name)
    globals()[name] = exported_symbol
    return exported_symbol
