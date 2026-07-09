"""Public Python API for FASTQ classifier tools."""

from __future__ import annotations

from fastq_classifier.build_matrix import (
    KmerMatrix,
    MatrixBuildError,
    build_kmer_matrix,
)
from fastq_classifier.evaluate_classifier import (
    ClassifierError,
    ClassifierEvaluation,
    evaluate_classifier,
)
from fastq_classifier.extract_features import (
    KmerDatabase,
    KmerExtraction,
    KmerExtractionError,
    KmerStats,
    extract_kmer_features,
)
from fastq_classifier.fetch_first_n import (
    DownloadedRun,
    DownloadReport,
    FetchError,
    fetch_first_n,
)
from fastq_classifier.utils import InputError, RejectedRow

__all__ = [
    "ClassifierError",
    "ClassifierEvaluation",
    "DownloadReport",
    "DownloadedRun",
    "FetchError",
    "InputError",
    "KmerDatabase",
    "KmerExtraction",
    "KmerExtractionError",
    "KmerMatrix",
    "KmerStats",
    "MatrixBuildError",
    "RejectedRow",
    "__version__",
    "build_kmer_matrix",
    "evaluate_classifier",
    "extract_kmer_features",
    "fetch_first_n",
]

__version__ = "0.1.0"
