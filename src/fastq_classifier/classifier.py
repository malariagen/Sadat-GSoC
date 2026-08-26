"""Load the fitted classifier and predict compatible count matrices."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fastq_classifier.features import (
    COUNT_NORMALIZATION_METADATA,
    COUNTS_PER_MILLION,
    kmer_feature_metadata,
    read_canonical_kmers,
)
from fastq_classifier.matrix import read_matrix_run_accessions
from fastq_classifier.species import SPECIES_LABELS

DEFAULT_PREDICTION_BATCH_ROWS = 64


@dataclass(frozen=True, slots=True)
class _LogisticClassifier:
    coefficients: NDArray[np.float64]
    intercept: NDArray[np.float64]


def normalize_kmer_counts(count_rows: NDArray[np.uint32]) -> NDArray[np.float32]:
    normalized_counts = np.asarray(count_rows, dtype=np.float32).copy()
    kmer_totals = normalized_counts.sum(axis=1, dtype=np.float64)
    if np.any(kmer_totals <= 0):
        raise ValueError("Every sample must contain at least one counted k-mer")
    normalized_counts *= (COUNTS_PER_MILLION / kmer_totals).astype(np.float32)[:, None]
    np.log1p(normalized_counts, out=normalized_counts)
    normalized_counts /= np.linalg.norm(normalized_counts, axis=1)[:, None]
    return normalized_counts


def predict_kmer_counts(
    classifier_dir: str | Path,
    count_rows: NDArray[np.uint32],
    *,
    batch_size: int = DEFAULT_PREDICTION_BATCH_ROWS,
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Predict labels and probabilities from fixed-vocabulary count rows."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if count_rows.dtype != np.uint32:
        raise ValueError("Counts must have dtype uint32")
    if count_rows.ndim != 2:
        raise ValueError("Counts must be a two-dimensional array")
    classifier = _load_logistic_classifier(Path(classifier_dir))
    feature_count = classifier.coefficients.shape[1]
    if count_rows.shape[1] != feature_count:
        raise ValueError(f"Counts must have {feature_count} columns")
    probabilities = np.empty((len(count_rows), len(SPECIES_LABELS)), dtype=np.float64)
    predicted_species: list[str] = []
    for batch_start in range(0, len(count_rows), batch_size):
        batch_stop = min(batch_start + batch_size, len(count_rows))
        batch_predictions, probabilities[batch_start:batch_stop] = _predict_count_batch(
            classifier,
            count_rows[batch_start:batch_stop],
        )
        predicted_species.extend(batch_predictions)
    return tuple(predicted_species), probabilities


def classify_count_matrix(
    classifier_dir: str | Path,
    count_matrix_dir: str | Path,
    predictions_path: str | Path,
    *,
    batch_size: int = DEFAULT_PREDICTION_BATCH_ROWS,
) -> Path:
    """Predict every row of a compatible count matrix."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    classifier_path = Path(classifier_dir)
    matrix_path = Path(count_matrix_dir)
    prediction_path = Path(predictions_path)
    if prediction_path.exists():
        raise FileExistsError(f"Prediction output already exists: {prediction_path}")

    classifier = _load_logistic_classifier(classifier_path)
    if (classifier_path / "kmers.txt").read_bytes() != (matrix_path / "kmers.txt").read_bytes():
        raise ValueError("Model and count matrix use different k-mer vocabularies")
    run_accessions = read_matrix_run_accessions(matrix_path / "runs.tsv")
    count_rows = np.load(matrix_path / "counts.npy", mmap_mode="r", allow_pickle=False)
    if count_rows.dtype != np.uint32 or count_rows.shape != (
        len(run_accessions),
        classifier.coefficients.shape[1],
    ):
        raise ValueError(
            f"Count matrix {matrix_path / 'counts.npy'} must have shape "
            f"({len(run_accessions)}, {classifier.coefficients.shape[1]}) and dtype uint32"
        )

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pending_prediction_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            prefix=f".{prediction_path.name}.",
            suffix=".tmp",
            dir=prediction_path.parent,
            delete=False,
        ) as prediction_stream:
            pending_prediction_path = Path(prediction_stream.name)
            prediction_rows = csv.writer(prediction_stream, delimiter="\t", lineterminator="\n")
            prediction_rows.writerow(
                (
                    "row_index",
                    "run_accession",
                    "predicted_label",
                    *(f"probability_{species_label}" for species_label in SPECIES_LABELS),
                )
            )
            for batch_start in range(0, len(run_accessions), batch_size):
                batch_stop = min(batch_start + batch_size, len(run_accessions))
                predicted_species, probabilities = _predict_count_batch(
                    classifier,
                    count_rows[batch_start:batch_stop],
                )
                for batch_index, species_label in enumerate(predicted_species):
                    row_index = batch_start + batch_index
                    prediction_rows.writerow(
                        (
                            row_index,
                            run_accessions[row_index],
                            species_label,
                            *(f"{probability:.12g}" for probability in probabilities[batch_index]),
                        )
                    )
        pending_prediction_path.replace(prediction_path)
    finally:
        del count_rows
        if pending_prediction_path is not None:
            pending_prediction_path.unlink(missing_ok=True)
    return prediction_path


def _load_logistic_classifier(classifier_dir: Path) -> _LogisticClassifier:
    model_metadata = _read_model_metadata(classifier_dir / "model.json")
    try:
        species_labels = model_metadata["classes"]
    except KeyError as error:
        raise ValueError(
            f"Classifier {classifier_dir} is missing {error.args[0]} metadata"
        ) from error
    if species_labels != list(SPECIES_LABELS):
        raise ValueError(f"Classifier {classifier_dir} has the wrong class order")

    kmers_path = classifier_dir / "kmers.txt"
    kmers = read_canonical_kmers(kmers_path)
    feature_metadata = _model_metadata_section(model_metadata, "features", classifier_dir)
    read_pairs = feature_metadata.get("read_pairs")
    if type(read_pairs) is not int or feature_metadata != kmer_feature_metadata(kmers, read_pairs):
        raise ValueError(f"Classifier {classifier_dir} has incompatible k-mer feature metadata")

    if (
        _model_metadata_section(model_metadata, "normalization", classifier_dir)
        != COUNT_NORMALIZATION_METADATA
    ):
        raise ValueError(f"Classifier {classifier_dir} has incompatible normalization metadata")

    with np.load(classifier_dir / "model.npz", allow_pickle=False) as model_archive:
        if not {"coefficients", "intercept"} <= set(model_archive.files):
            raise ValueError(f"Classifier {classifier_dir} is missing required numeric arrays")
        coefficients = np.asarray(model_archive["coefficients"])
        intercept = np.asarray(model_archive["intercept"])
    if coefficients.dtype != np.float64 or coefficients.shape != (
        len(SPECIES_LABELS),
        len(kmers),
    ):
        raise ValueError(f"Classifier {classifier_dir} has the wrong coefficient shape")
    if intercept.dtype != np.float64 or intercept.shape != (len(SPECIES_LABELS),):
        raise ValueError(f"Classifier {classifier_dir} has the wrong intercept shape")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(intercept)):
        raise ValueError(f"Classifier {classifier_dir} contains non-finite values")
    return _LogisticClassifier(coefficients, intercept)


def _softmax_probabilities(
    classifier: _LogisticClassifier,
    normalized_counts: NDArray[np.float32],
) -> NDArray[np.float64]:
    logits = np.asarray(normalized_counts @ classifier.coefficients.T, dtype=np.float64)
    logits += classifier.intercept
    logits -= logits.max(axis=1, keepdims=True)
    np.exp(logits, out=logits)
    logits /= logits.sum(axis=1, keepdims=True)
    return logits


def _predict_count_batch(
    classifier: _LogisticClassifier,
    count_rows: NDArray[np.uint32],
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    probabilities = _softmax_probabilities(classifier, normalize_kmer_counts(count_rows))
    predicted_indices = np.asarray(np.argmax(probabilities, axis=1), dtype=np.int64)
    predicted_species = tuple(
        SPECIES_LABELS[int(species_index)] for species_index in predicted_indices
    )
    return predicted_species, probabilities


def _read_model_metadata(metadata_path: Path) -> dict[str, object]:
    try:
        parsed_metadata = cast(object, json.loads(metadata_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read classifier metadata {metadata_path}: {error}") from error
    if not isinstance(parsed_metadata, dict):
        raise ValueError(f"Classifier metadata {metadata_path} must be a JSON object")
    return cast("dict[str, object]", parsed_metadata)


def _model_metadata_section(
    model_metadata: dict[str, object],
    section_name: str,
    classifier_dir: Path,
) -> dict[str, object]:
    try:
        section_metadata = model_metadata[section_name]
    except KeyError as error:
        raise ValueError(
            f"Classifier {classifier_dir} is missing {section_name} metadata"
        ) from error
    if not isinstance(section_metadata, dict):
        raise ValueError(f"Classifier {classifier_dir} has invalid {section_name} metadata")
    return cast("dict[str, object]", section_metadata)
