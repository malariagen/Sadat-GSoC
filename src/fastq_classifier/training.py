"""Train the five-class logistic classifier on development data."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning  # pyright: ignore[reportMissingTypeStubs]
from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingTypeStubs]

from fastq_classifier.classifier import normalize_kmer_counts
from fastq_classifier.features import (
    CANONICAL_KMER_COUNT,
    CANONICAL_KMERS,
    COUNT_NORMALIZATION_METADATA,
    KMER_FEATURE_METADATA,
)
from fastq_classifier.folds import (
    DEVELOPMENT_FOLD_COUNT,
    DevelopmentFold,
    read_development_folds,
)
from fastq_classifier.matrix import read_matrix_run_accessions
from fastq_classifier.species import SPECIES_LABELS

CANDIDATE_C_VALUES = (0.01, 0.1, 1.0, 10.0)
NORMALIZATION_BATCH_ROWS = 64
LOGISTIC_MAX_ITERATIONS = 1_000
LOGISTIC_TOLERANCE = 1e-6
CALIBRATION_BIN_COUNT = 10


class _FittedLogisticRegression(Protocol):
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_iter_: NDArray[np.int32]

    def fit(
        self,
        normalized_counts: NDArray[np.float32],
        species_indices: NDArray[np.int64],
    ) -> object: ...

    def predict_proba(
        self,
        normalized_counts: NDArray[np.float32],
    ) -> NDArray[np.float64]: ...


def train_classifier(
    count_matrix_dir: str | Path,
    folds_path: str | Path,
    classifier_dir: str | Path,
) -> Path:
    """Select L2 strength with grouped folds and fit one classifier."""
    matrix_path = Path(count_matrix_dir)
    development_folds_path = Path(folds_path)
    classifier_path = Path(classifier_dir)
    if classifier_path.exists():
        raise FileExistsError(f"Training output already exists: {classifier_path}")

    run_accessions = read_matrix_run_accessions(matrix_path / "runs.tsv")
    development_folds = read_development_folds(development_folds_path, run_accessions)
    kmers_path = matrix_path / "kmers.txt"
    if tuple(kmers_path.read_text(encoding="ascii").splitlines()) != CANONICAL_KMERS:
        raise ValueError(f"K-mer file {kmers_path} does not contain the fixed vocabulary")
    count_rows = np.load(matrix_path / "counts.npy", mmap_mode="r", allow_pickle=False)
    if count_rows.dtype != np.uint32 or count_rows.shape != (
        len(development_folds),
        CANONICAL_KMER_COUNT,
    ):
        raise ValueError(
            f"Count matrix {matrix_path / 'counts.npy'} must have shape "
            f"({len(development_folds)}, {CANONICAL_KMER_COUNT}) and dtype uint32"
        )

    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    pending_classifier_dir = Path(
        tempfile.mkdtemp(prefix=f".{classifier_path.name}.", dir=classifier_path.parent)
    )
    try:
        normalized_counts_path = pending_classifier_dir / ".normalized.npy"
        normalized_counts = _write_normalized_counts(count_rows, normalized_counts_path)
        species_indices = np.asarray(
            [
                SPECIES_LABELS.index(development_fold.species_label)
                for development_fold in development_folds
            ],
            dtype=np.int64,
        )
        fold_indices = np.asarray(
            [development_fold.fold_index for development_fold in development_folds],
            dtype=np.int8,
        )
        cross_validation_results = [
            _cross_validate_c(normalized_counts, species_indices, fold_indices, c_value)
            for c_value in CANDIDATE_C_VALUES
        ]
        converged_results = [
            candidate_result
            for candidate_result in cross_validation_results
            if candidate_result[0]["converged"]
        ]
        if not converged_results:
            raise RuntimeError("Logistic regression did not converge for any C value")
        selected_candidate_summary, selected_probabilities = max(
            converged_results,
            key=lambda candidate_result: _candidate_rank(candidate_result[0]),
        )
        selected_c = cast(float, selected_candidate_summary["C"])

        fitted_classifier, convergence_messages = _fit_logistic_regression(
            normalized_counts,
            species_indices,
            selected_c,
        )
        if convergence_messages:
            convergence_detail = "; ".join(convergence_messages)
            raise RuntimeError(
                f"Full-development logistic regression did not converge: {convergence_detail}"
            )

        _write_classifier(
            pending_classifier_dir,
            matrix_path,
            development_folds,
            selected_c,
            fitted_classifier,
        )
        _write_development_results(
            pending_classifier_dir,
            development_folds,
            cross_validation_results,
            selected_c,
            selected_probabilities,
        )
        del normalized_counts
        normalized_counts_path.unlink()
        pending_classifier_dir.replace(classifier_path)
    finally:
        del count_rows
        shutil.rmtree(pending_classifier_dir, ignore_errors=True)

    return classifier_path / "model.npz"


def _write_normalized_counts(
    count_rows: NDArray[np.uint32],
    normalized_counts_path: Path,
) -> NDArray[np.float32]:
    normalized_counts = np.lib.format.open_memmap(
        normalized_counts_path,
        mode="w+",
        dtype=np.float32,
        shape=count_rows.shape,
    )
    for batch_start in range(0, len(count_rows), NORMALIZATION_BATCH_ROWS):
        batch_stop = min(batch_start + NORMALIZATION_BATCH_ROWS, len(count_rows))
        normalized_counts[batch_start:batch_stop] = normalize_kmer_counts(
            count_rows[batch_start:batch_stop]
        )
    normalized_counts.flush()
    del normalized_counts
    return cast(
        "NDArray[np.float32]",
        np.load(normalized_counts_path, mmap_mode="r", allow_pickle=False),
    )


def _configure_logistic_regression(c_value: float) -> _FittedLogisticRegression:
    return cast(
        "_FittedLogisticRegression",
        LogisticRegression(
            C=c_value,
            l1_ratio=0.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=LOGISTIC_MAX_ITERATIONS,
            tol=LOGISTIC_TOLERANCE,
        ),
    )


def _fit_logistic_regression(
    normalized_counts: NDArray[np.float32],
    species_indices: NDArray[np.int64],
    c_value: float,
) -> tuple[_FittedLogisticRegression, tuple[str, ...]]:
    classifier = _configure_logistic_regression(c_value)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(normalized_counts, species_indices)
    convergence_messages = tuple(
        str(warning.message)
        for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    )
    return classifier, convergence_messages


def _cross_validate_c(
    normalized_counts: NDArray[np.float32],
    species_indices: NDArray[np.int64],
    fold_indices: NDArray[np.int8],
    c_value: float,
) -> tuple[dict[str, object], NDArray[np.float64]]:
    probabilities = np.zeros((len(species_indices), len(SPECIES_LABELS)), dtype=np.float64)
    fold_summaries: list[dict[str, object]] = []
    convergence_messages: list[str] = []
    for fold_index in range(DEVELOPMENT_FOLD_COUNT):
        training_rows = fold_indices != fold_index
        validation_rows = fold_indices == fold_index
        classifier, fold_convergence_messages = _fit_logistic_regression(
            normalized_counts[training_rows],
            species_indices[training_rows],
            c_value,
        )
        probabilities[validation_rows] = np.asarray(
            classifier.predict_proba(normalized_counts[validation_rows]),
            dtype=np.float64,
        )
        convergence_messages.extend(fold_convergence_messages)
        fold_summaries.append(
            {
                "fold": fold_index,
                "training_samples": int(np.count_nonzero(training_rows)),
                "validation_samples": int(np.count_nonzero(validation_rows)),
                "iterations": int(np.max(classifier.n_iter_)),
                "metrics": _classification_metrics(
                    species_indices[validation_rows], probabilities[validation_rows]
                ),
            }
        )

    candidate_summary: dict[str, object] = {
        "C": c_value,
        "converged": not convergence_messages,
        "convergence_warnings": convergence_messages,
        "metrics": _classification_metrics(species_indices, probabilities),
        "folds": fold_summaries,
    }
    return candidate_summary, probabilities


def _classification_metrics(
    species_indices: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> dict[str, object]:
    predicted_indices = np.asarray(np.argmax(probabilities, axis=1), dtype=np.int64)
    confusion_matrix = np.zeros(
        (len(SPECIES_LABELS), len(SPECIES_LABELS)),
        dtype=np.int64,
    )
    np.add.at(confusion_matrix, (species_indices, predicted_indices), 1)
    species_support = confusion_matrix.sum(axis=1)
    predicted_species_support = confusion_matrix.sum(axis=0)
    true_positive_counts = np.asarray(np.diag(confusion_matrix), dtype=np.int64)
    precision = np.divide(
        true_positive_counts,
        predicted_species_support,
        out=np.zeros(len(SPECIES_LABELS), dtype=np.float64),
        where=predicted_species_support != 0,
    )
    recall = true_positive_counts / species_support
    f1_score = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(SPECIES_LABELS), dtype=np.float64),
        where=(precision + recall) != 0,
    )
    true_species_probabilities = probabilities[np.arange(len(species_indices)), species_indices]
    squared_probability_sum = np.square(probabilities).sum(dtype=np.float64)
    brier_score = (
        squared_probability_sum - 2.0 * true_species_probabilities.sum() + len(species_indices)
    ) / len(species_indices)

    return {
        "sample_count": len(species_indices),
        "accuracy": np.count_nonzero(predicted_indices == species_indices) / len(species_indices),
        "balanced_accuracy": float(recall.sum(dtype=np.float64) / len(SPECIES_LABELS)),
        "macro_f1": float(f1_score.sum(dtype=np.float64) / len(SPECIES_LABELS)),
        "log_loss": float(
            -np.log(np.clip(true_species_probabilities, 1e-15, 1.0)).sum() / len(species_indices)
        ),
        "brier_score": float(brier_score),
        "expected_calibration_error": _expected_calibration_error(
            species_indices,
            predicted_indices,
            probabilities,
        ),
        "per_class": {
            species_label: {
                "precision": float(precision[species_index]),
                "recall": float(recall[species_index]),
                "f1": float(f1_score[species_index]),
                "support": int(species_support[species_index]),
            }
            for species_index, species_label in enumerate(SPECIES_LABELS)
        },
        "confusion_matrix": confusion_matrix.tolist(),
    }


def _expected_calibration_error(
    species_indices: NDArray[np.int64],
    predicted_indices: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> float:
    confidence_by_row = probabilities.max(axis=1)
    calibration_bin_by_row = np.minimum(
        (confidence_by_row * CALIBRATION_BIN_COUNT).astype(np.int64),
        CALIBRATION_BIN_COUNT - 1,
    )
    correct_prediction = predicted_indices == species_indices
    calibration_error = 0.0
    for bin_index in range(CALIBRATION_BIN_COUNT):
        rows_in_bin = calibration_bin_by_row == bin_index
        row_count = int(np.count_nonzero(rows_in_bin))
        if row_count:
            observed_accuracy = np.count_nonzero(correct_prediction & rows_in_bin) / row_count
            mean_confidence = float(confidence_by_row[rows_in_bin].sum() / row_count)
            calibration_error += (
                row_count / len(species_indices) * abs(observed_accuracy - mean_confidence)
            )
    return calibration_error


def _candidate_rank(candidate_summary: dict[str, object]) -> tuple[float, float, float, float]:
    candidate_metrics = cast("dict[str, object]", candidate_summary["metrics"])
    return (
        cast(float, candidate_metrics["balanced_accuracy"]),
        cast(float, candidate_metrics["macro_f1"]),
        -cast(float, candidate_metrics["log_loss"]),
        -cast(float, candidate_summary["C"]),
    )


def _write_classifier(
    classifier_dir: Path,
    count_matrix_dir: Path,
    development_folds: tuple[DevelopmentFold, ...],
    selected_c: float,
    classifier: _FittedLogisticRegression,
) -> None:
    coefficients = np.asarray(classifier.coef_, dtype=np.float64)
    intercept = np.asarray(classifier.intercept_, dtype=np.float64)
    np.savez_compressed(
        classifier_dir / "model.npz",
        coefficients=coefficients,
        intercept=intercept,
    )

    kmers_path = count_matrix_dir / "kmers.txt"
    shutil.copyfile(kmers_path, classifier_dir / "kmers.txt")
    model_metadata = {
        "classes": list(SPECIES_LABELS),
        "features": KMER_FEATURE_METADATA,
        "normalization": COUNT_NORMALIZATION_METADATA,
        "classifier": {
            "family": "multiclass_logistic_regression",
            "penalty": "l2",
            "C": selected_c,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "iterations": int(np.max(classifier.n_iter_)),
            "maximum_iterations": LOGISTIC_MAX_ITERATIONS,
            "convergence_tolerance": LOGISTIC_TOLERANCE,
        },
        "development": {
            "samples": len(development_folds),
            "folds": DEVELOPMENT_FOLD_COUNT,
            "C_values": list(CANDIDATE_C_VALUES),
            "selection_order": [
                "balanced_accuracy",
                "macro_f1",
                "log_loss",
                "smaller_C",
            ],
        },
        "software": {
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
    }
    _write_classifier_json(classifier_dir / "model.json", model_metadata)


def _write_development_results(
    classifier_dir: Path,
    development_folds: tuple[DevelopmentFold, ...],
    cross_validation_results: list[tuple[dict[str, object], NDArray[np.float64]]],
    selected_c: float,
    probabilities: NDArray[np.float64],
) -> None:
    _write_classifier_json(
        classifier_dir / "development_metrics.json",
        {
            "selected_C": selected_c,
            "candidates": [
                candidate_summary
                for candidate_summary, _candidate_probabilities in cross_validation_results
            ],
        },
    )
    _write_out_of_fold_predictions(
        classifier_dir / "oof_predictions.tsv",
        development_folds,
        probabilities,
    )
    _write_domain_metrics(
        classifier_dir / "domain_metrics.tsv",
        development_folds,
        probabilities,
    )


def _write_out_of_fold_predictions(
    predictions_path: Path,
    development_folds: tuple[DevelopmentFold, ...],
    probabilities: NDArray[np.float64],
) -> None:
    predicted_indices = np.asarray(np.argmax(probabilities, axis=1), dtype=np.int64)
    prediction_columns = (
        "row_index",
        "specimen_id",
        "run_accession",
        "blocking_group",
        "fold",
        "true_label",
        "predicted_label",
        *(f"probability_{species_label}" for species_label in SPECIES_LABELS),
    )
    with predictions_path.open("w", encoding="utf-8", newline="") as prediction_stream:
        prediction_rows = csv.writer(prediction_stream, delimiter="\t", lineterminator="\n")
        prediction_rows.writerow(prediction_columns)
        for row_index, development_fold in enumerate(development_folds):
            prediction_rows.writerow(
                (
                    row_index,
                    development_fold.specimen_id,
                    development_fold.run_accession,
                    development_fold.blocking_group,
                    development_fold.fold_index,
                    development_fold.species_label,
                    SPECIES_LABELS[int(predicted_indices[row_index])],
                    *(f"{probability:.12g}" for probability in probabilities[row_index]),
                )
            )


def _write_domain_metrics(
    metrics_path: Path,
    development_folds: tuple[DevelopmentFold, ...],
    probabilities: NDArray[np.float64],
) -> None:
    row_indices_by_group: dict[str, list[int]] = {}
    for row_index, development_fold in enumerate(development_folds):
        row_indices_by_group.setdefault(development_fold.blocking_group, []).append(row_index)

    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_stream:
        metric_rows = csv.writer(metrics_stream, delimiter="\t", lineterminator="\n")
        metric_rows.writerow(
            (
                "label",
                "blocking_group",
                "fold",
                "sample_count",
                "accuracy",
                "mean_true_probability",
                "log_loss",
            )
        )
        for blocking_group in sorted(row_indices_by_group):
            row_indices = np.asarray(row_indices_by_group[blocking_group], dtype=np.int64)
            representative_fold = development_folds[int(row_indices[0])]
            true_species_index = SPECIES_LABELS.index(representative_fold.species_label)
            predicted_indices = np.asarray(
                np.argmax(probabilities[row_indices], axis=1),
                dtype=np.int64,
            )
            true_species_probabilities = probabilities[row_indices, true_species_index]
            accuracy = np.count_nonzero(predicted_indices == true_species_index) / len(row_indices)
            mean_true_probability = float(true_species_probabilities.sum() / len(row_indices))
            log_loss = float(
                -np.log(np.clip(true_species_probabilities, 1e-15, 1.0)).sum() / len(row_indices)
            )
            metric_rows.writerow(
                (
                    representative_fold.species_label,
                    blocking_group,
                    representative_fold.fold_index,
                    len(row_indices),
                    f"{accuracy:.12g}",
                    f"{mean_true_probability:.12g}",
                    f"{log_loss:.12g}",
                )
            )


def _write_classifier_json(json_path: Path, json_document: object) -> None:
    json_path.write_text(
        json.dumps(json_document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
