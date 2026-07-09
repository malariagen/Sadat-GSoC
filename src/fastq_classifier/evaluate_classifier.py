"""Evaluate a sparse k-mer classifier."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from fastq_classifier.build_matrix import (
    FEATURES_FILE_NAME,
    MATRIX_FILE_NAME,
    SAMPLES_FILE_NAME,
)
from fastq_classifier.utils import InputError, read_validated_tsv, write_rows

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    import numpy as np

METRICS_FILE_NAME = "metrics.tsv"
PREDICTIONS_FILE_NAME = "predictions.tsv"
CONFUSION_MATRIX_FILE_NAME = "confusion_matrix.tsv"
CONFUSION_MATRIX_PLOT_FILE_NAME = "confusion_matrix.png"
MIN_CLASSES = 2
MIN_SAMPLES_PER_CLASS = 2
PLOT_DPI = 200
PLOT_BASE_SIZE = 4.0
PLOT_WIDTH_PER_CLASS = 0.8
__all__ = [
    "ClassifierError",
    "ClassifierEvaluation",
    "evaluate_classifier",
]


class ClassifierError(Exception):
    """Classifier evaluation failed."""


@dataclass(frozen=True, slots=True)
class ClassifierEvaluation:
    """Classifier evaluation summary.

    Attributes
    ----------
    sample_count : int
        Number of labeled samples.
    feature_count : int
        Number of k-mer features.
    train_count : int
        Number of training samples.
    test_count : int
        Number of test samples.
    class_count : int
        Number of labels.
    accuracy : float
        Test-set accuracy.
    balanced_accuracy : float
        Test-set balanced accuracy.
    metrics_path, predictions_path, confusion_matrix_path, plot_path : pathlib.Path
        Output file paths.
    """

    sample_count: int
    feature_count: int
    train_count: int
    test_count: int
    class_count: int
    accuracy: float
    balanced_accuracy: float
    metrics_path: Path
    predictions_path: Path
    confusion_matrix_path: Path
    plot_path: Path


@dataclass(frozen=True, slots=True)
class LabeledSamples:
    """Validated inputs for classifier evaluation.

    Attributes
    ----------
    samples : tuple of dict of str to str
        Sample metadata rows from ``samples.tsv``.
    labels : tuple of str
        Label for each sample.
    classes : tuple of str
        Unique labels in their first-seen order.
    feature_count : int
        Number of matrix columns.
    """

    samples: tuple[dict[str, str], ...]
    labels: tuple[str, ...]
    classes: tuple[str, ...]
    feature_count: int


def evaluate_classifier(
    matrix_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
    label_column: str,
    *,
    test_size: float = 0.25,
    random_seed: int = 1,
) -> ClassifierEvaluation:
    """Train and evaluate a sparse k-mer classifier.

    Parameters
    ----------
    matrix_dir : path-like
        Directory containing ``matrix.npz``, ``samples.tsv``, and
        ``features.tsv`` from ``build_kmer_matrix``.
    output_dir : path-like
        Directory where evaluation outputs will be written.
    label_column : str
        Column in ``samples.tsv`` containing class labels.
    test_size : float, optional
        Fraction of samples used for the test set.
    random_seed : int, optional
        Random seed used for the stratified split and classifier.

    Returns
    -------
    ClassifierEvaluation
        Evaluation metrics and output paths.
    """
    check_test_size(test_size)
    matrix_root = Path(matrix_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / METRICS_FILE_NAME
    predictions_path = output / PREDICTIONS_FILE_NAME
    confusion_matrix_path = output / CONFUSION_MATRIX_FILE_NAME
    plot_path = output / CONFUSION_MATRIX_PLOT_FILE_NAME
    output_paths = (metrics_path, predictions_path, confusion_matrix_path, plot_path)
    for path in output_paths:
        path.unlink(missing_ok=True)

    require_matrix_files(matrix_root)
    data = read_labeled_samples(matrix_root, label_column)
    check_split_size(data.labels, data.classes, test_size)
    matrix = load_npz(matrix_root / MATRIX_FILE_NAME).tocsr()
    matrix_sample_count, matrix_feature_count = matrix.shape
    if matrix_sample_count != len(data.samples):
        msg = (
            f"matrix row count does not match samples.tsv: {matrix_sample_count} "
            f"!= {len(data.samples)}"
        )
        raise InputError(msg)
    if matrix_feature_count != data.feature_count:
        msg = (
            "matrix column count does not match features.tsv: "
            f"{matrix_feature_count} != {data.feature_count}"
        )
        raise InputError(msg)

    indices = list(range(len(data.labels)))
    try:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=data.labels,
        )
    except ValueError as error:
        raise ClassifierError(str(error)) from error
    y_train = [data.labels[index] for index in train_indices]
    y_test = [data.labels[index] for index in test_indices]

    model = make_pipeline(
        MaxAbsScaler(),
        LogisticRegression(
            C=100.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed,
        ),
    )
    try:
        model.fit(matrix[train_indices], y_train)
        predictions = tuple(str(label) for label in model.predict(matrix[test_indices]))
    except ValueError as error:
        raise ClassifierError(str(error)) from error
    accuracy = float(accuracy_score(y_test, predictions))
    balanced_accuracy = float(balanced_accuracy_score(y_test, predictions))
    raw_counts = confusion_matrix(y_test, predictions, labels=data.classes)
    counts = tuple(tuple(row) for row in raw_counts.tolist())

    try:
        metric_rows = (
            {"metric": name, "value": value}
            for name, value in (
                ("sample_count", str(len(data.samples))),
                ("feature_count", str(data.feature_count)),
                ("train_count", str(len(train_indices))),
                ("test_count", str(len(test_indices))),
                ("class_count", str(len(data.classes))),
                ("accuracy", f"{accuracy:.6f}"),
                ("balanced_accuracy", f"{balanced_accuracy:.6f}"),
            )
        )
        write_rows(
            metrics_path,
            metric_rows,
            ("metric", "value"),
        )
        write_predictions(predictions_path, data.samples, y_test, predictions, test_indices)
        write_confusion_matrix(confusion_matrix_path, data.classes, counts)
        write_confusion_matrix_plot(plot_path, data.classes, raw_counts)
    except OSError:
        for path in output_paths:
            path.unlink(missing_ok=True)
        raise
    return ClassifierEvaluation(
        sample_count=len(data.samples),
        feature_count=data.feature_count,
        train_count=len(train_indices),
        test_count=len(test_indices),
        class_count=len(data.classes),
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        confusion_matrix_path=confusion_matrix_path,
        plot_path=plot_path,
    )


def read_labeled_samples(matrix_dir: Path, label_column: str) -> LabeledSamples:
    """Read and validate matrix metadata.

    Parameters
    ----------
    matrix_dir : pathlib.Path
        Matrix directory.
    label_column : str
        Column in ``samples.tsv`` containing class labels.

    Returns
    -------
    LabeledSamples
        Sample metadata, labels, classes, and feature count.
    """
    samples = read_samples(matrix_dir / SAMPLES_FILE_NAME, label_column)
    labels = tuple(row[label_column].strip() for row in samples)
    label_counts = Counter(labels)
    classes = tuple(dict.fromkeys(labels))
    if len(classes) < MIN_CLASSES:
        msg = "label column must contain at least two classes"
        raise InputError(msg)
    small_classes = [
        label for label, count in label_counts.items() if count < MIN_SAMPLES_PER_CLASS
    ]
    if small_classes:
        small_class_names = ", ".join(sorted(small_classes))
        msg = f"each class needs at least two samples; too small: {small_class_names}"
        raise InputError(msg)
    return LabeledSamples(
        samples=samples,
        labels=labels,
        classes=classes,
        feature_count=count_features(matrix_dir / FEATURES_FILE_NAME),
    )


def require_matrix_files(matrix_dir: Path) -> None:
    """Check that a matrix directory contains required files.

    Parameters
    ----------
    matrix_dir : pathlib.Path
        Matrix directory.

    Raises
    ------
    InputError
        If a required matrix file is missing.
    """
    for file_name in (MATRIX_FILE_NAME, SAMPLES_FILE_NAME, FEATURES_FILE_NAME):
        path = matrix_dir / file_name
        if not path.is_file():
            msg = f"matrix directory is missing {file_name}: {path}"
            raise InputError(msg)


def read_samples(path: Path, label_column: str) -> tuple[dict[str, str], ...]:
    """Read sample metadata and labels.

    Parameters
    ----------
    path : pathlib.Path
        ``samples.tsv`` path.
    label_column : str
        Column containing class labels.

    Returns
    -------
    tuple of dict of str to str
        Sample metadata rows.

    Raises
    ------
    InputError
        If the file has no header, the label column is missing, or the label
        column contains empty values.
    """
    rows, _ = read_validated_tsv(path, (label_column,), file_label="samples.tsv")
    blank_rows = [row.get("sample_index", "") for row in rows if not row[label_column].strip()]
    if blank_rows:
        msg = f"label column contains empty values for sample_index: {', '.join(blank_rows)}"
        raise InputError(msg)
    return rows


def count_features(path: Path) -> int:
    """Count feature rows.

    Parameters
    ----------
    path : pathlib.Path
        ``features.tsv`` path.

    Returns
    -------
    int
        Number of k-mer features.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            msg = f"{path} does not contain a header row"
            raise InputError(msg)
        if "feature_index" not in header or "kmer" not in header:
            msg = "features.tsv must contain feature_index and kmer columns"
            raise InputError(msg)
        return sum(1 for _row in reader)


def write_predictions(
    path: Path,
    samples: Sequence[dict[str, str]],
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    test_indices: Sequence[int],
) -> None:
    """Write test-set predictions.

    Parameters
    ----------
    path : pathlib.Path
        Output TSV path.
    samples : sequence of dict of str to str
        Sample metadata rows.
    true_labels : sequence of str
        True labels for test samples.
    predicted_labels : sequence of str
        Predicted labels for test samples.
    test_indices : sequence of int
        Zero-based sample row indices in the test split.
    """
    rows = (
        {
            "sample_index": samples[index].get("sample_index", ""),
            "run_accession": samples[index].get("run_accession", ""),
            "true_label": true_label,
            "predicted_label": predicted_label,
        }
        for index, true_label, predicted_label in zip(
            test_indices,
            true_labels,
            predicted_labels,
            strict=True,
        )
    )
    write_rows(path, rows, ("sample_index", "run_accession", "true_label", "predicted_label"))


def write_confusion_matrix(
    path: Path,
    classes: Sequence[str],
    counts: Sequence[Sequence[int]],
) -> None:
    """Write a confusion matrix table.

    Parameters
    ----------
    path : pathlib.Path
        Output TSV path.
    classes : sequence of str
        Class labels in matrix order.
    counts : sequence of sequence of int
        Square count matrix returned by scikit-learn.
    """
    rows = []
    for row_index, true_label in enumerate(classes):
        row = {"true_label": true_label}
        for column_index, predicted_label in enumerate(classes):
            row[predicted_label] = str(counts[row_index][column_index])
        rows.append(row)
    write_rows(path, rows, ("true_label", *classes))


def write_confusion_matrix_plot(path: Path, classes: Sequence[str], counts: np.ndarray) -> None:
    """Plot a confusion matrix with scikit-learn and matplotlib.

    Parameters
    ----------
    path : pathlib.Path
        Output image path.
    classes : sequence of str
        Class labels in matrix order.
    counts : numpy.ndarray
        Square count matrix returned by scikit-learn.
    """
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")

    from matplotlib import pyplot as plt  # noqa: PLC0415

    figure, axes = plt.subplots(
        figsize=(max(PLOT_BASE_SIZE, len(classes) * PLOT_WIDTH_PER_CLASS), PLOT_BASE_SIZE),
    )
    try:
        display = ConfusionMatrixDisplay(confusion_matrix=counts, display_labels=classes)
        display.plot(ax=axes, cmap="Blues", colorbar=False, values_format="d")
        figure.tight_layout()
        figure.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    finally:
        plt.close(figure)


def check_test_size(test_size: float) -> None:
    """Validate a test-set fraction.

    Parameters
    ----------
    test_size : float
        Fraction of samples used for testing.

    Raises
    ------
    InputError
        If ``test_size`` is not between zero and one.
    """
    if 0.0 < test_size < 1.0:
        return
    msg = "test_size must be greater than 0 and less than 1"
    raise InputError(msg)


def check_split_size(labels: Sequence[str], classes: Sequence[str], test_size: float) -> None:
    """Check that a stratified split can contain every class.

    Parameters
    ----------
    labels : sequence of str
        Sample labels.
    classes : sequence of str
        Unique labels.
    test_size : float
        Fraction of samples used for testing.

    Raises
    ------
    InputError
        If the train or test split would have fewer rows than classes.
    """
    test_count = math.ceil(len(labels) * test_size)
    train_count = len(labels) - test_count
    class_count = len(classes)
    if test_count >= class_count and train_count >= class_count:
        return
    msg = (
        "test_size leaves too few samples for a stratified split; "
        f"train={train_count}, test={test_count}, classes={class_count}"
    )
    raise InputError(msg)
