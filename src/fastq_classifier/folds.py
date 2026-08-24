"""Assign development samples to grouped cross-validation folds."""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastq_classifier.matrix import read_matrix_run_accessions
from fastq_classifier.species import SPECIES_LABELS

DEVELOPMENT_FOLD_COUNT = 4

_BLOCKING_FIELDS_BY_SPECIES = {
    "gambiae_complex": ("GAMBIAE_COMPLEX_COUNTRY", ("country",)),
    "darlingi": ("DARLINGI_SOURCE", ("source",)),
    "minimus": ("MINIMUS_LOCATION", ("location",)),
    "stephensi": ("STEPHENSI_STUDY", ("study_id",)),
    "funestus": ("FUNESTUS_LOCATION_YEAR", ("location", "year")),
}
_SAMPLE_COLUMNS = {
    "specimen_id",
    "run_accession",
    "label",
    "country",
    "source",
    "location",
    "year",
    "study_id",
}


@dataclass(frozen=True, slots=True)
class _DevelopmentSample:
    specimen_id: str
    run_accession: str
    species_label: str
    blocking_group: str


@dataclass(frozen=True, slots=True)
class DevelopmentFold:
    specimen_id: str
    run_accession: str
    species_label: str
    blocking_group: str
    fold_index: int


def assign_development_folds(
    matrix_dir: str | Path,
    development_samples_path: str | Path,
    fold_dir: str | Path,
) -> Path:
    """Assign complete sampling domains to four development folds."""
    matrix_run_accessions = read_matrix_run_accessions(Path(matrix_dir) / "runs.tsv")
    development_samples = _read_development_samples(Path(development_samples_path))
    sample_by_accession = {sample.run_accession: sample for sample in development_samples}
    _validate_sample_runs(matrix_run_accessions, sample_by_accession)

    fold_by_blocking_group = _balance_blocking_groups(development_samples)
    fold_directory = Path(fold_dir)
    fold_directory.mkdir(parents=True, exist_ok=True)
    folds_path = fold_directory / "folds.tsv"
    pending_folds_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            prefix=".folds.",
            suffix=".tmp",
            dir=fold_directory,
            delete=False,
        ) as fold_stream:
            pending_folds_path = Path(fold_stream.name)
            fold_rows = csv.writer(fold_stream, delimiter="\t", lineterminator="\n")
            fold_rows.writerow(
                (
                    "row_index",
                    "specimen_id",
                    "run_accession",
                    "label",
                    "blocking_group",
                    "fold",
                )
            )
            for row_index, run_accession in enumerate(matrix_run_accessions):
                sample = sample_by_accession[run_accession]
                fold_rows.writerow(
                    (
                        row_index,
                        sample.specimen_id,
                        run_accession,
                        sample.species_label,
                        sample.blocking_group,
                        fold_by_blocking_group[sample.blocking_group],
                    )
                )
        pending_folds_path.replace(folds_path)
    finally:
        if pending_folds_path is not None:
            pending_folds_path.unlink(missing_ok=True)
    return folds_path


def read_development_folds(
    folds_path: str | Path,
    matrix_run_accessions: tuple[str, ...],
) -> tuple[DevelopmentFold, ...]:
    fold_index_path = Path(folds_path)
    with fold_index_path.open(encoding="utf-8-sig", newline="") as fold_stream:
        fold_rows = csv.DictReader(fold_stream, delimiter="\t")
        fold_columns = tuple(fold_rows.fieldnames or ())
        required_columns = {
            "row_index",
            "specimen_id",
            "run_accession",
            "label",
            "blocking_group",
            "fold",
        }
        missing_columns = required_columns - set(fold_columns)
        if missing_columns:
            column_names = ", ".join(sorted(missing_columns))
            raise ValueError(f"Fold file {fold_index_path} is missing columns: {column_names}")

        development_folds: list[DevelopmentFold] = []
        species_and_fold_by_group: dict[str, tuple[str, int]] = {}
        seen_specimens: set[str] = set()
        for fold_row in fold_rows:
            if not any((fold_row.get(column_name) or "").strip() for column_name in fold_columns):
                continue
            line_number = fold_rows.line_num
            expected_row_index = len(development_folds)
            try:
                row_index = int((fold_row["row_index"] or "").strip())
                fold_index = int((fold_row["fold"] or "").strip())
            except ValueError as error:
                raise ValueError(
                    f"Fold file {fold_index_path}, line {line_number}: invalid row_index or fold"
                ) from error
            if row_index != expected_row_index:
                raise ValueError(
                    f"Fold file {fold_index_path}, line {line_number}: expected row_index "
                    f"{expected_row_index}, got {row_index}"
                )
            if not 0 <= fold_index < DEVELOPMENT_FOLD_COUNT:
                raise ValueError(
                    f"Fold file {fold_index_path}, line {line_number}: invalid fold {fold_index}"
                )

            specimen_id = (fold_row["specimen_id"] or "").strip()
            run_accession = (fold_row["run_accession"] or "").strip()
            species_label = (fold_row["label"] or "").strip()
            blocking_group = (fold_row["blocking_group"] or "").strip()
            if not specimen_id or not run_accession or not species_label or not blocking_group:
                raise ValueError(
                    f"Fold file {fold_index_path}, line {line_number}: empty required value"
                )
            if (
                expected_row_index >= len(matrix_run_accessions)
                or run_accession != matrix_run_accessions[expected_row_index]
            ):
                raise ValueError(
                    f"Fold file {fold_index_path}, line {line_number} "
                    "does not match the matrix row order"
                )
            if specimen_id in seen_specimens:
                raise ValueError(
                    f"Fold file {fold_index_path}, line {line_number}: duplicate specimen"
                )
            if species_label not in SPECIES_LABELS:
                raise ValueError(
                    f"Fold file {fold_index_path}, line {line_number}: "
                    f"unknown label {species_label}"
                )
            species_and_fold = (species_label, fold_index)
            existing_species_and_fold = species_and_fold_by_group.setdefault(
                blocking_group, species_and_fold
            )
            if existing_species_and_fold != species_and_fold:
                raise ValueError(
                    f"Blocking group {blocking_group} has inconsistent labels or folds"
                )

            development_folds.append(
                DevelopmentFold(
                    specimen_id,
                    run_accession,
                    species_label,
                    blocking_group,
                    fold_index,
                )
            )
            seen_specimens.add(specimen_id)

    if len(development_folds) != len(matrix_run_accessions):
        raise ValueError(
            f"Fold file {fold_index_path} has {len(development_folds)} rows; "
            f"expected {len(matrix_run_accessions)}"
        )
    expected_species = set(SPECIES_LABELS)
    for fold_index in range(DEVELOPMENT_FOLD_COUNT):
        observed_species = {
            development_fold.species_label
            for development_fold in development_folds
            if development_fold.fold_index == fold_index
        }
        if observed_species != expected_species:
            missing_species = ", ".join(
                species_label
                for species_label in SPECIES_LABELS
                if species_label not in observed_species
            )
            raise ValueError(f"Fold {fold_index} is missing labels: {missing_species}")
    return tuple(development_folds)


def _read_development_samples(sample_path: Path) -> tuple[_DevelopmentSample, ...]:
    with sample_path.open(encoding="utf-8-sig", newline="") as sample_stream:
        sample_rows = csv.DictReader(sample_stream, delimiter="\t")
        sample_columns = tuple(sample_rows.fieldnames or ())
        if "split" in sample_columns:
            raise ValueError(
                f"Development sample file {sample_path} must not contain a split column; "
                "provide a development-only file"
            )
        missing_columns = _SAMPLE_COLUMNS - set(sample_columns)
        if missing_columns:
            column_names = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Development sample file {sample_path} is missing columns: {column_names}"
            )

        development_samples: list[_DevelopmentSample] = []
        seen_specimens: set[str] = set()
        seen_accessions: set[str] = set()
        for sample_row in sample_rows:
            if not any(
                (sample_row.get(column_name) or "").strip() for column_name in sample_columns
            ):
                continue
            line_number = sample_rows.line_num
            specimen_id = _required_sample_field(
                sample_row, "specimen_id", sample_path, line_number
            )
            run_accession = _required_sample_field(
                sample_row, "run_accession", sample_path, line_number
            )
            species_label = _required_sample_field(sample_row, "label", sample_path, line_number)
            if specimen_id in seen_specimens:
                raise ValueError(
                    f"Development sample file {sample_path}, line {line_number}: "
                    f"duplicate specimen {specimen_id}"
                )
            if run_accession in seen_accessions:
                raise ValueError(
                    f"Development sample file {sample_path}, line {line_number}: "
                    f"duplicate run {run_accession}"
                )
            if species_label not in _BLOCKING_FIELDS_BY_SPECIES:
                raise ValueError(
                    f"Development sample file {sample_path}, line {line_number}: "
                    f"unknown label {species_label}"
                )

            group_prefix, blocking_columns = _BLOCKING_FIELDS_BY_SPECIES[species_label]
            blocking_values = tuple(
                _required_sample_field(sample_row, column_name, sample_path, line_number)
                for column_name in blocking_columns
            )
            development_samples.append(
                _DevelopmentSample(
                    specimen_id,
                    run_accession,
                    species_label,
                    f"{group_prefix}:{'|'.join(blocking_values)}",
                )
            )
            seen_specimens.add(specimen_id)
            seen_accessions.add(run_accession)

    if not development_samples:
        raise ValueError(f"Development sample file {sample_path} contains no samples")
    return tuple(development_samples)


def _required_sample_field(
    sample_row: dict[str, str | None],
    column_name: str,
    sample_path: Path,
    line_number: int,
) -> str:
    field_value = (sample_row[column_name] or "").strip()
    if not field_value:
        raise ValueError(
            f"Development sample file {sample_path}, line {line_number}: {column_name} is empty"
        )
    return field_value


def _validate_sample_runs(
    matrix_run_accessions: tuple[str, ...],
    sample_by_accession: dict[str, _DevelopmentSample],
) -> None:
    matrix_accessions = set(matrix_run_accessions)
    sample_accessions = set(sample_by_accession)
    missing_samples = matrix_accessions - sample_accessions
    unmatched_samples = sample_accessions - matrix_accessions
    if missing_samples or unmatched_samples:
        discrepancies: list[str] = []
        if missing_samples:
            discrepancies.append(f"missing development rows for {len(missing_samples)} matrix runs")
        if unmatched_samples:
            discrepancies.append(
                f"found {len(unmatched_samples)} development rows without matrix runs"
            )
        raise ValueError(
            "Development samples do not match the count matrix: " + "; ".join(discrepancies)
        )


def _balance_blocking_groups(
    development_samples: tuple[_DevelopmentSample, ...],
) -> dict[str, int]:
    blocking_group_sizes_by_species: dict[str, dict[str, int]] = {
        species_label: {} for species_label in SPECIES_LABELS
    }
    for sample in development_samples:
        blocking_group_sizes = blocking_group_sizes_by_species[sample.species_label]
        blocking_group_sizes[sample.blocking_group] = (
            blocking_group_sizes.get(sample.blocking_group, 0) + 1
        )

    fold_by_blocking_group: dict[str, int] = {}
    for species_label in SPECIES_LABELS:
        blocking_group_sizes = blocking_group_sizes_by_species[species_label]
        if len(blocking_group_sizes) < DEVELOPMENT_FOLD_COUNT:
            raise ValueError(
                f"Label {species_label} has {len(blocking_group_sizes)} blocking groups; "
                f"at least {DEVELOPMENT_FOLD_COUNT} are required"
            )

        sample_count_by_fold = [0] * DEVELOPMENT_FOLD_COUNT
        largest_groups_first = sorted(
            blocking_group_sizes.items(),
            key=lambda group_and_size: (-group_and_size[1], group_and_size[0]),
        )
        for blocking_group, sample_count in largest_groups_first:
            fold_index = min(
                range(DEVELOPMENT_FOLD_COUNT),
                key=lambda candidate_fold_index: (
                    sample_count_by_fold[candidate_fold_index],
                    candidate_fold_index,
                ),
            )
            fold_by_blocking_group[blocking_group] = fold_index
            sample_count_by_fold[fold_index] += sample_count

    return fold_by_blocking_group
