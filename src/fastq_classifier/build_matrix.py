"""Build sparse k-mer feature matrices from KMC databases.

The input is the ``feature_results.tsv`` file written by the KMC feature
extractor. Completed and skipped rows become matrix samples. Rows that cannot
be converted are written to ``invalid_rows.tsv``.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from fastq_classifier.utils import (
    InputError,
    RejectedRow,
    check_executable,
    format_process_error,
    kmc_database_exists,
    read_validated_tsv,
    safe_path_name,
    write_invalid_rows,
    write_rows,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from os import PathLike

ACCEPTED_FEATURE_STATUSES = frozenset(("completed", "skipped"))
KMC_DUMP_FIELD_COUNT = 2
REQUIRED_COLUMNS = ("run_accession", "k", "kmc_database", "kmer_status")
MATRIX_FILE_NAME = "matrix.npz"
FEATURES_FILE_NAME = "features.tsv"
SAMPLES_FILE_NAME = "samples.tsv"
PARALLEL_PARSE_MIN_DUMP_SIZE = 100 * 1024 * 1024
__all__ = [
    "KmerMatrix",
    "MatrixBuildError",
    "build_kmer_matrix",
]


class MatrixBuildError(Exception):
    """Sparse matrix building failed."""


@dataclass(frozen=True, slots=True)
class MatrixRow:
    """One KMC database ready to become a matrix row.

    Attributes
    ----------
    row_number : int
        One-based line number in the TSV file, including the header row.
    sample_index : int
        One-based row index in the sparse matrix.
    values : dict of str to str
        Original TSV values keyed by column name.
    database_prefix : pathlib.Path
        KMC database prefix without ``.kmc_pre`` or ``.kmc_suf``.
    """

    row_number: int
    sample_index: int
    values: dict[str, str]
    database_prefix: Path

    def to_row(self, original_columns: Sequence[str]) -> dict[str, str]:
        """Build the row written to ``samples.tsv``.

        Parameters
        ----------
        original_columns : sequence of str
            Input columns, in their original order.

        Returns
        -------
        dict of str to str
            Original values plus the matrix sample index.
        """
        row = {column: self.values.get(column, "") for column in original_columns}
        row["sample_index"] = str(self.sample_index)
        return row


@dataclass(frozen=True, slots=True)
class KmerMatrix:
    """Sparse k-mer matrix written by ``build_kmer_matrix``.

    Attributes
    ----------
    sample_count : int
        Number of matrix rows.
    feature_count : int
        Number of distinct k-mers.
    entry_count : int
        Number of non-zero matrix entries.
    elapsed_seconds : float
        Wall-clock seconds spent building the matrix.
    matrix_path, features_path, samples_path : pathlib.Path
        Output file paths.
    """

    samples: tuple[MatrixRow, ...]
    invalid_rows: tuple[RejectedRow, ...]
    original_columns: tuple[str, ...]
    sample_count: int
    feature_count: int
    entry_count: int
    elapsed_seconds: float
    matrix_path: Path
    features_path: Path
    samples_path: Path


@dataclass(frozen=True, slots=True)
class MatrixFiles:
    """Matrix files and dimensions produced by KMC dump."""

    sample_count: int
    feature_count: int
    entry_count: int
    elapsed_seconds: float
    matrix_path: Path
    features_path: Path
    samples_path: Path


def read_matrix_rows(
    feature_results_path: Path,
) -> tuple[tuple[MatrixRow, ...], tuple[RejectedRow, ...], tuple[str, ...]]:
    """Read feature results and choose rows that can become matrix samples.

    Parameters
    ----------
    feature_results_path : pathlib.Path
        ``feature_results.tsv`` written by ``extract_kmer_features``.

    Returns
    -------
    tuple
        Matrix rows, rejected rows, and original column order.

    Raises
    ------
    InputError
        If the input has no header or a required column is missing.
    """
    rows, original_columns = read_validated_tsv(
        feature_results_path,
        REQUIRED_COLUMNS,
        file_label="feature_results.tsv",
    )

    matrix_rows: list[MatrixRow] = []
    invalid_rows: list[RejectedRow] = []
    matrix_k = ""
    for row_number, row in enumerate(rows, start=2):
        try:
            matrix_row = parse_matrix_row(row_number, len(matrix_rows) + 1, row)
        except InputError as error:
            invalid_rows.append(
                RejectedRow(row_number=row_number, values=row, reason=str(error)),
            )
        else:
            if matrix_k and matrix_row.values["k"] != matrix_k:
                msg = (
                    "feature_results.tsv contains multiple k values: "
                    f"{matrix_k}, {matrix_row.values['k']}"
                )
                raise InputError(msg)
            matrix_k = matrix_row.values["k"]
            matrix_rows.append(matrix_row)

    return tuple(matrix_rows), tuple(invalid_rows), original_columns


def build_kmer_matrix(
    feature_results_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    kmc_dump: tuple[str, ...] = ("kmc_dump",),
    jobs: int = 1,
) -> KmerMatrix:
    """Build a sparse exact k-mer count matrix from KMC databases.

    Parameters
    ----------
    feature_results_path : path-like
        ``feature_results.tsv`` written by ``extract_kmer_features``.
    output_dir : path-like
        Directory where matrix files will be written.
    kmc_dump : tuple of str, optional
        Command used to run KMC dump.
    jobs : int, optional
        Number of parallel workers for KMC dumps and matrix parsing.

    Returns
    -------
    KmerMatrix
        Matrix paths, dimensions, accepted rows, and rejected rows.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrix_rows, invalid_rows, original_columns = read_matrix_rows(Path(feature_results_path))
    check_executable(kmc_dump, MatrixBuildError)

    write_invalid_rows(output, invalid_rows, original_columns)
    matrix = write_sparse_matrix(matrix_rows, output, kmc_dump=kmc_dump, jobs=jobs)
    write_rows(
        matrix.samples_path,
        (sample.to_row(original_columns) for sample in matrix_rows),
        ("sample_index", *original_columns),
    )
    return KmerMatrix(
        samples=matrix_rows,
        invalid_rows=invalid_rows,
        original_columns=original_columns,
        sample_count=matrix.sample_count,
        feature_count=matrix.feature_count,
        entry_count=matrix.entry_count,
        elapsed_seconds=matrix.elapsed_seconds,
        matrix_path=matrix.matrix_path,
        features_path=matrix.features_path,
        samples_path=matrix.samples_path,
    )


def parse_matrix_row(
    row_number: int,
    sample_index: int,
    row: dict[str, str],
) -> MatrixRow:
    """Convert one feature-result row into a matrix sample.

    Parameters
    ----------
    row_number : int
        One-based line number in the TSV file, including the header row.
    sample_index : int
        One-based matrix row index.
    row : dict of str to str
        K-mer result values keyed by column name.

    Returns
    -------
    MatrixRow
        Matrix sample for the row.

    Raises
    ------
    InputError
        If the row is not usable for matrix building.
    """
    run_accession = row["run_accession"].strip()
    if not run_accession:
        msg = "run_accession is empty"
        raise InputError(msg)
    status = row["kmer_status"].strip()
    if status not in ACCEPTED_FEATURE_STATUSES:
        msg = f"feature status is {status or 'empty'}"
        raise InputError(msg)
    k = row["k"].strip()
    if not k:
        msg = "k is empty"
        raise InputError(msg)

    database_prefix = Path(row["kmc_database"])
    if not kmc_database_exists(database_prefix):
        msg = f"KMC database does not exist: {database_prefix}"
        raise InputError(msg)

    return MatrixRow(
        row_number=row_number,
        sample_index=sample_index,
        values={**row, "run_accession": run_accession, "k": k},
        database_prefix=database_prefix,
    )


def write_sparse_matrix(
    samples: Sequence[MatrixRow],
    output_dir: Path,
    *,
    kmc_dump: tuple[str, ...],
    jobs: int = 1,
) -> MatrixFiles:
    """Write sparse matrix, feature, and sample files.

    Parameters
    ----------
    samples : sequence of MatrixRow
        Matrix rows to write.
    output_dir : pathlib.Path
        Output directory.
    kmc_dump : tuple of str
        Command used to run KMC dump.
    jobs : int, optional
        Number of parallel workers for KMC dumps and matrix parsing.

    Returns
    -------
    MatrixFiles
        Output paths and matrix dimensions.
    """
    start = time.perf_counter()
    matrix_path = output_dir / MATRIX_FILE_NAME
    features_path = output_dir / FEATURES_FILE_NAME
    samples_path = output_dir / SAMPLES_FILE_NAME
    vocabulary: dict[str, int] = {}
    data = array("I")
    indices = array("I")
    indptr = array("Q", [0])

    matrix_paths = (matrix_path, features_path, samples_path)
    for path in matrix_paths:
        path.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="kmc_dump_", dir=output_dir) as dump_dir_name:
            dump_dir = Path(dump_dir_name)
            dump_paths = [
                dump_dir / f"{safe_path_name(sample.values['run_accession'])}.txt"
                for sample in samples
            ]
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
                futures = [
                    executor.submit(
                        run_kmc_dump,
                        sample.database_prefix,
                        dump_path,
                        kmc_dump=kmc_dump,
                    )
                    for sample, dump_path in zip(samples, dump_paths, strict=True)
                ]
                for future in futures:
                    future.result()

            total_dump_size = sum(
                dp.stat().st_size for dp in dump_paths if dp.is_file()
            )
            if total_dump_size > PARALLEL_PARSE_MIN_DUMP_SIZE and jobs > 1:
                chunk_size = (len(dump_paths) + jobs - 1) // jobs
                chunks = [
                    dump_paths[i:i + chunk_size]
                    for i in range(0, len(dump_paths), chunk_size)
                ]
                with ProcessPoolExecutor(max_workers=jobs) as executor:
                    worker_results = list(
                        executor.map(_parse_dump_files_worker, chunks)
                    )
                vocabulary, data, indices, indptr = _merge_parse_results(
                    worker_results,
                )
            else:
                for dump_path in dump_paths:
                    for kmer, count in read_kmc_dump(dump_path):
                        feature_index = vocabulary.get(kmer)
                        if feature_index is None:
                            feature_index = len(vocabulary)
                            vocabulary[kmer] = feature_index
                        indices.append(feature_index)
                        data.append(count)
                    indptr.append(len(data))

            for dump_path in dump_paths:
                dump_path.unlink(missing_ok=True)

        write_features(features_path, vocabulary)
        matrix = csr_matrix(
            (
                np.frombuffer(data, dtype=np.uint32),
                np.frombuffer(indices, dtype=np.uint32),
                np.frombuffer(indptr, dtype=np.uint64),
            ),
            shape=(len(samples), len(vocabulary)),
        )
        save_npz(matrix_path, matrix, compressed=False)
    except (MatrixBuildError, OSError):
        for path in matrix_paths:
            path.unlink(missing_ok=True)
        raise

    return MatrixFiles(
        sample_count=len(samples),
        feature_count=len(vocabulary),
        entry_count=len(data),
        elapsed_seconds=time.perf_counter() - start,
        matrix_path=matrix_path,
        features_path=features_path,
        samples_path=samples_path,
    )


def run_kmc_dump(
    database_prefix: Path,
    output_path: Path,
    *,
    kmc_dump: tuple[str, ...],
) -> None:
    """Dump one KMC database to a text file.

    Parameters
    ----------
    database_prefix : pathlib.Path
        KMC database prefix without ``.kmc_pre`` or ``.kmc_suf``.
    output_path : pathlib.Path
        Temporary text dump path.
    kmc_dump : tuple of str
        Command used to run KMC dump.

    Raises
    ------
    MatrixBuildError
        If KMC dump exits with a non-zero status.
    """
    output_path.unlink(missing_ok=True)
    command = (*kmc_dump, str(database_prefix), str(output_path))
    completed = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        msg = format_process_error("kmc_dump", completed.returncode, completed.stderr)
        raise MatrixBuildError(msg)


def read_kmc_dump(path: Path) -> Iterable[tuple[str, int]]:
    """Read k-mer counts from a KMC dump file.

    Parameters
    ----------
    path : pathlib.Path
        Text file written by KMC dump.

    Yields
    ------
    tuple of str and int
        K-mer and count.

    Raises
    ------
    MatrixBuildError
        If a dump line is malformed.
    """
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) != KMC_DUMP_FIELD_COUNT:
                msg = f"Malformed KMC dump line {line_number} in {path}"
                raise MatrixBuildError(msg)
            try:
                yield parts[0], int(parts[1])
            except ValueError as error:
                msg = f"Malformed KMC count on line {line_number} in {path}"
                raise MatrixBuildError(msg) from error


def _parse_dump_files_worker(
    dump_paths: list[Path],
) -> tuple[list[tuple[str, int]], bytes, bytes, list[int]]:
    """Parse dump files in a worker process.

    Returns
    -------
    tuple
        Vocabulary items (kmer, local_index sorted by index), data bytes,
        indices bytes, and indptr list.
    """
    local_vocab: dict[str, int] = {}
    data = array("I")
    indices = array("I")
    indptr = array("Q", [0])

    for dump_path in dump_paths:
        for kmer, count in read_kmc_dump(dump_path):
            local_index = local_vocab.get(kmer)
            if local_index is None:
                local_index = len(local_vocab)
                local_vocab[kmer] = local_index
            indices.append(local_index)
            data.append(count)
        indptr.append(len(data))

    vocab_items = sorted(local_vocab.items(), key=lambda item: item[1])
    return vocab_items, bytes(data), bytes(indices), list(indptr)


def _merge_parse_results(
    worker_results: list[tuple[list[tuple[str, int]], bytes, bytes, list[int]]],
) -> tuple[dict[str, int], array, array, array]:
    """Merge parallel parse results into a global vocabulary and arrays.

    Processes workers in order so that the global vocabulary matches the
    sequential first-seen ordering.
    """
    global_vocab: dict[str, int] = {}
    all_data = array("I")
    all_indices = array("I")
    all_indptr = array("Q", [0])
    offset = 0

    for vocab_items, data_bytes, indices_bytes, indptr_list in worker_results:
        n_local = len(vocab_items)
        local_to_global = np.empty(n_local, dtype=np.uint32)
        for kmer, local_index in vocab_items:
            global_index = global_vocab.get(kmer)
            if global_index is None:
                global_index = len(global_vocab)
                global_vocab[kmer] = global_index
            local_to_global[local_index] = global_index

        local_indices = np.frombuffer(indices_bytes, dtype=np.uint32)
        remapped = local_to_global[local_indices]

        all_data.frombytes(data_bytes)
        all_indices.frombytes(remapped.tobytes())

        for i in range(1, len(indptr_list)):
            all_indptr.append(offset + indptr_list[i])
        offset += len(data_bytes) // 4

    return global_vocab, all_data, all_indices, all_indptr


def write_features(path: Path, vocabulary: dict[str, int]) -> None:
    """Write feature indices and k-mers.

    Parameters
    ----------
    path : pathlib.Path
        Output TSV path.
    vocabulary : dict of str to int
        K-mer to zero-based feature index.
    """
    rows = ({"feature_index": str(index), "kmer": kmer} for kmer, index in vocabulary.items())
    write_rows(path, rows, ("feature_index", "kmer"))
