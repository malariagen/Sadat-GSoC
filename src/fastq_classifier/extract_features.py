"""Extract exact canonical k-mer features with KMC.

The input is the ``fetch_results.tsv`` file written by the first-N fetcher.
Rows with completed or skipped FASTQ downloads are counted with KMC.
Rows that cannot be counted are written to ``invalid_rows.tsv``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastq_classifier.utils import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    InputError,
    RejectedRow,
    check_executable,
    format_process_error,
    kmc_database_exists,
    partition_rows,
    read_validated_tsv,
    run_ordered,
    safe_path_name,
    write_invalid_rows,
    write_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

ACCEPTED_FETCH_STATUSES = frozenset(("completed", "skipped"))
KMC_COUNTER_MAX = 1_000_000_000
KMC_MIN_MEMORY_GB = 2
KMER_MIN_LENGTH = 1
KMER_MAX_LENGTH = 256
REQUIRED_COLUMNS = ("run_accession", "output_r1", "output_r2", "status")
KMER_COLUMNS = (
    "row_number",
    "k",
    "r1_input",
    "r2_input",
    "kmc_database",
    "kmc_stats",
)
RESULT_COLUMNS = (
    "kmer_status",
    "unique_kmers",
    "total_kmers",
    "total_reads",
    "kmer_elapsed_seconds",
    "kmc_pre_bytes",
    "kmc_suf_bytes",
    "kmer_error",
)
__all__ = [
    "KmerDatabase",
    "KmerExtraction",
    "KmerExtractionError",
    "KmerStats",
    "extract_kmer_features",
]


class KmerExtractionError(Exception):
    """K-mer feature extraction failed."""


@dataclass(frozen=True, slots=True)
class KmerJob:
    """One run ready for k-mer counting.

    Attributes
    ----------
    row_number : int
        One-based line number in the TSV file, including the header row.
    values : dict of str to str
        Original TSV values keyed by column name.
    k : int
        K-mer length.
    r1_input, r2_input : pathlib.Path
        Input FASTQ files for mate 1 and mate 2.
    database_prefix : pathlib.Path
        KMC database prefix without ``.kmc_pre`` or ``.kmc_suf``.
    stats_path : pathlib.Path
        Path where KMC writes JSON execution statistics.
    """

    row_number: int
    values: dict[str, str]
    k: int
    r1_input: Path
    r2_input: Path
    database_prefix: Path
    stats_path: Path


@dataclass(frozen=True, slots=True)
class KmerStats:
    """KMC count summary.

    Attributes
    ----------
    unique_kmers : int
        Number of unique counted k-mers in the KMC database.
    total_kmers : int
        Total number of k-mers processed by KMC.
    total_reads : int
        Total number of reads processed by KMC.
    """

    unique_kmers: int
    total_kmers: int
    total_reads: int


@dataclass(frozen=True, slots=True)
class KmerDatabase:
    """One KMC database from ``extract_kmer_features``.

    Attributes
    ----------
    run_accession : str
        ENA run accession.
    status : str
        ``"completed"``, ``"skipped"``, or ``"failed"``.
    stats : KmerStats or None
        KMC summary for successful runs.
    elapsed_seconds : float
        Wall-clock seconds spent counting the run.
    kmc_pre_bytes, kmc_suf_bytes : int
        Sizes of the KMC database files. Failed runs report ``0``.
    error : str
        Failure message, or an empty string when counting succeeded.
    """

    row_number: int
    values: dict[str, str]
    run_accession: str
    k: int
    r1_input: Path
    r2_input: Path
    database_prefix: Path
    stats_path: Path
    status: str
    stats: KmerStats | None
    elapsed_seconds: float
    kmc_pre_bytes: int
    kmc_suf_bytes: int
    error: str

    def to_row(self, original_columns: Sequence[str]) -> dict[str, str]:
        """Build the row written to ``feature_results.tsv``.

        Parameters
        ----------
        original_columns : sequence of str
            Input columns, in their original order.

        Returns
        -------
        dict of str to str
            Original values plus KMC paths and count results.
        """
        row = {column: self.values.get(column, "") for column in original_columns}
        stats = self.stats
        row.update(
            {
                "row_number": str(self.row_number),
                "k": str(self.k),
                "r1_input": str(self.r1_input),
                "r2_input": str(self.r2_input),
                "kmc_database": str(self.database_prefix),
                "kmc_stats": str(self.stats_path),
                "kmer_status": self.status,
                "unique_kmers": "" if stats is None else str(stats.unique_kmers),
                "total_kmers": "" if stats is None else str(stats.total_kmers),
                "total_reads": "" if stats is None else str(stats.total_reads),
                "kmer_elapsed_seconds": f"{self.elapsed_seconds:.3f}",
                "kmc_pre_bytes": str(self.kmc_pre_bytes),
                "kmc_suf_bytes": str(self.kmc_suf_bytes),
                "kmer_error": self.error,
            },
        )
        return row


@dataclass(frozen=True, slots=True)
class KmerExtraction:
    """KMC databases and rejected rows from ``extract_kmer_features``."""

    databases: tuple[KmerDatabase, ...]
    invalid_rows: tuple[RejectedRow, ...]
    original_columns: tuple[str, ...]


def read_kmer_jobs(
    fetch_results_path: Path,
    output_dir: Path,
    k: int,
) -> tuple[tuple[KmerJob, ...], tuple[RejectedRow, ...], tuple[str, ...]]:
    """Read fetch results and choose rows that can be counted.

    Parameters
    ----------
    fetch_results_path : pathlib.Path
        ``fetch_results.tsv`` written by ``fetch_first_n``.
    output_dir : pathlib.Path
        Root directory used to build KMC database paths.
    k : int
        K-mer length.

    Returns
    -------
    tuple
        KMC jobs, rejected rows, and original column order.

    Raises
    ------
    InputError
        If ``k`` is outside the KMC-supported range, the input has no header,
        or a required column is missing.
    """
    check_k(k)
    rows, original_columns = read_validated_tsv(
        fetch_results_path,
        REQUIRED_COLUMNS,
        file_label="fetch_results.tsv",
    )

    def parse(row_number: int, row: dict[str, str]) -> KmerJob:
        return kmer_job_from_row(row_number, row, output_dir, k)

    kmer_jobs, invalid_rows = partition_rows(rows, parse)
    return kmer_jobs, invalid_rows, original_columns


def extract_kmer_features(
    fetch_results_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    k: int,
    *,
    jobs: int = 1,
    kmc: tuple[str, ...] = ("kmc",),
    memory_gb: int = KMC_MIN_MEMORY_GB,
) -> KmerExtraction:
    """Count exact canonical k-mers for fetched paired FASTQs.

    The function writes ``invalid_rows.tsv`` for input rows that cannot be
    counted and ``feature_results.tsv`` for completed, skipped, and failed KMC
    runs.

    Parameters
    ----------
    fetch_results_path : path-like
        ``fetch_results.tsv`` written by ``fetch_first_n``.
    output_dir : path-like
        Directory for KMC databases and summary TSV files.
    k : int
        K-mer length.
    jobs : int, optional
        Number of runs to count at once.
    kmc : tuple of str, optional
        Command used to run KMC.
    memory_gb : int, optional
        Memory limit passed to each KMC process.

    Returns
    -------
    KmerExtraction
        KMC databases, rejected rows, and original TSV column order.
    """
    output = Path(output_dir)
    kmer_jobs, invalid_rows, original_columns = read_kmer_jobs(Path(fetch_results_path), output, k)

    output.mkdir(parents=True, exist_ok=True)
    write_invalid_rows(output, invalid_rows, original_columns)

    databases = count_runs(kmer_jobs, workers=jobs, kmc=kmc, memory_gb=memory_gb)
    write_rows(
        output / "feature_results.tsv",
        (database.to_row(original_columns) for database in databases),
        (*original_columns, *KMER_COLUMNS, *RESULT_COLUMNS),
    )
    return KmerExtraction(
        databases=databases,
        invalid_rows=invalid_rows,
        original_columns=original_columns,
    )


def count_runs(
    kmer_jobs: Sequence[KmerJob],
    *,
    workers: int,
    kmc: tuple[str, ...] = ("kmc",),
    memory_gb: int = KMC_MIN_MEMORY_GB,
) -> tuple[KmerDatabase, ...]:
    """Count k-mers for paired-end runs.

    Parameters
    ----------
    kmer_jobs : sequence of KmerJob
        Runs to count.
    workers : int
        Number of runs to count at once.
    kmc : tuple of str, optional
        Command used to run KMC.
    memory_gb : int, optional
        Memory limit passed to each KMC process.

    Returns
    -------
    tuple of KmerDatabase
        KMC databases in the same order as ``kmer_jobs``.

    Raises
    ------
    InputError
        If ``workers`` or ``memory_gb`` is too small.
    KmerExtractionError
        If KMC cannot be found on ``PATH``.
    """
    if memory_gb < KMC_MIN_MEMORY_GB:
        msg = f"memory_gb must be at least {KMC_MIN_MEMORY_GB}"
        raise InputError(msg)
    check_executable(kmc, KmerExtractionError)
    threads = max(1, (os.cpu_count() or 1) // max(1, workers))
    return run_ordered(
        kmer_jobs,
        count_one_run,
        workers=workers,
        kmc=kmc,
        memory_gb=memory_gb,
        threads=threads,
    )


def kmer_job_from_row(
    row_number: int,
    row: dict[str, str],
    output_dir: Path,
    k: int,
) -> KmerJob:
    """Convert one fetch-result row into a KMC job.

    Parameters
    ----------
    row_number : int
        One-based line number in the TSV file, including the header row.
    row : dict of str to str
        Fetch-result values keyed by column name.
    output_dir : pathlib.Path
        Root directory used to build KMC output paths.
    k : int
        K-mer length.

    Returns
    -------
    KmerJob
        KMC job for the row.

    Raises
    ------
    InputError
        If the row is not usable for feature extraction.
    """
    run_accession = row["run_accession"].strip()
    if not run_accession:
        msg = "run_accession is empty"
        raise InputError(msg)
    status = row["status"].strip()
    if status not in ACCEPTED_FETCH_STATUSES:
        msg = f"fetch status is {status or 'empty'}"
        raise InputError(msg)

    r1_input = Path(row["output_r1"])
    r2_input = Path(row["output_r2"])
    for path in (r1_input, r2_input):
        if not path.is_file():
            msg = f"FASTQ file does not exist: {path}"
            raise InputError(msg)

    run_name = safe_path_name(run_accession)
    run_dir = output_dir / "runs" / run_name
    database_prefix = run_dir / f"{run_name}.k{k}"
    return KmerJob(
        row_number=row_number,
        values={**row, "run_accession": run_accession},
        k=k,
        r1_input=r1_input,
        r2_input=r2_input,
        database_prefix=database_prefix,
        stats_path=database_prefix.with_suffix(".stats.json"),
    )


def count_one_run(
    job: KmerJob,
    *,
    kmc: tuple[str, ...] = ("kmc",),
    memory_gb: int = KMC_MIN_MEMORY_GB,
    threads: int = 1,
) -> KmerDatabase:
    """Count k-mers for one run.

    If complete KMC outputs already exist, the run is reported as
    ``"skipped"``. KMC failures are returned as ``KmerDatabase(status="failed")``
    instead of being raised.

    Parameters
    ----------
    job : KmerJob
        Run to count.
    kmc : tuple of str, optional
        Command used to run KMC.
    memory_gb : int, optional
        Memory limit passed to KMC.
    threads : int, optional
        Number of threads passed to KMC.

    Returns
    -------
    KmerDatabase
        Status, KMC summary, file sizes, and any failure message.
    """
    start = time.perf_counter()
    try:
        if kmc_outputs_exist(job):
            return kmer_database(job, STATUS_SKIPPED, start)

        run_kmc(job, kmc=kmc, memory_gb=memory_gb, threads=threads)
        return kmer_database(job, STATUS_COMPLETED, start)
    except (KmerExtractionError, OSError) as error:
        cleanup_partial_outputs(job)
        return KmerDatabase(
            row_number=job.row_number,
            values=job.values,
            run_accession=job.values["run_accession"],
            k=job.k,
            r1_input=job.r1_input,
            r2_input=job.r2_input,
            database_prefix=job.database_prefix,
            stats_path=job.stats_path,
            status=STATUS_FAILED,
            stats=None,
            elapsed_seconds=time.perf_counter() - start,
            kmc_pre_bytes=0,
            kmc_suf_bytes=0,
            error=str(error),
        )


def run_kmc(job: KmerJob, *, kmc: tuple[str, ...], memory_gb: int, threads: int) -> None:
    """Run KMC for one paired FASTQ run.

    Parameters
    ----------
    job : KmerJob
        Run to count.
    kmc : tuple of str
        Command used to run KMC.
    memory_gb : int
        Memory limit passed to KMC.
    threads : int
        Number of threads passed to KMC.

    Raises
    ------
    KmerExtractionError
        If KMC exits with a non-zero status.
    """
    job.database_prefix.parent.mkdir(parents=True, exist_ok=True)
    cleanup_partial_outputs(job)

    with tempfile.TemporaryDirectory(
        prefix=f"{job.database_prefix.name}.",
        dir=job.database_prefix.parent,
    ) as work_dir_name:
        work_dir = Path(work_dir_name)
        input_list = work_dir / "inputs.txt"
        input_list.write_text(f"{job.r1_input}\n{job.r2_input}\n", encoding="utf-8")
        command = (
            *kmc,
            f"-k{job.k}",
            "-fq",
            "-ci1",
            f"-cs{KMC_COUNTER_MAX}",
            f"-m{memory_gb}",
            f"-t{threads}",
            "-hp",
            f"-j{job.stats_path}",
            f"@{input_list}",
            str(job.database_prefix),
            str(work_dir),
        )
        completed = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            msg = format_process_error("kmc", completed.returncode, completed.stderr)
            raise KmerExtractionError(msg)


def kmer_database(job: KmerJob, status: str, start: float) -> KmerDatabase:
    """Read KMC output metadata for a completed or skipped run.

    Parameters
    ----------
    job : KmerJob
        Run whose KMC outputs exist.
    status : str
        Status to record.
    start : float
        ``time.perf_counter`` value captured before KMC began.

    Returns
    -------
    KmerDatabase
        Result with KMC stats, file sizes, and elapsed time.
    """
    stats = read_kmc_stats(job.stats_path)
    return KmerDatabase(
        row_number=job.row_number,
        values=job.values,
        run_accession=job.values["run_accession"],
        k=job.k,
        r1_input=job.r1_input,
        r2_input=job.r2_input,
        database_prefix=job.database_prefix,
        stats_path=job.stats_path,
        status=status,
        stats=stats,
        elapsed_seconds=time.perf_counter() - start,
        kmc_pre_bytes=Path(f"{job.database_prefix}.kmc_pre").stat().st_size,
        kmc_suf_bytes=Path(f"{job.database_prefix}.kmc_suf").stat().st_size,
        error="",
    )


def read_kmc_stats(path: Path) -> KmerStats:
    """Read KMC JSON statistics.

    Parameters
    ----------
    path : pathlib.Path
        KMC JSON stats file.

    Returns
    -------
    KmerStats
        Unique k-mers, total k-mers, and total reads.
    """
    data = cast("object", json.loads(path.read_text(encoding="utf-8")))
    stats = require_mapping(data, "KMC stats file", path).get("Stats")
    stats_mapping = require_mapping(stats, "KMC Stats field", path)
    return KmerStats(
        unique_kmers=read_int_stat(stats_mapping, "#Unique_counted_k-mers", path),
        total_kmers=read_int_stat(stats_mapping, "#Total no. of k-mers", path),
        total_reads=read_int_stat(stats_mapping, "#Total_reads", path),
    )


def require_mapping(value: object, label: str, path: Path) -> Mapping[object, object]:
    """Require a parsed JSON value to be an object.

    Parameters
    ----------
    value : object
        Parsed JSON value.
    label : str
        Value name used in error messages.
    path : pathlib.Path
        Source JSON path used in error messages.

    Returns
    -------
    mapping of object to object
        Parsed JSON object.

    Raises
    ------
    KmerExtractionError
        If ``value`` is not a mapping.
    """
    if isinstance(value, Mapping):
        return cast("Mapping[object, object]", value)
    msg = f"{label} is not a JSON object: {path}"
    raise KmerExtractionError(msg)


def read_int_stat(stats: Mapping[object, object], key: str, path: Path) -> int:
    """Read one integer from a KMC stats object.

    Parameters
    ----------
    stats : mapping of object to object
        Parsed ``Stats`` object from KMC JSON.
    key : str
        Statistic key to read.
    path : pathlib.Path
        Source JSON path used in error messages.

    Returns
    -------
    int
        Parsed integer statistic.

    Raises
    ------
    KmerExtractionError
        If the key is missing or does not contain an integer.
    """
    value = stats.get(key)
    if isinstance(value, int):
        return value
    msg = f"KMC stats file has no integer {key!r}: {path}"
    raise KmerExtractionError(msg)


def kmc_outputs_exist(job: KmerJob) -> bool:
    """Check whether all reusable KMC output files exist.

    Parameters
    ----------
    job : KmerJob
        Run whose output paths are checked.

    Returns
    -------
    bool
        ``True`` when the KMC database and stats files exist.
    """
    return kmc_database_exists(job.database_prefix) and job.stats_path.is_file()


def cleanup_partial_outputs(job: KmerJob) -> None:
    """Remove KMC output files for one run.

    Parameters
    ----------
    job : KmerJob
        Run whose output files should be removed.
    """
    for path in (
        Path(f"{job.database_prefix}.kmc_pre"),
        Path(f"{job.database_prefix}.kmc_suf"),
        job.stats_path,
    ):
        path.unlink(missing_ok=True)


def check_k(k: int) -> None:
    """Validate a k-mer length.

    Parameters
    ----------
    k : int
        K-mer length.

    Raises
    ------
    InputError
        If ``k`` is outside the KMC-supported range.
    """
    if KMER_MIN_LENGTH <= k <= KMER_MAX_LENGTH:
        return
    msg = f"k must be between {KMER_MIN_LENGTH} and {KMER_MAX_LENGTH}"
    raise InputError(msg)
