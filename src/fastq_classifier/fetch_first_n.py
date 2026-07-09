"""Download read prefixes from paired-end ENA FASTQ runs.

The input is an ENA run report TSV with ``run_accession`` and ``fastq_ftp``
columns. Valid paired-end rows are downloaded; rows with
missing fields or malformed FASTQ URLs are kept in ``invalid_rows.tsv`` so the
caller can inspect them.

Downloaded files are written under ``<output_dir>/runs/<run_accession>/``.
The result TSV keeps the original ENA report columns and adds the URLs, output
paths, counts, file sizes, and error text produced by this module.
"""

from __future__ import annotations

import gzip
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
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

FASTQ_RECORD_LINES = 4
PAIRED_URL_COUNT = 2
# SeqKit stops reading after the requested number of records. curl reports that
# closed pipe as exit 23, which is expected for this pipeline.
CURL_EARLY_PIPE_CLOSE_EXIT_CODE = 23
REQUIRED_COLUMNS = ("run_accession", "fastq_ftp")
DOWNLOAD_COLUMNS = (
    "row_number",
    "r1_url",
    "r2_url",
    "output_r1",
    "output_r2",
    "requested_read_pairs",
)
RESULT_COLUMNS = (
    "status",
    "written_read_pairs",
    "elapsed_seconds",
    "r1_bytes",
    "r2_bytes",
    "error",
)
__all__ = [
    "DownloadReport",
    "DownloadedRun",
    "FetchError",
    "fetch_first_n",
]


class FetchError(Exception):
    """A download, file check, or external command failed."""


@dataclass(frozen=True, slots=True)
class DownloadJob:
    """A paired-end run ready to download.

    Attributes
    ----------
    row_number : int
        One-based line number in the TSV file, including the header row.
    values : dict of str to str
        Original TSV values keyed by column name.
    r1_url : str
        HTTPS URL for mate 1.
    r2_url : str
        HTTPS URL for mate 2.
    output_r1 : pathlib.Path
        Destination for the mate 1 FASTQ subset.
    output_r2 : pathlib.Path
        Destination for the mate 2 FASTQ subset.
    requested_read_pairs : int
        Number of records requested from each mate.
    """

    row_number: int
    values: dict[str, str]
    r1_url: str
    r2_url: str
    output_r1: Path
    output_r2: Path
    requested_read_pairs: int


@dataclass(frozen=True, slots=True)
class DownloadedRun:
    """One run from ``fetch_first_n``.

    Attributes
    ----------
    run_accession : str
        ENA run accession.
    status : str
        ``"completed"``, ``"skipped"``, or ``"failed"``.
    written_read_pairs : int
        Number of paired records in the output files.
    elapsed_seconds : float
        Wall-clock seconds spent on the download.
    r1_bytes, r2_bytes : int
        Output file sizes in bytes. Failed downloads report ``0``.
    error : str
        Failure message, or an empty string when the download succeeded.
    """

    row_number: int
    values: dict[str, str]
    run_accession: str
    r1_url: str
    r2_url: str
    output_r1: Path
    output_r2: Path
    requested_read_pairs: int
    status: str
    written_read_pairs: int
    elapsed_seconds: float
    r1_bytes: int
    r2_bytes: int
    error: str

    def to_row(self, original_columns: Sequence[str]) -> dict[str, str]:
        """Build the row written to ``fetch_results.tsv``.

        Parameters
        ----------
        original_columns : sequence of str
            ENA report columns, in their original order.

        Returns
        -------
        dict of str to str
            Original values plus download paths and results.
        """
        row = {column: self.values.get(column, "") for column in original_columns}
        row.update(
            {
                "row_number": str(self.row_number),
                "r1_url": self.r1_url,
                "r2_url": self.r2_url,
                "output_r1": str(self.output_r1),
                "output_r2": str(self.output_r2),
                "requested_read_pairs": str(self.requested_read_pairs),
                "status": self.status,
                "written_read_pairs": str(self.written_read_pairs),
                "elapsed_seconds": f"{self.elapsed_seconds:.3f}",
                "r1_bytes": str(self.r1_bytes),
                "r2_bytes": str(self.r2_bytes),
                "error": self.error,
            },
        )
        return row


@dataclass(frozen=True, slots=True)
class DownloadReport:
    """Downloads and rejected rows from ``fetch_first_n``."""

    downloads: tuple[DownloadedRun, ...]
    invalid_rows: tuple[RejectedRow, ...]
    original_columns: tuple[str, ...]


def read_download_jobs(
    report_path: Path,
    output_dir: Path,
    read_pairs: int,
) -> tuple[tuple[DownloadJob, ...], tuple[RejectedRow, ...], tuple[str, ...]]:
    """Read an ENA report and choose the rows that can be downloaded.

    The report must contain ``run_accession`` and ``fastq_ftp``. Each valid
    ``fastq_ftp`` value must contain exactly two semicolon-separated FASTQ
    paths or URLs.

    Parameters
    ----------
    report_path : pathlib.Path
        ENA run report TSV.
    output_dir : pathlib.Path
        Root directory used to build per-run FASTQ output paths.
    read_pairs : int
        Number of records to request from each mate.

    Returns
    -------
    tuple
        Download jobs, rejected rows, and original column order.

    Raises
    ------
    InputError
        If ``read_pairs`` is less than one, the report has no header, or a
        required column is missing.
    """
    if read_pairs < 1:
        msg = "read_pairs must be greater than zero"
        raise InputError(msg)

    rows, original_columns = read_validated_tsv(
        report_path,
        REQUIRED_COLUMNS,
        file_label="ENA report",
    )

    def parse(row_number: int, row: dict[str, str]) -> DownloadJob:
        return download_job_from_row(row_number, row, output_dir, read_pairs)

    jobs, invalid_rows = partition_rows(rows, parse)
    return jobs, invalid_rows, original_columns


def fetch_first_n(
    report_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    read_pairs: int,
    *,
    jobs: int = 1,
    curl: tuple[str, ...] = ("curl",),
    seqkit: tuple[str, ...] = ("seqkit",),
) -> DownloadReport:
    """Download the first records from each paired-end run in an ENA report.

    The function writes ``invalid_rows.tsv`` for rejected input rows and
    ``fetch_results.tsv`` for completed, skipped, and failed downloads.

    Parameters
    ----------
    report_path : path-like
        ENA run report TSV.
    output_dir : path-like
        Directory for FASTQ subsets and summary TSV files.
    read_pairs : int
        Number of records to request from each mate.
    jobs : int, optional
        Number of runs to download at once.
    curl : tuple of str, optional
        Command used to stream remote FASTQ files.
    seqkit : tuple of str, optional
        Command used to keep the first ``read_pairs`` records.

    Returns
    -------
    DownloadReport
        Downloads, rejected rows, and original TSV column order.
    """
    output = Path(output_dir)
    download_jobs, invalid_rows, original_columns = read_download_jobs(
        Path(report_path),
        output,
        read_pairs,
    )

    output.mkdir(parents=True, exist_ok=True)
    write_invalid_rows(output, invalid_rows, original_columns)

    downloads = download_runs(download_jobs, workers=jobs, curl=curl, seqkit=seqkit)
    write_rows(
        output / "fetch_results.tsv",
        (download.to_row(original_columns) for download in downloads),
        (*original_columns, *DOWNLOAD_COLUMNS, *RESULT_COLUMNS),
    )
    return DownloadReport(
        downloads=downloads,
        invalid_rows=invalid_rows,
        original_columns=original_columns,
    )


def download_runs(
    download_jobs: Sequence[DownloadJob],
    *,
    workers: int,
    curl: tuple[str, ...] = ("curl",),
    seqkit: tuple[str, ...] = ("seqkit",),
) -> tuple[DownloadedRun, ...]:
    """Download paired-end runs.

    Parameters
    ----------
    download_jobs : sequence of DownloadJob
        Runs to download.
    workers : int
        Number of runs to download at once.
    curl : tuple of str, optional
        Command used to stream remote FASTQ files.
    seqkit : tuple of str, optional
        Command used to keep the requested number of records.

    Returns
    -------
    tuple of DownloadedRun
        Downloads in the same order as ``download_jobs``.

    Raises
    ------
    InputError
        If ``workers`` is less than one.
    FetchError
        If curl or SeqKit cannot be found on ``PATH``.
    """
    check_tools(curl, seqkit)
    return run_ordered(
        download_jobs,
        download_one,
        workers=workers,
        curl=curl,
        seqkit=seqkit,
    )


def download_job_from_row(
    row_number: int,
    row: dict[str, str],
    output_dir: Path,
    read_pairs: int,
) -> DownloadJob:
    """Convert one ENA report row into a download job.

    Parameters
    ----------
    row_number : int
        One-based line number in the TSV file, including the header row.
    row : dict of str to str
        ENA report values keyed by column name.
    output_dir : pathlib.Path
        Root directory used to build per-run FASTQ output paths.
    read_pairs : int
        Number of records to request from each mate.

    Returns
    -------
    DownloadJob
        Download job for the row.

    Raises
    ------
    InputError
        If the run accession or FASTQ URL list is empty or malformed.
    """
    run_accession = row["run_accession"].strip()
    if not run_accession:
        msg = "run_accession is empty"
        raise InputError(msg)

    r1_url, r2_url = (https_url(url) for url in parse_paired_urls(row["fastq_ftp"]))
    run_name = safe_path_name(run_accession)
    run_dir = output_dir / "runs" / run_name
    return DownloadJob(
        row_number=row_number,
        values={**row, "run_accession": run_accession},
        r1_url=r1_url,
        r2_url=r2_url,
        output_r1=run_dir / f"{run_name}_R1.first_{read_pairs}.fastq.gz",
        output_r2=run_dir / f"{run_name}_R2.first_{read_pairs}.fastq.gz",
        requested_read_pairs=read_pairs,
    )


def parse_paired_urls(value: str) -> tuple[str, str]:
    """Split a paired-end ``fastq_ftp`` field.

    Parameters
    ----------
    value : str
        Semicolon-separated value from the ENA report.

    Returns
    -------
    tuple of str
        Mate 1 and mate 2 FASTQ paths, in report order.

    Raises
    ------
    InputError
        If ``value`` does not contain exactly two non-empty paths.
    """
    urls = tuple(part.strip() for part in value.split(";") if part.strip())
    if len(urls) != PAIRED_URL_COUNT:
        msg = f"fastq_ftp must contain exactly two FASTQ paths, found {len(urls)}"
        raise InputError(msg)
    return urls[0], urls[1]


def https_url(value: str) -> str:
    """Return an HTTPS URL for an ENA FASTQ path.

    Parameters
    ----------
    value : str
        Either an ``https://`` URL or an ``ftp.sra.ebi.ac.uk/`` path.

    Returns
    -------
    str
        URL that curl can fetch over HTTPS.

    Raises
    ------
    InputError
        If ``value`` is not one of the supported ENA URL forms.
    """
    if value.startswith("https://"):
        return value
    if value.startswith("ftp.sra.ebi.ac.uk/"):
        return f"https://{value}"

    msg = f"Unsupported FASTQ URL: {value}"
    raise InputError(msg)


def download_one(
    job: DownloadJob,
    *,
    curl: tuple[str, ...] = ("curl",),
    seqkit: tuple[str, ...] = ("seqkit",),
) -> DownloadedRun:
    """Download both mates for one run.

    If both output files already exist and have the same record count, the download
    is reported as ``"skipped"``. Download failures are returned as
    ``DownloadedRun(status="failed")`` instead of being raised.

    Parameters
    ----------
    job : DownloadJob
        Paired-end run to download.
    curl : tuple of str
        Command used to stream remote FASTQ files.
    seqkit : tuple of str
        Command used to keep the requested number of records.

    Returns
    -------
    DownloadedRun
        Status, counts, file sizes, and any failure message.
    """
    start = time.perf_counter()
    job.output_r1.parent.mkdir(parents=True, exist_ok=True)

    try:
        if job.output_r1.exists() and job.output_r2.exists():
            with ThreadPoolExecutor(max_workers=2) as executor:
                r1_future = executor.submit(count_fastq_records, job.output_r1)
                r2_future = executor.submit(count_fastq_records, job.output_r2)
                r1_count = r1_future.result()
                r2_count = r2_future.result()
            if r1_count != r2_count:
                msg = f"Existing R1/R2 record counts differ: {r1_count} != {r2_count}"
                raise FetchError(msg)  # noqa: TRY301
            return download_from_outputs(job, STATUS_SKIPPED, start, r1_count)

        with tempfile.TemporaryDirectory(
            prefix=f"{safe_path_name(job.values['run_accession'])}.",
            dir=job.output_r1.parent,
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            temp_r1 = temp_dir / job.output_r1.name
            temp_r2 = temp_dir / job.output_r2.name

            with ThreadPoolExecutor(max_workers=2) as executor:
                r1_future = executor.submit(
                    download_prefix,
                    job.r1_url,
                    temp_r1,
                    job.requested_read_pairs,
                    curl=curl,
                    seqkit=seqkit,
                )
                r2_future = executor.submit(
                    download_prefix,
                    job.r2_url,
                    temp_r2,
                    job.requested_read_pairs,
                    curl=curl,
                    seqkit=seqkit,
                )
                r1_future.result()
                r2_future.result()

            with ThreadPoolExecutor(max_workers=2) as executor:
                r1_future = executor.submit(count_fastq_records, temp_r1)
                r2_future = executor.submit(count_fastq_records, temp_r2)
                r1_count = r1_future.result()
                r2_count = r2_future.result()
            if r1_count != r2_count:
                msg = f"R1/R2 record counts differ: {r1_count} != {r2_count}"
                raise FetchError(msg)  # noqa: TRY301

            temp_r1.replace(job.output_r1)
            temp_r2.replace(job.output_r2)
        return download_from_outputs(job, STATUS_COMPLETED, start, r1_count)
    except (FetchError, OSError) as error:
        return DownloadedRun(
            row_number=job.row_number,
            values=job.values,
            run_accession=job.values["run_accession"],
            r1_url=job.r1_url,
            r2_url=job.r2_url,
            output_r1=job.output_r1,
            output_r2=job.output_r2,
            requested_read_pairs=job.requested_read_pairs,
            status=STATUS_FAILED,
            written_read_pairs=0,
            elapsed_seconds=time.perf_counter() - start,
            r1_bytes=0,
            r2_bytes=0,
            error=str(error),
        )


def download_from_outputs(
    job: DownloadJob,
    status: str,
    start: float,
    written_read_pairs: int,
) -> DownloadedRun:
    """Read output metadata for a completed or skipped download.

    Parameters
    ----------
    job : DownloadJob
        Download whose output files exist.
    status : str
        Status to record.
    start : float
        ``time.perf_counter`` value captured before the download began.
    written_read_pairs : int
        Number of paired records in the output files.

    Returns
    -------
    DownloadedRun
        Result with file sizes and elapsed time.
    """
    return DownloadedRun(
        row_number=job.row_number,
        values=job.values,
        run_accession=job.values["run_accession"],
        r1_url=job.r1_url,
        r2_url=job.r2_url,
        output_r1=job.output_r1,
        output_r2=job.output_r2,
        requested_read_pairs=job.requested_read_pairs,
        status=status,
        written_read_pairs=written_read_pairs,
        elapsed_seconds=time.perf_counter() - start,
        r1_bytes=job.output_r1.stat().st_size,
        r2_bytes=job.output_r2.stat().st_size,
        error="",
    )


def download_prefix(
    url: str,
    output_path: Path,
    read_count: int,
    *,
    curl: tuple[str, ...],
    seqkit: tuple[str, ...],
) -> None:
    """Stream one compressed FASTQ through ``seqkit head``.

    Parameters
    ----------
    url : str
        HTTPS URL for a gzipped FASTQ file.
    output_path : pathlib.Path
        Destination for the gzipped FASTQ subset.
    read_count : int
        Number of records to write.
    curl : tuple of str
        Command used to stream the remote file.
    seqkit : tuple of str
        Command used to keep the first ``read_count`` records.

    Raises
    ------
    FetchError
        If curl or SeqKit exits with an unexpected status.
    """
    curl_command = (*curl, "--fail", "--location", "--silent", "--show-error", url)
    seqkit_command = (
        *seqkit,
        "head",
        "--quiet",
        "-j",
        "1",
        "-n",
        str(read_count),
        "-o",
        str(output_path),
        "-",
    )

    curl_process: subprocess.Popen[bytes] = subprocess.Popen(  # noqa: S603
        curl_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if curl_process.stdout is None or curl_process.stderr is None:
        msg = "Failed to open curl pipes"
        raise FetchError(msg)

    try:
        seqkit_process: subprocess.Popen[bytes] = subprocess.Popen(  # noqa: S603
            seqkit_command,
            stdin=curl_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        curl_process.stdout.close()
        _, seqkit_stderr = seqkit_process.communicate()
        curl_stderr = cast("bytes", curl_process.stderr.read())
        curl_process.wait()
    finally:
        if curl_process.poll() is None:
            curl_process.terminate()

    if seqkit_process.returncode != 0:
        msg = format_process_error(
            "seqkit",
            seqkit_process.returncode,
            seqkit_stderr.decode("utf-8", "replace"),
        )
        raise FetchError(msg)
    if curl_process.returncode not in (0, CURL_EARLY_PIPE_CLOSE_EXIT_CODE):
        msg = format_process_error(
            "curl",
            curl_process.returncode,
            curl_stderr.decode("utf-8", "replace"),
        )
        raise FetchError(msg)


def count_fastq_records(path: Path) -> int:
    """Count complete records in a gzipped FASTQ file.

    Parameters
    ----------
    path : pathlib.Path
        FASTQ file to inspect.

    Returns
    -------
    int
        Number of records in ``path``.

    Raises
    ------
    FetchError
        If the file does not contain complete four-line FASTQ records.
    """
    line_count = 0
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
    if line_count % FASTQ_RECORD_LINES != 0:
        msg = f"{path} does not contain a complete FASTQ record set"
        raise FetchError(msg)
    return line_count // FASTQ_RECORD_LINES


def check_tools(curl: tuple[str, ...], seqkit: tuple[str, ...]) -> None:
    """Check that curl and SeqKit commands can be resolved.

    Parameters
    ----------
    curl : tuple of str
        Command used to stream remote FASTQ files.
    seqkit : tuple of str
        Command used to keep the requested number of records.

    Raises
    ------
    FetchError
        If a command executable cannot be found.
    """
    check_executable(curl, FetchError)
    check_executable(seqkit, FetchError)
