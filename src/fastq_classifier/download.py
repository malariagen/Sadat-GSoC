"""Download paired FASTQ reads listed in an ENA report."""

from __future__ import annotations

import csv
import gzip
import io
import re
import shutil
import tempfile
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from http.client import IncompleteRead
from pathlib import Path
from typing import BinaryIO, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from fastq_classifier.features import DEFAULT_READ_PAIRS

DEFAULT_DOWNLOAD_JOBS = 4

_RUN_ACCESSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_FASTQ_READ_BUFFER_BYTES = 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 60
_RETRY_DELAYS_SECONDS = (1, 2, 4)


@dataclass(frozen=True, slots=True)
class _EnaRun:
    run_accession: str
    read1_url: str
    read2_url: str


def download_read_pairs(
    ena_report: str | Path,
    download_dir: str | Path,
    *,
    read_pairs: int = DEFAULT_READ_PAIRS,
    jobs: int = DEFAULT_DOWNLOAD_JOBS,
) -> Path:
    """Download the requested read-pair prefix from every run in an ENA report."""
    if read_pairs <= 0:
        raise ValueError(f"read_pairs must be positive, got {read_pairs}")
    if jobs <= 0:
        raise ValueError(f"jobs must be positive, got {jobs}")

    ena_runs = _read_ena_runs(Path(ena_report))
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    download_one_run = partial(
        _download_ena_run,
        download_dir=download_path,
        read_pairs=read_pairs,
    )
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for _ in pool.map(download_one_run, ena_runs):
            pass

    return _write_fastq_manifest(download_path, ena_runs, read_pairs)


def _read_ena_runs(report_path: Path) -> tuple[_EnaRun, ...]:
    with report_path.open(encoding="utf-8-sig", newline="") as report_stream:
        report_rows = csv.DictReader(report_stream, delimiter="\t")
        missing_columns = {"run_accession", "fastq_ftp"} - set(report_rows.fieldnames or ())
        if missing_columns:
            column_names = ", ".join(sorted(missing_columns))
            raise ValueError(f"ENA report {report_path} is missing columns: {column_names}")

        ena_runs: list[_EnaRun] = []
        seen_accessions: set[str] = set()
        for report_row in report_rows:
            if not any((field_value or "").strip() for field_value in report_row.values()):
                continue

            line_number = report_rows.line_num
            run_accession = (report_row["run_accession"] or "").strip()
            if _RUN_ACCESSION_PATTERN.fullmatch(run_accession) is None:
                raise ValueError(
                    f"ENA report {report_path}, line {line_number}: invalid run accession"
                )
            if run_accession in seen_accessions:
                raise ValueError(
                    f"ENA report {report_path}, line {line_number}: duplicate run {run_accession}"
                )

            fastq_urls = [
                fastq_url.strip() for fastq_url in (report_row["fastq_ftp"] or "").split(";")
            ]
            if len(fastq_urls) != 2 or not all(fastq_urls):
                raise ValueError(
                    f"ENA report {report_path}, line {line_number}: "
                    f"{run_accession} must have two FASTQ URLs"
                )

            ena_runs.append(
                _EnaRun(
                    run_accession,
                    _ena_fastq_url(fastq_urls[0], report_path, line_number),
                    _ena_fastq_url(fastq_urls[1], report_path, line_number),
                )
            )
            seen_accessions.add(run_accession)

    if not ena_runs:
        raise ValueError(f"ENA report {report_path} contains no runs")
    return tuple(ena_runs)


def _ena_fastq_url(address: str, report_path: Path, line_number: int) -> str:
    if address.startswith("ftp://"):
        address = f"https://{address[6:]}"
    elif "://" not in address:
        address = f"https://{address}"

    parsed_url = urlsplit(address)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError(
            f"ENA report {report_path}, line {line_number}: invalid FASTQ URL {address!r}"
        )
    if parsed_url.fragment or not parsed_url.path.lower().endswith(".fastq.gz"):
        raise ValueError(
            f"ENA report {report_path}, line {line_number}: invalid FASTQ URL {address!r}"
        )
    return address


def _download_ena_run(ena_run: _EnaRun, *, download_dir: Path, read_pairs: int) -> None:
    run_dir = download_dir / ena_run.run_accession
    if run_dir.exists():
        _validate_downloaded_run(run_dir, ena_run.run_accession, read_pairs)
        return

    pending_run_dir = Path(tempfile.mkdtemp(prefix=f".{ena_run.run_accession}.", dir=download_dir))
    read1_path = pending_run_dir / f"{ena_run.run_accession}_1.fastq.gz"
    read2_path = pending_run_dir / f"{ena_run.run_accession}_2.fastq.gz"
    try:
        _download_fastq(ena_run.read1_url, read1_path, read_pairs)
        _download_fastq(ena_run.read2_url, read2_path, read_pairs)
        _validate_read_pair(read1_path, read2_path, read_pairs)
        pending_run_dir.replace(run_dir)
    finally:
        shutil.rmtree(pending_run_dir, ignore_errors=True)


def _download_fastq(url: str, destination_path: Path, read_pairs: int) -> None:
    for request_attempt in range(len(_RETRY_DELAYS_SECONDS) + 1):
        try:
            _download_fastq_once(url, destination_path, read_pairs)
            return
        except HTTPError as error:
            if error.code not in {408, 429} and not 500 <= error.code < 600:
                raise OSError(f"Could not download {url}: HTTP {error.code}") from error
            last_download_error: Exception = error
        except (URLError, TimeoutError, ConnectionError, IncompleteRead, EOFError) as error:
            last_download_error = error

        if request_attempt == len(_RETRY_DELAYS_SECONDS):
            request_count = len(_RETRY_DELAYS_SECONDS) + 1
            raise OSError(
                f"Could not download {url} after {request_count} attempts: {last_download_error}"
            ) from last_download_error
        time.sleep(_RETRY_DELAYS_SECONDS[request_attempt])


def _download_fastq_once(url: str, destination_path: Path, read_pairs: int) -> None:
    with _open_fastq_response(url) as fastq_response:
        try:
            with (
                io.BufferedReader(
                    gzip.GzipFile(fileobj=fastq_response, mode="rb"),
                    buffer_size=_FASTQ_READ_BUFFER_BYTES,
                ) as decompressed_fastq,
                gzip.open(destination_path, mode="wb") as compressed_fastq,
            ):
                for read_number in range(1, read_pairs + 1):
                    compressed_fastq.writelines(
                        _read_fastq_record(decompressed_fastq, url, read_number)
                    )
        except (EOFError, gzip.BadGzipFile) as error:
            raise ValueError(f"Invalid FASTQ data from {url}: {error}") from error


@contextmanager
def _open_fastq_response(url: str) -> Generator[BinaryIO, None, None]:
    request = Request(url, headers={"User-Agent": "fastq-classifier/0.1"})
    with urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as fastq_response:
        yield cast(BinaryIO, fastq_response)


def _read_fastq_record(
    fastq_stream: io.BufferedReader[gzip.GzipFile] | gzip.GzipFile,
    fastq_name: str,
    read_number: int,
) -> tuple[bytes, bytes, bytes, bytes]:
    header = fastq_stream.readline()
    sequence = fastq_stream.readline()
    separator = fastq_stream.readline()
    quality = fastq_stream.readline()

    if not header:
        raise ValueError(f"{fastq_name} ended before read {read_number}")
    if not sequence or not separator or not quality:
        raise ValueError(f"{fastq_name} has an incomplete read at read {read_number}")

    fastq_record = (header, sequence, separator, quality)
    if any(not record_line.endswith(b"\n") for record_line in fastq_record):
        raise ValueError(f"{fastq_name} has an unterminated line at read {read_number}")
    if not header.startswith(b"@") or not header[1:].strip():
        raise ValueError(f"{fastq_name} has an invalid read name at read {read_number}")
    if not separator.startswith(b"+"):
        raise ValueError(f"{fastq_name} has an invalid separator at read {read_number}")

    sequence_length = len(sequence.rstrip(b"\r\n"))
    quality_length = len(quality.rstrip(b"\r\n"))
    if sequence_length == 0:
        raise ValueError(f"{fastq_name} has an empty sequence at read {read_number}")
    if sequence_length != quality_length:
        raise ValueError(
            f"{fastq_name} has different sequence and quality lengths at read {read_number}"
        )
    return fastq_record


def _validate_downloaded_run(run_dir: Path, run_accession: str, read_pairs: int) -> None:
    read1_path = run_dir / f"{run_accession}_1.fastq.gz"
    read2_path = run_dir / f"{run_accession}_2.fastq.gz"
    if {run_file.name for run_file in run_dir.iterdir()} != {
        read1_path.name,
        read2_path.name,
    }:
        raise ValueError(f"Run directory {run_dir} is incomplete or contains unexpected files")

    _validate_read_pair(read1_path, read2_path, read_pairs)


def _validate_read_pair(read1_path: Path, read2_path: Path, read_pairs: int) -> None:
    try:
        with (
            gzip.open(read1_path, mode="rb") as read1_stream,
            gzip.open(read2_path, mode="rb") as read2_stream,
        ):
            for read_number in range(1, read_pairs + 1):
                read1_record = _read_fastq_record(read1_stream, str(read1_path), read_number)
                read2_record = _read_fastq_record(read2_stream, str(read2_path), read_number)
                if _read_identifier(read1_record[0]) != _read_identifier(read2_record[0]):
                    raise ValueError(f"{read1_path} and {read2_path} differ at read {read_number}")
            if read1_stream.read(1) or read2_stream.read(1):
                raise ValueError(f"FASTQ pair contains more than {read_pairs} reads")
    except (EOFError, OSError) as error:
        raise ValueError(f"Invalid FASTQ pair {read1_path}, {read2_path}: {error}") from error


def _read_identifier(header: bytes) -> bytes:
    identifier = header[1:].split(maxsplit=1)[0]
    return identifier[:-2] if identifier.endswith((b"/1", b"/2")) else identifier


def _write_fastq_manifest(
    download_dir: Path,
    ena_runs: tuple[_EnaRun, ...],
    read_pairs: int,
) -> Path:
    manifest_path = download_dir / "fastq_manifest.tsv"
    pending_manifest = download_dir / ".fastq_manifest.tsv.tmp"
    try:
        with pending_manifest.open("w", encoding="utf-8", newline="") as manifest_stream:
            manifest_rows = csv.writer(manifest_stream, delimiter="\t", lineterminator="\n")
            manifest_rows.writerow(("run_accession", "read1_path", "read2_path", "read_pairs"))
            for ena_run in ena_runs:
                run_dir = (download_dir / ena_run.run_accession).resolve()
                manifest_rows.writerow(
                    (
                        ena_run.run_accession,
                        run_dir / f"{ena_run.run_accession}_1.fastq.gz",
                        run_dir / f"{ena_run.run_accession}_2.fastq.gz",
                        read_pairs,
                    )
                )
        pending_manifest.replace(manifest_path)
    finally:
        pending_manifest.unlink(missing_ok=True)
    return manifest_path
