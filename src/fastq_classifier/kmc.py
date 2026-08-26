"""Count canonical k-mers with KMC."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast

from fastq_classifier.features import (
    DEFAULT_KMER_SIZE,
    DEFAULT_READ_PAIRS,
    MAXIMUM_KMER_SIZE,
    MINIMUM_KMER_SIZE,
)

DEFAULT_KMC_JOBS = 4

_KMC_COMMAND = ("kmc",)
_KMC_MEMORY_GB = 2
_KMC_THREADS = 1
_MINIMUM_KMER_COUNT = 1
_MAXIMUM_KMER_COUNT = 1_000_000_000


@dataclass(frozen=True, slots=True)
class _FastqRun:
    run_accession: str
    read1_path: Path
    read2_path: Path


@dataclass(frozen=True, slots=True)
class _KmcStatistics:
    unique_kmers: int
    total_kmers: int
    total_reads: int


def count_kmers(
    fastq_manifest: str | Path,
    count_dir: str | Path,
    *,
    k: int = DEFAULT_KMER_SIZE,
    jobs: int = DEFAULT_KMC_JOBS,
) -> Path:
    """Count canonical k-mers for each run in a FASTQ manifest."""
    if not MINIMUM_KMER_SIZE <= k <= MAXIMUM_KMER_SIZE:
        raise ValueError(f"k must be between {MINIMUM_KMER_SIZE} and {MAXIMUM_KMER_SIZE}, got {k}")
    if jobs <= 0:
        raise ValueError(f"jobs must be positive, got {jobs}")

    fastq_runs = _read_fastq_manifest(Path(fastq_manifest))
    kmc_version = _installed_kmc_version()
    count_path = Path(count_dir)
    count_path.mkdir(parents=True, exist_ok=True)

    count_one_run = partial(
        _count_fastq_run,
        count_dir=count_path,
        k=k,
        kmc_version=kmc_version,
    )
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        run_statistics = tuple(pool.map(count_one_run, fastq_runs))

    return _write_kmc_manifest(count_path, fastq_runs, run_statistics, k, kmc_version)


def _read_fastq_manifest(manifest_path: Path) -> tuple[_FastqRun, ...]:
    with manifest_path.open(encoding="utf-8-sig", newline="") as manifest_stream:
        manifest_rows = csv.DictReader(manifest_stream, delimiter="\t")
        required_columns = {"run_accession", "read1_path", "read2_path"}
        missing_columns = required_columns - set(manifest_rows.fieldnames or ())
        if missing_columns:
            column_names = ", ".join(sorted(missing_columns))
            raise ValueError(f"FASTQ manifest {manifest_path} is missing columns: {column_names}")

        fastq_runs: list[_FastqRun] = []
        seen_accessions: set[str] = set()
        for manifest_row in manifest_rows:
            if not any((field_value or "").strip() for field_value in manifest_row.values()):
                continue

            line_number = manifest_rows.line_num
            run_accession = (manifest_row["run_accession"] or "").strip()
            if (
                not run_accession
                or run_accession in {".", ".."}
                or "/" in run_accession
                or "\\" in run_accession
            ):
                raise ValueError(
                    f"FASTQ manifest {manifest_path}, line {line_number}: invalid run accession"
                )
            if run_accession in seen_accessions:
                raise ValueError(
                    f"FASTQ manifest {manifest_path}, line {line_number}: "
                    f"duplicate run {run_accession}"
                )

            read1_path = _existing_fastq_path(
                manifest_row["read1_path"], manifest_path, line_number
            )
            read2_path = _existing_fastq_path(
                manifest_row["read2_path"], manifest_path, line_number
            )

            fastq_runs.append(_FastqRun(run_accession, read1_path, read2_path))
            seen_accessions.add(run_accession)

    if not fastq_runs:
        raise ValueError(f"FASTQ manifest {manifest_path} contains no runs")
    return tuple(fastq_runs)


def _existing_fastq_path(
    path_text: str | None,
    manifest_path: Path,
    line_number: int,
) -> Path:
    fastq_path = Path((path_text or "").strip())
    if not fastq_path.is_absolute():
        raise ValueError(
            f"FASTQ manifest {manifest_path}, line {line_number}: FASTQ paths must be absolute"
        )
    if not fastq_path.is_file():
        raise FileNotFoundError(fastq_path)
    return fastq_path.resolve()


def _installed_kmc_version() -> str:
    try:
        version_probe = subprocess.run(
            _KMC_COMMAND,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise OSError("KMC is not installed or is not on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise OSError("KMC did not report its version within 30 seconds") from error

    version_lines = f"{version_probe.stdout}\n{version_probe.stderr}".splitlines()
    version_line = next(
        (
            version_output_line.strip()
            for version_output_line in version_lines
            if "KMC" in version_output_line
        ),
        "",
    )
    if not version_line:
        raise OSError("Could not determine the installed KMC version")
    return version_line


def _count_fastq_run(
    fastq_run: _FastqRun,
    *,
    count_dir: Path,
    k: int,
    kmc_version: str,
) -> _KmcStatistics:
    run_dir = count_dir / fastq_run.run_accession
    if run_dir.exists():
        return _validate_kmc_run(run_dir, fastq_run, k, kmc_version)

    pending_run_dir = Path(tempfile.mkdtemp(prefix=f".{fastq_run.run_accession}.", dir=count_dir))
    try:
        _run_kmc(fastq_run, pending_run_dir, k)
        _write_run_metadata(
            pending_run_dir / "run.json",
            _run_metadata(fastq_run, k, kmc_version),
        )
        statistics = _validate_kmc_run(pending_run_dir, fastq_run, k, kmc_version)
        pending_run_dir.replace(run_dir)
        return statistics
    finally:
        shutil.rmtree(pending_run_dir, ignore_errors=True)


def _run_kmc(fastq_run: _FastqRun, pending_run_dir: Path, k: int) -> None:
    input_list_path = pending_run_dir / "inputs.txt"
    with input_list_path.open("w", encoding="utf-8", newline="\n") as input_list_stream:
        input_list_stream.write(f"{fastq_run.read1_path}\n{fastq_run.read2_path}\n")

    kmc_scratch_dir = pending_run_dir / "temporary"
    kmc_scratch_dir.mkdir()
    database_path = pending_run_dir / fastq_run.run_accession
    statistics_path = pending_run_dir / "stats.json"
    kmc_command = (
        *_KMC_COMMAND,
        f"-k{k}",
        "-fq",
        f"-ci{_MINIMUM_KMER_COUNT}",
        f"-cs{_MAXIMUM_KMER_COUNT}",
        f"-m{_KMC_MEMORY_GB}",
        f"-t{_KMC_THREADS}",
        "-hp",
        f"-j{statistics_path}",
        f"@{input_list_path}",
        str(database_path),
        str(kmc_scratch_dir),
    )
    kmc_process = subprocess.run(kmc_command, capture_output=True, text=True, check=False)
    if kmc_process.returncode != 0:
        error_message = (
            kmc_process.stderr.strip() or kmc_process.stdout.strip() or "no error message"
        )
        raise RuntimeError(
            f"KMC failed for {fastq_run.run_accession} "
            f"with exit code {kmc_process.returncode}: {error_message}"
        )

    input_list_path.unlink()
    shutil.rmtree(kmc_scratch_dir)


def _run_metadata(fastq_run: _FastqRun, k: int, kmc_version: str) -> dict[str, object]:
    return {
        "canonical": True,
        "counter_max": _MAXIMUM_KMER_COUNT,
        "k": k,
        "kmc_version": kmc_version,
        "memory_gb": _KMC_MEMORY_GB,
        "min_count": _MINIMUM_KMER_COUNT,
        "pairs": DEFAULT_READ_PAIRS,
        "read1_path": str(fastq_run.read1_path),
        "read2_path": str(fastq_run.read2_path),
        "run_accession": fastq_run.run_accession,
        "threads": _KMC_THREADS,
    }


def _validate_kmc_run(
    run_dir: Path,
    fastq_run: _FastqRun,
    k: int,
    kmc_version: str,
) -> _KmcStatistics:
    database_path = run_dir / fastq_run.run_accession
    prefix_path = Path(f"{database_path}.kmc_pre")
    suffix_path = Path(f"{database_path}.kmc_suf")
    statistics_path = run_dir / "stats.json"
    metadata_path = run_dir / "run.json"
    expected_files = {
        prefix_path.name,
        suffix_path.name,
        statistics_path.name,
        metadata_path.name,
    }
    if {kmc_file.name for kmc_file in run_dir.iterdir()} != expected_files:
        raise ValueError(f"KMC directory {run_dir} is incomplete or contains unexpected files")
    if prefix_path.stat().st_size == 0 or suffix_path.stat().st_size == 0:
        raise ValueError(f"KMC database {database_path} is empty")

    saved_metadata = _read_kmc_json(metadata_path)
    if saved_metadata != _run_metadata(fastq_run, k, kmc_version):
        raise ValueError(f"KMC directory {run_dir} was built from different inputs or settings")

    statistics = _read_kmc_statistics(statistics_path)
    if statistics.unique_kmers > statistics.total_kmers:
        raise ValueError(
            f"KMC statistics in {statistics_path} have more unique k-mers than total k-mers"
        )
    expected_read_count = DEFAULT_READ_PAIRS * 2
    if statistics.total_reads != expected_read_count:
        raise ValueError(
            f"KMC statistics in {statistics_path} report {statistics.total_reads} reads; "
            f"expected {expected_read_count}"
        )
    return statistics


def _read_kmc_json(json_path: Path) -> dict[str, object]:
    try:
        kmc_json = cast(object, json.loads(json_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read KMC JSON file {json_path}: {error}") from error
    if not isinstance(kmc_json, dict):
        raise ValueError(f"KMC JSON file {json_path} must contain an object")
    return cast("dict[str, object]", kmc_json)


def _read_kmc_statistics(statistics_path: Path) -> _KmcStatistics:
    statistics_json = _read_kmc_json(statistics_path)
    statistics_value = statistics_json.get("Stats")
    if not isinstance(statistics_value, dict):
        raise ValueError(f"KMC statistics file {statistics_path} has no Stats object")
    statistics = cast("dict[str, object]", statistics_value)
    return _KmcStatistics(
        unique_kmers=_kmc_statistic(statistics, "#Unique_counted_k-mers", statistics_path),
        total_kmers=_kmc_statistic(statistics, "#Total no. of k-mers", statistics_path),
        total_reads=_kmc_statistic(statistics, "#Total_reads", statistics_path),
    )


def _kmc_statistic(statistics: dict[str, object], field_name: str, statistics_path: Path) -> int:
    statistic_value = statistics.get(field_name)
    if type(statistic_value) is not int or statistic_value < 0:
        raise ValueError(f"KMC statistics file {statistics_path} has an invalid {field_name} value")
    return statistic_value


def _write_kmc_manifest(
    count_dir: Path,
    fastq_runs: tuple[_FastqRun, ...],
    run_statistics: tuple[_KmcStatistics, ...],
    k: int,
    kmc_version: str,
) -> Path:
    manifest_path = count_dir / "kmc_manifest.tsv"
    pending_manifest = count_dir / ".kmc_manifest.tsv.tmp"
    manifest_columns = (
        "run_accession",
        "database_path",
        "k",
        "unique_kmers",
        "total_kmers",
        "total_reads",
        "kmc_version",
    )

    try:
        with pending_manifest.open("w", encoding="utf-8", newline="") as manifest_stream:
            manifest_rows = csv.writer(manifest_stream, delimiter="\t", lineterminator="\n")
            manifest_rows.writerow(manifest_columns)
            for fastq_run, statistics in zip(fastq_runs, run_statistics, strict=True):
                run_dir = (count_dir / fastq_run.run_accession).resolve()
                manifest_rows.writerow(
                    (
                        fastq_run.run_accession,
                        run_dir / fastq_run.run_accession,
                        k,
                        statistics.unique_kmers,
                        statistics.total_kmers,
                        statistics.total_reads,
                        kmc_version,
                    )
                )
        pending_manifest.replace(manifest_path)
    finally:
        pending_manifest.unlink(missing_ok=True)
    return manifest_path


def _write_run_metadata(metadata_path: Path, run_metadata: dict[str, object]) -> None:
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
