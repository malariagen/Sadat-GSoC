"""Build the fixed canonical 8-mer count matrix."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastq_classifier.features import (
    CANONICAL_KMER_COUNT,
    CANONICAL_KMERS,
    READ_PAIRS_PER_RUN,
)

DEFAULT_MATRIX_JOBS = 4

_KMC_TOOLS_COMMAND = ("kmc_tools",)


@dataclass(frozen=True, slots=True)
class _KmcDatabase:
    run_accession: str
    database_path: Path
    unique_kmers: int
    total_kmers: int


def build_count_matrix(
    kmc_manifest: str | Path,
    matrix_dir: str | Path,
    *,
    jobs: int = DEFAULT_MATRIX_JOBS,
) -> Path:
    """Build a dense uint32 matrix from canonical KMC databases."""
    if jobs <= 0:
        raise ValueError(f"jobs must be positive, got {jobs}")

    matrix_path = Path(matrix_dir)
    if matrix_path.exists():
        raise FileExistsError(f"Matrix output already exists: {matrix_path}")

    kmc_databases = _read_kmc_manifest(Path(kmc_manifest))
    _require_kmc_tools()

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    pending_matrix_dir = Path(
        tempfile.mkdtemp(prefix=f".{matrix_path.name}.", dir=matrix_path.parent)
    )
    try:
        _write_count_matrix(pending_matrix_dir / "counts.npy", kmc_databases, jobs)
        (pending_matrix_dir / "kmers.txt").write_text(
            "\n".join(CANONICAL_KMERS) + "\n",
            encoding="ascii",
        )
        _write_run_index(pending_matrix_dir / "runs.tsv", kmc_databases)
        pending_matrix_dir.replace(matrix_path)
    finally:
        shutil.rmtree(pending_matrix_dir, ignore_errors=True)

    return matrix_path / "counts.npy"


def read_matrix_run_accessions(index_path: str | Path) -> tuple[str, ...]:
    """Read and validate the row order recorded for a count matrix."""
    run_index_path = Path(index_path)
    with run_index_path.open(encoding="utf-8-sig", newline="") as run_index_stream:
        index_rows = csv.DictReader(run_index_stream, delimiter="\t")
        index_columns = tuple(index_rows.fieldnames or ())
        missing_columns = {"row_index", "run_accession"} - set(index_columns)
        if missing_columns:
            column_names = ", ".join(sorted(missing_columns))
            raise ValueError(f"Matrix run file {run_index_path} is missing columns: {column_names}")

        run_accessions: list[str] = []
        seen_accessions: set[str] = set()
        for index_row in index_rows:
            if not any((index_row.get(column_name) or "").strip() for column_name in index_columns):
                continue
            line_number = index_rows.line_num
            run_accession = (index_row["run_accession"] or "").strip()
            if not run_accession or run_accession in seen_accessions:
                raise ValueError(
                    f"Matrix run file {run_index_path}, line {line_number}: "
                    "empty or duplicate run accession"
                )
            try:
                row_index = int((index_row["row_index"] or "").strip())
            except ValueError as error:
                raise ValueError(
                    f"Matrix run file {run_index_path}, line {line_number}: invalid row_index"
                ) from error
            if row_index != len(run_accessions):
                raise ValueError(
                    f"Matrix run file {run_index_path}, line {line_number}: expected row_index "
                    f"{len(run_accessions)}, got {row_index}"
                )
            run_accessions.append(run_accession)
            seen_accessions.add(run_accession)

    if not run_accessions:
        raise ValueError(f"Matrix run file {run_index_path} contains no runs")
    return tuple(run_accessions)


def _read_kmc_manifest(manifest_path: Path) -> tuple[_KmcDatabase, ...]:
    with manifest_path.open(encoding="utf-8-sig", newline="") as manifest_stream:
        manifest_rows = csv.DictReader(manifest_stream, delimiter="\t")
        manifest_columns = tuple(manifest_rows.fieldnames or ())
        required_columns = {
            "run_accession",
            "database_path",
            "unique_kmers",
            "total_kmers",
            "total_reads",
            "kmc_version",
        }
        missing_columns = required_columns - set(manifest_columns)
        if missing_columns:
            column_names = ", ".join(sorted(missing_columns))
            raise ValueError(f"KMC manifest {manifest_path} is missing columns: {column_names}")

        kmc_databases: list[_KmcDatabase] = []
        seen_accessions: set[str] = set()
        kmc_versions: set[str] = set()
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
                    f"KMC manifest {manifest_path}, line {line_number}: invalid run accession"
                )
            if run_accession in seen_accessions:
                raise ValueError(
                    f"KMC manifest {manifest_path}, line {line_number}: "
                    f"duplicate run {run_accession}"
                )

            unique_kmers = _manifest_integer(
                manifest_row, "unique_kmers", manifest_path, line_number
            )
            total_kmers = _manifest_integer(manifest_row, "total_kmers", manifest_path, line_number)
            total_reads = _manifest_integer(manifest_row, "total_reads", manifest_path, line_number)
            expected_read_count = READ_PAIRS_PER_RUN * 2
            if total_reads != expected_read_count:
                raise ValueError(
                    f"KMC manifest {manifest_path}, line {line_number}: "
                    f"expected {expected_read_count} reads"
                )
            if unique_kmers > CANONICAL_KMER_COUNT or unique_kmers > total_kmers:
                raise ValueError(
                    f"KMC manifest {manifest_path}, line {line_number}: invalid k-mer counts"
                )

            database_path = Path((manifest_row["database_path"] or "").strip())
            if not database_path.is_absolute():
                raise ValueError(
                    f"KMC manifest {manifest_path}, line {line_number}: "
                    "database paths must be absolute"
                )
            if (
                not Path(f"{database_path}.kmc_pre").is_file()
                or not Path(f"{database_path}.kmc_suf").is_file()
            ):
                raise FileNotFoundError(f"KMC database {database_path} is incomplete")

            kmc_version = (manifest_row["kmc_version"] or "").strip()
            if not kmc_version:
                raise ValueError(
                    f"KMC manifest {manifest_path}, line {line_number}: kmc_version is empty"
                )
            kmc_databases.append(
                _KmcDatabase(
                    run_accession,
                    database_path.resolve(),
                    unique_kmers,
                    total_kmers,
                )
            )
            seen_accessions.add(run_accession)
            kmc_versions.add(kmc_version)

    if not kmc_databases:
        raise ValueError(f"KMC manifest {manifest_path} contains no runs")
    if len(kmc_versions) != 1:
        raise ValueError(
            f"KMC manifest {manifest_path} contains databases from different KMC versions"
        )
    return tuple(kmc_databases)


def _manifest_integer(
    manifest_row: dict[str, str | None],
    column_name: str,
    manifest_path: Path,
    line_number: int,
) -> int:
    try:
        parsed_value = int((manifest_row[column_name] or "").strip())
    except ValueError as error:
        raise ValueError(
            f"KMC manifest {manifest_path}, line {line_number}: invalid {column_name} value"
        ) from error
    if parsed_value < 0:
        raise ValueError(
            f"KMC manifest {manifest_path}, line {line_number}: {column_name} must not be negative"
        )
    return parsed_value


def _require_kmc_tools() -> None:
    try:
        version_probe = subprocess.run(
            _KMC_TOOLS_COMMAND,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise OSError("kmc_tools is not installed or is not on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise OSError("kmc_tools did not report its version within 30 seconds") from error

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
        raise OSError("Could not determine the installed kmc_tools version")


def _write_count_matrix(
    counts_path: Path,
    kmc_databases: tuple[_KmcDatabase, ...],
    jobs: int,
) -> None:
    column_by_kmer = {kmer: column_index for column_index, kmer in enumerate(CANONICAL_KMERS)}
    count_matrix = np.lib.format.open_memmap(
        counts_path,
        mode="w+",
        dtype=np.uint32,
        shape=(len(kmc_databases), CANONICAL_KMER_COUNT),
    )
    kmc_dump_dir = counts_path.parent / "dumps"
    kmc_dump_dir.mkdir()

    try:
        _fill_count_matrix(count_matrix, kmc_databases, column_by_kmer, kmc_dump_dir, jobs)
        count_matrix.flush()
    finally:
        del count_matrix
        shutil.rmtree(kmc_dump_dir, ignore_errors=True)


def _fill_count_matrix(
    count_matrix: np.memmap[tuple[int, int], np.dtype[np.uint32]],
    kmc_databases: tuple[_KmcDatabase, ...],
    column_by_kmer: dict[str, int],
    kmc_dump_dir: Path,
    jobs: int,
) -> None:
    remaining_databases = iter(enumerate(kmc_databases))
    row_by_future: dict[Future[NDArray[np.uint32]], int] = {}

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for _ in range(min(jobs, len(kmc_databases))):
            row_number, kmc_database = next(remaining_databases)
            count_future = pool.submit(
                _read_kmc_counts,
                kmc_database,
                column_by_kmer,
                kmc_dump_dir / f"{row_number}.txt",
            )
            row_by_future[count_future] = row_number

        while row_by_future:
            completed_futures, _ = wait(row_by_future, return_when=FIRST_COMPLETED)
            for count_future in completed_futures:
                row_number = row_by_future.pop(count_future)
                count_matrix[row_number] = count_future.result()
                try:
                    next_row_number, kmc_database = next(remaining_databases)
                except StopIteration:
                    continue
                next_count_future = pool.submit(
                    _read_kmc_counts,
                    kmc_database,
                    column_by_kmer,
                    kmc_dump_dir / f"{next_row_number}.txt",
                )
                row_by_future[next_count_future] = next_row_number


def _read_kmc_counts(
    kmc_database: _KmcDatabase,
    column_by_kmer: dict[str, int],
    dump_path: Path,
) -> NDArray[np.uint32]:
    try:
        _dump_kmc_database(kmc_database.database_path, dump_path)
        count_row = np.zeros(CANONICAL_KMER_COUNT, dtype=np.uint32)
        with dump_path.open(encoding="ascii", newline="") as dump_stream:
            for line_number, dump_line in enumerate(dump_stream, start=1):
                dump_fields = dump_line.rstrip("\r\n").split("\t")
                if len(dump_fields) != 2:
                    raise ValueError(
                        f"KMC dump {dump_path}, line {line_number}: expected two fields"
                    )
                kmer, count_text = dump_fields
                column_index = column_by_kmer.get(kmer)
                if column_index is None:
                    raise ValueError(
                        f"KMC dump {dump_path}, line {line_number}: invalid canonical 8-mer"
                    )
                if count_row[column_index] != 0:
                    raise ValueError(
                        f"KMC dump {dump_path}, line {line_number}: duplicate k-mer {kmer}"
                    )
                try:
                    kmer_count = int(count_text)
                except ValueError as error:
                    raise ValueError(
                        f"KMC dump {dump_path}, line {line_number}: invalid count"
                    ) from error
                if not 0 < kmer_count <= np.iinfo(np.uint32).max:
                    raise ValueError(
                        f"KMC dump {dump_path}, line {line_number}: count is outside uint32"
                    )
                count_row[column_index] = kmer_count

        observed_kmer_total = int(count_row.sum(dtype=np.uint64))
        observed_unique_kmers = int(np.count_nonzero(count_row))
        if observed_unique_kmers != kmc_database.unique_kmers:
            raise ValueError(
                f"KMC dump for {kmc_database.run_accession} contains "
                f"{observed_unique_kmers} unique k-mers; expected {kmc_database.unique_kmers}"
            )
        if observed_kmer_total != kmc_database.total_kmers:
            raise ValueError(
                f"KMC dump for {kmc_database.run_accession} sums to {observed_kmer_total}; "
                f"expected {kmc_database.total_kmers}"
            )
        return count_row
    finally:
        dump_path.unlink(missing_ok=True)


def _dump_kmc_database(database_path: Path, dump_path: Path) -> None:
    dump_command = (
        *_KMC_TOOLS_COMMAND,
        "-hp",
        "transform",
        str(database_path),
        "dump",
        str(dump_path),
    )
    dump_process = subprocess.run(dump_command, capture_output=True, text=True, check=False)
    if dump_process.returncode != 0:
        error_message = (
            dump_process.stderr.strip() or dump_process.stdout.strip() or "no error message"
        )
        raise RuntimeError(
            f"kmc_tools failed for {database_path} with exit code "
            f"{dump_process.returncode}: {error_message}"
        )


def _write_run_index(index_path: Path, kmc_databases: tuple[_KmcDatabase, ...]) -> None:
    with index_path.open("w", encoding="utf-8", newline="") as index_stream:
        index_rows = csv.writer(index_stream, delimiter="\t", lineterminator="\n")
        index_rows.writerow(("row_index", "run_accession"))
        for row_index, kmc_database in enumerate(kmc_databases):
            index_rows.writerow((row_index, kmc_database.run_accession))
