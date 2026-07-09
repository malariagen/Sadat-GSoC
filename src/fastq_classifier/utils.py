"""Shared utilities for TSV I/O, path sanitization, and subprocess helpers."""

from __future__ import annotations

import csv
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

T = TypeVar("T")
R = TypeVar("R")

INVALID_ROWS_FILE_NAME = "invalid_rows.tsv"
STATUS_COMPLETED = "completed"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
__all__ = [
    "INVALID_ROWS_FILE_NAME",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "InputError",
    "RejectedRow",
    "check_executable",
    "format_process_error",
    "kmc_database_exists",
    "partition_rows",
    "read_validated_tsv",
    "run_ordered",
    "safe_path_name",
    "write_invalid_rows",
    "write_rows",
]


class InputError(Exception):
    """The input file or options cannot be used."""


@dataclass(frozen=True, slots=True)
class RejectedRow:
    """An input row that could not be used.

    Attributes
    ----------
    row_number : int
        One-based line number in the TSV file, including the header row.
    values : dict of str to str
        Original TSV values keyed by column name.
    reason : str
        Reason the row was rejected.
    """

    row_number: int
    values: dict[str, str]
    reason: str

    def to_row(self, original_columns: Sequence[str]) -> dict[str, str]:
        """Build the row written to ``invalid_rows.tsv``.

        Parameters
        ----------
        original_columns : sequence of str
            Input columns, in their original order.

        Returns
        -------
        dict of str to str
            Original values plus ``row_number`` and ``reason``.
        """
        row = {column: self.values.get(column, "") for column in original_columns}
        row["row_number"] = str(self.row_number)
        row["reason"] = self.reason
        return row


def write_rows(path: Path, rows: Iterable[Mapping[str, str]], columns: Sequence[str]) -> None:
    """Write dictionaries to a TSV file.

    Parameters
    ----------
    path : pathlib.Path
        File to create.
    rows : iterable of mapping of str to str
        Row values keyed by column name.
    columns : sequence of str
        Column names and output order.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_path_name(value: str) -> str:
    """Make a string safe to use as one path segment.

    Parameters
    ----------
    value : str
        Raw text.

    Returns
    -------
    str
        String containing only letters, digits, underscores, dots, and hyphens.
        Empty results are returned as ``"unnamed"``.
    """
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return name or "unnamed"


def read_validated_tsv(
    path: Path,
    required_columns: Sequence[str],
    *,
    file_label: str,
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...]]:
    """Read a TSV file, validate its header and required columns, and normalize rows.

    Parameters
    ----------
    path : pathlib.Path
        TSV file to read.
    required_columns : sequence of str
        Column names that must be present in the header.
    file_label : str
        File description used in error messages.

    Returns
    -------
    tuple
        Normalized rows (dicts keyed by column name) and original column order.

    Raises
    ------
    InputError
        If the file does not exist, has no header, or is missing required columns.
    """
    if not path.is_file():
        msg = f"{file_label} does not exist: {path}"
        raise InputError(msg)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            msg = f"{path} does not contain a header row"
            raise InputError(msg)
        original_columns = tuple(reader.fieldnames)
        missing_columns = [column for column in required_columns if column not in original_columns]
        if missing_columns:
            msg = f"Missing required {file_label} columns: {', '.join(missing_columns)}"
            raise InputError(msg)
        rows = tuple(
            {column: raw_row.get(column) or "" for column in original_columns} for raw_row in reader
        )
    return rows, original_columns


def check_executable(command: tuple[str, ...], error_cls: type[Exception]) -> None:
    """Check that a command's executable can be resolved on PATH.

    Parameters
    ----------
    command : tuple of str
        Command whose first element is the executable to check.
    error_cls : type of Exception
        Exception type to raise if the executable cannot be found.

    Raises
    ------
    Exception
        An instance of ``error_cls`` if the executable cannot be found.
    """
    executable = command[0]
    if shutil.which(executable) is None:
        msg = f"Required executable not found: {executable}"
        raise error_cls(msg)


def format_process_error(name: str, return_code: int | None, stderr: str) -> str:
    """Turn a failed subprocess result into one readable sentence.

    Parameters
    ----------
    name : str
        Command name to show to the caller.
    return_code : int or None
        Process return code.
    stderr : str
        Captured standard error from the process.

    Returns
    -------
    str
        Message with the command name, exit code, and stderr text when present.
    """
    message = stderr.strip()
    if message:
        return f"{name} failed with exit code {return_code}: {message}"
    return f"{name} failed with exit code {return_code}"


def run_ordered(
    items: Sequence[T],
    worker: Callable[..., R],
    *,
    workers: int,
    **kwargs: object,
) -> tuple[R, ...]:
    """Run a function over items in parallel, preserving input order.

    Parameters
    ----------
    items : sequence of T
        Items to process.
    worker : callable
        Function called as ``worker(item, **kwargs)`` for each item.
    workers : int
        Number of items to process at once.
    **kwargs
        Keyword arguments forwarded to ``worker``.

    Returns
    -------
    tuple of R
        Results in the same order as ``items``.

    Raises
    ------
    InputError
        If ``workers`` is less than one.
    """
    if workers < 1:
        msg = "jobs must be greater than zero"
        raise InputError(msg)
    if workers == 1:
        return tuple(worker(item, **kwargs) for item in items)

    results: dict[int, R] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(worker, item, **kwargs): index for index, item in enumerate(items)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return tuple(results[index] for index in range(len(items)))


def partition_rows(
    rows: Iterable[dict[str, str]],
    parse_fn: Callable[[int, dict[str, str]], T],
) -> tuple[tuple[T, ...], tuple[RejectedRow, ...]]:
    """Split TSV rows into accepted items and rejected rows.

    Parameters
    ----------
    rows : iterable of dict of str to str
        TSV rows keyed by column name.
    parse_fn : callable
        Function called as ``parse_fn(row_number, row)``. Must return the
        parsed item or raise ``InputError`` for rows that cannot be used.

    Returns
    -------
    tuple
        Accepted items and rejected rows, both in input order.
    """
    accepted: list[T] = []
    rejected: list[RejectedRow] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            accepted.append(parse_fn(row_number, row))
        except InputError as error:
            rejected.append(RejectedRow(row_number=row_number, values=row, reason=str(error)))
    return tuple(accepted), tuple(rejected)


def write_invalid_rows(
    output_dir: Path,
    invalid_rows: Iterable[RejectedRow],
    original_columns: Sequence[str],
) -> None:
    """Write rejected rows to ``invalid_rows.tsv``.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory where the file will be written.
    invalid_rows : iterable of RejectedRow
        Rows that could not be processed.
    original_columns : sequence of str
        Input columns, in their original order.
    """
    write_rows(
        output_dir / INVALID_ROWS_FILE_NAME,
        (row.to_row(original_columns) for row in invalid_rows),
        (*original_columns, "row_number", "reason"),
    )


def kmc_database_exists(database_prefix: Path) -> bool:
    """Check whether both KMC database files exist.

    Parameters
    ----------
    database_prefix : pathlib.Path
        KMC database prefix without ``.kmc_pre`` or ``.kmc_suf``.

    Returns
    -------
    bool
        ``True`` when both KMC database files exist.
    """
    return (
        Path(f"{database_prefix}.kmc_pre").is_file()
        and Path(f"{database_prefix}.kmc_suf").is_file()
    )
