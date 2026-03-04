"""
Check that downloaded data is correct, and decide what to retry.

This module handles three things:
1. Validation - open the extracted Zarr files and count chunks to make sure
   nothing is missing or corrupt.
2. Skip logic - if a sample was already downloaded successfully (has a
   _SUCCESS.json marker), skip it instead of re-downloading.
3. Retry policy - decide which errors are worth retrying (network glitches)
   vs which should fail immediately (broken archives).
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import requests
import zarr
from zarr.storage import DirectoryStore

from gt_extract._setup import NonRetryableError
from gt_extract.types import Config, Sample


def _count_local_chunks(gt_dir: Path) -> int:
    """Count data files in a GT folder (ignoring dotfiles like .zarray)."""
    if not gt_dir.exists():
        return 0
    n = 0
    with os.scandir(gt_dir) as it:
        for entry in it:
            if entry.is_file() and not entry.name.startswith("."):
                n += 1
    return n


def _validate_local_output(sample_dir: Path, contig_expected_chunks: dict[str, int], contigs: list[str]) -> None:
    """Check that the extracted data on disk is complete and valid.

    For each contig, verifies:
    1. The .zarray metadata file exists and is valid JSON
    2. Zarr can open the GT array successfully
    3. The number of chunk files matches what was in the archive

    Raises NonRetryableError if anything is wrong.
    """
    store = DirectoryStore(str(sample_dir))
    for contig in contigs:
        gt_dir = sample_dir / contig / "calldata" / "GT"
        zarray_path = gt_dir / ".zarray"
        if not zarray_path.exists():
            raise NonRetryableError(f"Missing local GT .zarray for contig {contig}: {zarray_path}")
        try:
            _ = json.loads(zarray_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise NonRetryableError(f"Invalid JSON in {zarray_path}: {exc}") from exc

        try:
            _ = zarr.open_array(store, path=f"{contig}/calldata/GT", mode="r")
        except Exception as exc:
            raise NonRetryableError(f"Cannot open local zarr array for contig {contig}: {exc}") from exc

        expected = contig_expected_chunks.get(contig)
        if expected is None:
            raise NonRetryableError(f"Missing expected chunk count for contig {contig} in expectations")
        actual = _count_local_chunks(gt_dir)
        if actual != expected:
            raise NonRetryableError(
                f"Chunk count mismatch for contig {contig}: expected {expected}, got {actual} (dir={gt_dir})"
            )


def _read_success_marker(sample_dir: Path) -> dict[str, Any] | None:
    """Read the _SUCCESS.json file from a sample folder. Returns None if missing or invalid."""
    marker_path = sample_dir / "_SUCCESS.json"
    if not marker_path.exists():
        return None
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def should_skip(final_dir: Path, sample: Sample, cfg: Config) -> bool:
    """Check if this sample was already downloaded correctly.

    Only skips if ALL of these are true:
    - There's a _SUCCESS.json marker with all the right fields
    - The sample ID and source URL match
    - The contig include/exclude filters match the current config
    - The chunk counts on disk match what's recorded in the marker

    Returns True to skip, False to re-download.
    """
    marker = _read_success_marker(final_dir)
    if not marker:
        return False
    required = ["sample_id", "run_id", "timestamp_utc", "source_url", "source_sample_id", "contigs", "expected_chunk_counts"]
    if any(k not in marker for k in required):
        return False
    if marker.get("sample_id") != sample.sample_id:
        return False
    if marker.get("source_url") != sample.url:
        return False
    # Make sure the contig filters match, otherwise we might skip a partial extraction
    _missing = object()
    marker_include = marker.get("contig_include_regex", _missing)
    marker_exclude = marker.get("contig_exclude_regex", _missing)
    if marker_include is _missing or marker_exclude is _missing:
        return False
    if (marker_include or None) != (cfg.contig_include_regex or None):
        return False
    if (marker_exclude or None) != (cfg.contig_exclude_regex or None):
        return False
    contigs = marker.get("contigs")
    expected = marker.get("expected_chunk_counts")
    if not isinstance(contigs, list) or not isinstance(expected, dict):
        return False
    try:
        _validate_local_output(final_dir, {str(k): int(v) for k, v in expected.items()}, [str(c) for c in contigs])
    except Exception:
        return False
    return True


def cleanup_stale_tmp_dirs(output_dir: Path, sample_id: str) -> None:
    """Delete any leftover temp folders from previous failed runs."""
    prefix = f"{sample_id}.zarr.__tmp__"
    for p in output_dir.glob(prefix + "*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def is_retryable_exception(exc: BaseException) -> bool:
    """Decide if an error is worth retrying (True) or should fail immediately (False).

    Worth retrying (transient problems):
    - Network timeouts and connection errors
    - Server errors (429, 500, 502, 503, 504)
    - Corrupt ZIP files (might be a network glitch)

    NOT worth retrying (permanent problems):
    - NonRetryableError (missing data, bad archive structure, etc.)
    - Everything else
    """
    # aiohttp is used under the hood by fsspec HTTPFileSystem
    try:
        import aiohttp  # type: ignore

        if isinstance(exc, aiohttp.ClientResponseError):
            return exc.status in {429, 500, 502, 503, 504}
        if isinstance(exc, aiohttp.ClientError):
            return True
    except Exception:
        pass

    if isinstance(exc, requests.RequestException):
        resp = getattr(exc, "response", None)
        if resp is not None and hasattr(resp, "status_code"):
            return int(resp.status_code) in {429, 500, 502, 503, 504}
        return True

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    if isinstance(exc, OSError):
        msg = str(exc).lower()
        if any(s in msg for s in ("timed out", "timeout", "connection reset", "connection aborted", "temporarily unavailable")):
            return True

    import zipfile

    if isinstance(exc, zipfile.BadZipFile):
        return True

    return False


def _backoff_sleep_s(attempt_idx: int) -> float:
    """Calculate how long to wait before retrying.

    Uses exponential backoff: 1s, 2s, 4s, 8s... capped at 60s,
    plus a random 0-1s jitter to avoid all workers retrying at once.
    """
    base = 1.0
    cap = 60.0
    sleep = min(cap, base * (2**attempt_idx))
    jitter = random.uniform(0.0, 1.0)
    return sleep + jitter
