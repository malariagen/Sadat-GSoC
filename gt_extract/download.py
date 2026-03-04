"""
Fast copying to download files from a remote ZIP efficiently.

Instead of downloading each file one by one, this module:
1. Figures out where each file sits inside the ZIP
2. Groups nearby files into big "spans"
3. Downloads each span in one shot (one HTTP request instead of many)
4. Slices the downloaded blob to extract individual files

Falls back to one-at-a-time downloads for compressed files or parsing errors.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gt_extract._setup import NonRetryableError


def _drop_internal_root(path: str, internal_root: str) -> str:
    """Remove the archive's internal root folder from a path."""
    if internal_root and path.startswith(internal_root + "/"):
        return path[len(internal_root) + 1 :]
    return path


@dataclass(frozen=True)
class CopyStats:
    """Stats from one copy_members_fast() call.

    members    : total files processed
    chunks     : data files (not metadata like .zarray)
    spans      : how many big HTTP requests we made
    bytes_read_spans : total bytes downloaded via spans
    fallback_members : files we had to download one at a time
    wrote_files      : files written to disk
    elapsed_s        : how long it took in seconds
    """

    members: int
    chunks: int
    spans: int
    bytes_read_spans: int
    fallback_members: int
    wrote_files: int
    elapsed_s: float


def _ensure_safe_key(dst_key: str) -> None:
    """Make sure a destination path doesn't try to escape the output folder.

    Blocks absolute paths and '..' tricks that could write files elsewhere.
    """
    if dst_key.startswith(("/", "\\")):
        raise NonRetryableError(f"Unsafe key (absolute path): {dst_key}")
    parts = [p for p in dst_key.split("/") if p]
    if any(p == ".." for p in parts):
        raise NonRetryableError(f"Unsafe key (parent traversal): {dst_key}")


def write_key_bytes(tmp_dir: Path, dst_key: str, data: Any, mkdir_cache: set[Path]) -> None:
    """Write data to a file inside tmp_dir, creating folders as needed.

    Uses mkdir_cache to remember which folders already exist (avoids
    making the same folder twice).
    """
    _ensure_safe_key(dst_key)
    out_path = tmp_dir / dst_key
    parent = out_path.parent
    if parent not in mkdir_cache:
        parent.mkdir(parents=True, exist_ok=True)
        mkdir_cache.add(parent)
    # Accept bytes-like objects (e.g., memoryview slices) to avoid extra copies.
    with out_path.open("wb") as f:
        f.write(data)


def _read_exact(fp, n: int) -> bytes:
    """Read exactly n bytes from a file. Raises OSError if the file ends early."""
    out = bytearray()
    while len(out) < n:
        chunk = fp.read(n - len(out))
        if not chunk:
            break
        out.extend(chunk)
    if len(out) != n:
        raise OSError(f"Short read: expected {n} bytes, got {len(out)}")
    return bytes(out)


def _read_range_bytes(zipfs, start: int, end: int) -> bytes:
    """Download bytes [start, end) from the remote ZIP file.

    For HTTP files, this sends a single Range request for the whole chunk
    instead of going through fsspec's caching layer (which would make
    many small requests).
    """
    if end < start:
        raise ValueError(f"Invalid range: start={start} end={end}")
    n = end - start
    if n == 0:
        return b""

    fo = getattr(zipfs, "fo", None)
    fs = getattr(fo, "fs", None)
    url = getattr(fo, "url", None) or getattr(fo, "path", None)
    cat_file = getattr(fs, "cat_file", None) if fs is not None else None
    if url and callable(cat_file):
        data = cat_file(url, start=start, end=end)
        if len(data) != n:
            raise OSError(f"Short range read via cat_file: expected {n} bytes, got {len(data)}")
        return data

    fo.seek(start)
    return _read_exact(fo, n)


def _group_spans(
    records: list[tuple[Any, ...]],
    gap_threshold_bytes: int,
    max_span_bytes: int,
) -> list[tuple[int, int, list[tuple[Any, ...]]]]:
    """Merge nearby file records into bigger spans to reduce HTTP requests.

    Two files are merged into one span if:
    - The gap between them is small enough (<= gap_threshold_bytes)
    - The combined span is not too large (<= max_span_bytes)

    Args:
        records : list of (start, end, ...) tuples, sorted by start
        gap_threshold_bytes : max gap to bridge between files
        max_span_bytes      : max size for a single span

    Returns:
        list of (span_start, span_end, records_in_span)
    """
    if not records:
        return []
    spans: list[tuple[int, int, list[tuple[Any, ...]]]] = []
    cur: list[tuple[Any, ...]] = [records[0]]
    span_start = int(records[0][0])
    span_end = int(records[0][1])

    for rec in records[1:]:
        rec_start, rec_end = int(rec[0]), int(rec[1])
        gap = rec_start - span_end
        new_end = max(span_end, rec_end)
        if gap <= gap_threshold_bytes and (new_end - span_start) <= max_span_bytes:
            cur.append(rec)
            span_end = new_end
        else:
            spans.append((span_start, span_end, cur))
            cur = [rec]
            span_start = rec_start
            span_end = rec_end

    spans.append((span_start, span_end, cur))
    return spans


def copy_members_fast(
    zipfs,
    members: list[Any],
    internal_root: str,
    tmp_dir: Path,
    mkdir_cache: set[Path],
    *,
    gap_threshold_bytes: int = 256 * 1024,
    max_span_bytes: int = 64 * 1024 * 1024,
) -> CopyStats:
    """Download files from a remote ZIP and save them locally.

    For uncompressed files (ZIP_STORED), groups nearby files into spans
    and downloads each span in one HTTP request. Then parses the ZIP
    headers in memory to extract individual files from the blob.

    For compressed files (or if header parsing fails), falls back to
    downloading one file at a time.

    Args:
        zipfs       : the open remote ZIP filesystem
        members     : list of files to extract
        internal_root : root folder inside the archive (stripped from output paths)
        tmp_dir     : local folder to write files into
        mkdir_cache : shared set of folders already created
        gap_threshold_bytes : max gap to merge (default 256KB)
        max_span_bytes      : max span size (default 64MB)

    Returns:
        CopyStats with download performance numbers.
    """
    t0 = time.perf_counter()
    stored: list[tuple[int, int, Any, str]] = []  # (header_offset, end_est, ZipInfo, dst_key)
    fallback: list[tuple[str, str]] = []  # (src_name, dst_key)

    chunks = 0
    for zi in members:
        src = getattr(zi, "filename", "")
        if not src or src.endswith("/"):
            continue
        dst_key = _drop_internal_root(src, internal_root)
        base = src.rsplit("/", 1)[-1]
        if base and not base.startswith("."):
            chunks += 1

        # Fast path only for uncompressed (stored) files
        if getattr(zi, "compress_type", None) == 0:
            try:
                header_offset = int(getattr(zi, "header_offset"))
                compress_size = int(getattr(zi, "compress_size"))
            except Exception:
                fallback.append((src, dst_key))
                continue

            # Estimate where the file data ends inside the ZIP
            try:
                name_len_est = len(src.encode("utf-8"))
            except Exception:
                name_len_est = len(src.encode("utf-8", errors="replace"))
            extra_len_est = len(getattr(zi, "extra", b"") or b"")
            header_slack = 512  # safety margin
            end_est = header_offset + 30 + name_len_est + extra_len_est + compress_size + header_slack
            stored.append((header_offset, end_est, zi, dst_key))
            continue

        fallback.append((src, dst_key))

    stored.sort(key=lambda r: r[0])  # sort by position in ZIP
    spans = _group_spans(stored, gap_threshold_bytes=gap_threshold_bytes, max_span_bytes=max_span_bytes)

    bytes_read = 0
    wrote_files = 0

    for span_start, span_end, span_recs in spans:
        # Don't read past the end of the file
        fo = getattr(zipfs, "fo", None)
        size = getattr(fo, "size", None)
        if not isinstance(size, int):
            try:
                if fo is not None and hasattr(fo, "fileno"):
                    size = os.fstat(fo.fileno()).st_size
            except Exception:
                size = None
        if isinstance(size, int):
            span_end = min(span_end, size)
        span_len = span_end - span_start
        if span_len <= 0:
            continue

        blob = memoryview(_read_range_bytes(zipfs, span_start, span_end))
        if len(blob) != span_len:
            raise OSError(f"Short span read: expected {span_len} bytes, got {len(blob)}")
        bytes_read += span_len

        for header_offset, _end_est, zi, dst_key in span_recs:
            local_rel = int(header_offset) - span_start
            compress_size = int(getattr(zi, "compress_size", 0))
            src = getattr(zi, "filename", "")

            # Parse the ZIP local file header to find where the actual data starts
            if local_rel < 0 or local_rel + 30 > len(blob):
                data = zipfs.cat_file(src)
                write_key_bytes(tmp_dir, dst_key, data, mkdir_cache)
                wrote_files += 1
                continue
            hdr = blob[local_rel : local_rel + 30]
            if hdr[:4].tobytes() != b"PK\x03\x04":
                data = zipfs.cat_file(src)
                write_key_bytes(tmp_dir, dst_key, data, mkdir_cache)
                wrote_files += 1
                continue
            try:
                fname_len, extra_len = struct.unpack("<HH", hdr[26:30].tobytes())
            except struct.error:
                data = zipfs.cat_file(src)
                write_key_bytes(tmp_dir, dst_key, data, mkdir_cache)
                wrote_files += 1
                continue

            data_rel = local_rel + 30 + int(fname_len) + int(extra_len)
            data_end_rel = data_rel + compress_size
            if data_rel < 0 or data_end_rel > len(blob):
                data = zipfs.cat_file(src)
                write_key_bytes(tmp_dir, dst_key, data, mkdir_cache)
                wrote_files += 1
                continue

            view = blob[data_rel:data_end_rel]
            write_key_bytes(tmp_dir, dst_key, view, mkdir_cache)
            wrote_files += 1

    # Download compressed files one at a time (fallback)
    for src, dst_key in fallback:
        data = zipfs.cat_file(src)
        write_key_bytes(tmp_dir, dst_key, data, mkdir_cache)
        wrote_files += 1

    return CopyStats(
        members=len([zi for zi in members if getattr(zi, "filename", "") and not getattr(zi, "filename", "").endswith("/")]),
        chunks=chunks,
        spans=len(spans),
        bytes_read_spans=bytes_read,
        fallback_members=len(fallback),
        wrote_files=wrote_files,
        elapsed_s=time.perf_counter() - t0,
    )
