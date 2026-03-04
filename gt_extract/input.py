"""
Read the input TSV and connect to remote ZIP archives.

This module handles the first steps of the pipeline:
1. Parse the sample list from a TSV file
2. Check that the server supports partial downloads (HTTP Range)
3. Open a remote .zip file so we can read parts of it without downloading the whole thing
"""

from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import urlparse

import fsspec
import requests

from gt_extract._setup import NonRetryableError
from gt_extract.types import Config, Sample


def derive_sample_id(url: str) -> str:
    """Extract a short sample name from a URL.

    Examples:
        'https://.../AB0085-Cx.gatk.zarr.zip'  ->  'AB0085-Cx'
        'https://.../sample1.zarr.zip'          ->  'sample1'
    """
    parsed = urlparse(url)
    name = Path(parsed.path).name
    for suffix in (".gatk.zarr.zip", ".zarr.zip", ".zip"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name.endswith(".zip"):
        return name[:-4]
    return name


def load_samples_tsv(path: str, limit: int | None) -> list[Sample]:
    """Read the input TSV and return a list of Sample objects.

    The TSV should have NO header row, and two tab-separated columns:
        URL    species

    Args:
        path  : path to the TSV file
        limit : stop after this many rows (None = read all)

    Raises:
        ValueError: if the file has bad formatting or duplicate sample IDs
    """
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{row_idx}: expected 2 columns (url, species), got {len(row)}")
            url = row[0].strip()
            species = row[1].strip()
            if not url:
                raise ValueError(f"{path}:{row_idx}: empty URL")
            if not (url.startswith("https://") and url.endswith(".zip")):
                raise ValueError(f"{path}:{row_idx}: URL must look like https://.../*.zip, got: {url}")
            sample_id = derive_sample_id(url)
            if not sample_id:
                raise ValueError(f"{path}:{row_idx}: could not derive sample_id from URL: {url}")
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{row_idx}: duplicate sample_id derived from URLs: {sample_id}")
            seen_ids.add(sample_id)
            samples.append(Sample(url=url, species=species, sample_id=sample_id))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def preflight_range_support(url: str, timeout_s: float = 30.0) -> None:
    """Check that the server lets us download parts of a file (HTTP Range requests).

    We send a tiny request asking for just 1 byte. If the server replies
    with status 206, it supports partial downloads. If not, we'd end up
    downloading the entire archive, so we raise an error.

    Raises:
        NonRetryableError: if the server doesn't support Range requests
    """
    try:
        resp = requests.get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=timeout_s,
            verify=False,
        )
    except requests.RequestException as exc:
        raise exc
    try:
        if resp.status_code != 206:
            raise NonRetryableError(
                f"Server does not appear to support HTTP range requests (expected 206, got {resp.status_code})"
            )
    finally:
        resp.close()


def open_remote_zipfs(url: str, cfg: Config):
    """Open a remote .zip file so we can read individual files inside it.

    Uses fsspec to treat the remote ZIP like a local filesystem.
    Only downloads the parts we actually need (the ZIP index + requested files),
    not the whole archive.

    Args:
        url : HTTPS link to the .zip archive
        cfg : pipeline config (controls download chunk size and caching)

    Returns:
        An fsspec ZipFileSystem that you can use like a local filesystem.
    """
    return fsspec.filesystem(
        "zip",
        fo=url,
        target_protocol="https",
        target_options={
            "ssl": False,
            "block_size": cfg.http_block_size_bytes,
            "cache_type": "blockcache",
            "cache_options": {"maxblocks": cfg.http_cache_maxblocks},
        },
    )
