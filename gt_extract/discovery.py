"""
Figure out which contigs and files to extract from a remote archive.

This module scans the list of files inside a .zip archive and:
1. Finds which contigs have GT (genotype) data
2. Filters contigs based on include/exclude patterns from the config
3. Builds a list of exactly which files to download for each contig,
   sorted by their position in the archive for faster downloading
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from gt_extract._setup import NonRetryableError
from gt_extract.types import Config


def _parse_gt_zarray_path(path: str) -> tuple[str, str] | None:
    """Check if a file path looks like a GT .zarray metadata file.

    We look for paths like:
        {root}/{contig}/calldata/GT/.zarray  ->  (root, contig)
        {contig}/calldata/GT/.zarray          ->  ('', contig)

    Returns (root, contig) if it matches, or None if it doesn't.
    """
    parts = [p for p in path.split("/") if p]
    if len(parts) == 5 and parts[2:] == ["calldata", "GT", ".zarray"]:
        return parts[0], parts[1]
    if len(parts) == 4 and parts[1:] == ["calldata", "GT", ".zarray"]:
        return "", parts[0]
    return None


def discover_root_and_contigs(zip_infos: Iterable[Any], cfg: Config) -> tuple[str, list[str]]:
    """Scan the archive to find which contigs have GT data.

    Looks for files matching the pattern */calldata/GT/.zarray to
    discover contigs automatically (no hardcoded contig names).

    Args:
        zip_infos : list of file entries from the ZIP archive
        cfg       : pipeline config (has the include/exclude regex filters)

    Returns:
        (internal_root, selected_contigs) - the archive's root folder name
        and the sorted list of contigs that passed filtering.

    Raises:
        NonRetryableError: if no GT contigs are found, or all are filtered out
    """
    roots: set[str] = set()
    contigs: set[str] = set()
    for zi in zip_infos:
        parsed = _parse_gt_zarray_path(getattr(zi, "filename", ""))
        if not parsed:
            continue
        root, contig = parsed
        roots.add(root)
        contigs.add(contig)

    if not contigs:
        raise NonRetryableError("No contigs found (no */calldata/GT/.zarray entries in archive)")

    if len(roots) > 1:
        raise NonRetryableError(f"Ambiguous internal roots in archive: {sorted(roots)}")
    internal_root = next(iter(roots))

    # Apply include/exclude filters
    include_re = re.compile(cfg.contig_include_regex) if cfg.contig_include_regex else None
    exclude_re = re.compile(cfg.contig_exclude_regex) if cfg.contig_exclude_regex else None

    selected = []
    for c in sorted(contigs):
        if include_re and not include_re.search(c):
            continue
        if exclude_re and exclude_re.search(c):
            continue
        selected.append(c)

    if not selected:
        raise NonRetryableError("No contigs selected after include/exclude filtering")
    return internal_root, selected


def _join_prefix(root: str, contig: str) -> str:
    """Build a path prefix like 'root/contig/' (or just 'contig/' if root is empty)."""
    if root:
        return f"{root}/{contig}/"
    return f"{contig}/"


def build_gt_member_index(
    zip_infos: list[Any],
    name_to_info: dict[str, Any],
    internal_root: str,
    selected_contigs: list[str],
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    """Build a list of files to download for each contig.

    For each selected contig, collects:
    - Parent metadata files (.zgroup, .zattrs)
    - All GT data files (calldata/GT/...)

    The files are sorted by their position inside the ZIP so that
    the download module can download them in order (fewer HTTP requests).

    Args:
        zip_infos         : all file entries in the archive
        name_to_info      : lookup dict mapping filename -> file entry
        internal_root     : the root folder inside the archive
        selected_contigs  : which contigs to index

    Returns:
        (members_by_contig, expected_chunk_counts) - per-contig file lists
        and the expected number of data chunks per contig.

    Raises:
        NonRetryableError: if a contig is missing its .zarray or has no chunks
    """
    selected_set = set(selected_contigs)
    members_by_contig: dict[str, list[Any]] = {c: [] for c in selected_contigs}
    expected_chunk_counts: dict[str, int] = {c: 0 for c in selected_contigs}

    # Grab small metadata files for each contig
    for contig in selected_contigs:
        contig_prefix = _join_prefix(internal_root, contig)
        for meta in (
            f"{contig_prefix}.zgroup",
            f"{contig_prefix}.zattrs",
            f"{contig_prefix}calldata/.zgroup",
            f"{contig_prefix}calldata/.zattrs",
        ):
            zi = name_to_info.get(meta)
            if zi is not None and not getattr(zi, "filename", "").endswith("/"):
                members_by_contig[contig].append(zi)

    # Scan all files and pick out the GT data files for our selected contigs
    root_prefix = f"{internal_root}/" if internal_root else ""
    for zi in zip_infos:
        name = getattr(zi, "filename", "")
        if not name or name.endswith("/"):
            continue
        if internal_root and not name.startswith(root_prefix):
            continue
        rel = name[len(root_prefix) :] if root_prefix else name

        # Quick check: does this file belong to {contig}/calldata/GT/...?
        parts = rel.split("/", 3)
        if len(parts) < 4:
            continue
        contig, lvl1, lvl2, rest = parts[0], parts[1], parts[2], parts[3]
        if contig not in selected_set:
            continue
        if lvl1 != "calldata" or lvl2 != "GT":
            continue

        members_by_contig[contig].append(zi)
        # Count actual data chunks (not metadata files like .zarray)
        base = rest.rsplit("/", 1)[-1]
        if base and not base.startswith("."):
            expected_chunk_counts[contig] += 1

    # Validate and sort by position in the ZIP file
    for contig in selected_contigs:
        contig_prefix = _join_prefix(internal_root, contig)
        gt_prefix = f"{contig_prefix}calldata/GT/"
        zarray_name = f"{gt_prefix}.zarray"
        if zarray_name not in name_to_info:
            raise NonRetryableError(f"Missing required GT .zarray for contig {contig}: {zarray_name}")
        if expected_chunk_counts[contig] <= 0:
            raise NonRetryableError(f"GT array for contig {contig} has no chunk files under {gt_prefix}")

        # Remove duplicates and sort by position in the ZIP for faster reading
        unique_by_name: dict[str, Any] = {}
        for zi in members_by_contig[contig]:
            unique_by_name[getattr(zi, "filename", "")] = zi
        members = list(unique_by_name.values())
        members.sort(key=lambda z: getattr(z, "header_offset", 0))
        members_by_contig[contig] = members

    return members_by_contig, expected_chunk_counts
