"""
Data types used throughout the pipeline.

Config       - all the settings for a pipeline run
Sample       - one row from the input TSV
SampleResult - what happened when we processed one sample
RunSummary   - overall results after processing all samples
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """All the settings for a pipeline run.

    Fields:
        input_tsv       : path to the TSV file (two columns: URL, species)
        output_dir      : where to save extracted .zarr folders
        limit_samples   : only process this many samples (None = all)
        workers         : how many samples to download in parallel
        retries         : how many extra attempts if a download fails
        contig_include_regex : regex to select contigs (only matching ones are extracted)
        contig_exclude_regex : regex to reject contigs (matching ones are skipped)
        http_block_size_bytes : download chunk size for HTTP reads
        http_cache_maxblocks  : how many chunks to keep in memory
    """

    input_tsv: str = "selected_samples.tsv"
    output_dir: str = "data/gt_extracted"
    limit_samples: int | None = None
    workers: int = field(default_factory=lambda: min(16, (os.cpu_count() or 1) * 2))
    retries: int = 3

    contig_include_regex: str | None = None
    contig_exclude_regex: str | None = None

    http_block_size_bytes: int = 2**20
    http_cache_maxblocks: int = 64


@dataclass(frozen=True)
class Sample:
    """One sample parsed from the input TSV.

    Fields:
        url       : download link for the .zarr.zip archive
        species   : species name from the TSV (e.g. 'gambiae')
        sample_id : short name extracted from the URL (e.g. 'AB0085-Cx')
    """

    url: str
    species: str
    sample_id: str


@dataclass(frozen=True)
class SampleResult:
    """What happened when we tried to process one sample.

    Fields:
        sample_id : which sample this is about
        status    : 'completed', 'skipped', or 'failed'
        contigs   : list of contigs we extracted (empty if skipped/failed)
        error     : error message if it failed, otherwise None
        elapsed_s : how many seconds it took
    """

    sample_id: str
    status: str  # completed | skipped | failed | cancelled
    contigs: list[str]
    error: str | None
    elapsed_s: float


@dataclass(frozen=True)
class RunSummary:
    """Overall results after processing all samples.

    Fields:
        run_id    : unique ID for this run
        results   : list of per-sample results
        completed : how many samples succeeded
        skipped   : how many were already done and skipped
        failed    : how many failed
        cancelled : how many were cancelled by Ctrl+C
        elapsed_s : total time in seconds
    """

    run_id: str
    results: list[SampleResult] = field(repr=False)
    completed: int
    skipped: int
    failed: int
    cancelled: int = 0
    elapsed_s: float = 0.0
