"""
CLI entry point for gt_extract.

Usage:
    python -m gt_extract --input-tsv selected_samples.tsv --output-dir data/gt_extracted
    python -m gt_extract --help
"""

from __future__ import annotations

import argparse
import sys

from gt_extract.types import Config
from gt_extract.pipeline import run_pipeline
from gt_extract.exports import format_run_summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gt_extract",
        description="Extract per-contig GT arrays from remote .zarr.zip archives over HTTPS.",
    )
    p.add_argument(
        "--input-tsv",
        default="selected_samples.tsv",
        help="Path to input TSV (URL<TAB>species, no header). Default: %(default)s",
    )
    p.add_argument(
        "--output-dir",
        default="data/gt_extracted",
        help="Directory for output .zarr stores. Default: %(default)s",
    )
    p.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Process at most N samples (default: all).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel sample workers. Default: min(16, cpu_count*2).",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries per sample on transient errors. Default: %(default)s",
    )
    p.add_argument(
        "--contig-include",
        default=None,
        dest="contig_include_regex",
        help="Regex: only include contigs matching this pattern.",
    )
    p.add_argument(
        "--contig-exclude",
        default=None,
        dest="contig_exclude_regex",
        help="Regex: exclude contigs matching this pattern.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    kwargs: dict = {
        "input_tsv": args.input_tsv,
        "output_dir": args.output_dir,
        "limit_samples": args.limit_samples,
        "retries": args.retries,
        "contig_include_regex": args.contig_include_regex,
        "contig_exclude_regex": args.contig_exclude_regex,
    }
    if args.workers is not None:
        kwargs["workers"] = args.workers

    cfg = Config(**kwargs)
    try:
        summary = run_pipeline(cfg)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        return 130
    print(format_run_summary(summary))
    return 1 if summary.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

