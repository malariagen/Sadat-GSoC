"""
Helper functions for showing results.

- demo_config()       : quick config for testing with just 1 sample
- format_run_summary(): turn a RunSummary into a readable string
- print_run_summary() : print the summary to the console

Example:
    from gt_extract import Config, run_pipeline, format_run_summary

    summary = run_pipeline(Config(input_tsv="selected_samples.tsv", limit_samples=1))
    print(format_run_summary(summary))
"""

from __future__ import annotations

from gt_extract.types import Config, RunSummary


def demo_config(
    *,
    input_tsv: str = "selected_samples.tsv",
    output_dir: str = "data/gt_extracted",
    limit_samples: int | None = 1,
    workers: int = 4,
    retries: int = 3,
    contig_include_regex: str | None = None,
    contig_exclude_regex: str | None = None,
) -> Config:
    """Create a Config with safe defaults for quick testing.

    Limits to 1 sample with 4 workers. Suitable for demos and smoke tests.
    Override any parameter as needed.
    """
    return Config(
        input_tsv=input_tsv,
        output_dir=output_dir,
        limit_samples=limit_samples,
        workers=workers,
        retries=retries,
        contig_include_regex=contig_include_regex,
        contig_exclude_regex=contig_exclude_regex,
    )


def format_run_summary(summary: RunSummary) -> str:
    """Turn a RunSummary into a readable multi-line string.

    First line: overall counts. Then one line per sample.
    """
    lines: list[str] = []
    lines.append(
        f"run_id={summary.run_id} completed={summary.completed} skipped={summary.skipped} "
        f"failed={summary.failed} cancelled={summary.cancelled} elapsed_s={summary.elapsed_s:.1f}"
    )
    for r in summary.results:
        contigs = ",".join(r.contigs) if r.contigs else "-"
        err = f" err={r.error}" if r.error else ""
        lines.append(f"{r.sample_id} status={r.status} contigs={contigs} elapsed_s={r.elapsed_s:.1f}{err}")
    return "\n".join(lines)


def print_run_summary(summary: RunSummary) -> None:
    """Print run results to the console."""
    print(format_run_summary(summary))
