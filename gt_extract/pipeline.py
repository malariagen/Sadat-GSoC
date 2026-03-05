"""
Main pipeline - ties everything together.

This module runs the full extraction:
1. Load samples from the TSV file
2. For each sample: connect, discover contigs, download GT data, validate, save
3. Retry on network errors, skip already-completed samples
4. Run multiple samples in parallel using threads

Usage:
    from gt_extract import Config, run_pipeline
    summary = run_pipeline(Config(input_tsv="selected_samples.tsv"))
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from gt_extract._setup import (
    NonRetryableError,
    install_signal_handler,
    log,
    make_run_id,
    shutdown_requested,
    utc_now_iso,
)
from gt_extract.types import Config, Sample, SampleResult, RunSummary
from gt_extract.input import load_samples_tsv, open_remote_zipfs, preflight_range_support
from gt_extract.discovery import build_gt_member_index, discover_root_and_contigs
from gt_extract.download import copy_members_fast
from gt_extract.validation import (
    _backoff_sleep_s,
    _validate_local_output,
    cleanup_stale_tmp_dirs,
    is_retryable_exception,
    should_skip,
)


def extract_sample_attempt(sample: Sample, cfg: Config, run_id: str) -> tuple[str, list[str]]:
    """Try to extract GT data for one sample. This is one attempt (no retries here).

    Steps:
    1. Clean up leftover temp folders from any previous failed attempt
    2. Check if this sample is already done (skip if so)
    3. Verify the server supports partial downloads
    4. Open the remote ZIP and figure out which contigs have GT data
    5. Download the GT files for each contig
    6. Validate that everything downloaded correctly
    7. Write a _SUCCESS.json marker
    8. Move the temp folder to its final location (atomic rename)

    Returns ('completed', contigs) or ('skipped', []).
    Raises an exception if something goes wrong.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_dir = output_dir / f"{sample.sample_id}.zarr"
    tmp_dir = output_dir / f"{sample.sample_id}.zarr.__tmp__{run_id}"

    # Always clean stale temp dirs before (re)building.
    cleanup_stale_tmp_dirs(output_dir, sample.sample_id)

    if final_dir.exists() and should_skip(final_dir, sample, cfg):
        return "skipped", []

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    zipfs = None
    contigs: list[str] = []
    expected_chunk_counts: dict[str, int] = {}
    internal_root = ""
    try:
        tmp_dir.mkdir(parents=True, exist_ok=False)

        preflight_range_support(sample.url)
        zipfs = open_remote_zipfs(sample.url, cfg)
        zip_infos = list(zipfs.zip.infolist())

        internal_root, contigs = discover_root_and_contigs(zip_infos, cfg)
        name_to_info = {getattr(zi, "filename", ""): zi for zi in zip_infos}
        members_by_contig, expected_chunk_counts = build_gt_member_index(zip_infos, name_to_info, internal_root, contigs)
        mkdir_cache: set[Path] = set()

        for contig in contigs:
            if shutdown_requested():
                raise KeyboardInterrupt("shutdown requested")
            members = members_by_contig[contig]
            stats = copy_members_fast(zipfs, members, internal_root, tmp_dir, mkdir_cache)
            mb = stats.bytes_read_spans / (1024 * 1024)
            log(
                f"{sample.sample_id} contig {contig} done "
                f"(members={stats.members} chunks={stats.chunks} spans={stats.spans} "
                f"bytes_read={mb:.1f}MB fallback={stats.fallback_members} elapsed={stats.elapsed_s:.1f}s)"
            )

        _validate_local_output(tmp_dir, expected_chunk_counts, contigs)

        marker = {
            "sample_id": sample.sample_id,
            "run_id": run_id,
            "timestamp_utc": utc_now_iso(),
            "source_url": sample.url,
            "source_sample_id": internal_root or sample.sample_id,
            "contig_include_regex": cfg.contig_include_regex,
            "contig_exclude_regex": cfg.contig_exclude_regex,
            "contigs": contigs,
            "expected_chunk_counts": expected_chunk_counts,
        }
        (tmp_dir / "_SUCCESS.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Replace final only after temp is validated and marked.
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        tmp_dir.rename(final_dir)

        return "completed", contigs
    except (Exception, KeyboardInterrupt) as exc:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        try:
            if zipfs is not None:
                zipfs.close()
        except Exception:
            pass


def process_sample_with_retries(sample: Sample, cfg: Config, run_id: str) -> SampleResult:
    """Process one sample, retrying on network errors.

    - NonRetryableError: fail immediately (no point retrying)
    - Network errors: wait (exponential backoff), then try again
    - Other errors: fail after all retries are used up
    """
    log(f"SAMPLE START {sample.sample_id}")
    attempts = cfg.retries + 1
    for attempt_idx in range(attempts):
        if shutdown_requested():
            log(f"SAMPLE CANCELLED {sample.sample_id} (shutdown requested before attempt {attempt_idx + 1})")
            return SampleResult(
                sample_id=sample.sample_id,
                status="cancelled",
                contigs=[],
                error=None,
                elapsed_s=time.perf_counter() - time.perf_counter(),
            )
        if attempt_idx > 0:
            sleep_s = _backoff_sleep_s(attempt_idx - 1)
            log(f"{sample.sample_id} retrying attempt {attempt_idx + 1}/{attempts} after {sleep_s:.1f}s")
            time.sleep(sleep_s)

        t0 = time.perf_counter()
        try:
            status, contigs = extract_sample_attempt(sample, cfg, run_id)
            elapsed = time.perf_counter() - t0
            log(f"SAMPLE {status.upper()} {sample.sample_id} (elapsed={elapsed:.1f}s)")
            return SampleResult(
                sample_id=sample.sample_id,
                status=status,
                contigs=contigs,
                error=None,
                elapsed_s=elapsed,
            )
        except NonRetryableError as exc:
            elapsed = time.perf_counter() - t0
            err = f"{type(exc).__name__}: {exc}"
            log(f"SAMPLE FAILED {sample.sample_id} (elapsed={elapsed:.1f}s) err={err}")
            return SampleResult(
                sample_id=sample.sample_id,
                status="failed",
                contigs=[],
                error=err,
                elapsed_s=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            err = f"{type(exc).__name__}: {exc}"
            if attempt_idx < attempts - 1 and is_retryable_exception(exc):
                log(f"{sample.sample_id} attempt {attempt_idx + 1}/{attempts} failed (retryable): {err}")
                continue
            log(f"SAMPLE FAILED {sample.sample_id} (elapsed={elapsed:.1f}s) err={err}")
            return SampleResult(
                sample_id=sample.sample_id,
                status="failed",
                contigs=[],
                error=err,
                elapsed_s=elapsed,
            )

    raise AssertionError("unreachable")


def run_pipeline(cfg: Config) -> RunSummary:
    """Run the full extraction pipeline.

    Loads samples from the TSV, downloads them in parallel, and returns
    a summary of what happened. Supports clean shutdown via Ctrl+C.

    Args:
        cfg: all the settings (input file, output dir, workers, etc.)

    Returns:
        RunSummary with per-sample results and overall counts.
    """
    install_signal_handler()
    run_id = make_run_id()
    started = time.perf_counter()

    if cfg.workers < 1:
        raise ValueError("workers must be >= 1")
    if cfg.retries < 0:
        raise ValueError("retries must be >= 0")
    if not Path(cfg.input_tsv).exists():
        raise FileNotFoundError(f"input TSV not found: {cfg.input_tsv}")

    samples = load_samples_tsv(cfg.input_tsv, cfg.limit_samples)
    if not samples:
        log("No samples to process.")
        return RunSummary(
            run_id=run_id,
            results=[],
            completed=0,
            skipped=0,
            failed=0,
            cancelled=0,
            elapsed_s=time.perf_counter() - started,
        )

    log(f"RUN START run_id={run_id} samples={len(samples)} workers={cfg.workers} retries={cfg.retries}")

    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    results: list[SampleResult] = []
    submitted_samples: dict[Any, Sample] = {}  # future -> Sample
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs = set()
        for s in samples:
            fut = ex.submit(process_sample_with_retries, s, cfg, run_id)
            futs.add(fut)
            submitted_samples[fut] = s

        # Poll with a timeout so the main thread can respond to Ctrl+C.
        remaining = set(futs)
        while remaining:
            if shutdown_requested():
                for pending in remaining:
                    pending.cancel()
                break

            done, remaining = wait(remaining, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    results.append(fut.result())
                except KeyboardInterrupt:
                    results.append(SampleResult(
                        sample_id=submitted_samples[fut].sample_id,
                        status="cancelled",
                        contigs=[],
                        error="interrupted by Ctrl+C",
                        elapsed_s=0.0,
                    ))

    # Any submitted sample not yet in results is cancelled.
    collected_ids = {r.sample_id for r in results}
    for fut, sample in submitted_samples.items():
        if sample.sample_id not in collected_ids:
            results.append(SampleResult(
                sample_id=sample.sample_id,
                status="cancelled",
                contigs=[],
                error=None,
                elapsed_s=0.0,
            ))

    # Clean up any leftover temp dirs after shutdown.
    if shutdown_requested():
        out = Path(cfg.output_dir)
        if out.exists():
            for p in out.glob("*.zarr.__tmp__*"):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    log(f"Cleaned up {p.name}")

    completed = sum(1 for r in results if r.status == "completed")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed_count = sum(1 for r in results if r.status == "failed")
    cancelled_count = sum(1 for r in results if r.status == "cancelled")

    elapsed = time.perf_counter() - started
    log(
        f"RUN SUMMARY completed={completed} skipped={skipped} "
        f"failed={failed_count} cancelled={cancelled_count} elapsed={elapsed:.1f}s"
    )

    return RunSummary(
        run_id=run_id,
        results=results,
        completed=completed,
        skipped=skipped,
        failed=failed_count,
        cancelled=cancelled_count,
        elapsed_s=elapsed,
    )
