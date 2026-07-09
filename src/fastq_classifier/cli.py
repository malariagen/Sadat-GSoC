"""Console entry points for the FASTQ classifier package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast


def build_parser() -> argparse.ArgumentParser:
    """Create the ``fastq-classifier`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with package subcommands.
    """
    parser = argparse.ArgumentParser(prog="fastq-classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser(
        "fetch-first-n",
        help="Fetch first N paired FASTQ reads from an ENA run report TSV.",
    )
    fetch.add_argument("--ena-report", required=True, type=Path)
    fetch.add_argument("--out-dir", required=True, type=Path)
    fetch.add_argument("--read-pairs", required=True, type=int)
    fetch.add_argument("--jobs", default=1, type=int)
    fetch.add_argument(
        "--curl",
        default="curl",
        help="curl executable or path. Defaults to the curl found on PATH.",
    )
    fetch.add_argument(
        "--seqkit",
        default="seqkit",
        help="SeqKit executable or path. Use this when seqkit is not on PATH.",
    )

    extract = subparsers.add_parser(
        "extract-kmers",
        help="Extract exact canonical k-mer counts with KMC.",
    )
    extract.add_argument("--fetch-results", required=True, type=Path)
    extract.add_argument("--out-dir", required=True, type=Path)
    extract.add_argument("--k", required=True, type=int)
    extract.add_argument("--jobs", default=1, type=int)
    extract.add_argument(
        "--kmc",
        default="kmc",
        help="KMC executable or path. Use this when kmc is not on PATH.",
    )
    extract.add_argument(
        "--memory-gb",
        default=2,
        type=int,
        help="Memory limit for each KMC process. KMC requires at least 2.",
    )

    matrix = subparsers.add_parser(
        "build-matrix",
        help="Build a sparse k-mer matrix from KMC feature results.",
    )
    matrix.add_argument("--feature-results", required=True, type=Path)
    matrix.add_argument("--out-dir", required=True, type=Path)
    matrix.add_argument("--jobs", default=1, type=int)
    matrix.add_argument(
        "--kmc-dump",
        default="kmc_dump",
        help="KMC dump executable or path. Use this when kmc_dump is not on PATH.",
    )

    evaluate = subparsers.add_parser(
        "evaluate-classifier",
        help="Train and evaluate a sparse k-mer classifier.",
    )
    evaluate.add_argument("--matrix-dir", required=True, type=Path)
    evaluate.add_argument("--out-dir", required=True, type=Path)
    evaluate.add_argument("--label-column", required=True)
    evaluate.add_argument("--test-size", default=0.25, type=float)
    evaluate.add_argument("--seed", default=1, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line program.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments without the program name. When ``None``,
        ``argparse`` reads from ``sys.argv``.

    Returns
    -------
    int
        ``0`` after the command finishes, or ``2`` for invalid input and fetch
        setup errors.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    command = cast("str", args.command)

    if command == "fetch-first-n":
        return run_fetch_command(args)

    if command == "extract-kmers":
        return run_extract_command(args)

    if command == "build-matrix":
        return run_matrix_command(args)

    if command == "evaluate-classifier":
        return run_evaluate_command(args)

    sys.stderr.write(f"error: Unknown command: {command}\n")
    return 2


def run_fetch_command(args: argparse.Namespace) -> int:
    """Run the ``fetch-first-n`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from fastq_classifier.fetch_first_n import FetchError, fetch_first_n  # noqa: PLC0415
    from fastq_classifier.utils import InputError  # noqa: PLC0415

    ena_report = cast("Path", args.ena_report)
    out_dir = cast("Path", args.out_dir)
    read_pairs = cast("int", args.read_pairs)
    jobs = cast("int", args.jobs)
    curl = cast("str", args.curl)
    seqkit = cast("str", args.seqkit)
    try:
        report = fetch_first_n(
            ena_report,
            out_dir,
            read_pairs,
            jobs=jobs,
            curl=(curl,),
            seqkit=(seqkit,),
        )
    except (FetchError, InputError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    summary = f"downloads={len(report.downloads)} invalid={len(report.invalid_rows)}"
    sys.stdout.write(f"{summary} out_dir={out_dir}\n")
    return 0


def run_extract_command(args: argparse.Namespace) -> int:
    """Run the ``extract-kmers`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from fastq_classifier.extract_features import (  # noqa: PLC0415
        KmerExtractionError,
        extract_kmer_features,
    )
    from fastq_classifier.utils import InputError  # noqa: PLC0415

    fetch_results = cast("Path", args.fetch_results)
    out_dir = cast("Path", args.out_dir)
    k = cast("int", args.k)
    jobs = cast("int", args.jobs)
    kmc = cast("str", args.kmc)
    memory_gb = cast("int", args.memory_gb)
    try:
        extraction = extract_kmer_features(
            fetch_results,
            out_dir,
            k,
            jobs=jobs,
            kmc=(kmc,),
            memory_gb=memory_gb,
        )
    except (KmerExtractionError, InputError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    summary = f"databases={len(extraction.databases)} invalid={len(extraction.invalid_rows)}"
    sys.stdout.write(f"{summary} out_dir={out_dir}\n")
    return 0


def run_matrix_command(args: argparse.Namespace) -> int:
    """Run the ``build-matrix`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from fastq_classifier.build_matrix import MatrixBuildError, build_kmer_matrix  # noqa: PLC0415
    from fastq_classifier.utils import InputError  # noqa: PLC0415

    feature_results = cast("Path", args.feature_results)
    out_dir = cast("Path", args.out_dir)
    kmc_dump = cast("str", args.kmc_dump)
    jobs = cast("int", args.jobs)
    try:
        matrix = build_kmer_matrix(
            feature_results,
            out_dir,
            kmc_dump=(kmc_dump,),
            jobs=jobs,
        )
    except (InputError, MatrixBuildError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    summary = f"samples={matrix.sample_count} invalid={len(matrix.invalid_rows)}"
    metrics = f"features={matrix.feature_count} entries={matrix.entry_count}"
    sys.stdout.write(f"{summary} {metrics} out_dir={out_dir}\n")
    return 0


def run_evaluate_command(args: argparse.Namespace) -> int:
    """Run the ``evaluate-classifier`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from fastq_classifier.evaluate_classifier import (  # noqa: PLC0415
        ClassifierError,
        evaluate_classifier,
    )
    from fastq_classifier.utils import InputError  # noqa: PLC0415

    matrix_dir = cast("Path", args.matrix_dir)
    out_dir = cast("Path", args.out_dir)
    label_column = cast("str", args.label_column)
    test_size = cast("float", args.test_size)
    seed = cast("int", args.seed)
    try:
        result = evaluate_classifier(
            matrix_dir,
            out_dir,
            label_column,
            test_size=test_size,
            random_seed=seed,
        )
    except (InputError, ClassifierError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    summary = f"samples={result.sample_count} features={result.feature_count}"
    metrics = f"accuracy={result.accuracy:.6f} balanced_accuracy={result.balanced_accuracy:.6f}"
    sys.stdout.write(f"{summary} {metrics} out_dir={out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
