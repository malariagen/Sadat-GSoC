"""Command-line interface for the mosquito classifier pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from fastq_classifier.features import DEFAULT_KMER_SIZE, DEFAULT_READ_PAIRS

_DEFAULT_PARALLEL_JOBS = 4
_DEFAULT_PREDICTION_BATCH_ROWS = 64


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one pipeline command."""
    argument_parser = _argument_parser()
    parsed_args = argument_parser.parse_args(arguments)
    try:
        if parsed_args.command == "download":
            from fastq_classifier.download import download_read_pairs

            command_artifact_path = download_read_pairs(
                parsed_args.ena_report,
                parsed_args.download_dir,
                read_pairs=parsed_args.read_pairs,
                jobs=parsed_args.jobs,
            )
        elif parsed_args.command == "count-kmers":
            from fastq_classifier.kmc import count_kmers

            command_artifact_path = count_kmers(
                parsed_args.fastq_manifest,
                parsed_args.count_dir,
                k=parsed_args.k,
                jobs=parsed_args.jobs,
            )
        elif parsed_args.command == "build-matrix":
            from fastq_classifier.matrix import build_count_matrix

            command_artifact_path = build_count_matrix(
                parsed_args.kmc_manifest,
                parsed_args.matrix_dir,
                jobs=parsed_args.jobs,
            )
        elif parsed_args.command == "assign-folds":
            from fastq_classifier.folds import assign_development_folds

            command_artifact_path = assign_development_folds(
                parsed_args.matrix_dir,
                parsed_args.development_samples,
                parsed_args.fold_dir,
            )
        elif parsed_args.command == "train":
            from fastq_classifier.training import train_classifier

            command_artifact_path = train_classifier(
                parsed_args.matrix_dir,
                parsed_args.folds_path,
                parsed_args.classifier_dir,
            )
        elif parsed_args.command == "predict":
            from fastq_classifier.classifier import classify_count_matrix

            command_artifact_path = classify_count_matrix(
                parsed_args.classifier_dir,
                parsed_args.matrix_dir,
                parsed_args.predictions_path,
                batch_size=parsed_args.batch_size,
            )
        else:
            raise AssertionError(f"Unknown command: {parsed_args.command}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(command_artifact_path)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(prog="fastq-classifier")
    subcommands = argument_parser.add_subparsers(dest="command", required=True)

    download_command = subcommands.add_parser(
        "download",
        help="download paired FASTQ prefixes",
    )
    download_command.add_argument("ena_report", type=Path)
    download_command.add_argument("download_dir", type=Path)
    download_command.add_argument(
        "--read-pairs",
        type=_positive_cli_integer,
        default=DEFAULT_READ_PAIRS,
    )
    download_command.add_argument(
        "--jobs",
        type=_positive_cli_integer,
        default=_DEFAULT_PARALLEL_JOBS,
    )

    count_command = subcommands.add_parser("count-kmers", help="count canonical k-mers with KMC")
    count_command.add_argument("fastq_manifest", type=Path)
    count_command.add_argument("count_dir", type=Path)
    count_command.add_argument("--k", type=int, default=DEFAULT_KMER_SIZE)
    count_command.add_argument(
        "--jobs",
        type=_positive_cli_integer,
        default=_DEFAULT_PARALLEL_JOBS,
    )

    matrix_command = subcommands.add_parser(
        "build-matrix",
        help="build the canonical 8-mer count matrix",
    )
    matrix_command.add_argument("kmc_manifest", type=Path)
    matrix_command.add_argument("matrix_dir", type=Path)
    matrix_command.add_argument(
        "--jobs",
        type=_positive_cli_integer,
        default=_DEFAULT_PARALLEL_JOBS,
    )

    folds_command = subcommands.add_parser(
        "assign-folds",
        help="assign grouped development folds",
    )
    folds_command.add_argument("matrix_dir", type=Path)
    folds_command.add_argument("development_samples", type=Path)
    folds_command.add_argument("fold_dir", type=Path)

    train_command = subcommands.add_parser(
        "train",
        help="train the five-class logistic classifier",
    )
    train_command.add_argument("matrix_dir", type=Path)
    train_command.add_argument("folds_path", type=Path)
    train_command.add_argument("classifier_dir", type=Path)

    predict_command = subcommands.add_parser(
        "predict",
        help="predict a compatible count matrix",
    )
    predict_command.add_argument("classifier_dir", type=Path)
    predict_command.add_argument("matrix_dir", type=Path)
    predict_command.add_argument("predictions_path", type=Path)
    predict_command.add_argument(
        "--batch-size",
        type=_positive_cli_integer,
        default=_DEFAULT_PREDICTION_BATCH_ROWS,
    )

    return argument_parser


def _positive_cli_integer(text: str) -> int:
    parsed_integer = int(text)
    if parsed_integer <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_integer
