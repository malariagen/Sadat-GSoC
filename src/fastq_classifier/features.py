"""K-mer feature definition shared by counting, training, and prediction."""

from __future__ import annotations

import itertools
from pathlib import Path

DEFAULT_KMER_SIZE = 8
MINIMUM_KMER_SIZE = 4
MAXIMUM_KMER_SIZE = 8
READ_PAIRS_PER_RUN = 25_000
COUNTS_PER_MILLION = 1_000_000.0
COUNT_NORMALIZATION_METADATA: dict[str, object] = {
    "counts_per_million": COUNTS_PER_MILLION,
    "transform": "log1p",
    "row_norm": "l2",
}

_DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(kmer: str) -> str:
    return kmer.translate(_DNA_COMPLEMENT)[::-1]


def canonical_kmers(k: int) -> tuple[str, ...]:
    return tuple(
        kmer
        for bases in itertools.product("ACGT", repeat=k)
        if (kmer := "".join(bases)) <= _reverse_complement(kmer)
    )


def read_canonical_kmers(kmers_path: Path) -> tuple[str, ...]:
    kmers = tuple(kmers_path.read_text(encoding="ascii").splitlines())
    k = len(kmers[0]) if kmers else 0
    if not MINIMUM_KMER_SIZE <= k <= MAXIMUM_KMER_SIZE or kmers != canonical_kmers(k):
        raise ValueError(f"K-mer file {kmers_path} does not contain a supported vocabulary")
    return kmers


def kmer_feature_metadata(kmers: tuple[str, ...]) -> dict[str, object]:
    return {
        "kind": "canonical_kmer_counts",
        "k": len(kmers[0]),
        "count": len(kmers),
        "read_pairs": READ_PAIRS_PER_RUN,
    }
