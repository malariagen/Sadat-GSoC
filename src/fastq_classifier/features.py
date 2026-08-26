"""K-mer feature definition shared by counting, training, and prediction."""

from __future__ import annotations

import itertools

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


CANONICAL_KMERS = canonical_kmers(DEFAULT_KMER_SIZE)
CANONICAL_KMER_COUNT = len(CANONICAL_KMERS)
KMER_FEATURE_METADATA: dict[str, object] = {
    "kind": "canonical_kmer_counts",
    "k": DEFAULT_KMER_SIZE,
    "count": CANONICAL_KMER_COUNT,
    "read_pairs": READ_PAIRS_PER_RUN,
}
