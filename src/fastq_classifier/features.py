"""K-mer feature definition shared by counting, training, and prediction."""

from __future__ import annotations

import itertools

KMER_SIZE = 8
CANONICAL_KMER_COUNT = 32_896
READ_PAIRS_PER_RUN = 25_000
COUNTS_PER_MILLION = 1_000_000.0

KMER_FEATURE_METADATA: dict[str, object] = {
    "kind": "canonical_kmer_counts",
    "k": KMER_SIZE,
    "count": CANONICAL_KMER_COUNT,
    "read_pairs": READ_PAIRS_PER_RUN,
}
COUNT_NORMALIZATION_METADATA: dict[str, object] = {
    "counts_per_million": COUNTS_PER_MILLION,
    "transform": "log1p",
    "row_norm": "l2",
}

_DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(kmer: str) -> str:
    return kmer.translate(_DNA_COMPLEMENT)[::-1]


CANONICAL_KMERS = tuple(
    kmer
    for bases in itertools.product("ACGT", repeat=KMER_SIZE)
    if (kmer := "".join(bases)) <= _reverse_complement(kmer)
)
if len(CANONICAL_KMERS) != CANONICAL_KMER_COUNT:
    raise AssertionError(
        f"Expected {CANONICAL_KMER_COUNT} canonical {KMER_SIZE}-mers, got {len(CANONICAL_KMERS)}"
    )
