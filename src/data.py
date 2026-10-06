"""Annotation loading and gallery construction following the UPAR protocol.

Protocol (README of the starter kit): every distinct 40-bit vector of a gallery
is one query, and each image maps to exactly one query. ``with_protocol_queries``
rebuilds that structure for any image subset (domain fold, resampled gallery),
so all evaluation code paths share the same construction as the hidden test.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .attributes import NUM_ATTRS

DOMAINS: tuple[str, ...] = ("Market1501", "PA100k", "PETA")


@dataclass(frozen=True)
class Split:
    images: np.ndarray       # (N,) relative paths, str
    labels: np.ndarray       # (N, 40) int8
    query_ids: np.ndarray    # (N,) int, row of ``queries``
    queries: np.ndarray      # (Q, 40) int8, unique vectors

    def __len__(self) -> int:
        return len(self.images)

    @property
    def domains(self) -> np.ndarray:
        return np.array([p.split("/", 1)[0] for p in self.images])

    def subset(self, mask: np.ndarray) -> "Split":
        """Image subset with its query vocabulary rebuilt per the protocol."""
        return with_protocol_queries(self.images[mask], self.labels[mask])

    def without_queries(self, drop: np.ndarray) -> "Split":
        """Open-set variant (experiment B): drop query rows, keep their images as distractors.

        Distractor images get ``query_ids == -1``.
        """
        keep = np.setdiff1d(np.arange(len(self.queries)), drop)
        remap = np.full(len(self.queries), -1)
        remap[keep] = np.arange(len(keep))
        return replace(self, queries=self.queries[keep], query_ids=remap[self.query_ids])


def with_protocol_queries(images: np.ndarray, labels: np.ndarray) -> Split:
    queries, query_ids = np.unique(labels, axis=0, return_inverse=True)
    return Split(images, labels, query_ids.reshape(-1), queries.astype(np.int8))


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [row for row in csv.reader(fh) if row and not row[0].startswith("#")]


def load_split(data_dir: str | Path, split: str) -> Split:
    root = Path(data_dir) / "annotations" / "task2" / split
    gt = _read_rows(root / "gt.csv")
    ids = _read_rows(root / "ids.csv")
    images = np.array([r[0] for r in gt])
    labels = np.array([r[1:] for r in gt], dtype=np.int8)
    queries = np.array(_read_rows(root / "queries.csv"), dtype=np.int8)
    if [r[0] for r in ids] != list(images):
        raise ValueError(f"{root}: ids.csv and gt.csv image order differ")
    query_ids = np.array([int(r[1]) for r in ids])
    if labels.shape[1] != NUM_ATTRS or not np.array_equal(queries[query_ids], labels):
        raise ValueError(f"{root}: inconsistent gt/ids/queries")
    return Split(images, labels, query_ids, queries)


def domain_fold(train: Split, val: Split, holdout: str) -> tuple[Split, Split]:
    """Leave-one-domain-out: train on the two other domains, evaluate on ``holdout``."""
    if holdout not in DOMAINS:
        raise ValueError(f"holdout must be one of {DOMAINS}")
    return train.subset(train.domains != holdout), val.subset(val.domains == holdout)


def query_mask(split: Split, frac: float = 1.0, num_queries: int = 0, seed: int = 0,
               by_size: bool = False) -> np.ndarray:
    """Image mask selecting a random subset of queries (all their images).

    Use ``frac`` for a query-disjoint split (e.g. calibration / evaluation halves of
    a held-out domain, with ``~mask``), or ``num_queries`` for test-sized galleries;
    ``by_size`` draws queries in proportion to their image count, which yields the
    many-images-per-query galleries of the hidden test (2024: ~77 per query).
    Apply with ``split.subset(mask)`` and the same mask on the predictions.
    """
    rng = np.random.default_rng(seed)
    n = len(split.queries)
    k = min(num_queries, n) if num_queries else round(frac * n)
    p = None
    if by_size:
        counts = np.bincount(split.query_ids, minlength=n).astype(np.float64)
        p = counts / counts.sum()
    chosen = np.zeros(n, dtype=bool)
    chosen[rng.choice(n, size=k, replace=False, p=p)] = True
    return chosen[split.query_ids]


def novel_queries(split: Split, reference: Split) -> np.ndarray:
    """Boolean per query of ``split``: True if the vector never occurs in ``reference``."""
    known = {row.tobytes() for row in reference.queries}
    return np.array([row.tobytes() not in known for row in split.queries])


def sample_reliability(split: Split, strength: float) -> np.ndarray:
    """Per-image weight exp(-strength * #attributes disagreeing with the identity majority).

    Market1501 labels are per identity but set per image, so an image whose vector
    departs from its identity's majority is likely mislabelled. Other domains get 1.
    """
    w = np.ones(len(split), dtype=np.float32)
    market = split.domains == "Market1501"
    if not strength or not market.any():
        return w
    ids = np.array([Path(p).name.split("_", 1)[0] for p in split.images[market]])
    labels = split.labels[market].astype(np.float32)
    _, inv = np.unique(ids, return_inverse=True)
    sums = np.zeros((inv.max() + 1, labels.shape[1]))
    np.add.at(sums, inv, labels)
    majority = sums / np.bincount(inv)[:, None] >= 0.5
    disagree = np.abs(labels - majority[inv]).sum(1)
    w[market] = np.exp(-strength * disagree)
    return w


def sample_weights(split: Split, domain_balanced: bool = True, query_alpha: float = 0.0) -> np.ndarray:
    """Per-image sampling weights: equal mass per domain, times n(query)^-alpha."""
    w = np.ones(len(split), dtype=np.float64)
    if domain_balanced:
        _, inv, counts = np.unique(split.domains, return_inverse=True, return_counts=True)
        w /= counts[inv]
    if query_alpha:
        w *= np.bincount(split.query_ids)[split.query_ids] ** -query_alpha
    return w / w.sum()


def hamming_neighbors(queries: np.ndarray, max_dist: int = 2, max_per_query: int = 32,
                      block: int = 1024) -> list[np.ndarray]:
    """For each query, other vocabulary entries within ``max_dist`` bits, closest first."""
    q = queries.astype(np.int32)
    out: list[np.ndarray] = []
    for s in range(0, len(q), block):
        b = q[s:s + block]
        dist = NUM_ATTRS - (b @ q.T + (1 - b) @ (1 - q).T)
        for i, row in enumerate(dist):
            row[s + i] = NUM_ATTRS + 1  # exclude self
            cand = np.flatnonzero(row <= max_dist)
            out.append(cand[np.argsort(row[cand], kind="stable")][:max_per_query])
    return out
