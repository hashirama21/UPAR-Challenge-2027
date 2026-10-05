"""Local re-implementation of the Track 2 metrics: mADM, mAP, Rank-1/5/10, mINP.

A gallery image is a match iff all 40 attributes equal the query. ADM credits
each image of the top-r_k with its normalised degree of match. Ties keep the
gallery order (stable sort), as the official scorer does.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

KEYS = ("distances", "similarities", "ranking")
METRICS = ("mADM", "mAP", "R1", "R5", "R10", "mINP")


def _order(row: np.ndarray, key: str) -> np.ndarray:
    if key == "distances":
        return np.argsort(row, kind="stable")
    if key == "similarities":
        return np.argsort(-row, kind="stable")
    if key == "ranking":
        return np.asarray(row)
    raise ValueError(f"key must be one of {KEYS}")


def evaluate_blocks(blocks: Iterable[np.ndarray], queries: np.ndarray, gallery: np.ndarray,
                    key: str = "similarities") -> dict[str, float]:
    """``blocks`` yields consecutive (b, G) score rows covering all queries in order."""
    q_all = queries.astype(np.int32)
    g = gallery.astype(np.int32)
    n_attr = q_all.shape[1]
    res: dict[str, list[float]] = {m: [] for m in METRICS}
    s = 0
    for mat in blocks:
        qb = q_all[s:s + len(mat)]
        s += len(mat)
        agree = qb @ g.T + (1 - qb) @ (1 - g).T
        for row, ag in zip(np.asarray(mat), agree):
            order = _order(row, key)
            ranks = np.flatnonzero(ag[order] == n_attr) + 1
            if len(ranks) == 0:
                continue
            dom = ag / n_attr
            mean, top = dom.mean(), dom.max()
            ndom = np.maximum(0.0, (dom - mean) / (top - mean)) if top > mean else np.zeros_like(dom)
            cum = np.cumsum(ndom[order])
            n = len(ranks)
            res["mADM"].append(float(np.mean(cum[ranks - 1] / ranks)))
            res["mAP"].append(float(np.mean(np.arange(1, n + 1) / ranks)))
            res["mINP"].append(float(n / ranks[-1]))
            for k in (1, 5, 10):
                res[f"R{k}"].append(float(ranks[0] <= k))
    if s != len(q_all):
        raise ValueError(f"blocks covered {s} queries, expected {len(q_all)}")
    return {m: float(np.mean(v)) if v else float("nan") for m, v in res.items()}


def evaluate_output(output: dict, queries: np.ndarray, gallery: np.ndarray, block: int = 256) -> dict[str, float]:
    """Evaluate a ``rank_gallery`` output dict (exactly one of KEYS)."""
    (key, mat), = output.items()
    mat = np.asarray(mat)
    return evaluate_blocks((mat[s:s + block] for s in range(0, len(mat), block)), queries, gallery, key)
