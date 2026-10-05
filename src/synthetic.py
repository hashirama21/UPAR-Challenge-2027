"""Tiny UPAR-formatted dataset with random images, to run the whole pipeline without downloads.

Labels are real (sampled from the official annotations), images are noise: useful
for smoke tests of train -> evaluate -> export -> run.py, meaningless for scores.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .attributes import ATTRIBUTE_NAMES
from .data import Split, load_split, with_protocol_queries


def write_split(root: Path, name: str, split: Split, seed: int = 0) -> None:
    """Write ``annotations/task2/<name>/{gt,ids,queries}.csv`` and one random image per row."""
    d = root / "annotations" / "task2" / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "gt.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# image", *ATTRIBUTE_NAMES])
        w.writerows([img, *row] for img, row in zip(split.images, split.labels.tolist()))
    with (d / "ids.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(zip(split.images, split.query_ids.tolist()))
    with (d / "queries.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# " + ATTRIBUTE_NAMES[0], *ATTRIBUTE_NAMES[1:]])
        w.writerows(split.queries.tolist())
    rng = np.random.default_rng(seed)
    for img in split.images:
        path = root / img
        path.parent.mkdir(parents=True, exist_ok=True)
        w, h = int(rng.integers(40, 90)), int(rng.integers(100, 200))
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)


def make_synthetic_data(source_dir: str | Path, out_dir: str | Path, n_train: int = 96, n_val: int = 48,
                        seed: int = 0) -> Path:
    """Sample ``n_train``/``n_val`` labelled rows from the official val split into ``out_dir``."""
    source = load_split(source_dir, "val")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(source))
    out = Path(out_dir)
    for name, rows in (("train", idx[:n_train]), ("val", idx[n_train:n_train + n_val])):
        write_split(out, name, with_protocol_queries(source.images[rows], source.labels[rows]), seed)
    return out
