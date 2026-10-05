"""Attribute vocabulary and its group structure: the single source of truth.

Groups are *almost* categorical on the official annotations (98-99.99 % of
vectors have exactly one active attribute per group, see README_impl.md).
A group is encoded as a class index:

    0..n-1   the single active attribute
    n        "none" (only when ``allow_none``; e.g. 9.7 % of train hair labels)
    -1       not representable (two colours, illegal "none"); callers fall back
             to independent Bernoulli terms for that group.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ATTRIBUTE_NAMES: tuple[str, ...] = (
    "Age-Young", "Age-Adult", "Age-Old",
    "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short",
    "UpperBody-Color-Black", "UpperBody-Color-Blue", "UpperBody-Color-Brown",
    "UpperBody-Color-Green", "UpperBody-Color-Grey", "UpperBody-Color-Orange",
    "UpperBody-Color-Pink", "UpperBody-Color-Purple", "UpperBody-Color-Red",
    "UpperBody-Color-White", "UpperBody-Color-Yellow", "UpperBody-Color-Other",
    "LowerBody-Length-Short",
    "LowerBody-Color-Black", "LowerBody-Color-Blue", "LowerBody-Color-Brown",
    "LowerBody-Color-Green", "LowerBody-Color-Grey", "LowerBody-Color-Orange",
    "LowerBody-Color-Pink", "LowerBody-Color-Purple", "LowerBody-Color-Red",
    "LowerBody-Color-White", "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag",
    "Accessory-Glasses-Normal", "Accessory-Glasses-Sun",
    "Accessory-Hat",
)
NUM_ATTRS = len(ATTRIBUTE_NAMES)


@dataclass(frozen=True)
class Group:
    name: str
    indices: tuple[int, ...]
    allow_none: bool

    @property
    def num_classes(self) -> int:
        return len(self.indices) + int(self.allow_none)


GROUPS: tuple[Group, ...] = (
    Group("age", (0, 1, 2), allow_none=False),
    Group("hair", (4, 5, 6), allow_none=True),
    Group("upper_color", tuple(range(8, 20)), allow_none=False),
    Group("lower_color", tuple(range(21, 33)), allow_none=False),
    Group("lower_type", (33, 34), allow_none=True),
    Group("glasses", (37, 38), allow_none=True),
)

GROUPED = frozenset(i for g in GROUPS for i in g.indices)
INDEPENDENT: tuple[int, ...] = tuple(i for i in range(NUM_ATTRS) if i not in GROUPED)


def encode_groups(labels: np.ndarray | torch.Tensor) -> torch.Tensor:
    """(N, 40) binary vectors -> (N, len(GROUPS)) class indices, -1 if not representable."""
    y = torch.as_tensor(labels).float()
    out = torch.full((y.shape[0], len(GROUPS)), -1, dtype=torch.long)
    for k, g in enumerate(GROUPS):
        sub = y[:, list(g.indices)]
        active = sub.sum(1)
        out[active == 1, k] = sub.argmax(1)[active == 1]
        if g.allow_none:
            out[active == 0, k] = len(g.indices)
    return out


def align_columns(attribute_names: list[str] | tuple[str, ...]) -> np.ndarray:
    """Column permutation mapping an external attribute order onto ATTRIBUTE_NAMES."""
    pos = {n: i for i, n in enumerate(attribute_names)}
    missing = [n for n in ATTRIBUTE_NAMES if n not in pos]
    if missing:
        raise ValueError(f"attribute_names lacks {missing}")
    return np.array([pos[n] for n in ATTRIBUTE_NAMES])
