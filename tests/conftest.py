from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import compose, loss_config, model_config, runtime_config, score_config  # noqa: E402
from src.data import Split, load_split, query_mask  # noqa: E402

HAS_DATA = (ROOT / "data" / "annotations" / "task2" / "val" / "gt.csv").exists()


@pytest.fixture(scope="session")
def cfg():
    """configs/config.yaml with the smoke profile: the single source of every default."""
    return compose(["profile=smoke"])


@pytest.fixture(scope="session")
def make_model_cfg(cfg):
    return lambda **kw: replace(model_config(cfg), **kw)


@pytest.fixture(scope="session")
def make_score_cfg(cfg):
    return lambda **kw: replace(score_config(cfg), **kw)


@pytest.fixture(scope="session")
def make_loss_cfg(cfg):
    return lambda **kw: replace(loss_config(cfg), **kw)


@pytest.fixture(scope="session")
def make_runtime(cfg):
    return lambda **kw: replace(runtime_config(cfg), **kw)


@pytest.fixture(scope="session")
def val() -> Split:
    if not HAS_DATA:
        pytest.skip("annotations not available")
    return load_split(ROOT / "data", "val")


@pytest.fixture(scope="session")
def train_split() -> Split:
    if not HAS_DATA:
        pytest.skip("annotations not available")
    return load_split(ROOT / "data", "train")


@pytest.fixture(scope="session")
def small(val) -> Split:
    return val.subset(query_mask(val, num_queries=60, seed=1))
