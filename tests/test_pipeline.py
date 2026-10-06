from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from conftest import ROOT
from src import experiments
from src.attributes import ATTRIBUTE_NAMES, align_columns
from src.checkpoint import Bundle
from src.config import compose, with_values
from src.data import load_split
from src.metrics import evaluate_output
from src.synthetic import make_synthetic_data


@pytest.mark.slow
def test_end_to_end(tmp_path, val):
    """configs/experiment/smoke.yaml through Hydra: candidates -> yardstick -> ensemble + WiSE -> LODO
    for the winner -> export -> run.py called like the ingestion program, in degraded mode."""
    data = make_synthetic_data(ROOT / "data", tmp_path / "data")
    cfg = with_values(compose(["profile=smoke", "experiment=smoke", "eval.tune_group_weights=true"]), {
        "data.dir": data, "experiment.root": tmp_path / "exp", "experiment.export": tmp_path / "sub",
        "runtime.scales": [1.0, 1.25], "runtime.time_budget_s": 1e-6})
    summary = experiments.run(cfg)

    kinds = [r["kind"] for r in summary["rows"]]
    assert kinds.count("single") == len(cfg.experiment.candidates) and "ensemble" in kinds and "wise" in kinds
    winner = Bundle.load(summary["winner_checkpoint"])
    assert winner.score.group_weights is not None
    if next(r for r in summary["rows"] if r["name"] == summary["winner"])["kind"] == "single":
        assert set(winner.meta["lodo"]["folds"]) == {"Market1501", "PA100k", "PETA"}
    assert summary["archive"].exists()

    spec = importlib.util.spec_from_file_location("sub_run", tmp_path / "sub" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.WORKERS = 0
    gallery = load_split(data, "val")
    names = list(ATTRIBUTE_NAMES)[::-1]    # the platform order must not matter
    sample = {"attribute_names": names,
              "queries": gallery.queries[:, align_columns(names).argsort()].tolist(),
              "gallery": [{"image_path": str(data / p)} for p in gallery.images]}
    out = mod.rank_gallery(sample)
    sims = out["similarities"]
    assert sims.shape == (len(gallery.queries), len(gallery)) and np.isfinite(sims).all()
    assert all(len(np.unique(row)) == len(row) for row in sims)
    assert 0 <= evaluate_output(out, gallery.queries, gallery.labels)["mADM"] <= 1
