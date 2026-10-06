"""UPAR 2027 Track 2 submission: CSAR (Calibrated Structured Attribute Retrieval).

Packaged by ``python -m src.export``: the ``src`` package is shipped next to
this file and the checkpoint in ``assets/model.pt`` carries the members
(one model or an ensemble), frozen calibration, score configuration and runtime
(TTA, time budget). Only torch, torchvision, numpy and Pillow are needed. The test
gallery is used for per-image inference only, unless the checkpoint enables the
transductive score. ``CSAR_TIME_BUDGET`` (seconds) overrides the stored budget.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import replace
from typing import Any

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.attributes import align_columns  # noqa: E402
from src.checkpoint import Bundle  # noqa: E402
from src.inference import encode_queries, fit_to_budget, predict  # noqa: E402
from src.scoring import Scorer  # noqa: E402

MODEL_PATH = os.path.join(HERE, "assets", "model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = min(8, os.cpu_count() or 1)
log = logging.getLogger("submission")
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")

_BUNDLE: Bundle | None = None
_MODELS: list = []


def load_model() -> None:
    global _BUNDLE, _MODELS
    _BUNDLE = Bundle.load(MODEL_PATH)
    if os.environ.get("CSAR_TIME_BUDGET"):
        _BUNDLE.runtime = replace(_BUNDLE.runtime, time_budget_s=float(os.environ["CSAR_TIME_BUDGET"]))
    _MODELS = [m.to(DEVICE) for m in _BUNDLE.build_models()]
    log.info("loaded %d member(s) on %s: %s | score=%s | runtime=%s", len(_MODELS), DEVICE,
             [m.model_cfg.backbone for m in _BUNDLE.members], _BUNDLE.score, _BUNDLE.runtime)


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    if not _MODELS:
        load_model()
    t0 = time.time()
    paths = [g["image_path"] for g in sample["gallery"]]
    kwargs = {"device": DEVICE, "num_workers": WORKERS, "strict": False}
    models, runtime = fit_to_budget(_MODELS, _BUNDLE.runtime, paths, **kwargs)
    preds = _BUNDLE.calibration.apply(predict(models, paths, runtime=runtime, **kwargs))
    queries = np.asarray(sample["queries"], dtype=np.float32)[:, align_columns(sample["attribute_names"])]
    q_emb = encode_queries(models, queries, DEVICE) if _BUNDLE.score.gamma else None
    sims = Scorer(_BUNDLE.score, preds, queries, _BUNDLE.prior, q_emb, DEVICE).full()
    log.info("ranked %d queries x %d images in %.1fs", *sims.shape, time.time() - t0)
    return {"similarities": sims}
