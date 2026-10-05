"""UPAR 2027 Track 2 submission: CSAR (Calibrated Structured Attribute Retrieval).

Packaged by ``python -m src.export``: the ``src`` package is shipped next to
this file and the checkpoint in ``assets/model.pt`` carries the architecture,
weights, frozen calibration and score configuration. Only torch, torchvision,
numpy and Pillow are needed. The test gallery is used for per-image inference
only, unless the checkpoint enables the transductive score.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.attributes import align_columns  # noqa: E402
from src.checkpoint import Bundle  # noqa: E402
from src.inference import encode_queries, predict  # noqa: E402
from src.scoring import Scorer  # noqa: E402

MODEL_PATH = os.path.join(HERE, "assets", "model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = min(8, os.cpu_count() or 1)
log = logging.getLogger("submission")
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")

_BUNDLE: Bundle | None = None
_MODEL = None


def load_model() -> None:
    global _BUNDLE, _MODEL
    _BUNDLE = Bundle.load(MODEL_PATH)
    _MODEL = _BUNDLE.build_model().to(DEVICE)
    log.info("loaded %s on %s, score=%s", _BUNDLE.model_cfg, DEVICE, _BUNDLE.score)


def _calibrated_predictions(gallery: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if _MODEL is None:
        load_model()
    paths = [g["image_path"] for g in gallery]
    preds = predict(_MODEL, paths, device=DEVICE, num_workers=WORKERS, strict=False)
    return _BUNDLE.calibration.apply(preds)


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    preds = _calibrated_predictions(sample["gallery"])
    queries = np.asarray(sample["queries"], dtype=np.float32)[:, align_columns(sample["attribute_names"])]
    q_emb = encode_queries(_MODEL, queries, DEVICE) if _BUNDLE.score.gamma else None
    scorer = Scorer(_BUNDLE.score, preds, queries, _BUNDLE.prior, q_emb, DEVICE)
    sims = scorer.full()
    log.info("ranked %d queries x %d images in %.1fs", *sims.shape, time.time() - t0)
    return {"similarities": sims}
