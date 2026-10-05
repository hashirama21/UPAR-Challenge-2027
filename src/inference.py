"""Batched gallery inference shared by training-time evaluation and the submission."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .calibration import Predictions
from .dataset import ImageDataset
from .model import CSARNet
from .transforms import build_transform

log = logging.getLogger(__name__)


@torch.no_grad()
def predict(model: CSARNet, paths: Sequence[str | Path], root: str | Path | None = None, *,
            batch_size: int = 128, num_workers: int = 4, device: str | torch.device = "cpu",
            flip_tta: bool = True, strict: bool = True) -> Predictions:
    """Raw (uncalibrated) logits/embeddings for every image, on CPU in float32.

    With ``flip_tta`` logits are averaged over the image and its mirror.
    """
    device = torch.device(device)
    cfg = model.cfg
    ds = ImageDataset(paths, build_transform(cfg.height, cfg.width, train=False), root, strict=strict)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda")
    model.eval().to(device)
    chunks: dict[str, list[torch.Tensor]] = {}
    t0, done = time.time(), 0
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
            if flip_tta:
                mirrored = model(x.flip(-1))
                out = {k: (v + mirrored[k]) / 2 for k, v in out.items()}
        for k, v in out.items():
            chunks.setdefault(k, []).append(v.float().cpu())
        done += len(x)
        if done % (batch_size * 20) < batch_size or done == len(ds):
            log.info("inference %d/%d images (%.0f img/s)", done, len(ds), done / (time.time() - t0 + 1e-9))
    preds = {k: torch.cat(v) for k, v in chunks.items()}
    if "emb" in preds:  # averaging two unit vectors shrinks them
        preds["emb"] = torch.nn.functional.normalize(preds["emb"], dim=-1)
    return preds


@torch.no_grad()
def encode_queries(model: CSARNet, queries, device: str | torch.device = "cpu") -> torch.Tensor | None:
    if not model.has_retrieval:
        return None
    q = torch.as_tensor(queries, dtype=torch.float32, device=device)
    return model.eval().to(device).encode_queries(q).float().cpu()
