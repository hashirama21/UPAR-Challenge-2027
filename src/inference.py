"""Batched gallery inference shared by evaluation and the submission.

Views (scales x optional flip) are averaged per member, members are averaged in
logit space; embeddings of an ensemble are concatenated and scaled by 1/sqrt(M)
so their dot product is the mean of the members' cosine similarities.
``fit_to_budget`` implements the degraded mode: it measures throughput on a probe
and drops extra scales, then flip, then trailing members until the estimate fits.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .calibration import Predictions
from .checkpoint import Runtime
from .dataset import ImageDataset
from .model import CSARNet
from .transforms import build_transform

log = logging.getLogger(__name__)


@torch.no_grad()
def predict_model(model: CSARNet, paths: Sequence[str | Path], root: str | Path | None = None, *,
                  scales: Sequence[float] = (1.0,), flip: bool = True, batch_size: int = 128, num_workers: int = 4,
                  device: str | torch.device = "cpu", strict: bool = True) -> Predictions:
    """Raw (uncalibrated) outputs of one model, averaged over its views, on CPU in float32."""
    device = torch.device(device)
    model.eval().to(device)
    sums: dict[str, torch.Tensor] = {}
    for scale in scales:
        ds = ImageDataset(paths, build_transform(model.cfg, scale=scale), root, strict=strict)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda")
        chunks: dict[str, list[torch.Tensor]] = {}
        t0, done = time.time(), 0
        for x, _ in dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                views = [model(x)] + ([model(x.flip(-1))] if flip else [])
            for k in views[0]:
                chunks.setdefault(k, []).append(sum(v[k].float() for v in views).cpu())
            done += len(x)
            if done % (batch_size * 20) < batch_size or done == len(ds):
                log.info("inference scale %.2f: %d/%d images (%.0f img/s)", scale, done, len(ds),
                         done / (time.time() - t0 + 1e-9))
        for k, v in chunks.items():
            sums[k] = sums.get(k, 0) + torch.cat(v)
    n_views = len(scales) * (2 if flip else 1)
    preds = {k: v / n_views for k, v in sums.items()}
    if "emb" in preds:
        preds["emb"] = F.normalize(preds["emb"], dim=-1)
    return preds


def combine(per_model: list[Predictions]) -> Predictions:
    out = {k: torch.stack([p[k] for p in per_model]).mean(0) for k in per_model[0] if k != "emb"}
    if all("emb" in p for p in per_model):
        out["emb"] = torch.cat([p["emb"] for p in per_model], -1) / len(per_model) ** 0.5
    return out


def predict(models: list[CSARNet], paths: Sequence[str | Path], root: str | Path | None = None, *,
            runtime: Runtime, **kwargs) -> Predictions:
    return combine([predict_model(m, paths, root, scales=runtime.scales, flip=runtime.flip, **kwargs)
                    for m in models])


@torch.no_grad()
def encode_queries(models: list[CSARNet], queries, device: str | torch.device = "cpu") -> torch.Tensor | None:
    if not all(m.has_retrieval for m in models):
        return None
    q = torch.as_tensor(queries, dtype=torch.float32, device=device)
    return torch.cat([m.eval().to(device).encode_queries(q).float().cpu() for m in models], -1) / len(models) ** 0.5


def fit_to_budget(models: list[CSARNet], runtime: Runtime, paths: Sequence[str | Path],
                  **kwargs) -> tuple[list[CSARNet], Runtime]:
    """Degrade TTA, then the ensemble, until the estimated inference time fits ``runtime.time_budget_s``."""
    if runtime.time_budget_s <= 0 or not paths:
        return models, runtime
    sample = list(paths[:runtime.probe_images])
    costs = []
    for m in models:
        t0 = time.time()
        predict_model(m, sample, scales=(1.0,), flip=False, **kwargs)
        costs.append((time.time() - t0) / len(sample))

    def estimate(n_models: int, rt: Runtime) -> float:
        return len(paths) * sum(costs[:n_models]) * sum(s * s for s in rt.scales) * (2 if rt.flip else 1)

    n, rt = len(models), runtime
    steps = [lambda n, rt: (n, replace(rt, scales=[1.0])), lambda n, rt: (n, replace(rt, flip=False))]
    steps += [lambda n, rt, k=k: (k, rt) for k in range(len(models) - 1, 0, -1)]
    for step in steps:
        if estimate(n, rt) <= runtime.time_budget_s:
            break
        n, rt = step(n, rt)
    log.info("budget %.0fs: %d/%d members, scales=%s, flip=%s, estimated %.0fs",
             runtime.time_budget_s, n, len(models), rt.scales, rt.flip, estimate(n, rt))
    return models[:n], rt
