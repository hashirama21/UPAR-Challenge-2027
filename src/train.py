"""Reproducible training of a CSAR model (baseline recipe + structure-aware losses).

Recipe (UPAR baseline, Specker et al. 2023): ImageNet backbone, weighted BCE with
label smoothing, AdamW, plateau schedule, EMA of the weights, flip/crop, no random
erasing, no colour augmentation. Additions: group softmax loss, query-image
contrastive loss with Hamming hard negatives, domain-balanced sampling,
low-resolution simulation, per-source smoothing, reliability weights,
attribute-query head, LoRA / partial fine-tuning for foundation encoders.

With ``data.holdout`` the epoch is selected on one query-disjoint half of the
held-out domain (``query_mask(frac=0.5, seed)``); ``src.evaluate`` with
``eval.calib_frac=0.5`` calibrates on that same half and reports on the other one,
so LODO numbers stay unbiased. All settings: ``configs/config.yaml``.

    python -m src.train train.out=runs/final
    python -m src.train data.holdout=PETA train.out=runs/lodo_peta
    python -m src.train model.backbone=clip_vitb16 model.head=query model.text_init=true model.lora_rank=16
"""
from __future__ import annotations

import copy
import json
import logging
import math
import platform
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .calibration import fit_calibration
from .checkpoint import Bundle, Member
from .config import (augment_config, device as resolve_device, entrypoint, loss_config, model_config,
                     runtime_config, score_config, to_dict)
from .data import domain_fold, hamming_neighbors, load_split, query_mask, sample_reliability, sample_weights
from .dataset import ImageDataset
from .evaluate import report
from .finetune import apply_lora, freeze_backbone, merge_lora, param_groups
from .inference import encode_queries, predict
from .losses import CSARLoss, sample_smoothing
from .model import CSARNet
from .transforms import build_transform

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LRSchedule:
    """Linear warm-up, then cosine decay or plateau (x factor after ``patience`` epochs without gain)."""

    def __init__(self, opt: torch.optim.Optimizer, kind: str, warmup: int, total: int, patience: int, factor: float):
        self.opt, self.kind, self.warmup, self.total = opt, kind, warmup, total
        self.patience, self.decay = patience, factor
        self.base = [g["lr"] for g in opt.param_groups]
        self.factor, self.best, self.stale, self.step_no = 1.0, -math.inf, 0, 0
        self._apply()

    def _apply(self) -> None:
        scale = min(1.0, (self.step_no + 1) / max(1, self.warmup))
        if self.kind == "cosine" and self.step_no >= self.warmup:
            scale = 0.5 * (1 + math.cos(math.pi * (self.step_no - self.warmup) / max(1, self.total - self.warmup)))
        for g, base in zip(self.opt.param_groups, self.base):
            g["lr"] = base * scale * self.factor

    def step(self) -> None:
        self.step_no += 1
        self._apply()

    def epoch_end(self, metric: float) -> None:
        if self.kind != "plateau":
            return
        if metric > self.best:
            self.best, self.stale = metric, 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                self.factor, self.stale = self.factor * self.decay, 0
                log.info("plateau: learning rate x%.0e", self.factor)


def _snapshot(ema: AveragedModel) -> dict[str, torch.Tensor]:
    """EMA weights with LoRA adapters folded in, as plain CSARNet keys."""
    return {k: v.detach().cpu() for k, v in merge_lora(copy.deepcopy(ema.module)).state_dict().items()}


def build_model(cfg: DictConfig) -> CSARNet:
    model = CSARNet(model_config(cfg), pretrained=cfg.model.pretrained)
    if cfg.model.text_init:
        from .clip_text import init_queries_from_text
        init_queries_from_text(model)
    if cfg.model.lora_rank:
        n = apply_lora(model, cfg.model.lora_rank, cfg.model.lora_alpha)
        log.info("LoRA rank %d on %d layers", cfg.model.lora_rank, n)
    else:
        freeze_backbone(model, cfg.model.trainable_stages)
    return model


def run(cfg: DictConfig) -> Path:
    """Train with ``cfg`` and return the checkpoint path (``train.out``/model.pt); logs to train.log."""
    out = Path(cfg.train.out)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")
    handler = logging.FileHandler(out / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        return _train(cfg, out)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _train(cfg: DictConfig, out: Path) -> Path:
    seed_everything(cfg.run.seed)
    device = torch.device(resolve_device(cfg))
    workers = cfg.run.workers

    train, val = load_split(cfg.data.dir, "train"), load_split(cfg.data.dir, "val")
    if cfg.data.holdout:
        train, val = domain_fold(train, val, cfg.data.holdout)
        val = val.subset(query_mask(val, frac=0.5, seed=cfg.run.seed))
    if cfg.data.max_images:
        keep = np.zeros(len(train), dtype=bool)
        keep[np.random.default_rng(cfg.run.seed).permutation(len(train))[:cfg.data.max_images]] = True
        train = train.subset(keep)
    if cfg.data.eval_max_queries:
        val = val.subset(query_mask(val, num_queries=cfg.data.eval_max_queries, seed=cfg.run.seed))
    log.info("train %d images / %d queries, selection %d images / %d queries",
             len(train), len(train.queries), len(val), len(val.queries))

    model_cfg = model_config(cfg)
    model = build_model(cfg).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.optim.ema_decay), use_buffers=True)

    prior = train.labels.mean(0)
    loss_cfg = loss_config(cfg)
    neighbors = (hamming_neighbors(train.queries, loss_cfg.neighbor_max_dist, loss_cfg.neighbor_max_count)
                 if loss_cfg.w_ret and model_cfg.embed_dim else None)
    criterion = CSARLoss(loss_cfg, prior, train.queries, neighbors, seed=cfg.run.seed)
    smoothing = torch.as_tensor(sample_smoothing(loss_cfg, train.domains))
    reliability = torch.as_tensor(sample_reliability(train, cfg.data.reliability))

    bs = cfg.optim.batch_size
    loader = DataLoader(
        ImageDataset(train.images, build_transform(model_cfg, augment_config(cfg)), cfg.data.dir),
        batch_size=bs, num_workers=workers, drop_last=len(train) > bs,
        sampler=WeightedRandomSampler(sample_weights(train, cfg.data.domain_balance, cfg.data.query_alpha),
                                      len(train), generator=torch.Generator().manual_seed(cfg.run.seed)),
        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    labels = torch.as_tensor(train.labels, dtype=torch.float32)
    qids = torch.as_tensor(train.query_ids)

    o = cfg.optim
    opt = torch.optim.AdamW(param_groups(model, o.lr, o.backbone_lr_mult, o.weight_decay))
    sched = LRSchedule(opt, o.scheduler, int(o.warmup_epochs * len(loader)), o.epochs * len(loader),
                       o.plateau_patience, o.plateau_factor)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    score_cfg, runtime = score_config(cfg), runtime_config(cfg)
    select_cfg = replace(score_cfg, name=cfg.train.select_score)
    meta = {"config": to_dict(cfg), "torch": str(torch.__version__), "python": platform.python_version(),
            "history": []}

    best, best_preds = -1.0, None
    for epoch in range(1, o.epochs + 1):
        model.train()
        t0, running = time.time(), {}
        for x, idx in loader:
            x, y = x.to(device, non_blocking=True), labels[idx].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(x)
            loss, parts = criterion(model, outputs, y, qids[idx], smoothing[idx], reliability[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update_parameters(model)
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v / len(loader)

        net = ema.module
        preds = predict([net], val.images, cfg.data.dir, runtime=runtime, device=device, num_workers=workers,
                        batch_size=cfg.run.inference_batch)
        res = report(preds, val, select_cfg, prior, lambda qs: encode_queries([net], qs, device), device.type)
        sched.epoch_end(res["all"]["mADM"])
        meta["history"].append({"epoch": epoch, "loss": running, "metrics": res})
        log.info("epoch %d/%d %.0fs loss %s | %s", epoch, o.epochs, time.time() - t0,
                 json.dumps({k: round(v, 4) for k, v in running.items()}),
                 " ".join(f"{d}:mADM={r['mADM']:.4f}" for d, r in res.items()) + f" R1={res['all']['R1']:.4f}")
        if res["all"]["mADM"] > best:
            best, best_preds = res["all"]["mADM"], preds
            Bundle([Member(model_cfg, _snapshot(ema))], prior.tolist(), score_cfg, runtime,
                   meta=meta).save(out / "model.pt")

    bundle = Bundle.load(out / "model.pt")
    bundle.meta = meta | {"best_select_mADM": best}
    if cfg.train.calibration != "none":
        bundle.calibration = fit_calibration(best_preds, torch.as_tensor(val.labels), cfg.train.calibration)
        log.info("calibration: %s", json.dumps(bundle.calibration.to_dict()))
    bundle.save(out / "model.pt")
    log.info("done: best %s mADM %.4f -> %s", cfg.train.select_score, best, out / "model.pt")
    return out / "model.pt"


main = entrypoint(run)

if __name__ == "__main__":
    main()
