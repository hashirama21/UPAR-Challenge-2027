"""Reproducible training of a CSAR model (baseline recipe + structure-aware losses).

Recipe (UPAR baseline, Specker et al. 2023): ImageNet backbone, weighted BCE with
label smoothing, AdamW (lr 1e-4, wd 5e-4), EMA of the weights, flip/crop/AugMix,
no random erasing, no hue jitter. Additions: group softmax loss, query-image
contrastive loss with Hamming hard negatives, domain-balanced sampling.

Final model:      python -m src.train --out runs/final
LODO fold:        python -m src.train --holdout PETA --out runs/lodo_peta
Smoke test (CPU): python -m src.train --backbone resnet18 --max-images 256 --epochs 1 --no-pretrained
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .calibration import fit_calibration
from .checkpoint import Bundle
from .data import DOMAINS, domain_fold, hamming_neighbors, load_split, query_mask, sample_weights
from .dataset import ImageDataset
from .evaluate import report
from .inference import encode_queries, predict
from .losses import CSARLoss, LossConfig
from .model import BACKBONES, CSARNet, ModelConfig
from .scoring import SCORES, ScoreConfig
from .transforms import build_transform

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", type=Path, required=True, help="run directory (model.pt, config.json)")
    ap.add_argument("--holdout", choices=DOMAINS, help="LODO: drop this domain from train, evaluate on it")
    ap.add_argument("--backbone", default="convnext_base", choices=sorted(BACKBONES))
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--embed-dim", type=int, default=256, help="0 disables the retrieval branch")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--no-augmix", action="store_true")
    ap.add_argument("--no-domain-balance", action="store_true")
    ap.add_argument("--query-alpha", type=float, default=0.0, help="sample weight n(query)^-alpha")
    for f in LossConfig.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    ap.add_argument("--select-score", default="structured", choices=SCORES, help="score used for model selection")
    ap.add_argument("--eval-max-queries", type=int, default=0, help="subsample eval queries per epoch")
    ap.add_argument("--max-images", type=int, default=0, help="debug: random training subset")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return f


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "train.log")])
    seed_everything(args.seed)
    device = torch.device(args.device)

    train, val = load_split(args.data_dir, "train"), load_split(args.data_dir, "val")
    if args.holdout:
        train, val = domain_fold(train, val, args.holdout)
    if args.max_images:
        keep = np.zeros(len(train), dtype=bool)
        keep[np.random.default_rng(args.seed).permutation(len(train))[:args.max_images]] = True
        train = train.subset(keep)
        val = val.subset(query_mask(val, num_queries=max(8, args.max_images // 8), seed=args.seed))
    if args.eval_max_queries:
        val = val.subset(query_mask(val, num_queries=args.eval_max_queries, seed=args.seed))
    log.info("train %d images / %d queries, eval %d images / %d queries",
             len(train), len(train.queries), len(val), len(val.queries))

    model_cfg = ModelConfig(args.backbone, args.height, args.width, embed_dim=args.embed_dim)
    model = CSARNet(model_cfg, pretrained=not args.no_pretrained).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)

    prior = train.labels.mean(0)
    loss_cfg = LossConfig(**{k: getattr(args, k) for k in LossConfig.__dataclass_fields__})
    neighbors = hamming_neighbors(train.queries) if loss_cfg.w_ret and args.embed_dim else None
    criterion = CSARLoss(loss_cfg, prior, train.queries, neighbors, seed=args.seed)

    weights = sample_weights(train, not args.no_domain_balance, args.query_alpha)
    loader = DataLoader(
        ImageDataset(train.images, build_transform(args.height, args.width, True, not args.no_augmix), args.data_dir),
        batch_size=args.batch_size, num_workers=args.workers, drop_last=len(train) > args.batch_size,
        sampler=WeightedRandomSampler(weights, len(train), generator=torch.Generator().manual_seed(args.seed)),
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    labels = torch.as_tensor(train.labels, dtype=torch.float32)
    qids = torch.as_tensor(train.query_ids)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(int(args.warmup_epochs * len(loader)), steps))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    select_cfg = ScoreConfig(name=args.select_score)
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "torch": str(torch.__version__),
            "python": platform.python_version(), "history": []}
    (args.out / "config.json").write_text(json.dumps(meta["args"], indent=2))

    best, best_preds = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), {}
        for x, idx in loader:
            x, y, q = x.to(device, non_blocking=True), labels[idx].to(device), qids[idx]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
            loss, parts = criterion(model, out, y, q)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update_parameters(model)
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v / len(loader)

        net = ema.module
        preds = predict(net, val.images, args.data_dir, device=device, num_workers=args.workers)
        res = report(preds, val, select_cfg, prior, lambda qs: encode_queries(net, qs, device), device.type)
        meta["history"].append({"epoch": epoch, "loss": running, "metrics": res})
        log.info("epoch %d/%d %.0fs loss %s | %s", epoch, args.epochs, time.time() - t0,
                 json.dumps({k: round(v, 4) for k, v in running.items()}),
                 " ".join(f"{d}:mADM={r['mADM']:.4f}" for d, r in res.items()) + f" R1={res['all']['R1']:.4f}")
        if res["all"]["mADM"] > best:
            best, best_preds = res["all"]["mADM"], preds
            Bundle(model_cfg, {k: v.detach().cpu() for k, v in net.state_dict().items()},
                   prior.tolist(), meta=meta).save(args.out / "model.pt")

    bundle = Bundle.load(args.out / "model.pt")
    bundle.meta = meta | {"best_select_mADM": best}
    if not args.no_calibrate:
        # Frozen before submission; evaluate.py --calib-frac refits on a disjoint half for unbiased LODO numbers.
        bundle.calibration = fit_calibration(best_preds, torch.as_tensor(val.labels))
        log.info("calibration: %s", json.dumps(bundle.calibration.to_dict()))
    bundle.save(args.out / "model.pt")
    log.info("done: best %s mADM %.4f -> %s", args.select_score, best, args.out / "model.pt")


if __name__ == "__main__":
    main()
