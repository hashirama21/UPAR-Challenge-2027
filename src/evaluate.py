"""Evaluation: score functions x calibration x gallery protocols, per domain.

Examples
--------
Oracle simulation from ground truth (no images needed, reproduces the README study):
    python -m src.evaluate --simulate 2.0 --scores l1 loglik structured endom mix

Trained model on a LODO fold, calibration fitted on a query-disjoint half of the
held-out domain, sweep of lam, best config written back into the checkpoint:
    python -m src.evaluate --checkpoint runs/lodo_peta/model.pt --holdout PETA \
        --calib-frac 0.5 --scores loglik structured mix --lams 0.3 1 3 --save-best

Experiment B (open set): ``--open-set 0.2`` removes 20 % of the queries and keeps
their images as distractors. ``--resample 367`` mimics the 2024 test size.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .attributes import GROUPS
from .calibration import Calibration, Predictions, fit_calibration
from .checkpoint import Bundle
from .data import DOMAINS, Split, load_split, query_mask
from .inference import encode_queries, predict
from .metrics import METRICS, evaluate_blocks
from .scoring import SCORES, ScoreConfig, Scorer

log = logging.getLogger(__name__)
QueryEncoder = Callable[[np.ndarray], "torch.Tensor | None"]


def select(preds: Predictions, mask: np.ndarray) -> Predictions:
    idx = torch.as_tensor(np.flatnonzero(mask))
    return {k: v[idx] for k, v in preds.items()}


def score_split(preds: Predictions, split: Split, cfg: ScoreConfig, prior: np.ndarray,
                encoder: QueryEncoder | None = None, device: str = "cpu", block: int = 64) -> dict[str, float]:
    q_emb = encoder(split.queries) if encoder is not None and cfg.gamma else None
    scorer = Scorer(cfg, preds, split.queries, prior, q_emb, device)
    return evaluate_blocks(scorer.blocks(block), split.queries, split.labels)


def report(preds: Predictions, split: Split, cfg: ScoreConfig, prior: np.ndarray,
           encoder: QueryEncoder | None = None, device: str = "cpu", by_domain: bool = True) -> dict[str, dict]:
    """Metrics on the whole gallery and, separately, on each source domain (protocol rebuilt)."""
    out = {"all": score_split(preds, split, cfg, prior, encoder, device)}
    if by_domain:
        doms = split.domains
        for d in DOMAINS:
            mask = doms == d
            if mask.any() and mask.sum() < len(split):
                out[d] = score_split(select(preds, mask), split.subset(mask), cfg, prior, encoder, device)
    return out


def simulate(split: Split, sigma: float, signal: float = 3.0, seed: int = 0) -> Predictions:
    """Ground truth + independent Gaussian logit noise (oracle study of score functions)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.as_tensor(split.labels, dtype=torch.float32)

    def noisy(t: torch.Tensor) -> torch.Tensor:
        return signal * (2 * t - 1) + sigma * torch.randn(t.shape, generator=g)

    preds = {"attr": noisy(y)}
    # Group heads see the *same* noisy evidence (no free second view); "none" sits at logit 0.
    for grp in GROUPS:
        members = preds["attr"][:, list(grp.indices)]
        none = [torch.zeros(len(y), 1)] if grp.allow_none else []
        preds[f"group/{grp.name}"] = torch.cat([members, *none], 1)
    return preds


def format_table(rows: list[tuple[str, dict[str, dict]]]) -> str:
    domains = list(rows[0][1]) if rows else []
    head = f"{'config':<44}" + "".join(f"{d[:10] + ' mADM':>16}" for d in domains) + \
        "".join(f"{m:>8}" for m in METRICS if m != "mADM")
    lines = [head, "-" * len(head)]
    for name, res in rows:
        lines.append(f"{name:<44}" + "".join(f"{res[d]['mADM']:>16.4f}" for d in domains)
                     + "".join(f"{res['all'][m]:>8.4f}" for m in METRICS if m != "mADM"))
    return "\n".join(lines)


def candidate_configs(base: ScoreConfig, scores: list[str], lams: list[float], gammas: list[float],
                      transductive: bool) -> list[ScoreConfig]:
    cfgs = []
    for name in scores:
        lam_grid = lams if name == "mix" else [base.lam]
        gamma_grid = gammas if name in ("structured", "mix") else [0.0]
        for lam, gamma in itertools.product(lam_grid, gamma_grid):
            cfgs.append(replace(base, name=name, lam=lam, gamma=gamma, transductive=transductive))
    return cfgs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path)
    src.add_argument("--simulate", type=float, metavar="SIGMA", help="oracle: GT + logit noise")
    ap.add_argument("--holdout", choices=DOMAINS, help="evaluate on this domain of val (LODO fold)")
    ap.add_argument("--calib-frac", type=float, default=0.0,
                    help="fit calibration on this query-disjoint fraction, evaluate on the rest")
    ap.add_argument("--scores", nargs="+", default=["loglik", "structured", "mix"], choices=SCORES)
    ap.add_argument("--lams", nargs="+", type=float, default=[1.0])
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.0])
    ap.add_argument("--exact", choices=("loglik", "structured"), default="structured")
    ap.add_argument("--transductive", action="store_true", help="posterior over queries (needs organiser approval)")
    ap.add_argument("--open-set", type=float, default=0.0, help="experiment B: fraction of queries removed")
    ap.add_argument("--resample", type=int, default=0, help="keep N random queries and their images")
    ap.add_argument("--max-queries", type=int, default=0, help="speed: same as --resample, for quick sweeps")
    ap.add_argument("--no-domains", action="store_true")
    ap.add_argument("--cache", type=Path, help="predictions cache (.pt) to skip inference on re-runs")
    ap.add_argument("--save-best", action="store_true", help="write best score config + calibration to checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    val = load_split(args.data_dir, "val")
    train = load_split(args.data_dir, "train")
    if args.holdout:
        val = val.subset(val.domains == args.holdout)

    bundle = Bundle.load(args.checkpoint) if args.checkpoint else None
    prior = bundle.prior if bundle else train.labels.mean(0)
    model = bundle.build_model() if bundle else None
    if args.cache and args.cache.exists():
        preds = torch.load(args.cache, weights_only=True)
    elif model is not None:
        preds = predict(model, val.images, args.data_dir, device=args.device, num_workers=args.workers)
    else:
        preds = simulate(val, args.simulate, seed=args.seed)
    if args.cache and not args.cache.exists():
        torch.save(preds, args.cache)

    mask = np.ones(len(val), dtype=bool)
    calibration = bundle.calibration if bundle else Calibration()
    if args.calib_frac:
        cal = query_mask(val, frac=args.calib_frac, seed=args.seed)
        calibration = fit_calibration(select(preds, cal), torch.as_tensor(val.labels[cal]))
        log.info("calibration fitted on %d images: %s", cal.sum(), json.dumps(calibration.to_dict()))
        mask = ~cal
    gallery, gpreds = val.subset(mask), select(preds, mask)
    if args.resample or args.max_queries:
        keep = query_mask(gallery, num_queries=args.resample or args.max_queries, seed=args.seed)
        gallery, gpreds = gallery.subset(keep), select(gpreds, keep)
    if args.open_set:
        rng = np.random.default_rng(args.seed)
        n = len(gallery.queries)
        gallery = gallery.without_queries(rng.choice(n, round(args.open_set * n), replace=False))
    gpreds = calibration.apply(gpreds)
    log.info("gallery: %d images, %d queries", len(gallery), len(gallery.queries))

    encoder = (lambda q: encode_queries(model, q, args.device)) if model is not None else None
    base = bundle.score if bundle else ScoreConfig()
    base = replace(base, exact=args.exact)
    rows = []
    for cfg in candidate_configs(base, args.scores, args.lams, args.gammas, args.transductive):
        res = report(gpreds, gallery, cfg, prior, encoder, args.device, by_domain=not args.no_domains)
        name = f"{cfg.name} lam={cfg.lam:g} gamma={cfg.gamma:g}" + (" post" if cfg.transductive else "")
        rows.append((name, res, cfg))
        log.info("%s -> %s", name, json.dumps(res["all"]))
    print(format_table([(n, r) for n, r, _ in rows]))

    best_name, best_res, best_cfg = max(rows, key=lambda r: r[1]["all"]["mADM"])
    print(f"best: {best_name}  mADM={best_res['all']['mADM']:.4f}")
    if args.save_best and bundle is not None:
        bundle.score, bundle.calibration = best_cfg, calibration
        bundle.meta.setdefault("evaluations", []).append(
            {"args": {k: str(v) for k, v in vars(args).items()}, "best": best_name, "metrics": best_res})
        bundle.save(args.checkpoint)
        log.info("checkpoint updated: %s", args.checkpoint)


if __name__ == "__main__":
    main()
