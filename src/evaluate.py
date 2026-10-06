"""Evaluation: score functions x calibration x gallery protocols, per domain.

Reports mADM per source domain and, separately, for queries seen in train vs
novel ones; ECE per domain before/after calibration; optional coordinate-ascent
tuning of the S3 group weights w_k. Works for single models and ensembles.
All settings: ``configs/config.yaml`` (``eval``, ``score``).

Oracle simulation from ground truth (no images needed, reproduces the README study):
    python -m src.evaluate eval.simulate=2.0 "eval.scores=[l1,loglik,structured,endom,mix]"

Trained model on a LODO fold, calibration fitted on a query-disjoint half of the
held-out domain, sweep of lam / gamma / w_k, best config written back into the checkpoint:
    python -m src.evaluate eval.checkpoint=runs/lodo/PETA/model.pt eval.holdout=PETA eval.calib_frac=0.5 \
        "eval.lams=[1,30]" eval.tune_group_weights=true eval.save_best=true

Experiment B (open set): ``eval.open_set=0.2`` removes 20 % of the queries and keeps
their images as distractors. ``eval.resample=367`` builds a test-like gallery
(queries drawn by size, as in the 2024 test).
"""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from omegaconf import DictConfig

from .attributes import GROUPS
from .calibration import Calibration, Predictions, fit_calibration
from .checkpoint import Bundle
from .config import device as resolve_device, entrypoint, score_config, to_dict
from .data import DOMAINS, Split, load_split, novel_queries, query_mask
from .inference import encode_queries, predict
from .metrics import METRICS, expected_calibration_error, per_query_metrics, summarize
from .scoring import ScoreConfig, Scorer

log = logging.getLogger(__name__)
QueryEncoder = Callable[[np.ndarray], "torch.Tensor | None"]


def select(preds: Predictions, mask: np.ndarray) -> Predictions:
    idx = torch.as_tensor(np.flatnonzero(mask))
    return {k: v[idx] for k, v in preds.items()}


def score_split(preds: Predictions, split: Split, cfg: ScoreConfig, prior: np.ndarray,
                encoder: QueryEncoder | None = None, device: str = "cpu", block: int = 64) -> dict[str, np.ndarray]:
    """Per-query metrics of ``cfg`` on ``split``."""
    q_emb = encoder(split.queries) if encoder is not None and cfg.gamma else None
    scorer = Scorer(cfg, preds, split.queries, prior, q_emb, device)
    return per_query_metrics(scorer.blocks(block), split.queries, split.labels)


def report(preds: Predictions, split: Split, cfg: ScoreConfig, prior: np.ndarray,
           encoder: QueryEncoder | None = None, device: str = "cpu", by_domain: bool = True,
           novel: np.ndarray | None = None) -> dict[str, dict]:
    """Metrics on the whole gallery, per source domain (protocol rebuilt) and seen/novel queries."""
    per_query = score_split(preds, split, cfg, prior, encoder, device)
    out = {"all": summarize(per_query)}
    if novel is not None and novel.any() and not novel.all():
        out["seen"], out["novel"] = summarize(per_query, ~novel), summarize(per_query, novel)
    if by_domain:
        doms = split.domains
        for d in DOMAINS:
            mask = doms == d
            if mask.any() and mask.sum() < len(split):
                out[d] = summarize(score_split(select(preds, mask), split.subset(mask), cfg, prior, encoder, device))
    return out


def ece_report(preds: Predictions, split: Split) -> dict[str, float]:
    """Mean per-attribute ECE overall and per source domain."""
    probs = torch.sigmoid(preds["attr"]).numpy()
    out = {"all": float(expected_calibration_error(probs, split.labels).mean())}
    doms = split.domains
    for d in DOMAINS:
        mask = doms == d
        if mask.any() and mask.sum() < len(split):
            out[d] = float(expected_calibration_error(probs[mask], split.labels[mask]).mean())
    return out


def tune_group_weights(preds: Predictions, split: Split, cfg: ScoreConfig, prior: np.ndarray,
                       encoder: QueryEncoder | None, device: str, grid: list[float]) -> ScoreConfig:
    """One pass of coordinate ascent on the S3 group weights w_k (mADM on ``split``)."""
    def madm(weights: list[float]) -> float:
        return summarize(score_split(preds, split, replace(cfg, group_weights=weights), prior, encoder, device))["mADM"]

    weights = list(cfg.group_weights or [1.0] * len(GROUPS))
    best = madm(weights)
    for k, g in enumerate(GROUPS):
        for w in grid:
            trial = weights[:k] + [w] + weights[k + 1:]
            score = madm(trial)
            if score > best:
                best, weights = score, trial
        log.info("w[%s] = %g (mADM %.4f)", g.name, weights[k], best)
    return replace(cfg, group_weights=weights)


def simulate(split: Split, sigma: float, signal: float, seed: int) -> Predictions:
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
    columns = list(rows[0][1]) if rows else []
    head = f"{'config':<44}" + "".join(f"{c[:10] + ' mADM':>16}" for c in columns) + \
        "".join(f"{m:>8}" for m in METRICS if m != "mADM")
    lines = [head, "-" * len(head)]
    for name, res in rows:
        lines.append(f"{name:<44}" + "".join(f"{res[c]['mADM']:>16.4f}" for c in columns)
                     + "".join(f"{res['all'][m]:>8.4f}" for m in METRICS if m != "mADM"))
    return "\n".join(lines)


def candidate_configs(base: ScoreConfig, scores: list[str], lams: list[float],
                      gammas: list[float]) -> list[ScoreConfig]:
    cfgs = []
    for name in scores:
        lam_grid = lams if name == "mix" else [base.lam]
        gamma_grid = gammas if name in ("structured", "mix") else [0.0]
        for lam, gamma in itertools.product(lam_grid, gamma_grid):
            cfgs.append(replace(base, name=name, lam=lam, gamma=gamma))
    return cfgs


def _predictions(cfg: DictConfig, bundle: Bundle | None, models: list | None, val: Split,
                 needed: np.ndarray) -> Predictions:
    """Predictions for ``val.images[needed]``: simulated, cached (same image list) or inferred."""
    e = cfg.eval
    if bundle is None:
        return select(simulate(val, e.simulate, e.simulate_signal, cfg.run.seed), needed)
    images = list(val.images[needed])
    cache = Path(e.cache) if e.cache else None
    if cache and cache.exists():
        cached = torch.load(cache, weights_only=True)
        if cached["images"] == images:
            return cached["preds"]
        log.info("cache %s was built for other images: recomputing", cache)
    preds = predict(models, images, cfg.data.dir, runtime=bundle.runtime, device=resolve_device(cfg),
                    num_workers=cfg.run.workers, batch_size=cfg.run.inference_batch)
    if cache:
        torch.save({"images": images, "preds": preds}, cache)
    return preds


def run(cfg: DictConfig) -> dict:
    e = cfg.eval
    if (e.checkpoint is None) == (e.simulate is None):
        raise ValueError("set exactly one of eval.checkpoint and eval.simulate")
    dev, workers, seed = resolve_device(cfg), cfg.run.workers, cfg.run.seed
    val = load_split(cfg.data.dir, "val")
    train = load_split(cfg.data.dir, "train")
    if e.holdout:
        val = val.subset(val.domains == e.holdout)

    bundle = Bundle.load(e.checkpoint) if e.checkpoint else None
    prior = bundle.prior if bundle else train.labels.mean(0)
    models = bundle.build_models() if bundle else None
    # Masks first: inference only runs on the calibration half and the evaluated gallery.
    cal = query_mask(val, frac=e.calib_frac, seed=seed) if e.calib_frac else np.zeros(len(val), dtype=bool)
    gal = ~cal
    if e.resample or e.max_queries:
        keep = query_mask(val.subset(gal), num_queries=e.resample or e.max_queries, seed=seed,
                          by_size=bool(e.resample))
        gal = np.zeros(len(val), dtype=bool)
        gal[np.flatnonzero(~cal)[keep]] = True
    needed = cal | gal
    preds = _predictions(cfg, bundle, models, val, needed)

    calibration = bundle.calibration if bundle else Calibration()
    if e.calib_frac:
        calibration = fit_calibration(select(preds, cal[needed]), torch.as_tensor(val.labels[cal]), e.calibration)
        log.info("calibration fitted on %d images: %s", cal.sum(), json.dumps(calibration.to_dict()))
    gallery, gpreds = val.subset(gal), select(preds, gal[needed])
    if e.open_set:
        rng = np.random.default_rng(seed)
        n = len(gallery.queries)
        gallery = gallery.without_queries(rng.choice(n, round(e.open_set * n), replace=False))
    raw_preds = gpreds
    ece_raw = ece_report(gpreds, gallery)
    gpreds = calibration.apply(gpreds)
    ece_cal = ece_report(gpreds, gallery)
    log.info("gallery: %d images, %d queries", len(gallery), len(gallery.queries))
    print("ECE (mean over attributes)  " + "  ".join(
        f"{d}: {ece_raw[d]:.4f} -> {ece_cal[d]:.4f}" for d in ece_raw))

    encoder = (lambda q: encode_queries(models, q, dev)) if models is not None else None
    novel = novel_queries(gallery, train)
    base = score_config(cfg)
    if e.tune_group_weights:
        base = tune_group_weights(gpreds, gallery, replace(base, name="structured"), prior, encoder, dev,
                                  list(e.group_weight_grid))
    rows, raw_rows = [], []
    for score_cfg in candidate_configs(base, list(e.scores), list(e.lams), list(e.gammas)):
        res = report(gpreds, gallery, score_cfg, prior, encoder, dev, by_domain=e.by_domain, novel=novel)
        name = (f"{score_cfg.name} lam={score_cfg.lam:g} gamma={score_cfg.gamma:g}"
                + (" post" if score_cfg.transductive else ""))
        rows.append((name, res, score_cfg))
        log.info("%s -> %s", name, json.dumps(res["all"]))
        if e.compare_uncalibrated:
            raw = report(raw_preds, gallery, score_cfg, prior, encoder, dev, by_domain=e.by_domain, novel=novel)
            raw_rows.append((name + " (raw)", raw))
    print(format_table([(n, r) for n, r, _ in rows] + raw_rows))

    best_name, best_res, best_cfg = max(rows, key=lambda r: r[1]["all"]["mADM"])
    print(f"best: {best_name}  mADM={best_res['all']['mADM']:.4f}")
    if e.save_best and bundle is not None:
        bundle.score, bundle.calibration = best_cfg, calibration
        bundle.meta.setdefault("evaluations", []).append(
            {"eval": to_dict(e), "best": best_name, "metrics": best_res, "ece": {"raw": ece_raw, "calibrated": ece_cal}})
        bundle.save(e.checkpoint)
        log.info("checkpoint updated: %s", e.checkpoint)
    return {"best": best_cfg, "metrics": best_res, "calibration": calibration, "rows": rows, "raw_rows": raw_rows,
            "ece": {"raw": ece_raw, "calibrated": ece_cal}}


main = entrypoint(run)

if __name__ == "__main__":
    main()
