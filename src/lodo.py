"""Leave-one-domain-out protocol, end to end.

For each domain D in ``lodo.domains``: train on the two others (``data.holdout=D``),
calibrate on one query-disjoint half of D and evaluate on the other half. Then:

* print every fold and the mean over folds for each score configuration;
* pick the configuration with the best *mean* mADM over folds;
* with ``lodo.final``, write that configuration and the fold calibrations averaged
  (geometric mean of temperatures) into the final checkpoint, so its calibration
  was learnt out of domain and frozen before submission.

    python -m src.lodo lodo.final=runs/final/model.pt model.backbone=convnext_base "eval.lams=[1,30]"
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from . import evaluate, train
from .calibration import Calibration
from .checkpoint import Bundle
from .config import entrypoint, with_values
from .metrics import METRICS

log = logging.getLogger(__name__)


def run_folds(cfg: DictConfig) -> dict[str, dict]:
    root = Path(cfg.lodo.root)
    folds = {}
    for d in cfg.lodo.domains:
        ckpt = root / d / "model.pt"
        if not ckpt.exists():
            train.run(with_values(cfg, {"data.holdout": d, "train.out": ckpt.parent}))
        folds[d] = evaluate.run(with_values(cfg, {
            "eval.checkpoint": ckpt, "eval.simulate": None, "eval.holdout": d, "eval.calib_frac": 0.5,
            "eval.by_domain": False, "eval.save_best": False}))
    return folds


def aggregate(folds: dict[str, dict]) -> tuple[str, dict[str, float], dict[str, dict]]:
    """Mean metrics per configuration over folds; returns (best name, its mean, all means)."""
    by_name: dict[str, list[dict]] = {}
    for result in folds.values():
        for name, res, _ in result["rows"]:
            by_name.setdefault(name, []).append(res["all"])
    means = {name: {m: float(np.mean([r[m] for r in runs])) for m in METRICS}
             for name, runs in by_name.items() if len(runs) == len(folds)}
    best = max(means, key=lambda n: means[n]["mADM"])
    return best, means[best], means


def run(cfg: DictConfig) -> dict:
    folds = run_folds(cfg)
    best, best_mean, means = aggregate(folds)
    print(f"{'config':<44}" + "".join(f"{d[:10]:>12}" for d in folds) + f"{'mean mADM':>12}{'mean R1':>10}")
    for name, mean in sorted(means.items(), key=lambda kv: -kv[1]["mADM"]):
        per_fold = [next(r["all"]["mADM"] for n, r, _ in folds[d]["rows"] if n == name) for d in folds]
        print(f"{name:<44}" + "".join(f"{v:>12.4f}" for v in per_fold) + f"{mean['mADM']:>12.4f}{mean['R1']:>10.4f}")
    print(f"LODO best: {best}  mean mADM={best_mean['mADM']:.4f}")

    calibration = Calibration.average([f["calibration"] for f in folds.values()])
    summary = {"best": best, "mean": best_mean, "folds": {d: f["metrics"]["all"] for d, f in folds.items()},
               "per_config": means, "calibration": calibration.to_dict()}
    root = Path(cfg.lodo.root)
    (root / "lodo_summary.json").write_text(json.dumps(summary, indent=2))
    if cfg.lodo.final:
        bundle = Bundle.load(cfg.lodo.final)
        bundle.score = next(c for n, _, c in next(iter(folds.values()))["rows"] if n == best)
        bundle.calibration = calibration
        bundle.meta["lodo"] = summary
        bundle.save(cfg.lodo.final)
        log.info("final checkpoint updated with LODO score config and calibration: %s", cfg.lodo.final)
    return summary


main = entrypoint(run)

if __name__ == "__main__":
    main()
