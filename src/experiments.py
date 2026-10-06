"""Run the fine-tuning candidates of ``experiment`` end to end and package the winner.

Candidates are dotlist overrides of ``configs/config.yaml`` listed in
``configs/experiment/<name>.yaml``. For every candidate: train on the full training
split, then score it with one shared yardstick (official val, calibration fitted on
one query-disjoint half and measured on the other, best score config saved into the
checkpoint). Then:

* LODO (``experiment.lodo``): ``all`` runs it for every candidate (reported),
  ``best`` for the winner only; the winner receives its out-of-domain calibration
  and score config;
* logit ensemble of the top-k candidates and WiSE-FT of the best one, same yardstick;
* export of the winner as a Codabench submission (``experiment.export``).

    python -m src.experiments                                  # experiment=full
    python -m src.experiments profile=smoke experiment=smoke   # CPU smoke run
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from omegaconf import DictConfig

from . import evaluate, export, lodo, train
from .combine import ensemble, wise
from .config import entrypoint, to_dict, with_overrides, with_values

log = logging.getLogger(__name__)


def score(cfg: DictConfig, checkpoint: Path) -> dict[str, float]:
    """Shared yardstick: calibrate on half of the val queries, measure on the other half, save the best config."""
    result = evaluate.run(with_values(cfg, {"eval.checkpoint": checkpoint, "eval.simulate": None,
                                            "eval.holdout": None, "eval.calib_frac": 0.5, "eval.save_best": True}))
    return result["metrics"]["all"]


def run_lodo(cfg: DictConfig, checkpoint: Path) -> dict[str, float]:
    return lodo.run(with_values(cfg, {"lodo.root": checkpoint.parent / "lodo", "lodo.final": checkpoint}))["mean"]


def leaderboard(rows: list[dict]) -> str:
    lines = [f"{'candidate':<36}{'kind':<10}{'val mADM':>10}{'mAP':>8}{'R1':>8}{'LODO mADM':>11}", "-" * 83]
    for r in sorted(rows, key=lambda r: -r["val"]["mADM"]):
        lodo_madm = f"{r['lodo']['mADM']:>11.4f}" if r.get("lodo") else f"{'-':>11}"
        lines.append(f"{r['name']:<36}{r['kind']:<10}{r['val']['mADM']:>10.4f}{r['val']['mAP']:>8.4f}"
                     f"{r['val']['R1']:>8.4f}{lodo_madm}")
    return "\n".join(lines)


def run(cfg: DictConfig) -> dict:
    x = cfg.experiment
    if x.lodo not in ("none", "best", "all"):
        raise ValueError("experiment.lodo must be none, best or all")
    root = Path(x.root)
    rows = []
    for name, overrides in x.candidates.items():
        ccfg = with_values(with_overrides(cfg, list(overrides)), {"train.out": root / name})
        ckpt = root / name / "model.pt"
        if not ckpt.exists():
            train.run(ccfg)
        row = {"name": name, "kind": "single", "checkpoint": ckpt, "cfg": ccfg, "val": score(ccfg, ckpt)}
        if x.lodo == "all":
            row["lodo"] = run_lodo(ccfg, ckpt)
        rows.append(row)
        log.info("%s: val mADM %.4f", name, row["val"]["mADM"])

    singles = sorted(rows, key=lambda r: -r["val"]["mADM"])
    if x.ensemble_top_k >= 2 and len(singles) >= 2:
        top = singles[:x.ensemble_top_k]
        ckpt = root / "ensemble" / "model.pt"
        ensemble([r["checkpoint"] for r in top]).save(ckpt)
        rows.append({"name": "ensemble(" + "+".join(r["name"] for r in top) + ")", "kind": "ensemble",
                     "checkpoint": ckpt, "val": score(cfg, ckpt)})
    if x.wise_alpha is not None:
        best = singles[0]
        ckpt = root / f"{best['name']}_wise" / "model.pt"
        wise(best["checkpoint"], x.wise_alpha).save(ckpt)
        rows.append({"name": f"{best['name']}+wise{x.wise_alpha:g}", "kind": "wise", "checkpoint": ckpt,
                     "val": score(best["cfg"], ckpt)})

    winner = max(rows, key=lambda r: r["val"]["mADM"])
    if x.lodo != "none" and winner["kind"] == "single":
        winner["lodo"] = winner.get("lodo") or run_lodo(winner["cfg"], winner["checkpoint"])
    elif x.lodo != "none":
        log.info("winner %s is not a single model: keeping its val-half calibration", winner["name"])
    print(leaderboard(rows))
    print(f"winner: {winner['name']}")

    archive = None
    if x.export:
        archive = export.export(winner["checkpoint"], Path(x.export), half=not cfg.export.fp32, overwrite=True)
        print(f"submission: {archive} ({archive.stat().st_size / 2**20:.1f} MiB)")
    summary = {"experiment": to_dict(x), "winner": winner["name"], "archive": str(archive) if archive else None,
               "rows": [{"name": r["name"], "kind": r["kind"], "checkpoint": str(r["checkpoint"]),
                         "val": r["val"], "lodo": r.get("lodo")} for r in rows]}
    (root / "experiments.json").write_text(json.dumps(summary, indent=2))
    return summary | {"winner_checkpoint": winner["checkpoint"], "archive": archive}


main = entrypoint(run)

if __name__ == "__main__":
    main()
