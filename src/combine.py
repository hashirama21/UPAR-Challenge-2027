"""Combine checkpoints: logit ensemble, uniform model soup, WiSE-FT.

    python -m src.combine combine.mode=ensemble "combine.checkpoints=[runs/a/model.pt,runs/b/model.pt]" combine.out=runs/ens/model.pt
    python -m src.combine combine.mode=soup "combine.checkpoints=[runs/s0/model.pt,runs/s1/model.pt]" combine.out=runs/soup/model.pt
    python -m src.combine combine.mode=wise "combine.checkpoints=[runs/clip/model.pt]" combine.alpha=0.5 combine.out=runs/wise/model.pt

An ensemble needs a joint calibration afterwards
(``python -m src.evaluate eval.checkpoint=<out> eval.calib_frac=0.5 eval.save_best=true``).
WiSE-FT interpolates the backbone between its pre-trained and fine-tuned weights
(heads keep their fine-tuned values); it downloads the pre-trained weights.
"""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from .calibration import Calibration
from .checkpoint import Bundle, Member
from .config import entrypoint
from .model import CSARNet


def ensemble(paths: list[Path]) -> Bundle:
    bundles = [Bundle.load(p) for p in paths]
    first = bundles[0]
    return Bundle([m for b in bundles for m in b.members], first.attr_prior, first.score, first.runtime,
                  Calibration(), {"combined": "ensemble", "sources": [str(p) for p in paths]})


def soup(paths: list[Path]) -> Bundle:
    bundles = [Bundle.load(p) for p in paths]
    members = [b.members[0] for b in bundles]
    if len({str(m.model_cfg) for m in members}) != 1:
        raise ValueError("a soup needs checkpoints with the same architecture")
    avg = {k: torch.stack([m.state_dict[k].float() for m in members]).mean(0) for k in members[0].state_dict}
    first = bundles[0]
    return Bundle([Member(members[0].model_cfg, avg)], first.attr_prior, first.score, first.runtime,
                  Calibration(), {"combined": "soup", "sources": [str(p) for p in paths]})


def wise(path: Path, alpha: float) -> Bundle:
    """theta = alpha * fine-tuned + (1 - alpha) * pre-trained, on backbone weights only."""
    bundle = Bundle.load(path)
    member = bundle.members[0]
    zero_shot = CSARNet(member.model_cfg, pretrained=True).state_dict()
    mixed = {k: alpha * v.float() + (1 - alpha) * zero_shot[k].float()
             if k.startswith("backbone.") and v.is_floating_point() else v
             for k, v in member.state_dict.items()}
    bundle.members = [Member(member.model_cfg, mixed)]
    bundle.calibration = Calibration()
    bundle.meta = {**bundle.meta, "combined": f"wise alpha={alpha}", "sources": [str(path)]}
    return bundle


def run(cfg: DictConfig) -> Path:
    c = cfg.combine
    paths = [Path(p) for p in c.checkpoints]
    if not paths or not c.out:
        raise ValueError("set combine.checkpoints and combine.out")
    if c.mode == "wise":
        if len(paths) != 1:
            raise ValueError("wise takes exactly one checkpoint")
        bundle = wise(paths[0], c.alpha)
    elif c.mode in ("ensemble", "soup"):
        bundle = (ensemble if c.mode == "ensemble" else soup)(paths)
    else:
        raise ValueError(f"unknown combine.mode {c.mode!r}")
    bundle.save(c.out)
    print(f"{c.mode}: {len(bundle.members)} member(s) -> {c.out}")
    return Path(c.out)


main = entrypoint(run)

if __name__ == "__main__":
    main()
