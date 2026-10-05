"""Self-describing checkpoint: architecture, weights, frozen calibration, score config, prior.

The submission rebuilds everything from this single file, so training,
evaluation and inference cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .calibration import Calibration
from .model import CSARNet, ModelConfig
from .scoring import ScoreConfig


@dataclass
class Bundle:
    model_cfg: ModelConfig
    state_dict: dict[str, torch.Tensor]
    attr_prior: list[float]                     # training positive rates, used for E[dom_q]
    calibration: Calibration = field(default_factory=Calibration)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    meta: dict = field(default_factory=dict)    # training args, metrics, versions

    def build_model(self) -> CSARNet:
        model = CSARNet(self.model_cfg, pretrained=False)
        model.load_state_dict({k: v.float() for k, v in self.state_dict.items()})  # strict: fail loudly
        return model.eval()

    @property
    def prior(self) -> np.ndarray:
        return np.asarray(self.attr_prior, dtype=np.float32)

    def save(self, path: str | Path, half: bool = False) -> None:
        sd = {k: v.half() if half and v.is_floating_point() else v for k, v in self.state_dict.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_cfg": self.model_cfg.to_dict(),
            "state_dict": sd,
            "attr_prior": list(map(float, self.attr_prior)),
            "calibration": self.calibration.to_dict(),
            "score": self.score.to_dict(),
            "meta": self.meta,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "Bundle":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(ModelConfig(**d["model_cfg"]), d["state_dict"], d["attr_prior"],
                   Calibration.from_dict(d.get("calibration")), ScoreConfig(**d.get("score", {})),
                   d.get("meta", {}))
