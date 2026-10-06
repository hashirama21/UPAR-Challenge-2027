"""Self-describing checkpoint: members, frozen calibration, score config, runtime, prior.

A bundle holds one or more members (an ensemble is averaged in logit space and
calibrated jointly). The submission rebuilds everything from this single file,
so training, evaluation and inference cannot drift apart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .calibration import Calibration
from .model import CSARNet, ModelConfig
from .scoring import ScoreConfig


@dataclass
class Member:
    model_cfg: ModelConfig
    state_dict: dict[str, torch.Tensor]

    def build(self) -> CSARNet:
        model = CSARNet(self.model_cfg, pretrained=False)
        model.load_state_dict({k: v.float() for k, v in self.state_dict.items()})  # strict: fail loudly
        return model.eval()


@dataclass
class Runtime:
    """Test-time augmentation and degraded mode; values come from configs/config.yaml (``runtime``)."""
    flip: bool
    scales: list[float]
    time_budget_s: float        # 0 = unlimited
    probe_images: int           # images timed to estimate throughput in degraded mode

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Bundle:
    members: list[Member]
    attr_prior: list[float]                     # training positive rates, used for E[dom_q]
    score: ScoreConfig
    runtime: Runtime
    calibration: Calibration = field(default_factory=Calibration)  # identity until fitted
    meta: dict = field(default_factory=dict)    # resolved config, metrics, versions

    def build_models(self) -> list[CSARNet]:
        return [m.build() for m in self.members]

    @property
    def prior(self) -> np.ndarray:
        return np.asarray(self.attr_prior, dtype=np.float32)

    def save(self, path: str | Path, half: bool = False) -> None:
        def cast(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {k: v.half() if half and v.is_floating_point() else v for k, v in sd.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "members": [{"model_cfg": m.model_cfg.to_dict(), "state_dict": cast(m.state_dict)} for m in self.members],
            "attr_prior": list(map(float, self.attr_prior)),
            "calibration": self.calibration.to_dict(),
            "score": self.score.to_dict(),
            "runtime": self.runtime.to_dict(),
            "meta": self.meta,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "Bundle":
        d = torch.load(path, map_location="cpu", weights_only=True)
        members = [Member(ModelConfig(**m["model_cfg"]), m["state_dict"]) for m in d["members"]]
        return cls(members, d["attr_prior"], ScoreConfig(**d["score"]), Runtime(**d["runtime"]),
                   Calibration.from_dict(d.get("calibration")), d.get("meta", {}))
