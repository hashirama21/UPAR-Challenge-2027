"""Per-attribute and per-group temperature scaling (Guo et al., 2017).

Fit on held-out *training-side* data (official val or the left-out domain of a
LODO fold), then frozen in the checkpoint. Nothing is ever fitted on the test
gallery. What matters for the joint score is the *relative* calibration across
attributes, hence one temperature per attribute rather than a global one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .attributes import GROUPS, NUM_ATTRS, encode_groups

Predictions = dict[str, torch.Tensor]  # "attr", "group/<name>", optional "emb"


@dataclass
class Calibration:
    attr_t: list[float] = field(default_factory=lambda: [1.0] * NUM_ATTRS)
    group_t: dict[str, float] = field(default_factory=lambda: {g.name: 1.0 for g in GROUPS})

    def apply(self, preds: Predictions) -> Predictions:
        out = dict(preds)
        out["attr"] = preds["attr"] / torch.tensor(self.attr_t, dtype=preds["attr"].dtype)
        for name, t in self.group_t.items():
            out[f"group/{name}"] = preds[f"group/{name}"] / t
        return out

    def to_dict(self) -> dict:
        return {"attr_t": self.attr_t, "group_t": self.group_t}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Calibration":
        return cls(**d) if d else cls()


def _fit_log_t(loss_fn, n: int, steps: int = 100) -> torch.Tensor:
    log_t = torch.zeros(n, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn(log_t.exp())
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.detach().exp().clamp(0.05, 20.0)


def fit_calibration(preds: Predictions, labels: torch.Tensor) -> Calibration:
    y = labels.float()
    logits = preds["attr"].float()
    # Separable per attribute: summing per-attribute means fits each temperature independently.
    attr_t = _fit_log_t(lambda t: F.binary_cross_entropy_with_logits(logits / t, y, reduction="none").mean(0).sum(),
                        NUM_ATTRS)
    cls = encode_groups(labels)
    group_t = {}
    for k, g in enumerate(GROUPS):
        valid = cls[:, k] >= 0
        lg = preds[f"group/{g.name}"].float()[valid]
        group_t[g.name] = float(_fit_log_t(lambda t: F.cross_entropy(lg / t, cls[valid, k]), 1)[0])
    return Calibration(attr_t.tolist(), group_t)
