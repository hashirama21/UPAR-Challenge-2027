"""Per-attribute and per-group temperature scaling (Guo et al., 2017), optionally Platt.

``mode="platt"`` adds a per-attribute bias to the temperature (logit / t + b).

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
    attr_b: list[float] = field(default_factory=lambda: [0.0] * NUM_ATTRS)

    def apply(self, preds: Predictions) -> Predictions:
        out = dict(preds)
        dtype = preds["attr"].dtype
        out["attr"] = preds["attr"] / torch.tensor(self.attr_t, dtype=dtype) + torch.tensor(self.attr_b, dtype=dtype)
        for name, t in self.group_t.items():
            out[f"group/{name}"] = preds[f"group/{name}"] / t
        return out

    def to_dict(self) -> dict:
        return {"attr_t": self.attr_t, "group_t": self.group_t, "attr_b": self.attr_b}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Calibration":
        return cls(**d) if d else cls()

    @classmethod
    def average(cls, items: list["Calibration"]) -> "Calibration":
        """Geometric mean of temperatures, arithmetic mean of biases (e.g. over LODO folds)."""
        def gmean(values) -> float:
            return float(torch.tensor(values).log().mean().exp())
        return cls([gmean(ts) for ts in zip(*(c.attr_t for c in items))],
                   {n: gmean([c.group_t[n] for c in items]) for n in items[0].group_t},
                   [sum(bs) / len(bs) for bs in zip(*(c.attr_b for c in items))])


def _fit(loss_fn, n: int, with_bias: bool, steps: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    log_t = torch.zeros(n, requires_grad=True)
    bias = torch.zeros(n, requires_grad=with_bias)
    opt = torch.optim.LBFGS([log_t, bias] if with_bias else [log_t], lr=0.1, max_iter=steps,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn(log_t.exp(), bias)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.detach().exp().clamp(0.05, 20.0), bias.detach()


def fit_calibration(preds: Predictions, labels: torch.Tensor, mode: str = "temperature") -> Calibration:
    if mode not in ("temperature", "platt"):
        raise ValueError(f"unknown calibration mode {mode!r}")
    y = labels.float()
    logits = preds["attr"].float()
    # Separable per attribute: summing per-attribute means fits each parameter independently.
    attr_t, attr_b = _fit(lambda t, b: F.binary_cross_entropy_with_logits(logits / t + b, y, reduction="none")
                          .mean(0).sum(), NUM_ATTRS, with_bias=mode == "platt")
    cls = encode_groups(labels)
    group_t = {}
    for k, g in enumerate(GROUPS):
        valid = cls[:, k] >= 0
        lg = preds[f"group/{g.name}"].float()[valid]
        group_t[g.name] = float(_fit(lambda t, b: F.cross_entropy(lg / t, cls[valid, k]), 1, False)[0][0])
    return Calibration(attr_t.tolist(), group_t, attr_b.tolist())
