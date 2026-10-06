"""Training objective  L = L_PAR + w_group * L_group + w_ret * L_ret, per-sample weighted.

* L_PAR   weighted BCE (positive-rate weights, Li et al. 2015) with label smoothing;
* L_group cross-entropy per attribute group, ignoring non-representable labels;
* L_ret   image -> query contrastive loss over the batch queries plus Hamming-1/2
          hard negatives from the query vocabulary, with an angular (ArcFace, as in
          CLEAR) or additive cosine margin on the positive. ``soft_targets`` spreads
          the target over near-misses in proportion to their ndom, as mADM credits them.

Label smoothing and sample weights are per image (``sample_smoothing`` /
``data.sample_reliability``): noisier sources get more smoothing, images that
disagree with their identity's majority labels get less weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .attributes import GROUPS, NUM_ATTRS, encode_groups
from .model import CSARNet

@dataclass
class LossConfig:
    """Values come from configs/config.yaml (``loss``)."""
    smoothing: float
    noisy_smoothing: float
    noisy_domains: list[str]    # identity-level labels (Market1501), very low resolution (PETA)
    w_group: float
    w_ret: float
    ret_scale: float
    ret_margin: float
    margin_type: str            # arc | cos
    hard_negatives: int         # sampled Hamming neighbours per image
    neighbor_max_dist: int      # Hamming radius of the hard-negative pool
    neighbor_max_count: int
    soft_targets: bool


def sample_smoothing(cfg: LossConfig, domains: np.ndarray) -> np.ndarray:
    return np.where(np.isin(domains, cfg.noisy_domains), cfg.noisy_smoothing, cfg.smoothing).astype(np.float32)


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (values * weight).sum() / weight.sum().clamp(min=1e-8)


def _smoothed_ce(logits: torch.Tensor, target: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Per-sample CE with per-sample label smoothing; 0 where ``target`` is -1."""
    logp = F.log_softmax(logits, -1)
    valid = target >= 0
    nll = -logp.gather(1, target.clamp(min=0)[:, None])[:, 0]
    return torch.where(valid, (1 - eps) * nll - eps * logp.mean(-1), torch.zeros_like(nll))


class CSARLoss:
    def __init__(self, cfg: LossConfig, attr_prior: np.ndarray, vocab: np.ndarray,
                 neighbors: list[np.ndarray] | None = None, seed: int = 0):
        if cfg.margin_type not in ("arc", "cos"):
            raise ValueError(f"unknown margin_type {cfg.margin_type!r}")
        self.cfg = cfg
        r = torch.as_tensor(attr_prior, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
        self.w_pos, self.w_neg = torch.exp(1 - r), torch.exp(r)
        self.vocab = torch.as_tensor(vocab, dtype=torch.float32)
        self.neighbors = neighbors or []
        self.rng = np.random.default_rng(seed)
        self.ndom_cap = NUM_ATTRS * (1 - (self.vocab @ r + (1 - self.vocab) @ (1 - r)) / NUM_ATTRS)

    def __call__(self, model: CSARNet, out: dict[str, torch.Tensor], y: torch.Tensor, qid: torch.Tensor,
                 smoothing: torch.Tensor | None = None, weight: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, dict[str, float]]:
        y = y.float()
        dev = y.device
        eps = (smoothing if smoothing is not None else torch.full((len(y),), self.cfg.smoothing)).to(dev)
        w = (weight if weight is not None else torch.ones(len(y))).to(dev)

        target = y * (1 - eps[:, None]) + eps[:, None] / 2
        pos_neg = y * self.w_pos.to(dev) + (1 - y) * self.w_neg.to(dev)
        bce = F.binary_cross_entropy_with_logits(out["attr"].float(), target, pos_neg, reduction="none").mean(1)
        parts = {"par": _weighted_mean(bce, w)}

        if self.cfg.w_group:
            cls = encode_groups(y.cpu()).to(dev)
            ce = sum(_smoothed_ce(out[f"group/{g.name}"].float(), cls[:, k], eps) for k, g in enumerate(GROUPS))
            parts["group"] = _weighted_mean(ce / len(GROUPS), w)

        if self.cfg.w_ret and model.has_retrieval:
            parts["ret"] = _weighted_mean(self._retrieval(model, out["emb"].float(), qid.cpu().numpy()), w)

        total = parts["par"] + self.cfg.w_group * parts.get("group", 0) + self.cfg.w_ret * parts.get("ret", 0)
        return total, {k: float(v.detach()) for k, v in parts.items()}

    def _candidates(self, qid: np.ndarray) -> np.ndarray:
        cand = [qid]
        for q in qid:
            nb = self.neighbors[q] if q < len(self.neighbors) else ()
            if len(nb):
                cand.append(self.rng.choice(nb, size=min(self.cfg.hard_negatives, len(nb)), replace=False))
        return np.unique(np.concatenate(cand))

    def _retrieval(self, model: CSARNet, emb: torch.Tensor, qid: np.ndarray) -> torch.Tensor:
        """Per-sample contrastive loss."""
        cand = self._candidates(qid)
        vocab = self.vocab[cand].to(emb.device)
        cos = (emb @ model.encode_queries(vocab).T).clamp(-1 + 1e-6, 1 - 1e-6)
        pos = torch.as_tensor(np.searchsorted(cand, qid), device=emb.device)
        is_pos = F.one_hot(pos, len(cand)).bool()
        if self.cfg.margin_type == "arc":
            margined = torch.cos(torch.acos(cos) + self.cfg.ret_margin)
        else:
            margined = cos - self.cfg.ret_margin
        logits = self.cfg.ret_scale * torch.where(is_pos, margined, cos)
        if not self.cfg.soft_targets:
            return F.cross_entropy(logits, pos, reduction="none")
        own = self.vocab[qid].to(emb.device)
        ham = NUM_ATTRS - (own @ vocab.T + (1 - own) @ (1 - vocab).T)
        credit = (1 - ham / self.ndom_cap[qid].to(emb.device)[:, None]).clamp(min=0)
        soft = credit / credit.sum(-1, keepdim=True)
        return torch.sum(-soft * F.log_softmax(logits, -1), -1)
