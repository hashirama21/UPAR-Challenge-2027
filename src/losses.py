"""Training objective  L = L_PAR + w_group * L_group + w_ret * L_ret.

* L_PAR   weighted BCE (positive-rate weights, Li et al. 2015) with label smoothing;
* L_group cross-entropy per attribute group, ignoring non-representable labels;
* L_ret   image -> query contrastive loss over the batch queries plus
          Hamming-1/2 hard negatives from the query vocabulary, with an additive
          margin on the positive (CosFace style). ``soft_tau > 0`` spreads the
          target over near-misses, ~exp(-hamming / tau), mirroring mADM's credit.
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
    smoothing: float = 0.05
    w_group: float = 0.5
    w_ret: float = 0.2
    ret_scale: float = 20.0
    ret_margin: float = 0.2
    hard_negatives: int = 4     # sampled Hamming neighbours per image
    soft_tau: float = 0.0


class CSARLoss:
    def __init__(self, cfg: LossConfig, attr_prior: np.ndarray, vocab: np.ndarray,
                 neighbors: list[np.ndarray] | None = None, seed: int = 0):
        self.cfg = cfg
        r = torch.as_tensor(attr_prior, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
        self.w_pos, self.w_neg = torch.exp(1 - r), torch.exp(r)
        self.vocab = torch.as_tensor(vocab, dtype=torch.float32)
        self.neighbors = neighbors or []
        self.rng = np.random.default_rng(seed)

    def __call__(self, model: CSARNet, out: dict[str, torch.Tensor], y: torch.Tensor,
                 qid: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        y = y.float()
        dev = y.device
        target = y * (1 - self.cfg.smoothing) + self.cfg.smoothing / 2
        weight = y * self.w_pos.to(dev) + (1 - y) * self.w_neg.to(dev)
        parts = {"par": F.binary_cross_entropy_with_logits(out["attr"].float(), target, weight)}

        if self.cfg.w_group:
            cls = encode_groups(y.cpu()).to(dev)
            parts["group"] = sum(
                F.cross_entropy(out[f"group/{g.name}"].float(), cls[:, k], ignore_index=-1,
                                label_smoothing=self.cfg.smoothing)
                for k, g in enumerate(GROUPS)) / len(GROUPS)

        if self.cfg.w_ret and model.has_retrieval:
            parts["ret"] = self._retrieval(model, out["emb"].float(), qid.cpu().numpy())

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
        cand = self._candidates(qid)
        vocab = self.vocab[cand].to(emb.device)
        cos = emb @ model.encode_queries(vocab).T
        pos = torch.as_tensor(np.searchsorted(cand, qid), device=emb.device)
        logits = self.cfg.ret_scale * (cos - self.cfg.ret_margin * F.one_hot(pos, len(cand)))
        if self.cfg.soft_tau <= 0:
            return F.cross_entropy(logits, pos)
        own = self.vocab[qid].to(emb.device)
        ham = NUM_ATTRS - (own @ vocab.T + (1 - own) @ (1 - vocab).T)
        soft = torch.softmax(-ham / self.cfg.soft_tau, dim=-1)
        return torch.sum(-soft * F.log_softmax(logits, -1), -1).mean()
