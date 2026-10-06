"""Query x gallery scores, computed by query blocks (never a Q x G x 40 tensor).

Scores (higher = better match), selected by ``ScoreConfig.name``:

    l1          -||p - q||_1                         (starter-kit baseline)
    loglik      S2 = sum_a log P(q_a | g), independent Bernoullis
    structured  S3 = sum_groups log softmax P(q_k | g) + independent Bernoullis
                     + gamma * <e_img, e_query>     (Bernoulli fallback for
                     vectors a group softmax cannot represent)
    endom       E[ndom] under a Poisson-binomial on the number of wrong attributes
    mix         E[ndom] + lam * P_exact, P_exact = exp(exact score)

``group_weights`` are the w_k of S3 (one per group, independent attributes keep 1).
Ties never fall back to the gallery order: ``block`` sorts lexicographically by
the score, then by the *unclipped* log-likelihood (clipped probabilities and
E[ndom] = 0 tie otherwise), and returns the ranks as similarities (G = best).
``score`` gives the raw values.

The mean degree of match per query (needed by ndom) comes from the *training*
attribute prior, not the test gallery: E[dom_q] = mean_a q_a pi_a + (1-q_a)(1-pi_a).

``transductive=True`` replaces the exact score by the posterior over the query
set, log p(q | g), with a background class for images matching no query. It uses
the full set of test queries: keep it off unless organisers approve it in writing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .attributes import GROUPS, INDEPENDENT, NUM_ATTRS, encode_groups
from .calibration import Predictions

SCORES = ("l1", "loglik", "structured", "endom", "mix")
EPS = 1e-4


@dataclass
class ScoreConfig:
    """Values come from configs/config.yaml (``score``) and are frozen into the checkpoint."""
    name: str
    exact: str                  # exact-match term used by "mix": loglik | structured
    lam: float
    gamma: float                # weight of the learned query-image compatibility
    transductive: bool
    background: float           # log-score of the "no query" class (transductive only)
    group_weights: list[float] | None

    def __post_init__(self):
        if self.name not in SCORES or self.exact not in ("loglik", "structured"):
            raise ValueError(f"invalid score config {self}")
        if self.group_weights is not None and len(self.group_weights) != len(GROUPS):
            raise ValueError(f"group_weights needs {len(GROUPS)} values")

    def to_dict(self) -> dict:
        return asdict(self)


def _mask(indices, device) -> torch.Tensor:
    m = torch.zeros(NUM_ATTRS, device=device)
    m[list(indices)] = 1
    return m


class Scorer:
    """Binds calibrated predictions and queries, then yields score blocks.

    ``query_emb`` is the model's encoding of the queries (structured score with gamma > 0).
    """

    def __init__(self, cfg: ScoreConfig, preds: Predictions, queries: np.ndarray | torch.Tensor,
                 attr_prior: np.ndarray | torch.Tensor, query_emb: torch.Tensor | None = None,
                 device: str | torch.device = "cpu"):
        self.cfg = cfg
        dev = torch.device(device)
        attr = preds["attr"].float().to(dev)
        self.p = torch.sigmoid(attr).clamp(EPS, 1 - EPS)
        self.raw_logp, self.raw_log1mp = F.logsigmoid(attr).double(), F.logsigmoid(-attr).double()
        self.logp, self.log1mp = self.p.log(), (1 - self.p).log()
        self.group_logp = [F.log_softmax(preds[f"group/{g.name}"].float().to(dev), -1) for g in GROUPS]
        self.emb = preds.get("emb")
        self.emb = self.emb.float().to(dev) if self.emb is not None else None
        self.q = torch.as_tensor(np.asarray(queries), dtype=torch.float32, device=dev)
        self.q_groups = encode_groups(self.q.cpu()).to(dev)
        self.q_emb = query_emb.float().to(dev) if query_emb is not None else None
        prior = torch.as_tensor(np.asarray(attr_prior), dtype=torch.float32, device=dev)
        self.dom_bar = (self.q @ prior + (1 - self.q) @ (1 - prior)) / NUM_ATTRS
        self._indep = _mask(INDEPENDENT, dev)
        self._group_masks = [_mask(g.indices, dev) for g in GROUPS]
        self._group_w = cfg.group_weights or [1.0] * len(GROUPS)
        self._kind = cfg.name if cfg.name in ("loglik", "structured") else cfg.exact
        self._log_norm = None
        if cfg.transductive:
            self._log_norm = self._posterior_normalizer()

    @property
    def num_queries(self) -> int:
        return self.q.shape[0]

    def _bernoulli(self, q: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            return (q * mask) @ self.logp.T + ((1 - q) * mask) @ self.log1mp.T
        return q @ self.logp.T + (1 - q) @ self.log1mp.T

    def _structured(self, s: int, e: int) -> torch.Tensor:
        q = self.q[s:e]
        score = self._bernoulli(q, self._indep)
        for k, mask in enumerate(self._group_masks):
            cls = self.q_groups[s:e, k]
            term = self._bernoulli(q, mask)  # fallback for non-representable vectors
            ok = cls >= 0
            if ok.any():
                term[ok] = self.group_logp[k][:, cls[ok]].T
            score = score + self._group_w[k] * term
        if self.cfg.gamma and self.emb is not None and self.q_emb is not None:
            score = score + self.cfg.gamma * self.q_emb[s:e] @ self.emb.T
        return score

    def _raw_exact(self, s: int, e: int) -> torch.Tensor:
        return self._structured(s, e) if self._kind == "structured" else self._bernoulli(self.q[s:e])

    def _exact(self, s: int, e: int) -> torch.Tensor:
        """Exact-match log-score, or log posterior over the query set when transductive."""
        score = self._raw_exact(s, e)
        return score - self._log_norm if self._log_norm is not None else score

    def _expected_ndom(self, s: int, e: int) -> torch.Tensor:
        q = self.q[s:e]
        cap = NUM_ATTRS * (1 - self.dom_bar[s:e])            # ndom hits 0 at k = cap wrong attributes
        kmax = int(torch.ceil(cap.max()).item())
        # Poisson-binomial DP over the number of wrong attributes, truncated at kmax (k > cap scores 0).
        dist = torch.zeros(q.shape[0], self.p.shape[0], kmax + 1, device=q.device)
        dist[..., 0] = 1
        for a in range(NUM_ATTRS):
            wrong = (q[:, a:a + 1] * (1 - self.p[:, a]) + (1 - q[:, a:a + 1]) * self.p[:, a]).unsqueeze(-1)
            shifted = F.pad(dist[..., :-1], (1, 0))
            dist = dist * (1 - wrong) + shifted * wrong
        k = torch.arange(kmax + 1, device=q.device, dtype=torch.float32)
        credit = (1 - k[None, :] / cap[:, None]).clamp(min=0)
        return torch.einsum("bgk,bk->bg", dist, credit)

    def _posterior_normalizer(self, block: int = 64) -> torch.Tensor:
        """log sum_q exp(score(q, g)) + background, per gallery image (G,)."""
        lse = torch.full((self.p.shape[0],), self.cfg.background, device=self.p.device)
        for s in range(0, self.num_queries, block):
            lse = torch.logaddexp(lse, torch.logsumexp(self._raw_exact(s, s + block), 0))
        return lse

    def score(self, s: int, e: int) -> torch.Tensor:
        """(b, G) raw scores for queries s:e, higher is better (may contain ties)."""
        name = self.cfg.name
        if name == "l1":
            return -(self.p.sum(1)[None] + self.q[s:e] @ (1 - 2 * self.p).T)
        if name in ("loglik", "structured"):
            return self._exact(s, e)
        endom = self._expected_ndom(s, e)
        return endom if name == "endom" else endom + self.cfg.lam * self._exact(s, e).exp()

    def _tie_breaker(self, s: int, e: int) -> torch.Tensor:
        q = self.q[s:e].double()
        return q @ self.raw_logp.T + (1 - q) @ self.raw_log1mp.T

    def block(self, s: int, e: int) -> torch.Tensor:
        """(b, G) float64 tie-free similarities: G for the best image down to 1."""
        primary, secondary = self.score(s, e).double(), self._tie_breaker(s, e)
        by_secondary = torch.argsort(secondary, dim=1, descending=True, stable=True)
        order = by_secondary.gather(1, torch.argsort(primary.gather(1, by_secondary), dim=1,
                                                     descending=True, stable=True))
        n = primary.shape[1]
        ranks = torch.arange(n, 0, -1, dtype=torch.float64, device=primary.device).expand_as(primary)
        return torch.empty_like(primary).scatter_(1, order, ranks)

    def blocks(self, size: int = 64) -> Iterator[np.ndarray]:
        for s in range(0, self.num_queries, size):
            yield self.block(s, s + size).cpu().numpy()

    def full(self, size: int = 64) -> np.ndarray:
        return np.concatenate(list(self.blocks(size)), axis=0) if self.num_queries else \
            np.zeros((0, self.p.shape[0]), dtype=np.float64)
