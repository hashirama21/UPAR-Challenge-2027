from __future__ import annotations

import numpy as np
import pytest
import torch

from src.attributes import GROUPS, NUM_ATTRS
from src.evaluate import simulate
from src.metrics import evaluate_blocks
from src.scoring import Scorer


def test_expected_ndom_matches_untruncated_dp(make_score_cfg):
    rng = np.random.default_rng(0)
    preds = {"attr": torch.randn(7, NUM_ATTRS) * 2}
    preds.update({f"group/{g.name}": torch.zeros(7, g.num_classes) for g in GROUPS})
    q = (rng.random((5, NUM_ATTRS)) < 0.2).astype(np.float32)
    prior = rng.uniform(0.05, 0.4, NUM_ATTRS)
    got = Scorer(make_score_cfg(name="endom"), preds, q, prior)._expected_ndom(0, 5).numpy()

    p = torch.sigmoid(preds["attr"]).clamp(1e-4, 1 - 1e-4).numpy()
    dom_bar = (q @ prior + (1 - q) @ (1 - prior)) / NUM_ATTRS
    for i in range(5):
        for j in range(7):
            dist = np.zeros(NUM_ATTRS + 1)
            dist[0] = 1
            for a in range(NUM_ATTRS):
                m = q[i, a] * (1 - p[j, a]) + (1 - q[i, a]) * p[j, a]
                dist = dist * (1 - m) + np.concatenate([[0], dist[:-1]]) * m
            cap = NUM_ATTRS * (1 - dom_bar[i])
            ref = float(np.sum(dist * np.maximum(0, 1 - np.arange(NUM_ATTRS + 1) / cap)))
            assert got[i, j] == pytest.approx(ref, abs=1e-5)


@pytest.mark.parametrize("name", ["l1", "loglik", "structured", "endom", "mix"])
def test_scorer_blocks_are_consistent(small, name, make_score_cfg):
    preds = simulate(small, 2.0, 3.0, 0)
    scorer = Scorer(make_score_cfg(name=name), preds, small.queries, small.labels.mean(0))
    a, b = scorer.full(size=7), scorer.full(size=64)
    assert a.dtype == np.float64 and a.shape == (len(small.queries), len(small)) and np.array_equal(a, b)
    assert all(len(np.unique(row)) == len(row) for row in a)


@pytest.mark.parametrize("name", ["l1", "loglik", "structured", "endom", "mix"])
def test_no_ties_even_with_clipped_probabilities(name, make_score_cfg):
    attr = torch.full((4, NUM_ATTRS), -12.0)        # far beyond the 1e-4 clip
    attr[:, 0] = torch.tensor([30.0, 25.0, 20.0, 15.0])   # same clipped p, different raw logits
    preds = {"attr": attr, **{f"group/{g.name}": torch.zeros(4, g.num_classes) for g in GROUPS}}
    q = np.zeros((1, NUM_ATTRS), dtype=np.float32)
    q[0, 0] = 1
    row = Scorer(make_score_cfg(name=name), preds, q, np.full(NUM_ATTRS, 0.2)).full()[0]
    assert len(np.unique(row)) == 4 and np.all(np.diff(row) < 0)


def test_group_weights_change_structured_score(small, make_score_cfg):
    preds = simulate(small, 2.0, 3.0, 0)
    prior = small.labels.mean(0)
    n = len(small.queries)
    base = Scorer(make_score_cfg(), preds, small.queries, prior).score(0, n)
    weighted = Scorer(make_score_cfg(group_weights=[2.0] + [1.0] * (len(GROUPS) - 1)), preds, small.queries, prior).score(0, n)
    assert not torch.allclose(base, weighted)
    with pytest.raises(ValueError):
        make_score_cfg(group_weights=[1.0])
    assert torch.equal(base, Scorer(make_score_cfg(group_weights=[1.0] * len(GROUPS)), preds, small.queries,
                                    prior).score(0, n))


def test_structure_aware_scores_beat_l1_in_simulation(small, make_score_cfg):
    preds = simulate(small, 2.0, 3.0, 0)
    prior = small.labels.mean(0)
    madm = {n: evaluate_blocks(Scorer(make_score_cfg(name=n), preds, small.queries, prior).blocks(),
                               small.queries, small.labels)["mADM"] for n in ("l1", "loglik", "mix")}
    assert madm["loglik"] > madm["l1"] and madm["mix"] > madm["l1"]


def test_transductive_posterior_is_normalised(small, make_score_cfg):
    preds = simulate(small, 2.0, 3.0, 0)
    s = Scorer(make_score_cfg(name="loglik", transductive=True), preds, small.queries, small.labels.mean(0))
    assert torch.all(torch.logsumexp(s.score(0, s.num_queries), 0) <= 1e-4)
