from __future__ import annotations

import numpy as np
import pytest
import torch

from src.attributes import ATTRIBUTE_NAMES, GROUPS, NUM_ATTRS, align_columns, encode_groups
from src.calibration import Calibration, fit_calibration
from src.data import hamming_neighbors, novel_queries, query_mask, sample_reliability, sample_weights
from src.metrics import expected_calibration_error, per_query_metrics, evaluate_output, summarize


def test_load_split_matches_protocol(val):
    assert val.labels.shape[1] == NUM_ATTRS
    assert np.array_equal(val.queries[val.query_ids], val.labels)
    assert len(np.unique(val.queries, axis=0)) == len(val.queries)


def test_encode_groups():
    y = np.zeros((3, NUM_ATTRS), dtype=np.int8)
    y[0, [1, 4, 9]] = 1          # adult, short hair, upper blue
    y[1, [9, 10]] = 1            # two upper colours, no hair (allowed none)
    cls = encode_groups(y)
    hair, upper, age = (next(k for k, g in enumerate(GROUPS) if g.name == n) for n in ("hair", "upper_color", "age"))
    assert cls[0, age] == 1 and cls[0, hair] == 0 and cls[0, upper] == 1
    assert cls[1, upper] == -1 and cls[1, hair] == 3 and cls[1, age] == -1


def test_align_columns_roundtrip():
    names = list(ATTRIBUTE_NAMES)[::-1]
    q = np.arange(NUM_ATTRS)[::-1][None]       # column j holds the canonical index of names[j]
    assert np.array_equal(q[:, align_columns(names)][0], np.arange(NUM_ATTRS))


def test_metrics_perfect_and_reversed(small):
    agree = small.queries.astype(int) @ small.labels.T.astype(int) + \
        (1 - small.queries.astype(int)) @ (1 - small.labels.T.astype(int))
    perfect = evaluate_output({"similarities": agree.astype(float)}, small.queries, small.labels)
    assert all(v == pytest.approx(1.0) for v in perfect.values())
    worst = evaluate_output({"distances": agree.astype(float)}, small.queries, small.labels)
    assert worst["mAP"] < 0.2


def test_summarize_masks_queries(small):
    rng = np.random.default_rng(0)
    scores = rng.random((len(small.queries), len(small)))
    per_query = per_query_metrics([scores], small.queries, small.labels)
    mask = np.arange(len(small.queries)) % 2 == 0
    both = summarize(per_query)
    halves = summarize(per_query, mask), summarize(per_query, ~mask)
    assert both["mAP"] == pytest.approx((halves[0]["mAP"] * mask.sum() + halves[1]["mAP"] * (~mask).sum()) / len(mask))


def test_ece_separates_calibrated_from_overconfident():
    rng = np.random.default_rng(0)
    p = rng.random((20000, 3))
    y = (rng.random(p.shape) < p).astype(float)
    assert expected_calibration_error(p, y).max() < 0.02
    assert expected_calibration_error(np.where(p > 0.5, 0.99, 0.01), y).min() > 0.15


def test_calibration_temperature_and_platt():
    torch.manual_seed(0)
    true = 2 * torch.randn(4000, NUM_ATTRS)          # calibrated logits
    y = torch.bernoulli(torch.sigmoid(true))
    preds = {"attr": true * 4 - 2}                   # over-confident by 4x and shifted
    preds.update({f"group/{g.name}": torch.zeros(len(y), g.num_classes) for g in GROUPS})
    platt = fit_calibration(preds, y, "platt")
    assert np.median(platt.attr_t) == pytest.approx(4.0, rel=0.3)
    assert np.median(platt.attr_b) == pytest.approx(0.5, abs=0.2)
    assert Calibration.from_dict(platt.to_dict()) == platt
    avg = Calibration.average([Calibration(attr_t=[2.0] * NUM_ATTRS), Calibration(attr_t=[8.0] * NUM_ATTRS)])
    assert avg.attr_t[0] == pytest.approx(4.0)


def test_sampling_and_neighbors(small):
    w = sample_weights(small, domain_balanced=True)
    doms = small.domains
    per_domain = [w[doms == d].sum() for d in np.unique(doms)]
    assert np.allclose(per_domain, per_domain[0])
    nb = hamming_neighbors(small.queries, max_dist=2)
    for i, n in enumerate(nb):
        d = NUM_ATTRS - (small.queries[n] == small.queries[i]).sum(1)
        assert i not in n and np.all((d >= 1) & (d <= 2)) and np.all(np.diff(d) >= 0)


def test_test_like_galleries_and_novel_queries(val, train_split):
    uniform = val.subset(query_mask(val, num_queries=300, seed=0))
    by_size = val.subset(query_mask(val, num_queries=300, seed=0, by_size=True))
    assert len(by_size) / len(by_size.queries) > 2 * len(uniform) / len(uniform.queries)
    novel = novel_queries(val, train_split)
    assert 0.3 < novel.mean() < 0.4          # README: 35 % of val queries are absent from train


def test_reliability_only_down_weights_market(train_split):
    w = sample_reliability(train_split, 0.5)
    market = train_split.domains == "Market1501"
    assert np.all(w[~market] == 1) and 0 < w[market].min() and (w[market] < 1).any()
    assert np.all(sample_reliability(train_split, 0.0) == 1)
