from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.attributes import ATTRIBUTE_NAMES, GROUPS, NUM_ATTRS, align_columns, encode_groups  # noqa: E402
from src.calibration import Calibration, fit_calibration  # noqa: E402
from src.checkpoint import Bundle  # noqa: E402
from src.losses import CSARLoss, LossConfig  # noqa: E402
from src.model import BACKBONES, CSARNet, ModelConfig  # noqa: E402
from src.data import Split, hamming_neighbors, load_split, query_mask, sample_weights  # noqa: E402
from src.evaluate import simulate  # noqa: E402
from src.metrics import evaluate_blocks, evaluate_output  # noqa: E402
from src.scoring import ScoreConfig, Scorer  # noqa: E402
from src.synthetic import make_synthetic_data  # noqa: E402

HAS_DATA = (ROOT / "data" / "annotations" / "task2" / "val" / "gt.csv").exists()


@pytest.fixture(scope="module")
def val() -> Split:
    if not HAS_DATA:
        pytest.skip("annotations not available")
    return load_split(ROOT / "data", "val")


@pytest.fixture(scope="module")
def small(val) -> Split:
    return val.subset(query_mask(val, num_queries=60, seed=1))


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
    perm = align_columns(names)
    q = np.arange(NUM_ATTRS)[::-1][None]       # column j holds the canonical index of names[j]
    assert np.array_equal(q[:, perm][0], np.arange(NUM_ATTRS))


def test_metrics_perfect_and_reversed(small):
    agree = small.queries.astype(int) @ small.labels.T.astype(int) + \
        (1 - small.queries.astype(int)) @ (1 - small.labels.T.astype(int))
    perfect = evaluate_output({"similarities": agree.astype(float)}, small.queries, small.labels)
    assert all(v == pytest.approx(1.0) for v in perfect.values())
    worst = evaluate_output({"distances": agree.astype(float)}, small.queries, small.labels)
    assert worst["mAP"] < 0.2


def test_expected_ndom_matches_untruncated_dp():
    rng = np.random.default_rng(0)
    preds = {"attr": torch.randn(7, NUM_ATTRS) * 2}
    preds.update({f"group/{g.name}": torch.zeros(7, g.num_classes) for g in GROUPS})
    q = (rng.random((5, NUM_ATTRS)) < 0.2).astype(np.float32)
    prior = rng.uniform(0.05, 0.4, NUM_ATTRS)
    scorer = Scorer(ScoreConfig(name="endom"), preds, q, prior)
    got = scorer._expected_ndom(0, 5).numpy()

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
def test_scorer_blocks_are_consistent(small, name):
    preds = simulate(small, sigma=2.0)
    scorer = Scorer(ScoreConfig(name=name), preds, small.queries, small.labels.mean(0))
    a, b = scorer.full(size=7), scorer.full(size=64)
    assert a.shape == (len(small.queries), len(small)) and np.allclose(a, b, atol=1e-5)


def test_structure_aware_scores_beat_l1_in_simulation(small):
    preds = simulate(small, sigma=2.0)
    prior = small.labels.mean(0)
    madm = {n: evaluate_blocks(Scorer(ScoreConfig(name=n), preds, small.queries, prior).blocks(),
                               small.queries, small.labels)["mADM"] for n in ("l1", "loglik", "mix")}
    assert madm["loglik"] > madm["l1"] and madm["mix"] > madm["l1"]


def test_transductive_posterior_is_normalised(small):
    preds = simulate(small, sigma=2.0)
    s = Scorer(ScoreConfig(name="loglik", transductive=True), preds, small.queries, small.labels.mean(0))
    total = torch.logsumexp(torch.as_tensor(s.full()), 0)
    assert torch.all(total <= 1e-4)


def test_calibration_recovers_temperature():
    torch.manual_seed(0)
    true = 2 * torch.randn(4000, NUM_ATTRS)          # calibrated logits
    y = torch.bernoulli(torch.sigmoid(true))
    preds = {"attr": true * 4}                       # over-confident by 4x
    preds.update({f"group/{g.name}": torch.zeros(len(y), g.num_classes) for g in GROUPS})
    cal = fit_calibration(preds, y)
    assert np.median(cal.attr_t) == pytest.approx(4.0, rel=0.3)
    assert Calibration.from_dict(cal.to_dict()) == cal


def test_sampling_and_neighbors(small):
    w = sample_weights(small, domain_balanced=True)
    doms = small.domains
    per_domain = [w[doms == d].sum() for d in np.unique(doms)]
    assert np.allclose(per_domain, per_domain[0])
    nb = hamming_neighbors(small.queries, max_dist=2)
    for i, n in enumerate(nb):
        d = NUM_ATTRS - (small.queries[n] == small.queries[i]).sum(1)
        assert i not in n and np.all((d >= 1) & (d <= 2)) and np.all(np.diff(d) >= 0)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_backbone_end_to_end(tmp_path, name):
    """Build -> forward -> one training step -> fp16 checkpoint -> reload -> same predictions."""
    torch.manual_seed(0)
    model = CSARNet(ModelConfig(name, height=128, width=64, embed_dim=32))
    x = torch.randn(4, 3, 128, 64)
    y = torch.zeros(4, NUM_ATTRS)
    y[:, [1, 4, 8, 21, 33]] = 1
    vocab = np.unique(np.vstack([y.numpy(), np.eye(NUM_ATTRS)]), axis=0).astype(np.int8)
    qid = torch.as_tensor([int(np.flatnonzero((vocab == y[0].numpy()).all(1))[0])] * 4)
    loss_fn = CSARLoss(LossConfig(), y.numpy().mean(0), vocab, hamming_neighbors(vocab))
    out = model(x)
    assert out["attr"].shape == (4, NUM_ATTRS) and out["emb"].shape == (4, 32)
    assert all(out[f"group/{g.name}"].shape == (4, g.num_classes) for g in GROUPS)
    loss, parts = loss_fn(model, out, y, qid)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"par", "group", "ret"}
    assert all(p.grad is not None for p in model.attr_head.parameters())

    model.eval()
    Bundle(model.cfg, model.state_dict(), [0.1] * NUM_ATTRS).save(tmp_path / "m.pt", half=True)
    reloaded = Bundle.load(tmp_path / "m.pt").build_model()
    with torch.no_grad():
        ref, got = model(x)["attr"], reloaded(x)["attr"]
    assert torch.allclose(ref, got, atol=0.05 * ref.abs().max().item() + 1e-3)


# ---- end to end: train -> export -> run.py as the ingestion program would call it ------------------

@pytest.mark.slow
def test_end_to_end(tmp_path, val):  # noqa: ARG001 (val: skip without annotations)
    from src import evaluate, export, train

    data = make_synthetic_data(ROOT / "data", tmp_path / "data")

    run = tmp_path / "run"
    train.main(["--data-dir", str(data), "--out", str(run), "--backbone", "resnet18", "--no-pretrained",
                "--epochs", "1", "--batch-size", "16", "--workers", "0", "--height", "64", "--width", "32"])
    ckpt = run / "model.pt"
    evaluate.main(["--data-dir", str(data), "--checkpoint", str(ckpt), "--scores", "loglik", "mix",
                   "--gammas", "0", "0.5", "--workers", "0", "--no-domains", "--save-best"])
    archive = export.export(ckpt, tmp_path / "sub")
    assert archive.exists()

    spec = importlib.util.spec_from_file_location("sub_run", tmp_path / "sub" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.WORKERS = 0
    gallery_split = load_split(data, "val")
    names = list(ATTRIBUTE_NAMES)[::-1]    # the platform order must not matter
    sample = {"attribute_names": names,
              "queries": gallery_split.queries[:, align_columns(names).argsort()].tolist(),
              "gallery": [{"image_path": str(data / p)} for p in gallery_split.images]}
    out = mod.rank_gallery(sample)
    sims = out["similarities"]
    assert sims.shape == (len(gallery_split.queries), len(gallery_split)) and np.isfinite(sims).all()
    res = evaluate_output(out, gallery_split.queries, gallery_split.labels)
    assert 0 <= res["mADM"] <= 1
