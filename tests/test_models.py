from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from src.attributes import GROUPS, NUM_ATTRS
from src.checkpoint import Bundle, Member
from src.config import augment_config
from src.data import hamming_neighbors
from src.finetune import LoRALinear, apply_lora, freeze_backbone, merge_lora, param_groups
from src.inference import combine, encode_queries, fit_to_budget
from src.losses import CSARLoss
from src.model import BACKBONES, CSARNet
from src.transforms import build_transform, input_size
from src.vit import ViTConfig, VisionTransformer


@pytest.fixture
def model_cfg(make_model_cfg):
    """Model config at a patch-compatible size for ``name`` (128x64 rounded to the patch)."""
    def make(name: str, **kw):
        probe = make_model_cfg(backbone=name, height=128, width=64, **kw)
        h, w = input_size(probe)
        return make_model_cfg(backbone=name, height=h, width=w, **kw)
    return make


def _batch(n: int = 4):
    y = torch.zeros(n, NUM_ATTRS)
    y[:, [1, 4, 8, 21, 33]] = 1
    near_misses = np.repeat(y[:1].numpy(), 3, 0)
    near_misses[[0, 1, 2], [2, 9, 36]] = 1          # Hamming-1 neighbours, as in real galleries
    vocab = np.unique(np.vstack([y.numpy(), near_misses, np.eye(NUM_ATTRS)]), axis=0).astype(np.int8)
    qid = torch.as_tensor([int(np.flatnonzero((vocab == y[0].numpy()).all(1))[0])] * n)
    return y, vocab, qid


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_backbone_end_to_end(tmp_path, name, model_cfg, make_loss_cfg, make_score_cfg, make_runtime):
    """Build -> forward -> one training step -> fp16 checkpoint -> reload -> same predictions."""
    torch.manual_seed(0)
    cfg = model_cfg(name, embed_dim=32)
    model = CSARNet(cfg)
    x = torch.randn(2, 3, cfg.height, cfg.width)
    y, vocab, qid = _batch(2)
    out = model(x)
    assert out["attr"].shape == (2, NUM_ATTRS) and out["emb"].shape == (2, 32)
    assert all(out[f"group/{g.name}"].shape == (2, g.num_classes) for g in GROUPS)
    loss, parts = CSARLoss(make_loss_cfg(), y.numpy().mean(0), vocab, hamming_neighbors(vocab))(model, out, y, qid)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"par", "group", "ret"}
    assert all(p.grad is not None for p in model.head.parameters())

    model.eval()
    Bundle([Member(cfg, model.state_dict())], [0.1] * NUM_ATTRS, make_score_cfg(), make_runtime()).save(
        tmp_path / "m.pt", half=True)
    reloaded = Bundle.load(tmp_path / "m.pt").build_models()[0]
    with torch.no_grad():
        ref, got = model(x)["attr"], reloaded(x)["attr"]
    assert torch.allclose(ref, got, atol=0.05 * ref.abs().max().item() + 1e-3)


@pytest.mark.parametrize("backbone", ["resnet18", "dinov2_vits14", "dinov3_vits16"])
def test_query_head_trains(backbone, model_cfg, make_loss_cfg):
    torch.manual_seed(0)
    model = CSARNet(model_cfg(backbone, head="query", embed_dim=0))
    x = torch.randn(2, 3, model.cfg.height, model.cfg.width)
    y, vocab, qid = _batch(2)
    loss, _ = CSARLoss(make_loss_cfg(w_ret=0.0), y.numpy().mean(0), vocab)(model, model(x), y, qid)
    loss.backward()
    assert model.head.queries.grad is not None and model.head.queries.grad.abs().sum() > 0


def test_vit_position_interpolation_and_rope():
    vit = VisionTransformer(ViTConfig(16, 64, 1, 4, 128, (14, 14))).eval()
    tokens, cls = vit(torch.randn(1, 3, 224, 112))
    assert tokens.shape == (1, 14 * 7, 64) and cls.shape == (1, 64)
    assert torch.equal(vit._pos((14, 14)), vit.pos_embed)
    rope = VisionTransformer(ViTConfig(16, 64, 1, 4, 128, (14, 14), registers=4, pos_embed=False,
                                       rope_theta=100.0, layerscale=True)).eval()
    tokens, cls = rope(torch.randn(1, 3, 256, 128))
    assert tokens.shape == (1, 16 * 8, 64) and rope.pos_embed is None


def test_lora_merge_is_exact_and_only_adapters_train(model_cfg):
    torch.manual_seed(0)
    model = CSARNet(model_cfg("dinov2_vits14", embed_dim=0)).eval()
    assert apply_lora(model, rank=4) > 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.up, std=0.05)
    trainable_backbone = [name for name, p in model.backbone.named_parameters() if p.requires_grad]
    assert trainable_backbone and all(".down" in t or ".up" in t for t in trainable_backbone)
    x = torch.randn(1, 3, model.cfg.height, model.cfg.width)
    with torch.no_grad():
        before = model(x)["attr"]
        after = merge_lora(model)(x)["attr"]
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert torch.allclose(before, after, atol=1e-4)
    assert set(model.state_dict()) == set(CSARNet(model.cfg).state_dict())


def test_partial_finetuning_and_param_groups(model_cfg):
    model = CSARNet(model_cfg("resnet18"))
    freeze_backbone(model, 1)
    trainable = {n.split(".")[1] for n, p in model.backbone.named_parameters() if p.requires_grad}
    assert trainable == {"7"}                     # body index of layer4
    groups = param_groups(model, 1e-4, 0.1, 5e-4)
    assert [g["lr"] for g in groups] == [pytest.approx(1e-5), 1e-4]


def test_training_transform_keeps_colours(cfg, make_model_cfg):
    img = Image.new("RGB", (40, 100), (200, 20, 20))
    mcfg = make_model_cfg(backbone="resnet18", height=128, width=64)
    spec = mcfg.spec
    augment = augment_config(cfg)
    augment.low_res = 1.0
    for _ in range(20):
        x = build_transform(mcfg, augment)(img)
        rgb = x * torch.tensor(spec.std)[:, None, None] + torch.tensor(spec.mean)[:, None, None]
        person = (rgb[0] - rgb[1:].max(0).values) > 0.3
        assert person.float().mean() > 0.1      # still mostly a red figure on a grey canvas


def test_ensemble_embeddings_average_member_cosines(model_cfg):
    torch.manual_seed(0)
    members = [{"attr": torch.randn(3, NUM_ATTRS), "emb": F.normalize(torch.randn(3, 8), dim=-1)} for _ in range(2)]
    merged = combine(members)
    assert torch.allclose(merged["attr"], (members[0]["attr"] + members[1]["attr"]) / 2)
    models = [CSARNet(model_cfg("resnet18", embed_dim=8)).eval() for _ in range(2)]
    q = torch.eye(NUM_ATTRS)[:5]
    mean_cosine = sum(encode_queries([m], q) @ p["emb"].T for m, p in zip(models, members)) / 2
    assert torch.allclose(encode_queries(models, q) @ merged["emb"].T, mean_cosine, atol=1e-6)


def test_degraded_mode_drops_tta_then_members(tmp_path, make_model_cfg, make_runtime):
    paths = []
    for i in range(8):
        paths.append(tmp_path / f"{i}.jpg")
        Image.new("RGB", (40, 100), (i * 20, 50, 50)).save(paths[-1])
    models = [CSARNet(make_model_cfg(backbone="resnet18", height=64, width=32)).eval() for _ in range(3)]
    rt = make_runtime(flip=True, scales=[1.0, 1.25], time_budget_s=1e-9)
    kept, degraded = fit_to_budget(models, rt, paths, num_workers=0)
    assert len(kept) == 1 and degraded.flip is False and degraded.scales == [1.0]
    unlimited = make_runtime(time_budget_s=0.0)
    kept, same = fit_to_budget(models, unlimited, paths, num_workers=0)
    assert len(kept) == 3 and same == unlimited


def test_retrieval_loss_variants(model_cfg, make_loss_cfg):
    torch.manual_seed(0)
    model = CSARNet(model_cfg("resnet18", embed_dim=16))
    x = torch.randn(4, 3, model.cfg.height, model.cfg.width)
    y, vocab, qid = _batch(4)
    for loss_cfg in (make_loss_cfg(margin_type="arc"), make_loss_cfg(margin_type="cos"),
                     make_loss_cfg(soft_targets=True)):
        loss, parts = CSARLoss(loss_cfg, y.numpy().mean(0), vocab, hamming_neighbors(vocab))(model, model(x), y, qid)
        assert torch.isfinite(loss) and parts["ret"] > 0
    model.eval()
    weighted = CSARLoss(make_loss_cfg(), y.numpy().mean(0), vocab)(
        model, model(x), y, qid, weight=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    first_only = CSARLoss(make_loss_cfg(), y.numpy().mean(0), vocab)(model, model(x[:1]), y[:1], qid[:1])
    assert weighted[1]["par"] == pytest.approx(first_only[1]["par"], rel=1e-4)
