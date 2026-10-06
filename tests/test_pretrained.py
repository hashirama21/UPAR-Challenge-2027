"""Conversions against the reference implementations (need cached weights; skipped otherwise).

Reference libraries are training-time only and not dependencies of this repo:
DINOv2 via torch.hub, CLIP via ``open_clip``, SigLIP 2 and DINOv3 via ``transformers``.
DINOv3 is checked with random weights (its released weights are gated).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.checkpoint import Bundle, Member
from src.model import CSARNet
from src.pretrained import CLIP_RELEASES, SOURCES, clip_state, convert_dinov3_state, read_safetensors
from src.vit import VisionTransformer

CACHE = Path(torch.hub.get_dir()) / "checkpoints"


def _cached(name: str) -> Path:
    path = CACHE / name
    if not path.is_file():
        pytest.skip(f"{name} not cached")
    return path


def _ours(name: str) -> VisionTransformer:
    vit = VisionTransformer(SOURCES[name].cfg)
    vit.load_state_dict(SOURCES[name].convert())
    return vit.eval()


@torch.no_grad()
def test_dinov2_matches_torch_hub():
    _cached("dinov2_vits14_pretrain.pth")
    try:
        ref = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()
    except Exception as exc:  # no network / hub cache
        pytest.skip(f"torch.hub reference unavailable: {exc}")
    x = torch.randn(1, 3, 518, 518)
    tokens, cls = _ours("dinov2_vits14")(x)
    r = ref.forward_features(x)
    assert torch.allclose(cls, r["x_norm_clstoken"], atol=1e-5)
    assert torch.allclose(tokens, r["x_norm_patchtokens"], atol=1e-5)


@torch.no_grad()
def test_clip_matches_open_clip():
    open_clip = pytest.importorskip("open_clip")
    name, digest = CLIP_RELEASES["clip_vitb16"]
    path = _cached(f"clip-{name}.pt")
    ref = open_clip.load_openai_model(str(path), precision="fp32", device="cpu").eval()
    x = torch.randn(2, 3, 224, 224)
    _, cls = _ours("clip_vitb16")(x)
    assert torch.allclose(cls @ clip_state(name, digest)["visual.proj"].float(), ref.encode_image(x), atol=1e-4)

    from src.clip_text import TextEncoder, Tokenizer, query_prompts
    _cached("clip_bpe_simple_vocab_16e6.txt.gz")
    tokens = Tokenizer()(query_prompts())
    assert torch.equal(tokens, open_clip.get_tokenizer("ViT-B-16")(query_prompts()))
    ours = TextEncoder(clip_state(name, digest)).eval()(tokens)
    assert torch.allclose(ours, ref.encode_text(tokens), atol=1e-4)


@torch.no_grad()
def test_siglip2_matches_transformers():
    transformers = pytest.importorskip("transformers")
    repo = "google/siglip2-base-patch16-224"
    weights = _cached(f"{repo.replace('/', '--')}.safetensors")
    cfg = json.loads(_cached("siglip2-base-config.json").read_text())
    ref = transformers.SiglipVisionModel(transformers.SiglipVisionConfig(**cfg["vision_config"])).eval()
    ref.vision_model.load_state_dict(read_safetensors(weights, prefix="vision_model."))
    x = torch.randn(1, 3, 224, 224)
    tokens, _ = _ours("siglip2_vitb16")(x)
    assert torch.allclose(tokens, ref(pixel_values=x).last_hidden_state, atol=1e-4)


@torch.no_grad()
@pytest.mark.parametrize("name", ["dinov3_vits16", "dinov3_vitb16"])
def test_dinov3_matches_transformers(name):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "DINOv3ViTModel"):
        pytest.skip("transformers without DINOv3")
    vc = SOURCES[name].cfg
    ref = transformers.DINOv3ViTModel(transformers.DINOv3ViTConfig(
        hidden_size=vc.dim, num_hidden_layers=vc.depth, num_attention_heads=vc.heads, intermediate_size=vc.mlp_dim,
        num_register_tokens=vc.registers, patch_size=vc.patch, rope_theta=vc.rope_theta)).eval()
    torch.manual_seed(0)
    for p in ref.parameters():
        p.normal_(0, 0.05)
    ours = VisionTransformer(vc).eval()
    ours.load_state_dict(convert_dinov3_state(ref.state_dict()))
    x = torch.randn(1, 3, 256, 128)                 # rectangular pedestrian grid
    tokens, cls = ours(x)
    r = ref(pixel_values=x).last_hidden_state
    assert torch.allclose(cls, r[:, 0], atol=1e-5) and torch.allclose(tokens, r[:, vc.prefix:], atol=1e-5)


def test_wise_ft_interpolates_backbone_only(tmp_path, make_model_cfg, make_score_cfg, make_runtime):
    _cached("resnet18-f37072fd.pth")
    from src.combine import wise
    cfg = make_model_cfg(backbone="resnet18", height=64, width=32)
    tuned = CSARNet(cfg).state_dict()          # random weights stand in for a fine-tuned model
    path = tmp_path / "m.pt"
    Bundle([Member(cfg, tuned)], [0.1] * 40, make_score_cfg(), make_runtime()).save(path)
    zero_shot = CSARNet(cfg, pretrained=True).state_dict()
    for alpha, expected in ((0.0, zero_shot), (1.0, tuned)):
        mixed = wise(path, alpha).members[0].state_dict
        key = "backbone.body.0.weight"
        assert torch.allclose(mixed[key], expected[key].float())
        assert torch.equal(mixed["head.attr.weight"], tuned["head.attr.weight"])
