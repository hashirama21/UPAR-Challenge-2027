"""Foundation ViT weights: static configs + one-off download/conversion to ``src.vit`` keys.

Configs are static so the submission rebuilds the architecture offline; weights are
only fetched at training time and then live in the CSAR checkpoint.

    DINOv2   https://dl.fbaipublicfiles.com/dinov2 (Apache-2.0)
    DINOv3   https://huggingface.co/facebook (DINOv3 License, gated: accept it on the
             model page, then set HF_TOKEN to an access token of that account)
    CLIP     OpenAI release (TorchScript archives, MIT)
    SigLIP 2 https://huggingface.co/google (Apache-2.0, safetensors)
"""
from __future__ import annotations

import json
import os
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .vit import ViTConfig

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
_CLIP_URL = "https://openaipublic.azureedge.net/clip/models/{}/{}.pt"
_HF_URL = "https://huggingface.co/{}/resolve/main/{}"


def _dinov2(dim: int, depth: int, heads: int) -> ViTConfig:
    return ViTConfig(14, dim, depth, heads, 4 * dim, (37, 37), layerscale=True)


def _dinov3(dim: int, depth: int, heads: int) -> ViTConfig:
    return ViTConfig(16, dim, depth, heads, 4 * dim, (14, 14), registers=4, pos_embed=False, rope_theta=100.0,
                     layerscale=True, eps=1e-5)


def _clip(patch: int, dim: int, depth: int, heads: int) -> ViTConfig:
    return ViTConfig(patch, dim, depth, heads, 4 * dim, (224 // patch,) * 2, ln_pre=True,
                     patch_bias=False, act="quick_gelu", eps=1e-5)


@dataclass(frozen=True)
class Source:
    cfg: ViTConfig
    convert: Callable[[], dict[str, torch.Tensor]]
    mean: tuple[float, ...]
    std: tuple[float, ...]


def _download(url: str, name: str, token_env: str | None = None) -> Path:
    """Cached download into the torch hub folder; ``token_env`` names a bearer-token variable (gated repos)."""
    path = Path(torch.hub.get_dir()) / "checkpoints" / name
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if token_env is None:
        torch.hub.download_url_to_file(url, str(path))
        return path
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"{url} is gated: accept its licence on the model page and set {token_env}")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{url}: HTTP {exc.code}; is the licence accepted for this {token_env}?") from exc
    partial.replace(path)
    return path


def read_safetensors(path: Path, prefix: str = "") -> dict[str, torch.Tensor]:
    """Minimal safetensors reader (8-byte header size, JSON header, raw little-endian data)."""
    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + size])
    out = {}
    for name, meta in header.items():
        if name == "__metadata__" or not name.startswith(prefix):
            continue
        start, end = meta["data_offsets"]
        buf = bytearray(raw[8 + size + start:8 + size + end])
        out[name[len(prefix):]] = torch.frombuffer(buf, dtype=dtypes[meta["dtype"]]).reshape(meta["shape"]).float()
    return out


def _convert_dinov2(name: str) -> Callable[[], dict[str, torch.Tensor]]:
    def convert() -> dict[str, torch.Tensor]:
        sd = torch.load(_download(f"https://dl.fbaipublicfiles.com/dinov2/{name}/{name}_pretrain.pth",
                                  f"{name}_pretrain.pth"), map_location="cpu", weights_only=True)
        out = {"cls_token": sd["cls_token"], "pos_embed": sd["pos_embed"],
               "patch_embed.weight": sd["patch_embed.proj.weight"], "patch_embed.bias": sd["patch_embed.proj.bias"],
               "norm.weight": sd["norm.weight"], "norm.bias": sd["norm.bias"]}
        depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        for i in range(depth):
            b = f"blocks.{i}."
            for src, dst in (("norm1", "norm1"), ("norm2", "norm2"), ("attn.qkv", "attn.qkv"),
                             ("attn.proj", "attn.proj"), ("mlp.fc1", "fc1"), ("mlp.fc2", "fc2")):
                out[f"{b}{dst}.weight"], out[f"{b}{dst}.bias"] = sd[f"{b}{src}.weight"], sd[f"{b}{src}.bias"]
            out[f"{b}ls1"], out[f"{b}ls2"] = sd[f"{b}ls1.gamma"], sd[f"{b}ls2.gamma"]
        return out
    return convert


def convert_dinov3_state(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Hugging Face DINOv3ViTModel keys -> ``src.vit`` keys (the unused key bias becomes zeros)."""
    out = {"cls_token": sd["embeddings.cls_token"], "register_tokens": sd["embeddings.register_tokens"],
           "patch_embed.weight": sd["embeddings.patch_embeddings.weight"],
           "patch_embed.bias": sd["embeddings.patch_embeddings.bias"],
           "norm.weight": sd["norm.weight"], "norm.bias": sd["norm.bias"]}
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layer."))
    for i in range(depth):
        s, b = f"layer.{i}.", f"blocks.{i}."
        q, v = sd[f"{s}attention.q_proj.bias"], sd[f"{s}attention.v_proj.bias"]
        out[f"{b}attn.qkv.weight"] = torch.cat([sd[f"{s}attention.{x}_proj.weight"] for x in "qkv"])
        out[f"{b}attn.qkv.bias"] = torch.cat([q, torch.zeros_like(q), v])
        for src, dst in (("norm1", "norm1"), ("norm2", "norm2"), ("attention.o_proj", "attn.proj"),
                         ("mlp.up_proj", "fc1"), ("mlp.down_proj", "fc2")):
            out[f"{b}{dst}.weight"], out[f"{b}{dst}.bias"] = sd[f"{s}{src}.weight"], sd[f"{s}{src}.bias"]
        out[f"{b}ls1"], out[f"{b}ls2"] = sd[f"{s}layer_scale1.lambda1"], sd[f"{s}layer_scale2.lambda1"]
    return out


def _convert_dinov3(repo: str) -> Callable[[], dict[str, torch.Tensor]]:
    def convert() -> dict[str, torch.Tensor]:
        path = _download(_HF_URL.format(repo, "model.safetensors"), f"{repo.replace('/', '--')}.safetensors",
                         token_env="HF_TOKEN")
        return convert_dinov3_state(read_safetensors(path))
    return convert


def clip_state(name: str, digest: str) -> dict[str, torch.Tensor]:
    """All tensors of an OpenAI CLIP archive (visual and text towers)."""
    path = _download(_CLIP_URL.format(digest, name), f"clip-{name}.pt")
    return torch.jit.load(str(path), map_location="cpu").state_dict()


def _convert_clip(name: str, digest: str) -> Callable[[], dict[str, torch.Tensor]]:
    def convert() -> dict[str, torch.Tensor]:
        sd = {k[len("visual."):]: v.float() for k, v in clip_state(name, digest).items()
              if k.startswith("visual.")}
        out = {"cls_token": sd["class_embedding"].view(1, 1, -1), "pos_embed": sd["positional_embedding"][None],
               "patch_embed.weight": sd["conv1.weight"],
               "ln_pre.weight": sd["ln_pre.weight"], "ln_pre.bias": sd["ln_pre.bias"],
               "norm.weight": sd["ln_post.weight"], "norm.bias": sd["ln_post.bias"]}
        depth = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("transformer.resblocks."))
        for i in range(depth):
            s, b = f"transformer.resblocks.{i}.", f"blocks.{i}."
            out[f"{b}attn.qkv.weight"], out[f"{b}attn.qkv.bias"] = sd[f"{s}attn.in_proj_weight"], sd[f"{s}attn.in_proj_bias"]
            for src, dst in (("ln_1", "norm1"), ("ln_2", "norm2"), ("attn.out_proj", "attn.proj"),
                             ("mlp.c_fc", "fc1"), ("mlp.c_proj", "fc2")):
                out[f"{b}{dst}.weight"], out[f"{b}{dst}.bias"] = sd[f"{s}{src}.weight"], sd[f"{s}{src}.bias"]
        return out
    return convert


def _convert_siglip(repo: str) -> Callable[[], dict[str, torch.Tensor]]:
    def convert() -> dict[str, torch.Tensor]:
        sd = read_safetensors(_download(_HF_URL.format(repo, "model.safetensors"), f"{repo.replace('/', '--')}.safetensors"),
                              prefix="vision_model.")
        out = {"pos_embed": sd["embeddings.position_embedding.weight"][None],
               "patch_embed.weight": sd["embeddings.patch_embedding.weight"],
               "patch_embed.bias": sd["embeddings.patch_embedding.bias"],
               "norm.weight": sd["post_layernorm.weight"], "norm.bias": sd["post_layernorm.bias"]}
        depth = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers."))
        for i in range(depth):
            s, b = f"encoder.layers.{i}.", f"blocks.{i}."
            for part in ("weight", "bias"):
                out[f"{b}attn.qkv.{part}"] = torch.cat([sd[f"{s}self_attn.{x}_proj.{part}"] for x in "qkv"])
            for src, dst in (("layer_norm1", "norm1"), ("layer_norm2", "norm2"), ("self_attn.out_proj", "attn.proj"),
                             ("mlp.fc1", "fc1"), ("mlp.fc2", "fc2")):
                out[f"{b}{dst}.weight"], out[f"{b}{dst}.bias"] = sd[f"{s}{src}.weight"], sd[f"{s}{src}.bias"]
        return out
    return convert


CLIP_RELEASES = {
    "clip_vitb16": ("ViT-B-16", "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"),
    "clip_vitl14": ("ViT-L-14", "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"),
}

SOURCES: dict[str, Source] = {
    "dinov2_vits14": Source(_dinov2(384, 12, 6), _convert_dinov2("dinov2_vits14"), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "dinov2_vitb14": Source(_dinov2(768, 12, 12), _convert_dinov2("dinov2_vitb14"), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "dinov2_vitl14": Source(_dinov2(1024, 24, 16), _convert_dinov2("dinov2_vitl14"), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    **{f"dinov3_vit{size}16": Source(_dinov3(dim, depth, heads),
                                     _convert_dinov3(f"facebook/dinov3-vit{size}16-pretrain-lvd1689m"),
                                     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
       for size, dim, depth, heads in (("s", 384, 12, 6), ("b", 768, 12, 12), ("l", 1024, 24, 16))},
    "clip_vitb16": Source(_clip(16, 768, 12, 12), _convert_clip(*CLIP_RELEASES["clip_vitb16"]), CLIP_MEAN, CLIP_STD),
    "clip_vitl14": Source(_clip(14, 1024, 24, 16), _convert_clip(*CLIP_RELEASES["clip_vitl14"]), CLIP_MEAN, CLIP_STD),
    "siglip2_vitb16": Source(ViTConfig(16, 768, 12, 12, 3072, (14, 14), cls_token=False, act="gelu_tanh"),
                             _convert_siglip("google/siglip2-base-patch16-224"), SIGLIP_MEAN, SIGLIP_STD),
}
