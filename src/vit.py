"""Pure-torch Vision Transformer covering DINOv2, DINOv3, CLIP and SigLIP 2 vision towers.

The submission container has no timm/open_clip/transformers, so foundation encoders
are re-implemented here and their weights converted once at training time
(``src.pretrained``). Learned position tables are resampled on the fly and DINOv3's
2D rotary embeddings are computed for the actual grid, so pedestrian crops
(e.g. 224x112) and multi-scale TTA work unchanged.

``ViTConfig`` describes published architectures (fixed by the released weights),
not experiment settings.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ViTConfig:
    patch: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    grid: tuple[int, int]          # grid of the stored position embeddings
    cls_token: bool = True
    registers: int = 0             # DINOv3 register tokens (after CLS)
    pos_embed: bool = True         # learned absolute position table
    rope_theta: float = 0.0        # > 0: 2D rotary embeddings on patch tokens (DINOv3)
    layerscale: bool = False
    ln_pre: bool = False
    patch_bias: bool = True
    act: str = "gelu"              # gelu | gelu_tanh | quick_gelu
    eps: float = 1e-6

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def prefix(self) -> int:
        return int(self.cls_token) + self.registers


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def _activation(name: str) -> nn.Module:
    return {"gelu": nn.GELU(), "gelu_tanh": nn.GELU(approximate="tanh"), "quick_gelu": QuickGELU()}[name]


def rope_tables(grid: tuple[int, int], head_dim: int, theta: float,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """DINOv3 2D RoPE: (cos, sin) of shape (H*W, head_dim) from patch centres normalised to [-1, 1]."""
    inv_freq = 1 / theta ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32, device=device)
    h, w = grid
    coords_h = torch.arange(0.5, h, dtype=torch.float32, device=device) / h
    coords_w = torch.arange(0.5, w, dtype=torch.float32, device=device) / w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), -1).flatten(0, 1) * 2 - 1
    angles = (2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]).flatten(1, 2).tile(2)
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, causal: bool = False,
                rope: tuple[torch.Tensor, torch.Tensor] | None = None, prefix: int = 0) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if rope is not None:
            cos, sin = (t.to(q.dtype) for t in rope)
            q = torch.cat([q[..., :prefix, :], q[..., prefix:, :] * cos + _rotate_half(q[..., prefix:, :]) * sin], -2)
            k = torch.cat([k[..., :prefix, :], k[..., prefix:, :] * cos + _rotate_half(k[..., prefix:, :]) * sin], -2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim, eps=cfg.eps)
        self.attn = Attention(cfg.dim, cfg.heads)
        self.norm2 = nn.LayerNorm(cfg.dim, eps=cfg.eps)
        self.fc1 = nn.Linear(cfg.dim, cfg.mlp_dim)
        self.act = _activation(cfg.act)
        self.fc2 = nn.Linear(cfg.mlp_dim, cfg.dim)
        self.ls1 = nn.Parameter(torch.ones(cfg.dim)) if cfg.layerscale else None
        self.ls2 = nn.Parameter(torch.ones(cfg.dim)) if cfg.layerscale else None

    def forward(self, x: torch.Tensor, causal: bool = False,
                rope: tuple[torch.Tensor, torch.Tensor] | None = None, prefix: int = 0) -> torch.Tensor:
        a = self.attn(self.norm1(x), causal, rope, prefix)
        x = x + (a * self.ls1 if self.ls1 is not None else a)
        m = self.fc2(self.act(self.fc1(self.norm2(x))))
        return x + (m * self.ls2 if self.ls2 is not None else m)


class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(3, cfg.dim, cfg.patch, cfg.patch, bias=cfg.patch_bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim)) if cfg.cls_token else None
        self.register_tokens = nn.Parameter(torch.zeros(1, cfg.registers, cfg.dim)) if cfg.registers else None
        self.pos_embed = None
        if cfg.pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.grid[0] * cfg.grid[1] + int(cfg.cls_token), cfg.dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.ln_pre = nn.LayerNorm(cfg.dim, eps=cfg.eps) if cfg.ln_pre else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.dim, eps=cfg.eps)

    def _pos(self, grid: tuple[int, int]) -> torch.Tensor:
        return interpolate_pos(self.pos_embed, int(self.cfg.cls_token), self.cfg.grid, grid)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(patch tokens (B, N, D), CLS token (B, D) or None), after the final LayerNorm."""
        x = self.patch_embed(x)
        grid = tuple(x.shape[-2:])
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1)
        if self.pos_embed is not None:
            x = x + self._pos(grid)
        if self.register_tokens is not None:
            x = torch.cat([x[:, :1], self.register_tokens.expand(len(x), -1, -1), x[:, 1:]], 1)
        if self.ln_pre is not None:
            x = self.ln_pre(x)
        rope = (rope_tables(grid, self.cfg.dim // self.cfg.heads, self.cfg.rope_theta, x.device)
                if self.cfg.rope_theta else None)
        for blk in self.blocks:
            x = blk(x, rope=rope, prefix=self.cfg.prefix)
        x = self.norm(x)
        tokens = x[:, self.cfg.prefix:]
        return tokens, (x[:, 0] if self.cls_token is not None else None)


def interpolate_pos(pos_embed: torch.Tensor, n_cls: int, src: tuple[int, int],
                    dst: tuple[int, int]) -> torch.Tensor:
    """Bicubic resampling of a (1, n_cls + H*W, D) position table from grid ``src`` to ``dst``."""
    if tuple(src) == tuple(dst):
        return pos_embed
    cls, patches = pos_embed[:, :n_cls], pos_embed[:, n_cls:]
    patches = patches.reshape(1, *src, -1).permute(0, 3, 1, 2)
    patches = F.interpolate(patches.float(), size=tuple(dst), mode="bicubic", align_corners=False)
    patches = patches.to(pos_embed.dtype).permute(0, 2, 3, 1).flatten(1, 2)
    return torch.cat([cls, patches], 1)
