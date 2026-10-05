"""CSAR network: one visual encoder, two branches.

* attribute branch: independent sigmoid logits (40) + one softmax head per group;
* retrieval branch (optional): image embedding vs. an encoding of the full
  query vector, which captures co-occurrences the per-attribute product ignores.

Only torchvision backbones are registered: they need no vendored code in the
submission container (torch/torchvision/numpy/Pillow only).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .attributes import GROUPS, NUM_ATTRS


Builder = Callable[[bool], tuple[nn.Module, int]]


def _torchvision(ctor: Callable, weights, head: tuple[str | int, ...]) -> Builder:
    """Wrap a torchvision classifier: replace its final Linear (at ``head``) by Identity.

    Earlier head layers (ConvNeXt LayerNorm2d + Flatten, Swin norm + pooling) are kept.
    """
    def build(pretrained: bool) -> tuple[nn.Module, int]:
        net = ctor(weights=weights if pretrained else None)
        *path, last = head
        parent = net
        for step in path:
            parent = parent[step] if isinstance(step, int) else getattr(parent, step)
        linear = parent[last] if isinstance(last, int) else getattr(parent, last)
        if isinstance(last, int):
            parent[last] = nn.Identity()
        else:
            setattr(parent, last, nn.Identity())
        return net, linear.in_features
    return build


BACKBONES: dict[str, Builder] = {
    "resnet18": _torchvision(models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, ("fc",)),
    "resnet50": _torchvision(models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, ("fc",)),
    "convnext_tiny": _torchvision(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                                  ("classifier", 2)),
    "convnext_base": _torchvision(models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
                                  ("classifier", 2)),
    "efficientnet_v2_s": _torchvision(models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
                                      ("classifier", 1)),
    "swin_t": _torchvision(models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1, ("head",)),
    "swin_b": _torchvision(models.swin_b, models.Swin_B_Weights.IMAGENET1K_V1, ("head",)),
}


@dataclass
class ModelConfig:
    backbone: str = "convnext_base"
    height: int = 256
    width: int = 128
    dropout: float = 0.2
    embed_dim: int = 256     # 0 disables the retrieval branch

    def to_dict(self) -> dict:
        return asdict(self)


class CSARNet(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained: bool = False):
        super().__init__()
        if cfg.backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {cfg.backbone!r}, choose from {sorted(BACKBONES)}")
        self.cfg = cfg
        self.backbone, dim = BACKBONES[cfg.backbone](pretrained)
        self.dropout = nn.Dropout(cfg.dropout)
        self.attr_head = nn.Linear(dim, NUM_ATTRS)
        self.group_heads = nn.ModuleDict({g.name: nn.Linear(dim, g.num_classes) for g in GROUPS})
        if cfg.embed_dim:
            self.img_proj = nn.Linear(dim, cfg.embed_dim)
            self.query_encoder = nn.Sequential(
                nn.Linear(NUM_ATTRS, 512), nn.GELU(), nn.Linear(512, cfg.embed_dim))

    @property
    def has_retrieval(self) -> bool:
        return bool(self.cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.dropout(self.backbone(x).flatten(1))
        out = {"attr": self.attr_head(feat)}
        out.update({f"group/{name}": head(feat) for name, head in self.group_heads.items()})
        if self.has_retrieval:
            out["emb"] = F.normalize(self.img_proj(feat), dim=-1)
        return out

    def encode_queries(self, queries: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_encoder(queries.float()), dim=-1)
