"""CSAR network: one visual trunk, attribute heads and an optional retrieval branch.

Every backbone is wrapped as a ``Trunk`` returning patch tokens (B, N, C) and a
pooled vector (B, P), plus its ordered ``stages`` for partial fine-tuning.

Heads (``ModelConfig.head``):
    linear  sigmoid logits (40) + one softmax per group, on the pooled vector;
    query   one learned query per attribute (+ one "none" query per group allowing
            it) cross-attending to the tokens (VTB/PromptPAR style), optionally
            initialised from CLIP text embeddings of the attribute names.

Only torch/torchvision code: foundation ViTs are re-implemented in ``src.vit``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .attributes import GROUPS, NUM_ATTRS
from .pretrained import SOURCES
from .vit import VisionTransformer

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NONE_GROUPS = tuple(k for k, g in enumerate(GROUPS) if g.allow_none)


class Trunk(nn.Module):
    """``forward(x) -> (tokens (B, N, token_dim), pooled (B, pooled_dim))``; ``stages`` input to output."""

    token_dim: int
    pooled_dim: int
    stages: list[nn.Module]


class FeatureMapTrunk(Trunk):
    def __init__(self, body: nn.Module, stages: list[nn.Module], dim: int, channels_last: bool = False,
                 post: nn.Module | None = None):
        super().__init__()
        self.body, self.post = body, post
        self.stages, self.token_dim, self.pooled_dim = stages, dim, dim
        self.channels_last = channels_last

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.body(x)
        tokens = fmap.flatten(1, 2) if self.channels_last else fmap.flatten(2).transpose(1, 2)
        pooled = tokens.mean(1)
        if self.post is not None:
            pooled = self.post(pooled[:, :, None, None]).flatten(1)
        return tokens, pooled


class ViTTrunk(Trunk):
    """Pooled = [CLS, mean(patch tokens)] when the model has a CLS token, else the mean."""

    def __init__(self, vit: VisionTransformer):
        super().__init__()
        self.vit = vit
        self.stages, self.token_dim = list(vit.blocks), vit.cfg.dim
        self.pooled_dim = vit.cfg.dim * (2 if vit.cfg.cls_token else 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, cls = self.vit(x)
        pooled = tokens.mean(1) if cls is None else torch.cat([cls, tokens.mean(1)], -1)
        return tokens, pooled


def _resnet(ctor, weights) -> Callable[[bool], Trunk]:
    def build(pretrained: bool) -> Trunk:
        net = ctor(weights=weights if pretrained else None)
        stages = [net.layer1, net.layer2, net.layer3, net.layer4]
        body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, *stages)
        return FeatureMapTrunk(body, stages, net.fc.in_features)
    return build


def _convnext(ctor, weights) -> Callable[[bool], Trunk]:
    def build(pretrained: bool) -> Trunk:
        net = ctor(weights=weights if pretrained else None)
        return FeatureMapTrunk(net.features, list(net.features), net.classifier[2].in_features,
                               post=net.classifier[0]) 
    return build


def _efficientnet(ctor, weights) -> Callable[[bool], Trunk]:
    def build(pretrained: bool) -> Trunk:
        net = ctor(weights=weights if pretrained else None)
        return FeatureMapTrunk(net.features, list(net.features), net.classifier[1].in_features)
    return build


def _swin(ctor, weights) -> Callable[[bool], Trunk]:
    def build(pretrained: bool) -> Trunk:
        net = ctor(weights=weights if pretrained else None)
        return FeatureMapTrunk(nn.Sequential(net.features, net.norm), list(net.features), net.head.in_features,
                               channels_last=True)
    return build


def _vit(name: str) -> Callable[[bool], Trunk]:
    def build(pretrained: bool) -> Trunk:
        vit = VisionTransformer(SOURCES[name].cfg)
        if pretrained:
            vit.load_state_dict(SOURCES[name].convert())
        return ViTTrunk(vit)
    return build


@dataclass(frozen=True)
class BackboneSpec:
    build: Callable[[bool], Trunk]
    mean: tuple[float, ...] = IMAGENET_MEAN
    std: tuple[float, ...] = IMAGENET_STD
    patch: int = 1  


BACKBONES: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(_resnet(models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1)),
    "resnet50": BackboneSpec(_resnet(models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2)),
    "convnext_tiny": BackboneSpec(_convnext(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)),
    "convnext_base": BackboneSpec(_convnext(models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1)),
    "efficientnet_v2_s": BackboneSpec(_efficientnet(models.efficientnet_v2_s,
                                                    models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)),
    "swin_t": BackboneSpec(_swin(models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1)),
    "swin_b": BackboneSpec(_swin(models.swin_b, models.Swin_B_Weights.IMAGENET1K_V1)),
    **{name: BackboneSpec(_vit(name), src.mean, src.std, src.cfg.patch) for name, src in SOURCES.items()},
}


@dataclass
class ModelConfig:
    """Architecture, stored in every checkpoint; values come from configs/config.yaml (``model``)."""
    backbone: str
    height: int
    width: int
    dropout: float
    embed_dim: int           # 0 disables the retrieval branch
    head: str                # linear | query
    query_layers: int
    query_heads: int
    query_hidden: int        # hidden width of the query-vector encoder

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def spec(self) -> BackboneSpec:
        if self.backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {self.backbone!r}, choose from {sorted(BACKBONES)}")
        return BACKBONES[self.backbone]


class LinearHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attr = nn.Linear(dim, NUM_ATTRS)
        self.groups = nn.ModuleDict({g.name: nn.Linear(dim, g.num_classes) for g in GROUPS})

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {"attr": self.attr(pooled)}
        out.update({f"group/{n}": head(pooled) for n, head in self.groups.items()})
        return out


class QueryHead(nn.Module):
    """Attribute queries cross-attending to the trunk tokens; one scoring vector per query."""

    def __init__(self, dim: int, layers: int, heads: int):
        super().__init__()
        n = NUM_ATTRS + len(NONE_GROUPS)
        self.queries = nn.Parameter(torch.randn(n, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        layer = nn.TransformerDecoderLayer(dim, heads, 2 * dim, dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.attr_w, self.attr_b = nn.Parameter(torch.zeros(NUM_ATTRS, dim)), nn.Parameter(torch.zeros(NUM_ATTRS))
        self.group_w, self.group_b = nn.Parameter(torch.zeros(n, dim)), nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.attr_w, std=0.02)
        nn.init.trunc_normal_(self.group_w, std=0.02)
        none_idx = {k: NUM_ATTRS + i for i, k in enumerate(NONE_GROUPS)}
        self.class_index = [list(g.indices) + ([none_idx[k]] if g.allow_none else []) for k, g in enumerate(GROUPS)]

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.decoder(self.queries.expand(len(tokens), -1, -1), self.norm(tokens))
        out = {"attr": torch.einsum("bqd,qd->bq", h[:, :NUM_ATTRS], self.attr_w) + self.attr_b}
        group_scores = torch.einsum("bqd,qd->bq", h, self.group_w) + self.group_b
        out.update({f"group/{g.name}": group_scores[:, idx] for g, idx in zip(GROUPS, self.class_index)})
        return out


class CSARNet(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained: bool = False):
        super().__init__()
        spec = cfg.spec
        if cfg.height % spec.patch or cfg.width % spec.patch:
            raise ValueError(f"{cfg.backbone} needs height/width multiples of {spec.patch}")
        self.cfg = cfg
        self.backbone = spec.build(pretrained)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.head == "linear":
            self.head = LinearHead(self.backbone.pooled_dim)
        elif cfg.head == "query":
            self.head = QueryHead(self.backbone.token_dim, cfg.query_layers, cfg.query_heads)
        else:
            raise ValueError(f"unknown head {cfg.head!r}")
        if cfg.embed_dim:
            self.img_proj = nn.Linear(self.backbone.pooled_dim, cfg.embed_dim)
            self.query_encoder = nn.Sequential(
                nn.Linear(NUM_ATTRS, cfg.query_hidden), nn.GELU(), nn.Linear(cfg.query_hidden, cfg.embed_dim))

    @property
    def has_retrieval(self) -> bool:
        return bool(self.cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, pooled = self.backbone(x)
        out = self.head(self.dropout(tokens), self.dropout(pooled))
        if self.has_retrieval:
            out["emb"] = F.normalize(self.img_proj(self.dropout(pooled)), dim=-1)
        return out

    def encode_queries(self, queries: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_encoder(queries.float()), dim=-1)
