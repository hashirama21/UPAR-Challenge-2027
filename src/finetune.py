"""Fine-tuning controls that preserve pre-trained generalisation.

* ``freeze_backbone``: train only the last ``n`` trunk stages (blocks for ViTs);
* LoRA on the backbone's Linear layers, merged back into plain weights before
  saving, so checkpoints and the submission never need LoRA code;
* ``param_groups``: lower learning rate for the backbone than for the heads.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import CSARNet


def freeze_backbone(model: CSARNet, trainable_stages: int) -> None:
    """Freeze the trunk except its last ``trainable_stages`` stages (-1 keeps everything trainable)."""
    if trainable_stages < 0:
        return
    model.backbone.requires_grad_(False)
    stages = model.backbone.stages
    for stage in stages[len(stages) - trainable_stages:] if trainable_stages else []:
        stage.requires_grad_(True)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.down = nn.Parameter(torch.empty(rank, base.in_features))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.down.T @ self.up.T) * self.scale

    def merged(self) -> nn.Linear:
        with torch.no_grad():
            self.base.weight += (self.up @ self.down) * self.scale
        return self.base


def apply_lora(model: CSARNet, rank: int, alpha: float | None = None) -> int:
    """Freeze the backbone and wrap its Linear layers with LoRA adapters; returns how many."""
    model.backbone.requires_grad_(False)
    targets = [(parent, name) for parent in model.backbone.modules()
               for name, child in parent.named_children() if isinstance(child, nn.Linear)]
    for parent, name in targets:
        setattr(parent, name, LoRALinear(getattr(parent, name), rank, alpha or float(rank)))
    return len(targets)


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every LoRA adapter into its base Linear, in place."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
    return model


def param_groups(model: CSARNet, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
    backbone = {id(p) for p in model.backbone.parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in trainable if id(p) in backbone], "lr": lr * backbone_lr_mult},
        {"params": [p for p in trainable if id(p) not in backbone], "lr": lr},
    ]
    return [dict(g, weight_decay=weight_decay) for g in groups if g["params"]]
