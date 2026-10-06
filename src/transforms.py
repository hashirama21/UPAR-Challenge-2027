"""Image preprocessing shared by training and the submission.

Letterboxing keeps the pedestrian aspect ratio. Training augmentations are
geometric only (flip, crop, small affine) plus a random low-resolution
round-trip that mimics far or compressed footage: no colour, contrast or
solarisation ops (colour attributes must survive) and no random erasing, both
of which hurt cross-domain retrieval in UPAR 2023/2024.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torchvision.transforms as T
from PIL import Image

from .model import ModelConfig


@dataclass
class AugmentConfig:
    """Training augmentation; values come from configs/config.yaml (``augment``)."""
    low_res: float              # probability of the low-resolution round-trip
    low_res_min_scale: float
    crop_scale_min: float
    rotate_deg: float
    translate: float
    scale_jitter: float


class Letterbox:
    """Resize to fit inside (height, width) keeping the ratio, pad with ``fill``."""

    def __init__(self, height: int, width: int, fill: tuple[int, int, int]):
        self.size, self.fill = (width, height), fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = self.size
        scale = min(w / img.width, h / img.height)
        nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
        canvas = Image.new("RGB", self.size, self.fill)
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
        return canvas


class RandomLowResolution:
    """With probability ``p``, downscale by a factor in [min_scale, 1] and upscale back."""

    def __init__(self, p: float, min_scale: float):
        self.p, self.min_scale = p, min_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(()) >= self.p:
            return img
        s = self.min_scale + (1 - self.min_scale) * float(torch.rand(()))
        small = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)
        return small.resize(img.size, Image.BILINEAR)


def input_size(cfg: ModelConfig, scale: float = 1.0) -> tuple[int, int]:
    """(height, width) at ``scale``, rounded to the backbone's patch multiple."""
    m = cfg.spec.patch
    return max(m, round(cfg.height * scale / m) * m), max(m, round(cfg.width * scale / m) * m)


def build_transform(cfg: ModelConfig, augment: AugmentConfig | None = None, scale: float = 1.0) -> T.Compose:
    """Evaluation transform, or training transform when ``augment`` is given."""
    height, width = input_size(cfg, scale)
    spec = cfg.spec
    fill = tuple(round(255 * m) for m in spec.mean)
    steps: list = []
    if augment is not None and augment.low_res:
        steps.append(RandomLowResolution(augment.low_res, augment.low_res_min_scale))
    steps.append(Letterbox(height, width, fill))
    if augment is not None:
        a = augment
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((height, width), scale=(a.crop_scale_min, 1.0),
                                ratio=(width / height * 0.9, width / height * 1.1)),
            T.RandomAffine(degrees=a.rotate_deg, translate=(a.translate, a.translate),
                           scale=(1 - a.scale_jitter, 1 + a.scale_jitter), fill=fill),
        ]
    steps += [T.ToTensor(), T.Normalize(spec.mean, spec.std)]
    return T.Compose(steps)
