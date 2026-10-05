"""Image preprocessing shared by training and the submission.

Letterboxing keeps the pedestrian aspect ratio (crops vary widely across
domains). Training augmentations deliberately avoid hue shifts and random
erasing: both hurt cross-domain retrieval in UPAR 2023/2024.
"""
from __future__ import annotations

from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_PAD = tuple(round(255 * m) for m in IMAGENET_MEAN)


class Letterbox:
    """Resize to fit inside (height, width) keeping the ratio, pad with the mean colour."""

    def __init__(self, height: int, width: int):
        self.size = (width, height)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = self.size
        scale = min(w / img.width, h / img.height)
        nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
        canvas = Image.new("RGB", self.size, _PAD)
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
        return canvas


def build_transform(height: int, width: int, train: bool, augmix: bool = True) -> T.Compose:
    steps: list = [Letterbox(height, width)]
    if train:
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((height, width), scale=(0.8, 1.0), ratio=(width / height * 0.9, width / height * 1.1)),
        ]
        if augmix:
            steps.append(T.AugMix(all_ops=False))  # all_ops=False: no colour/brightness/contrast ops
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)
