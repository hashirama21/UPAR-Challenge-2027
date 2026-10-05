from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Yields ``(image_tensor, index)``; labels stay with the caller, indexed by position.

    ``strict=True`` (training) refuses missing files up front instead of silently
    learning from blank images. ``strict=False`` (submission) replaces unreadable
    images with a blank one so a single corrupt file cannot sink the whole run.
    """

    def __init__(self, paths: Sequence[str | Path], transform: Callable, root: str | Path | None = None,
                 strict: bool = True):
        root = Path(root) if root is not None else None
        self.paths = [root / p if root is not None else Path(p) for p in paths]
        self.transform = transform
        self.strict = strict
        if strict:
            missing = [p for p in self.paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"{len(missing)}/{len(self.paths)} images missing, e.g. {missing[0]} "
                                        "(run download_datasets.py)")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        try:
            with Image.open(self.paths[idx]) as im:
                img = im.convert("RGB")
        except OSError:
            if self.strict:
                raise
            log.warning("unreadable image, using a blank one: %s", self.paths[idx])
            img = Image.new("RGB", (64, 160))
        return self.transform(img), idx
