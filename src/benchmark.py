"""Load backbones with their pre-trained weights and measure what the time limit allows.

    python -m src.benchmark                                        # all backbones
    python -m src.benchmark "benchmark.backbones=[convnext_base,clip_vitb16]" benchmark.images=28000

Reports parameters, fp16 checkpoint size (zip budget) and gallery inference
throughput with flip TTA at the ``model`` input size, i.e. the estimated time
for ``benchmark.images`` test images.
"""
from __future__ import annotations

import io
import time
from dataclasses import replace

import torch
from omegaconf import DictConfig

from .config import device as resolve_device, entrypoint, model_config
from .model import BACKBONES, CSARNet, ModelConfig
from .transforms import input_size


@torch.no_grad()
def benchmark(cfg: ModelConfig, device: torch.device, batch_size: int, iters: int) -> dict:
    height, width = input_size(cfg)
    model = CSARNet(replace(cfg, height=height, width=width), pretrained=True).eval().to(device)
    buf = io.BytesIO()
    torch.save({k: v.half() for k, v in model.state_dict().items()}, buf)
    x = torch.randn(batch_size, 3, height, width, device=device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(x)
            model(x.flip(-1))
        if device.type == "cuda":
            torch.cuda.synchronize()
    return {"params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "ckpt_fp16_MiB": buf.tell() / 2**20,
            "img_per_s": batch_size * iters / (time.time() - t0)}


def run(cfg: DictConfig) -> dict[str, dict]:
    b = cfg.benchmark
    device = torch.device(resolve_device(cfg))
    results = {}
    print(f"{'backbone':<20}{'params(M)':>10}{'ckpt fp16(MiB)':>16}{'img/s':>9}{'est. gallery(s)':>17}")
    for name in list(b.backbones) or sorted(BACKBONES):
        r = benchmark(replace(model_config(cfg), backbone=name), device, b.batch_size, b.iters)
        results[name] = r
        print(f"{name:<20}{r['params_M']:>10.1f}{r['ckpt_fp16_MiB']:>16.1f}{r['img_per_s']:>9.1f}"
              f"{b.images / r['img_per_s']:>17.0f}", flush=True)
    return results


main = entrypoint(run)

if __name__ == "__main__":
    main()
