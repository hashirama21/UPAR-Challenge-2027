"""Load every backbone with its ImageNet weights and measure what the time limit allows.

    python -m src.benchmark                         # all backbones, current device
    python -m src.benchmark --backbones convnext_base swin_t --images 28000

Reports parameters, fp16 checkpoint size (zip budget) and gallery inference
throughput with flip TTA, i.e. the estimated time for ``--images`` test images.
"""
from __future__ import annotations

import argparse
import io
import time

import torch

from .model import BACKBONES, CSARNet, ModelConfig


@torch.no_grad()
def benchmark(name: str, device: torch.device, batch_size: int, iters: int, height: int, width: int) -> dict:
    model = CSARNet(ModelConfig(name, height, width), pretrained=True).eval().to(device)
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=sorted(BACKBONES), choices=sorted(BACKBONES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--images", type=int, default=28_095, help="gallery size for the time estimate (2024 test)")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    print(f"{'backbone':<20}{'params(M)':>10}{'ckpt fp16(MiB)':>16}{'img/s':>9}{'est. gallery(s)':>17}")
    for name in args.backbones:
        r = benchmark(name, device, args.batch_size, args.iters, args.height, args.width)
        print(f"{name:<20}{r['params_M']:>10.1f}{r['ckpt_fp16_MiB']:>16.1f}{r['img_per_s']:>9.1f}"
              f"{args.images / r['img_per_s']:>17.0f}", flush=True)


if __name__ == "__main__":
    main()
