"""Codabench container check, run by .github/workflows/codabench-container.yml.

    build   synthetic data + smoke training + export, with the training dependencies
    ingest  import the exported run.py and call rank_gallery like the ingestion program;
            meant for the official image with --network none and only its own packages

    python scripts/container_check.py build --work ci
    python scripts/container_check.py ingest --work ci
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(work: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from src import export, train
    from src.config import compose, with_values
    from src.synthetic import make_synthetic_data

    data = make_synthetic_data(ROOT / "data", work / "data")
    cfg = with_values(compose(["profile=smoke", "model.backbone=resnet18"]),
                      {"data.dir": data, "train.out": work / "model"})
    archive = export.export(train.run(cfg), work / "submission", overwrite=True)
    print(f"built {archive}")


def ingest(work: Path) -> None:
    submission = work / "submission"
    spec = importlib.util.spec_from_file_location("run", submission / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    from src.attributes import ATTRIBUTE_NAMES
    from src.data import load_split
    from src.metrics import evaluate_output

    gallery = load_split(work / "data", "val")
    sample = {"attribute_names": list(ATTRIBUTE_NAMES), "queries": gallery.queries.tolist(),
              "gallery": [{"image_path": str(work / "data" / p)} for p in gallery.images]}
    t0 = time.time()
    out = run.rank_gallery(sample)
    elapsed = time.time() - t0
    (key, matrix), = out.items()
    assert key == "similarities" and matrix.shape == (len(gallery.queries), len(gallery))
    assert all(len(set(row)) == len(row) for row in matrix.tolist()), "tied scores"
    metrics = evaluate_output(out, gallery.queries, gallery.labels)
    print(json.dumps({"seconds": round(elapsed, 2), "queries": len(gallery.queries), "images": len(gallery),
                      "metrics": metrics}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("build", "ingest"))
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    (build if args.step == "build" else ingest)(args.work.resolve())


if __name__ == "__main__":
    main()
