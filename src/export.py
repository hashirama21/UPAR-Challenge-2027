"""Build the Codabench code submission (directory + zip) from a checkpoint.

    python -m src.export --checkpoint runs/final/model.pt --out submissions/final

Layout (run.py at the zip root, as the ingestion program expects):
    run.py  metadata.yaml  assets/model.pt  src/*.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .checkpoint import Bundle

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path(__file__).resolve().parent
RUN_TEMPLATE = PACKAGE / "submission_run.py"
METADATA = ROOT / "examples" / "task2" / "sample_code_submission" / "metadata.yaml"


def export(checkpoint: Path, out: Path, half: bool = True, overwrite: bool = False) -> Path:
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists, pass --overwrite to replace it")
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    shutil.copytree(PACKAGE, out / PACKAGE.name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", RUN_TEMPLATE.name))
    shutil.copy2(RUN_TEMPLATE, out / "run.py")
    shutil.copy2(METADATA, out / "metadata.yaml")
    Bundle.load(checkpoint).save(out / "assets" / "model.pt", half=half)
    return Path(shutil.make_archive(str(out), "zip", root_dir=out))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("submissions/csar"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="keep fp32 weights (fp16 halves the zip)")
    args = ap.parse_args(argv)
    archive = export(args.checkpoint, args.out, half=not args.fp32, overwrite=args.overwrite)
    print(f"{archive} ({archive.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
