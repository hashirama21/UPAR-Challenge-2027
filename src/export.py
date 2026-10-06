"""Build the Codabench code submission (directory + zip) from a checkpoint.

    python -m src.export export.checkpoint=runs/final/model.pt export.out=submissions/final

Layout (run.py at the zip root, as the ingestion program expects):
    run.py  metadata.yaml  assets/model.pt  src/*.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import DictConfig

from .checkpoint import Bundle
from .config import entrypoint

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


def run(cfg: DictConfig) -> Path:
    e = cfg.export
    if not e.checkpoint:
        raise ValueError("set export.checkpoint")
    archive = export(Path(e.checkpoint), Path(e.out), half=not e.fp32, overwrite=e.overwrite)
    print(f"{archive} ({archive.stat().st_size / 2**20:.1f} MiB)")
    return archive


main = entrypoint(run)

if __name__ == "__main__":
    main()
