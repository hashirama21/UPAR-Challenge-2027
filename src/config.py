"""Hydra / OmegaConf configuration of the command-line entry points.

Every value lives in ``configs/`` (YAML). This module only declares the typed
schema (OmegaConf rejects wrong types and unknown keys), composes configurations
and converts sections into the library dataclasses, which double as the schema
of their section so each field is declared once. Only the CLIs, the notebook and
the tests import it: the submission reads the frozen checkpoint and needs neither
Hydra nor OmegaConf.

    python -m src.train model.backbone=clip_vitb16 model.head=query optim.epochs=10
    python -m src.train -m model.backbone=resnet50,convnext_base          # Hydra sweep
    python -m src.experiments profile=smoke experiment=smoke
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import hydra
import torch
from hydra import compose as hydra_compose
from hydra import initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .checkpoint import Runtime
from .losses import LossConfig
from .model import ModelConfig
from .scoring import ScoreConfig
from .transforms import AugmentConfig, input_size

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")
T = TypeVar("T")


@dataclass
class DataCfg:
    dir: str
    holdout: str | None
    max_images: int
    eval_max_queries: int
    domain_balance: bool
    query_alpha: float
    reliability: float


@dataclass
class ModelCfg(ModelConfig):
    pretrained: bool
    text_init: bool
    lora_rank: int
    lora_alpha: float | None
    trainable_stages: int


@dataclass
class OptimCfg:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    scheduler: str
    warmup_epochs: float
    plateau_patience: int
    plateau_factor: float
    ema_decay: float
    backbone_lr_mult: float


@dataclass
class TrainCfg:
    out: str
    select_score: str
    calibration: str


@dataclass
class EvalCfg:
    checkpoint: str | None
    simulate: float | None
    simulate_signal: float
    holdout: str | None
    calib_frac: float
    calibration: str
    scores: list[str]
    lams: list[float]
    gammas: list[float]
    tune_group_weights: bool
    group_weight_grid: list[float]
    open_set: float
    resample: int
    max_queries: int
    by_domain: bool
    compare_uncalibrated: bool
    cache: str | None
    save_best: bool


@dataclass
class LodoCfg:
    root: str
    domains: list[str]
    final: str | None


@dataclass
class ExperimentCfg:
    root: str
    candidates: dict[str, list[str]]
    lodo: str
    ensemble_top_k: int
    wise_alpha: float | None
    export: str | None


@dataclass
class ExportCfg:
    checkpoint: str | None
    out: str
    fp32: bool
    overwrite: bool


@dataclass
class CombineCfg:
    mode: str
    checkpoints: list[str]
    alpha: float
    out: str | None


@dataclass
class BenchmarkCfg:
    backbones: list[str]
    batch_size: int
    iters: int
    images: int


@dataclass
class RunCfg:
    device: str
    workers: int
    seed: int
    inference_batch: int


@dataclass
class Config:
    data: DataCfg
    augment: AugmentConfig
    model: ModelCfg
    optim: OptimCfg
    loss: LossConfig
    train: TrainCfg
    score: ScoreConfig
    eval: EvalCfg
    runtime: Runtime
    lodo: LodoCfg
    experiment: ExperimentCfg
    export: ExportCfg
    combine: CombineCfg
    benchmark: BenchmarkCfg
    run: RunCfg


ConfigStore.instance().store(name="schema", node=Config)


def compose(overrides: Sequence[str] = ()) -> DictConfig:
    """Compose ``configs/config.yaml`` with Hydra overrides, outside a Hydra app (notebook, tests)."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return hydra_compose(config_name="config", overrides=list(overrides))


def with_overrides(cfg: DictConfig, overrides: Sequence[str]) -> DictConfig:
    """Copy of ``cfg`` with dotlist overrides (``key.sub=value``), validated against the schema."""
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))


def with_values(cfg: DictConfig, values: dict[str, object]) -> DictConfig:
    """Copy of ``cfg`` with ``{"key.sub": value}`` set as Python values (paths, None), schema-validated."""
    out = OmegaConf.merge(cfg, {})
    for key, value in values.items():
        OmegaConf.update(out, key, str(value) if isinstance(value, Path) else value, merge=False)
    return out


def section(node: DictConfig, cls: type[T]) -> T:
    """Library dataclass built from the matching fields of a config section."""
    data = OmegaConf.to_container(node, resolve=True)
    return cls(**{f.name: data[f.name] for f in fields(cls)})


def model_config(cfg: DictConfig) -> ModelConfig:
    """``model`` section with height/width rounded to the backbone's patch multiple."""
    mcfg = section(cfg.model, ModelConfig)
    height, width = input_size(mcfg)
    return replace(mcfg, height=height, width=width)


def loss_config(cfg: DictConfig) -> LossConfig:
    return section(cfg.loss, LossConfig)


def score_config(cfg: DictConfig) -> ScoreConfig:
    return section(cfg.score, ScoreConfig)


def runtime_config(cfg: DictConfig) -> Runtime:
    return section(cfg.runtime, Runtime)


def augment_config(cfg: DictConfig) -> AugmentConfig:
    return section(cfg.augment, AugmentConfig)


def device(cfg: DictConfig) -> str:
    if cfg.run.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.run.device


def to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


def entrypoint(run: Callable[[DictConfig], object]) -> Callable[[], None]:
    """Hydra CLI around ``run(cfg)``: ``python -m src.<module> key=value ...``."""
    @hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
    def main(cfg: DictConfig) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
        run(cfg)
    return main
