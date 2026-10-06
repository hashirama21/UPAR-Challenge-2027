from __future__ import annotations

import pytest
from hydra.errors import ConfigCompositionException

from src.config import compose, model_config, with_overrides, with_values
from src.model import BACKBONES


def test_default_backbone_is_dinov3_with_token_else_dinov2(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))      # empty weight cache
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert compose().model.backbone == "dinov2_vitb14"
    assert (model_config(compose()).height, model_config(compose()).width) == (252, 126)
    monkeypatch.setenv("HF_TOKEN", "token")
    assert compose().model.backbone == "dinov3_vitb16"
    assert compose(["model.backbone=resnet50"]).model.backbone == "resnet50"


@pytest.mark.parametrize("override", ["optim.epochs=abc", "model.backbon=resnet18", "eval.scores=loglik"])
def test_schema_rejects_bad_overrides(override):
    with pytest.raises(ConfigCompositionException):
        compose([override])


@pytest.mark.parametrize("experiment", ["full", "smoke"])
def test_experiment_candidates_compose(experiment):
    cfg = compose([f"profile={experiment}", f"experiment={experiment}"])
    for name, overrides in cfg.experiment.candidates.items():
        candidate = with_overrides(cfg, list(overrides))
        assert candidate.model.backbone in BACKBONES, name
        model_config(candidate)                       # patch rounding and schema conversion


def test_with_values_keeps_types_and_paths(tmp_path):
    cfg = with_values(compose(), {"data.dir": tmp_path, "optim.epochs": 3, "data.holdout": None})
    assert cfg.data.dir == str(tmp_path) and cfg.optim.epochs == 3 and cfg.data.holdout is None
