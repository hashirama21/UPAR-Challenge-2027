# CSAR — Calibrated Structured Attribute Retrieval (UPAR 2027, Track 2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/UPAR-Challenge-2027/blob/main/notebooks/pipeline.ipynb)

One package, `src/`, serves training, evaluation and the submission. The exported `run.py`
imports the same code (shipped in the zip), and the checkpoint holds the architecture, weights,
frozen calibration and score configuration.

## Modules

| Module | Role |
| --- | --- |
| `attributes.py` | 40 attributes, groups ("none" class for hair, lower-body type, glasses), group encoding, column alignment |
| `data.py` | Validated loading, "one distinct vector = one query" protocol, LODO folds, query masks, domain-balanced sampling, Hamming neighbours |
| `transforms.py` | Letterbox (keeps aspect ratio), flip/crop/AugMix without colour ops, no random erasing |
| `model.py` | torchvision backbone + 40 sigmoid logits + per-group softmax heads + query-image retrieval branch |
| `losses.py` | Weighted BCE with label smoothing, per-group CE, image→query contrastive loss with Hamming 1-2 negatives and margin |
| `calibration.py` | Per-attribute and per-group temperature, fitted off-test then frozen |
| `scoring.py` | `l1`, `loglik`, `structured`, `endom` (truncated Poisson-binomial), `mix`, optional transductive posterior; blockwise, continuous tie-breaker |
| `metrics.py` | mADM, mAP, R1/5/10, mINP (blockwise re-implementation) |
| `inference.py` | Batched DataLoader, fp16 on GPU, flip TTA, throughput logs |
| `checkpoint.py` | Self-describing bundle, strict loading |
| `train.py`, `evaluate.py`, `export.py` | Training, evaluation/ablation and packaging CLIs |
| `benchmark.py` | Loads each backbone with ImageNet weights, measures throughput and checkpoint size |
| `synthetic.py` | Small UPAR-formatted set (real labels, random images) for end-to-end tests |
| `submission_run.py` | Codabench template (`load_model`, `rank_gallery`), copied to `run.py` by `export.py` |

## Notebook

`notebooks/pipeline.ipynb` runs the whole chain and opens directly in Colab with the badge above
(it clones the repository when opened alone). `MODE = "smoke"` takes about a minute on CPU with
synthetic data; `MODE = "full"` downloads and uses the real images.

## Workflow

```bash
python -m src.evaluate --simulate 3.0 --scores l1 loglik structured mix --lams 1 30 --max-queries 600

python -m src.train --holdout PETA --out runs/lodo_peta
python -m src.evaluate --checkpoint runs/lodo_peta/model.pt --holdout PETA --calib-frac 0.5 \
    --scores loglik structured mix --lams 1 30 --gammas 0 0.5 1 --cache runs/lodo_peta/val_preds.pt

python -m src.train --out runs/final
python -m src.evaluate --checkpoint runs/final/model.pt --scores loglik structured --gammas 0 0.5 1 --save-best
python -m src.export --checkpoint runs/final/model.pt --out submissions/final

python -m pytest
```

Open-set experiment: `--open-set 0.2`. Test-sized gallery (2024): `--resample 367`.

## Backbones

All come from torchvision, so nothing has to be vendored into the container. Each is tested end to
end (forward, training step with the real loss, fp16 checkpoint, reload). CPU throughput from
`python -m src.benchmark` (flip TTA, batch 16):

| Backbone | Params (M) | fp16 checkpoint (MiB) | img/s CPU | 28,000-image gallery (s) |
| --- | --- | --- | --- | --- |
| resnet18 | 11.5 | 22 | 45.7 | 615 |
| efficientnet_v2_s | 20.8 | 40 | 22.7 | 1,240 |
| convnext_tiny | 28.2 | 54 | 18.3 | 1,533 |
| resnet50 | 24.3 | 47 | 16.0 | 1,755 |
| swin_t | 27.9 | 53 | 8.0 | 3,512 |
| convnext_base | 88.1 | 168 | 5.4 | 5,198 |
| swin_b | 87.2 | 167 | 2.6 | 10,669 |

None fits a 600 s limit on CPU: the choice depends on the Codabench hardware, to be measured with
the first submission. New backbones are one entry in `src/model.py: BACKBONES`.

## Score study (official val, 600 queries, Gaussian logit noise σ)

| Score | mADM σ=3 | mAP σ=3 | mADM σ=2 | mAP σ=2 |
| --- | --- | --- | --- | --- |
| `l1` (official example) | 0.490 | 0.161 | 0.798 | 0.491 |
| `loglik` | 0.582 | 0.252 | 0.897 | 0.696 |
| **`structured` (default)** | **0.680** | **0.371** | **0.937** | **0.795** |
| `mix` λ=1 / λ=30 | 0.532 / 0.617 | 0.208 / 0.316 | 0.860 / 0.928 | 0.628 / 0.779 |
| transductive `loglik` (open-set 0 % / 20 %) | 0.762 / 0.758 | 0.534 / 0.521 | — | — |

- `structured` uses the same logits as `loglik`; the gain comes only from the one-value-per-group constraint.
- E[ndom] never improves on its exact term here, so `mix` is not the default. Re-measure on LODO with a real model, whose errors are correlated.

## Rule safeguards

- By default the test gallery is only used for per-image inference. The mean degree of match per query
  comes from the training prior (0.811, vs 0.81 measured on the val gallery).
- `ScoreConfig.transductive` (posterior over the query set, with a background class) is off; enable it
  only with written approval from the organisers and declare it in the fact sheet.
- No EM on priors and no Sinkhorn assignment.

## Not covered yet

- Foundation encoders (SigLIP 2, DINOv3): their code must be vendored (timm is not in the container).
- Model ensembles and WiSE-FT.
- Timed run in the official Docker image with `--network none`.
