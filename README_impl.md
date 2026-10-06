# CSAR — Calibrated Structured Attribute Retrieval (UPAR 2027, Track 2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/UPAR-Challenge-2027/blob/main/notebooks/pipeline.ipynb)
[![codabench-container](https://github.com/hashirama21/UPAR-Challenge-2027/actions/workflows/codabench-container.yml/badge.svg)](https://github.com/hashirama21/UPAR-Challenge-2027/actions/workflows/codabench-container.yml)

One package, `src/`, serves training, evaluation and the submission. The exported `run.py`
imports the same code (shipped in the zip), and the checkpoint holds the members (one model or an
ensemble), frozen calibration, score configuration and runtime (TTA, time budget).

## Configuration (Hydra / OmegaConf)

Every value lives in `configs/`; nothing is hard-coded in the entry points.

| File | Content |
| --- | --- |
| `configs/config.yaml` | All settings: data, augmentation, model, optimisation, losses, score, evaluation, runtime, LODO, export, benchmark |
| `configs/profile/{full,smoke}.yaml` | Scale of a run (`smoke`: CPU, tiny images, one epoch) |
| `configs/experiment/{full,smoke}.yaml` | Fine-tuning candidates compared by `src.experiments` (dotlist overrides) |

`src/config.py` declares the typed schema (the library dataclasses `ModelConfig`, `LossConfig`,
`ScoreConfig`, `Runtime`, `AugmentConfig` double as their section's schema), so OmegaConf rejects
wrong types and unknown keys. Hydra and OmegaConf are only imported by the entry points: the
submission reads the frozen checkpoint and needs neither.

```bash
python -m src.train model.backbone=clip_vitb16 model.head=query model.text_init=true model.lora_rank=16
python -m src.train -m model.backbone=resnet50,convnext_base           # Hydra sweep
python -m src.experiments profile=smoke experiment=smoke                 # CPU smoke run of everything
python -m src.experiments                                                # full candidates, LODO, export
```

## Modules

| Module | Role |
| --- | --- |
| `attributes.py` | 40 attributes, groups ("none" class for hair, lower-body type, glasses), group encoding, column alignment |
| `data.py` | Validated loading, "one distinct vector = one query" protocol, LODO folds, query masks (uniform or by size), novel queries, domain-balanced sampling, Market1501 reliability weights, Hamming neighbours |
| `transforms.py` | Letterbox, geometric augmentation only (flip, crop, small affine), random low-resolution round-trip; no colour ops, no random erasing |
| `vit.py`, `pretrained.py` | Pure-torch ViT (learned or rotary position embeddings, register tokens) and one-off weight conversion for DINOv2, DINOv3, CLIP and SigLIP 2 |
| `model.py` | Trunks (tokens + pooled) for torchvision CNNs/Swin and ViTs; `linear` head or `query` head (attribute queries cross-attending to tokens); query-image retrieval branch |
| `clip_text.py` | CLIP tokenizer and text tower to initialise the attribute queries from their names (training only) |
| `finetune.py` | LoRA (merged before saving), partial fine-tuning of the last N stages, backbone learning-rate multiplier |
| `losses.py` | Weighted BCE + per-group CE with per-source label smoothing, image→query contrastive loss (Hamming 1-2 negatives, ArcFace or CosFace margin, optional ndom soft targets), per-image weights |
| `calibration.py` | Per-attribute (and per-group) temperature or Platt scaling, fitted off-test, averaged over LODO folds, frozen |
| `scoring.py` | `l1`, `loglik`, `structured` (group weights w_k), `endom` (truncated Poisson-binomial), `mix`, optional transductive posterior; blockwise; lexicographic tie-breaking on the unclipped log-likelihood |
| `metrics.py` | mADM, mAP, R1/5/10, mINP per query (blockwise re-implementation), ECE per attribute |
| `inference.py` | Batched DataLoader, fp16 on GPU, flip and multi-scale TTA, ensembles, degraded mode under a time budget |
| `checkpoint.py` | Self-describing bundle (members, calibration, score, runtime), strict loading |
| `config.py` | Hydra schema, composition (`compose`, `with_overrides`, `with_values`), entry-point decorator |
| `train.py` | Baseline recipe + all of the above; plateau or cosine schedule; unbiased epoch selection on LODO folds |
| `evaluate.py` | Score sweeps, calibration fit, per-domain / seen / novel reporting, ECE, calibrated vs raw, w_k tuning, open-set and test-like galleries; inference only on the images it needs |
| `lodo.py` | Three folds end to end; picks the score config on the mean over folds and writes it, with the averaged calibration, into the final checkpoint |
| `experiments.py` | Trains every candidate, shared yardstick, ensemble of the top-k, WiSE-FT, LODO of the winner, export |
| `combine.py` | Logit ensembles, uniform model soups, WiSE-FT |
| `export.py`, `submission_run.py` | Codabench package (`run.py` + `src/` + `assets/model.pt`) |
| `benchmark.py` | Loads backbones with their pre-trained weights, measures throughput and checkpoint size |
| `synthetic.py` | Small UPAR-formatted set (real labels, random images) for end-to-end tests |

## Notebook

`notebooks/pipeline.ipynb` opens directly in Colab with the badge above (it clones the repository
when opened alone). It selects a profile, fine-tunes every candidate of `configs/experiment/<profile>.yaml`,
shows the leaderboard and the LODO of the winner, evaluates it on a test-like gallery and runs the
exported `run.py` end to end. `PROFILE = "smoke"` runs on CPU with synthetic data.

## Backbones

Torchvision CNNs/Swin (ImageNet) and foundation ViTs re-implemented in pure torch, so the container
needs no timm/open_clip/transformers. Each is tested end to end (forward, training step with the
real loss, fp16 checkpoint, reload). CPU throughput from `python -m src.benchmark` with pre-trained
weights (256x128 input rounded to the patch size; flip TTA; batch 16):

| Backbone | Pre-training | Params (M) | fp16 checkpoint (MiB) | img/s CPU | 28,000-image gallery (s) |
| --- | --- | --- | --- | --- | --- |
| resnet18 | ImageNet | 11.5 | 22 | 45.7 | 615 |
| efficientnet_v2_s | ImageNet | 20.8 | 40 | 22.7 | 1,240 |
| convnext_tiny | ImageNet | 28.2 | 54 | 18.3 | 1,533 |
| resnet50 | ImageNet | 24.3 | 47 | 16.0 | 1,755 |
| dinov2_vits14 | DINOv2 | 22.5 | 43 | 16.3 | 1,720 |
| swin_t | ImageNet | 27.9 | 53 | 8.0 | 3,512 |
| clip_vitb16 | CLIP (OpenAI) | 86.5 | 165 | 6.2 | 4,533 |
| siglip2_vitb16 | SigLIP 2 | 86.2 | 165 | 5.5 | 5,123 |
| convnext_base | ImageNet | 88.1 | 168 | 5.4 | 5,198 |
| dinov2_vitb14 | DINOv2 | 87.2 | 167 | 4.8 | 5,850 |
| swin_b | ImageNet | 87.2 | 167 | 2.6 | 10,669 |
| dinov2_vitl14 | DINOv2 | 305.2 | 582 | 1.4 | 19,984 |
| clip_vitl14 | CLIP (OpenAI) | 304.0 | 580 | 1.3 | 21,376 |
| dinov3_vits16 / vitb16 / vitl16 | DINOv3 (gated) | 22 / 86 / 304 | — | — | — |

None fits a 600 s limit on this CPU: the choice depends on the Codabench hardware, to be measured
with the first submission.

The ViT conversions are checked against the official implementations (`tests/test_pretrained.py`,
skipped when the reference libraries are absent): DINOv2, DINOv3 and SigLIP 2 are bit-exact, CLIP
image and text towers match `open_clip` to 6e-6 and the tokenizer is identical. DINOv3 is checked
with random weights against `transformers.DINOv3ViTModel`, on square and pedestrian-shaped grids; its
released weights need the DINOv3 licence accepted on Hugging Face and `HF_TOKEN` set.

## Codabench container

`.github/workflows/codabench-container.yml` runs on every push in `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`,
the image of the ingestion program: unit tests, a smoke submission build, then the submission itself
with `--network none` and only the packages of the image (`scripts/container_check.py`).

## Leave-one-domain-out on real images

```bash
python -m src.lodo model.backbone=resnet18 data.max_images=6000 optim.epochs=3 lodo.root=runs/lodo_real_resnet18     "eval.scores=[l1,loglik,structured,mix]" "eval.lams=[1.0,30.0]"
```

Each fold trains on two domains, selects the epoch on one query half of the third and reports on the
other half, every score calibrated and raw. A first CPU run (ResNet-18, 6,000 training images per
fold, 3 epochs) is in progress; its per-fold table will be added here.

## Score study (official val, 600 queries, Gaussian logit noise σ)

| Score | mADM σ=3 | mAP σ=3 | mADM σ=2 | mAP σ=2 |
| --- | --- | --- | --- | --- |
| `l1` (official example) | 0.490 | 0.161 | 0.798 | 0.491 |
| `loglik` | 0.582 | 0.252 | 0.897 | 0.696 |
| **`structured` (default)** | **0.680** | **0.371** | **0.937** | **0.795** |
| `mix` λ=1 / λ=30 | 0.532 / 0.617 | 0.208 / 0.316 | 0.860 / 0.928 | 0.628 / 0.779 |
| transductive `loglik` (open-set 0 % / 20 %) | 0.762 / 0.758 | 0.534 / 0.521 | — | — |

## Rule safeguards

- By default the test gallery is only used for per-image inference. The mean degree of match per query
  comes from the training prior (0.811, vs 0.81 measured on the val gallery).
- `score.transductive` (posterior over the query set, with a background class) is off; enable it
  only with written approval from the organisers and declare it in the fact sheet.
- No EM on priors and no Sinkhorn assignment.

## Setup

```bash
pip install -r requirements.txt     # training, evaluation, notebook, tests
python download_datasets.py         # Market1501, PA100k, PETA (needs gdown<6)
python -m pytest                    # add PYTHONPATH with open_clip/transformers for the parity tests
```
