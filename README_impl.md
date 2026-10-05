# CSAR — Calibrated Structured Attribute Retrieval (UPAR 2027, Track 2)

Un seul paquet `src/` sert à l'entraînement, à l'évaluation et à la soumission :
le `run.py` livré à Codabench importe exactement le même code (copié dans le zip),
et le checkpoint contient architecture, poids, calibration figée et config de score.

## Modules

| Module | Rôle | Levier de l'analyse |
| --- | --- | --- |
| `attributes.py` | 40 attributs, groupes (classe « aucun » pour cheveux, type de bas, lunettes), encodage par groupe, réalignement de colonnes | Structure par groupes |
| `data.py` | Chargement validé, protocole « 1 vecteur distinct = 1 requête », plis LODO, masques par requêtes, échantillonnage équilibré par domaine, voisins de Hamming | Robustesse au domaine, négatifs difficiles |
| `transforms.py` | Letterbox (ratio conservé), flip/crop/AugMix sans couleur, pas de Random Erasing | Robustesse au domaine |
| `model.py` | Backbone torchvision + tête sigmoïde 40 + têtes softmax par groupe + branche requête-image | Têtes par groupe, tête de compatibilité |
| `losses.py` | BCE pondérée + label smoothing, CE par groupe, contraste image→requête avec négatifs Hamming 1-2 et marge | L = L_PAR + λ L_groupe + μ L_ret |
| `calibration.py` | Température par attribut et par groupe, apprise hors test puis figée | Calibration |
| `scoring.py` | `l1`, `loglik` (S2), `structured` (S3), `endom` (E[ndom], Poisson-binomiale tronquée), `mix`, option transductive ; calcul par blocs, tie-breaker continu | Score / ranking |
| `metrics.py` | mADM, mAP, R1/5/10, mINP (réimplémentation, par blocs) | Mesure |
| `inference.py` | DataLoader batché, fp16 sur GPU, TTA flip, logs de débit | Ingénierie de soumission |
| `checkpoint.py` | Bundle auto-descriptif (chargement strict : jamais de poids aléatoires silencieux) | Reproductibilité |
| `train.py` / `evaluate.py` / `export.py` | CLI d'entraînement, d'évaluation/ablation, d'empaquetage | Plan d'attaque |
| `benchmark.py` | Charge chaque backbone avec ses poids ImageNet, mesure débit et taille du checkpoint | Budget de temps |
| `synthetic.py` | Mini-jeu au format UPAR (labels réels, images aléatoires) pour les tests de bout en bout | Tests |
| `submission_run.py` | Gabarit Codabench (`load_model`, `rank_gallery`), copié en `run.py` par `export.py` | — |

## Notebook

`notebooks/pipeline.ipynb` lance toute la chaîne (données → étude des scores → backbone → entraînement →
évaluation/calibration → export → test du `run.py` exporté comme Codabench). `MODE = "smoke"` tourne en
~1 min sur CPU avec des données synthétiques ; `MODE = "full"` utilise les vraies images.

## Backbones

Tous viennent de torchvision : aucun code à embarquer dans le conteneur. Chacun est testé de bout en bout
(`tests/test_pipeline.py::test_backbone_end_to_end`) : forward, pas d'entraînement avec la vraie perte,
checkpoint fp16, rechargement. Débit mesuré par `python -m src.benchmark` sur le CPU de développement,
avec TTA flip, batch 16 :

| Backbone | Params (M) | Checkpoint fp16 (MiB) | img/s CPU | Galerie 28 000 images (s) |
| --- | --- | --- | --- | --- |
| resnet18 | 11,5 | 22 | 45,7 | 615 |
| efficientnet_v2_s | 20,8 | 40 | 22,7 | 1 240 |
| convnext_tiny | 28,2 | 54 | 18,3 | 1 533 |
| resnet50 | 24,3 | 47 | 16,0 | 1 755 |
| swin_t | 27,9 | 53 | 8,0 | 3 512 |
| convnext_base | 88,1 | 168 | 5,4 | 5 198 |
| swin_b | 87,2 | 167 | 2,6 | 10 669 |

Sur CPU, aucun ne tient 600 s : le choix dépend du matériel Codabench (GPU ou non), à mesurer
à la première soumission. Pour ajouter un modèle : une entrée dans `src/model.py: BACKBONES`.

## Workflow

```bash
# 0. Étude des scores sur la vérité terrain bruitée (sans images)
python -m src.evaluate --simulate 3.0 --scores l1 loglik structured mix --lams 1 30 --max-queries 600

# 1. Plis LODO (sélection d'hyperparamètres) puis évaluation non biaisée :
#    calibration sur une moitié des requêtes du domaine exclu, mesure sur l'autre
python -m src.train --holdout PETA --out runs/lodo_peta
python -m src.evaluate --checkpoint runs/lodo_peta/model.pt --holdout PETA --calib-frac 0.5 \
    --scores loglik structured mix --lams 1 30 --gammas 0 0.5 1 --cache runs/lodo_peta/val_preds.pt
#    Expérience B (open-set) : --open-set 0.2 ; galerie au format test 2024 : --resample 367

# 2. Modèle final (train complet, calibration sur val), choix du score, soumission
python -m src.train --out runs/final
python -m src.evaluate --checkpoint runs/final/model.pt --scores loglik structured --gammas 0 0.5 1 --save-best
python -m src.export --checkpoint runs/final/model.pt --out submissions/final   # -> submissions/final.zip

# Tests (dont un bout-en-bout train -> export -> run.py sur images synthétiques)
python -m pytest
```

## Résultats de la simulation (val officielle, 600 requêtes, bruit gaussien σ sur les logits)

| Score | mADM σ=3 | mAP σ=3 | mADM σ=2 | mAP σ=2 |
| --- | --- | --- | --- | --- |
| `l1` (exemple officiel) | 0,490 | 0,161 | 0,798 | 0,491 |
| `loglik` (S2) | 0,582 | 0,252 | 0,897 | 0,696 |
| **`structured` (S3, défaut)** | **0,680** | **0,371** | **0,937** | **0,795** |
| `mix` λ=1 / λ=30 | 0,532 / 0,617 | 0,208 / 0,316 | 0,860 / 0,928 | 0,628 / 0,779 |
| `loglik` transductif (exp. A / B 20 %) | 0,762 / 0,758 | 0,534 / 0,521 | — | — |

- S3 utilise **les mêmes logits** que S2 (pas de seconde vue) : le gain vient uniquement de la contrainte « une valeur par groupe ».
- E[ndom] n'améliore jamais son terme exact dans cette simulation : `mix` reste disponible, mais pas par défaut. À re-mesurer en LODO sur un vrai modèle (erreurs corrélées).
- Bruit indépendant entre attributs : à confirmer en LODO avant toute conclusion.

## Garde-fous du règlement

- La voie par défaut n'utilise la galerie test que pour l'inférence image par image. La moyenne de dom par requête vient du prior d'entraînement (0,811 contre 0,81 mesuré sur la galerie val).
- `ScoreConfig.transductive` (postérieure sur l'ensemble des requêtes, classe de fond) est **désactivé** : à n'activer qu'après accord écrit des organisateurs, en le déclarant dans la fact sheet.
- Pas d'EM sur les priors ni de Sinkhorn : écartés par l'analyse (règle 2024, R ≈ 77 au test).

## Non couvert pour l'instant

- Encodeurs fondation (SigLIP 2, DINOv3) : il faudrait embarquer leur code (timm n'est pas dans le conteneur), puis ajouter une entrée dans `model.BACKBONES`.
- Ensemble de modèles et WiSE-FT.
- Test chronométré dans l'image Docker officielle (`--network none`) : à faire avant la première soumission.
