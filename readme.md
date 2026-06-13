# TerraMind Semantic Segmentation — Agricultural Damage Detection

Change detection and semantic segmentation pipeline using **TerraMind**, a pretrained Earth observation foundation model, to predict 10-meter resolution agricultural damage from before/after satellite imagery.

---

## Overview

This project fine-tunes (or uses frozen) the **TerraMind** encoder as a feature extractor for paired before/after multi-modal satellite scenes, differences the resulting embeddings, and feeds them into a **U-Net 2D decoder** to produce a pixel-wise segmentation map.

| Class | Meaning |
|---|---|
| 0 | Background |
| 1 | Non-damaged agricultural land |
| 2 | Damaged agricultural land |

**Pipeline:**
1. Load paired before/after multi-modal patches (e.g. Sentinel-2, Sentinel-1, DEM, NDVI)
2. Encode both timesteps with TerraMind
3. Difference the before/after embeddings
4. Decode the differenced embeddings with a U-Net 2D into a 3-class segmentation map
5. Train with weighted cross-entropy and track IoU, accuracy, precision, recall, and F1

---

## Repository Structure

```
TerraMind_SemanticSeg/
├── configs/                  # Hydra configuration files (train/val params)
├── DataLoader.py              # Custom dataset for paired before/after multi-modal patches
├── Decoder_UNet2D.py           # U-Net 2D decoder applied to differenced embeddings
├── Encoder_TerraMind.py        # TerraMind backbone wrapper
├── Train.py                    # Main training loop (Hydra + TensorBoard)
├── Test.py                     # Evaluation script
├── utils.py                    # Metrics, checkpointing, helper functions
├── arm.slurm / slurm.slurm     # SLURM job scripts for cluster training
└── __init__.py
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/evan-leendertse/TerraMind_SemanticSeg.git
cd TerraMind_SemanticSeg
```

### 2. Install dependencies

This project relies on [TerraTorch](https://github.com/IBM/terratorch) for the TerraMind backbone, along with PyTorch, Hydra, and Albumentations.

```bash
pip install -r requirements.txt
```

### 3. Configure your run

Edit the Hydra config files under `configs/train_val/` to set data paths, modalities, batch sizes, learning rate, and number of epochs.

### 4. Train

```bash
python Train.py
```

Training and validation metrics, checkpoints, and configs are saved per run under the Hydra multirun output directory.

### 5. Monitor with TensorBoard

```bash
tensorboard --logdir=./multirun
```

### 6. Evaluate

```bash
python Test.py
```

---

## Method

### Encoder — TerraMind

`Encoder_TerraMind.py` wraps a pretrained TerraMind backbone (via TerraTorch's `BACKBONE_REGISTRY`). It accepts a configurable list of modalities (e.g. `S2L2A`, `S1GRD`, `S1RTC`, `DEM`, `NDVI`, `LULC`) and produces embeddings for a given timestep.

The encoder can either be **frozen** (decoder-only training) or **fine-tuned** jointly with the decoder, controlled via the `TM_finetune` config flag.

### Change Representation

For each patch, embeddings are computed separately for the "before" and "after" timesteps. The model then computes the **difference** between corresponding embeddings:

$$z_{\text{diff}} = z_{\text{after}} - z_{\text{before}}$$

This differenced representation is what's passed to the decoder, so the model focuses on *change* rather than absolute scene content.

### Decoder — U-Net 2D

`Decoder_UNet2D.py` takes the differenced multi-scale embeddings and upsamples them through a U-Net-style decoder to produce a per-pixel 3-class segmentation map.

### Loss & Class Imbalance

Cross-entropy loss is used, with optional class weighting (`apply_weight_loss`) to handle the typical imbalance between background, undamaged, and damaged classes — weights can be computed by pixel frequency or inverse frequency.

### Metrics

Per epoch, the model tracks:
- IoU
- Accuracy
- Precision
- Recall
- F1

The best checkpoint (lowest validation loss) is saved automatically.

---

## Configuration

All hyperparameters and paths are managed via [Hydra](https://hydra.cc/), located in `configs/train_val/`. This allows sweeping over modalities, learning rates, patch sizes, and fine-tuning strategies without editing code.

---

## Cluster Training

`arm.slurm` and `slurm.slurm` provide example SLURM job submission scripts for running training on HPC clusters.

---

## License

MIT
