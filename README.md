# HyperCBM

<div align="center">
<br>
<h3>Learning Label-Efficient Interpretable Medical Image Diagnosis via Semi-supervised Hypergraph Concept Bottleneck Model</h3>


<p align="center">
<!--   <a href="https://yijun-yang.github.io/MeWM/"><img src="https://img.shields.io/badge/project-page-red" alt="Project Page"></a> -->
  <a href="https://github.com/scott-yjyang/HyperCBM"><img src="https://img.shields.io/badge/ArXiv-<2507.22530>-<COLOR>.svg" alt="arXiv"></a>
<!--   <a href="https://huggingface.co/papers/2506.02327"><img src="https://img.shields.io/badge/huggingface-page-yellow.svg" alt="huggingface"></a> -->
 <p align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=scott-yjyang.HRVVS)

  
</div>

<p align="center"><img width="100%" src="assets/framework.pdf" /></p>


## Overview

Deep learning excels in medical image analysis but lacks the interpretability essential for clinical trust. **Concept Bottleneck Models (CBMs)** address this by routing predictions through human-understandable concepts, yet they (1) treat concepts independently, ignoring high-order inter-concept dependencies critical in medical reasoning, and (2) require expensive expert-level concept annotations that limit scalability.

**HyperCBM** is a semi-supervised concept bottleneck framework that overcomes both limitations through dual-level hypergraph learning:

- **HECRL** (Hypergraph-Enhanced Concept Representation Learning) constructs a concept-level hypergraph to capture high-order semantic relationships among concepts via adaptive hyperedge formation and attention-driven weighting, followed by HGNN+ propagation for structured reasoning.

- **HIDP** (Hypergraph Image Dynamic Pseudo-labeling) builds an image-level hypergraph over domain-adapted feature maps to generate reliable pseudo-labels for unlabeled data, bridging the domain gap that undermines existing pseudo-labeling strategies.

## Key Results

With only **10% concept labels**, HyperCBM matches or surpasses fully supervised baselines:

| Dataset | Label Ratio | Concept Acc. | Class Acc. | Concept AUC | Class AUC |
|---------|:---:|:---:|:---:|:---:|:---:|
| **PAS** | 10% | 81.61 | 76.89 | 64.43 | 88.40 |
| **PAS** | 40% | 84.19 | 78.82 | 68.20 | 90.48 |
| **BrEaST** | 10% | 72.49 | 65.49 | 56.66 | 70.31 |
| **BrEaST** | 40% | 76.25 | 75.29 | 61.93 | 80.11 |

## Project Structure

```
HyperCBM/
├── main.py                         # Training entry point
├── train_multi_seeds.py            # Multi-seed experiment runner
├── utils.py                        # Utilities (logging, visualization, seeding)
├── configs/
│   ├── basic_config.py             # CLI argument parser
│   ├── PAS_Large_Hypergraph.yaml   # PAS dataset config
│   └── BrEaST_Hypergraph.yaml     # BrEaST dataset config
├── models/
│   ├── cem_hypergraph.py           # HyperCBM model (core)
│   ├── hypergraph.py               # HECRL module (HyperConceptNet)
│   ├── cem.py                      # Concept Embedding Model (base class)
│   ├── cbm.py                      # Concept Bottleneck Model (base class)
│   └── construction.py             # Model factory
├── data/
│   ├── pas_loader.py               # PAS dataset loader + HIDP pseudo-labeling
│   └── breast_loader.py            # BrEaST dataset loader + HIDP pseudo-labeling
└── train/
    ├── training.py                 # Training loop with early stopping
    ├── evaluate.py                 # Evaluation (accuracy, AUC, F1, representation metrics)
    └── utils.py                    # Training utilities (backbone wrapping, accuracy computation)
```

## Installation

**Requirements:** Python 3.10, CUDA 11.7+

```bash
# PyTorch (CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Dependencies
pip install -r requirements.txt

# CEM base library (included as submodule)
pip install -e cem/ --no-deps
```

## Dataset Preparation

Place datasets under `./dataset/`:

```
dataset/
├── PAS_Large/
│   ├── PAS_cropped/                    # .bmp ultrasound images
│   ├── PAS_updated.xlsx                # 45 concept annotations, 3 severity levels
│   └── PAS_updated_split_indices.pth   # Train/val/test split
└── BrEaST/
    ├── BrEaST-Lesions_USG-images_and_masks/   # .png ultrasound images
    ├── image_dict.csv                  # 7 BI-RADS concepts, 2 classes
    └── split_indices.pth               # Train/val/test split
```

**PAS**: 671 ultrasound scans with 45 clinically curated concepts across 3 PAS severity levels (Normal / Accreta / Increta). Concepts are extracted by HuatuoGPT-Vision and validated by two board-certified obstetricians. *Available upon request.*

**BrEaST**: 254 breast ultrasound images with 7 BI-RADS descriptor concepts and binary malignant/benign labels. [Source](https://doi.org/10.1038/s41597-024-02984-z)

All images are center-cropped and resized to 224x224. Data is split into train/val/test with a 7:1:2 ratio.

## Training

**Single run:**

```bash
# PAS dataset, 10% labeled concepts, seed=42
python main.py --dataset PAS_Large_Hypergraph --labeled_ratio 0.1 --seed 42 --device gpu

# BrEaST dataset, 40% labeled concepts
python main.py --dataset BrEaST_Hypergraph --labeled_ratio 0.4 --seed 42 --device gpu
```

**Multi-seed evaluation** (5 seeds: 42, 2024, 2025, 1, 2):

```bash
python train_multi_seeds.py --dataset PAS_Large_Hypergraph --labeled_ratio 0.1
python train_multi_seeds.py --dataset BrEaST_Hypergraph --labeled_ratio 0.4
```

**Key arguments:**
| Argument | Description | Default |
|---|---|---|
| `--dataset` | Config name (without `.yaml`) | - |
| `--labeled_ratio` | Proportion of concept-labeled data | 0.1 |
| `--seed` | Random seed | 42 |
| `--device` | `gpu` or `cpu` | auto |

## Implementation Details

| | PAS | BrEaST |
|---|---|---|
| Backbone | ResNet34 | ResNet34 |
| Max epochs | 250 | 150 |
| Learning rate | 5e-4 | 5e-4 |
| Early stopping patience | 5 | 5 |
| Optimizer | Adam | Adam |
| HECRL k_min | 3 | 2 |
| Loss weights (lambda_1, lambda_2) | (1.0, 0.1) | (0.5, 0.1) |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
