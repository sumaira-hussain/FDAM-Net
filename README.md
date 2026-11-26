# FDAM-Net: Frequency-Domain Adaptive Mask Network for Domain-Generalizable Polyp Segmentation

This repository contains the implementation of **FDAM-Net**, a dual-domain deep learning framework proposed for robust and domain-generalizable colorectal polyp segmentation.

FDAM-Net integrates spatial and frequency-domain feature calibration to improve segmentation robustness under cross-dataset and cross-institution domain shifts.

### Key Components
- **Dynamic Frequency Decoupling Block (FDB):** Employs learnable, channel-wise Gaussian frequency masks to separate low- and high-frequency feature components while preserving spectral phase information.
- **Global Context Guidance (GCG):** Propagates high-level semantic information to earlier decoding stages to improve boundary delineation in challenging, low-contrast regions.
- **Dynamic Gating Fusion Module (DGFM):** Uses adaptive feature gating to suppress irrelevant background information during decoder fusion.
- **Adaptive Boundary-Guided Selection (ABGS):** Combines region and boundary predictions for refined segmentation.

### Evaluation Datasets
FDAM-Net is evaluated on five public polyp segmentation benchmarks:
- Kvasir-SEG
- CVC-ClinicDB
- CVC-300
- CVC-ColonDB
- ETIS-LaribPolypDB

### Main Results
- Unweighted cross-dataset average Dice: **0.8568**
- ETIS-LaribPolypDB Dice: **0.7885**
- Model parameters: **27.69 million**
- Computational complexity: **17.89 GFLOPs**
- Inference speed: **42.75 FPS** at 352 × 352 resolution on NVIDIA RTX 2080 Ti.

### Implementation
The repository provides the model architecture, training and evaluation scripts, and configuration details.

*Code, pretrained weights, and reproduction instructions will be provided as they become available.*

### Citation
If you use this work, please cite the associated publication.

**Paper:** FDAM-Net: Frequency-Domain Adaptive Mask Network for Domain-Generalizable Polyp Segmentation.

