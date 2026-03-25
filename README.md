# Evaluating Semantic Grounding in Open-Vocabulary Segmentation

**CS766 Computer Vision — University of Wisconsin-Madison**

> A comparative study of MaskCLIP and DenseCLIP with a novel text-driven region explanation module.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Authors
Yutesh Vishnu Addanki (`yaddanki`) · Dinesh Kumar Rajakumar (`drajakumar`)

---

## Abstract

> _Coming soon — will be filled in after midterm experiments._

---

## Results

> _To be updated as experiments complete._

| Model     | mIoU | CLIP Similarity | Robustness Variance | Explanation Alignment |
|-----------|------|-----------------|---------------------|-----------------------|
| MaskCLIP  | —    | —               | —                   | —                     |
| DenseCLIP | —    | —               | —                   | —                     |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/semantic-grounding-eval.git
cd semantic-grounding-eval
conda create -n segclip python=3.9 -y
conda activate segclip
pip install -r requirements.txt
```

---

## References

- MaskCLIP: [arXiv:2208.08984](https://arxiv.org/abs/2208.08984)
- DenseCLIP: [arXiv:2112.01518](https://arxiv.org/abs/2112.01518)
- CLIP: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

---

## License
MIT — see [LICENSE](LICENSE)


## MileStones

### MileStone 1: Setting up the skeleton repo
 - proper folder headers and structures in which the project continues
 - Read me file to get the basic layout and explanations and relevant details of miestones
 - license and requirements

### MileStone 2 : Adding all required files for the project
 #### Configs
    - `configs/maskclip.yaml` — model, dataset, and output settings for MaskCLIP inference
    - `configs/denseclip.yaml` — model, dataset, and output settings for DenseCLIP inference
    - `configs/prompt_variants.yaml` — 5 concept groups × 4 prompt phrasings for robustness testing
    
    #### Models
    - `models/maskclip/__init__.py` — exposes MaskCLIP class for clean imports
    - `models/maskclip/model.py` — MaskCLIP reimplementation: frozen CLIP patch tokens → cosine similarity mask
    - `models/denseclip/__init__.py` — exposes DenseCLIP class for clean imports
    - `models/denseclip/model.py` — DenseCLIP reimplementation: ContextDecoder cross-attention + conv segmentation head
    
    #### Evaluation
    - `evaluation/metrics.py` — all 4 metrics: IoU, CLIP similarity, robustness variance, explanation alignment
    - `evaluation/evaluate.py` — main runner: loads both models, dataset, runs all metrics, saves JSON results
    
    #### Scripts
    - `scripts/run_maskclip.py` — inference pipeline: runs MaskCLIP on image folder, saves masks and red overlays
    - `scripts/run_denseclip.py` — inference pipeline: runs DenseCLIP on image folder, saves masks and blue overlays
    - `scripts/compare_outputs.py` — generates 3-panel side-by-side figures (original | MaskCLIP | DenseCLIP)
    
    #### Data
    - `data/download_pascal_voc.sh` — downloads and extracts PASCAL VOC 2012 (~2GB), verifies file counts
    
    #### Explanation
    - `explanation/crop_regions.py` — crops predicted regions from masks, overlay and bounding box helpers
    
    #### Notebooks
    - `notebooks/midterm_analysis.ipynb` — results table, robustness charts, failure case taxonomy, observations

