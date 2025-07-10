# BOOST: Out-of-Distribution-Informed Adaptive Sampling for Bias Mitigation in Stylistic Convolutional Neural Networks

This repository contains code for our paper:

**"BOOST: Out-of-Distribution-Informed Adaptive Sampling for Bias Mitigation in Stylistic Convolutional Neural Networks"**

Authors: Mridula Vijendran, Shuang Chen, Jingjing Deng, Hubert P. H. Shum  
📄 *[Preprint Link Coming Soon]*

---

## 📦 Repository Structure

### 🔍 Core Modules

- **STSACLF (Style-Transfer Sensitive Art Classifier)**
  - `stclf_model.py` — STSACLF architecture
  - `attention.py` — Channel and spatial attention modules
  - `pretrained_models.py` — Backbone feature extractors (e.g., ResNet, VGG)

- **BOOST (Bias-Oriented OOD Sampling and Tuning)**
  - `sampler_utils.py` *(if present)* — Adaptive batch sampling logic using ODIN-based confidence scoring and perturbation

### 📁 Datasets

- `kaokore_ds.py` — Dataset wrapper for the [KaoKore dataset](https://github.com/rois-codh/kaokore)
  - Download the Kaokore dataset using the download.py script and add into data/

> 📌 This project uses the [KaoKore](https://github.com/rois-codh/kaokore) and [stylize-datasets](https://github.com/facebookresearch/stylized-imageNet) repositories as dependencies.

### 🛠 Utilities

- `utils.py` — Metric computation, logging, reproducibility helpers
- `clf_vis_utils.py' — BOOST metrics such as OOD score, MAB, SDB used here for inference

---

## 🚀 Getting Started

### Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

###  How to train a model on the Kaokore dataset and get the performance results

```bash
python -m main
```
