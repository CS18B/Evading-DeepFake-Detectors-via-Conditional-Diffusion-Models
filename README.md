# Evading-DeepFake-Detectors-via-Conditional-Diffusion-Models


This repository provides the official PyTorch implementation of our paper:

> **Evading DeepFake Detectors via Conditional Diffusion Models**  
> Wenhao Wang, Fangjun Huang  
> _IH&MMSec 2024, ACM Workshop on Information Hiding and Multimedia Security_  
> [[Paper (PDF)]](./01论文pdf版本.pdf) | [[ACM DOI]](https://doi.org/10.1145/3658664.3659653)

## 🧠 Overview

DeepFake detectors play a critical role in identifying manipulated facial images. However, they are vulnerable to adversarial attacks. This work proposes a **semantic reconstruction-based adversarial attack** that leverages conditional diffusion models to generate imperceptible perturbations in the latent space.

![](./assets/framework.png)

### Key Highlights

- 🌀 **Latent space attack** using DDIM-guided conditional sampling.
- 🕵️ **Evades both spatial and frequency domain detectors** (e.g., XceptionNet, SRM).
- 🖼️ **High visual quality** — minimal artifacts, low LPIPS, and low FID.
- 🎯 Supports white-box and black-box settings with **meta-learning transferability**.

---

## 🔧 Method

The proposed attack optimizes the latent representation of fake images in a semantic latent space extracted by a diffusion autoencoder. A conditional DDIM decoder reconstructs adversarial faces guided by:

- **Attack loss**: Cross-entropy to mislead the detector.
- **Perceptual loss**: LPIPS + L2 to preserve structure and realism.
