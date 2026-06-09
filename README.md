# MedVLM — Can AI Be a Doctor?

An interactive Google Colab notebook demonstrating how large vision-language models tackle real ophthalmology tasks, built around the [LMOD benchmark](https://arxiv.org/abs/2410.01620) (Qin et al., 2025).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaJinWakeUp/MedVLM/blob/main/colab_demo.ipynb)

---

## What it does

The notebook runs **InternVL2-4B** on 10 curated ophthalmology samples and walks through three clinical tasks:

| Task | Description | Data |
|------|-------------|------|
| **Anatomical Recognition** | Label numbered bounding boxes in a retinal cross-section | 5 OCT scans (OIMHS dataset) |
| **Glaucoma Diagnosis** | Classify a fundus photo as Glaucoma / Non-Glaucoma | 5 color fundus photos (REFUGE dataset) |
| **Macular Hole Staging** | Grade the severity of a macular hole (Stage 1–4) | Same 5 OCT scans |

Results are compared against paper baselines and specialist-trained CNNs to show the gap between general LVLMs and domain-specific models.

---

## How to run

Click the **Open in Colab** badge above — the notebook clones this repo automatically. No setup needed.

> **GPU requirement:** InternVL2-4B needs ~8 GB GPU RAM. A free **T4 GPU** is sufficient.
> Go to **Runtime → Change runtime type → T4 GPU** before running.

---

## Repository structure

```
MedVLM/
├── colab_demo.ipynb          ← The notebook (open this in Colab)
└── samples/
    ├── OIMHS/                ← 5 OCT scans with macular hole annotations
    │   ├── 35_8/             ← Stage 1
    │   ├── 28_13/            ← Stage 2
    │   ├── 102_32/           ← Stage 3
    │   ├── 100_17/           ← Stage 4
    │   └── 100_18/           ← Stage 4 (second example)
    └── REFUGE/               ← 5 color fundus photos
        ├── V0001/            ← Non-Glaucoma
        ├── V0002/            ← Non-Glaucoma
        ├── V0003/            ← Non-Glaucoma
        ├── V0006/            ← Glaucoma
        └── V0026/            ← Glaucoma
```

Each sample folder contains:
- `visualization.png` — clean image
- `annotated/<annotated_image>.png` — image with expert bounding boxes
- `information.json` — ground-truth labels and annotation metadata

---

## Datasets

- **OIMHS** — 3,859 OCT images with macular hole annotations (stages 1–4). [Paper](https://arxiv.org/abs/2410.01620)
- **REFUGE** — Color fundus photos with glaucoma labels (train: 40 glaucoma / 360 healthy). [Challenge page](https://refuge.grand-challenge.org/)

The 10 samples in this repo are a curated subset for educational use.

---

## Paper

> Qin et al., *"LMOD: A Large Multimodal Ophthalmology Dataset and Benchmark for Large Vision-Language Models"*, arXiv:2410.01620, 2025.
> Project page: [kfzyqin.github.io/lmod](https://kfzyqin.github.io/lmod/)
