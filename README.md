# MedVLM — Can AI Be a Doctor?

An interactive Google Colab notebook demonstrating how large vision-language models tackle real ophthalmology tasks, built around the [LMOD benchmark](https://arxiv.org/abs/2410.01620) (Qin et al., 2025).

| Notebook | Purpose | Data Source |
|---|---|---|
| **Quick Demo**<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaJinWakeUp/MedVLM/blob/main/colab_demo.ipynb)<br>[`colab_demo.ipynb`](colab_demo.ipynb) | Interactive 10-sample walkthrough | Cloned from GitHub (No Drive needed) |
| **Full Dataset Benchmark**<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaJinWakeUp/MedVLM/blob/main/colab_full_dataset_inference.ipynb)<br>[`colab_full_dataset_inference.ipynb`](colab_full_dataset_inference.ipynb) | Full-scale batch inference, metrics & evaluation | Mounted from Google Drive (`Datasets/LMOD`) |

---

## What it does

The repository provides two notebooks:
1. **Interactive Demo (`colab_demo.ipynb`)**: Runs **InternVL2-4B** on 10 curated ophthalmology samples.
2. **Full Dataset Benchmark (`colab_full_dataset_inference.ipynb`)**: Mounts the dataset from Google Drive (`Datasets/LMOD`), extracts to high-speed local disk, performs batch inference with auto-checkpointing on all dataset parts supporting **both Anatomical Recognition and Diagnosis Analysis**, and exports all results back to Drive.

### Dual-Purpose Modalities & Datasets Evaluated:

| Modality | Source Dataset(s) | Sample Count | Anatomical Recognition | Diagnosis Analysis |
|:---|:---|:---:|:---:|:---|
| **OCT** | **OIMHS** | **3,859** | `irc`, `retina`, `choroid`, `mh` | **Macular Hole Staging** (Stages 1–4) |
| **Color Fundus (CFP)** | **REFUGE, ORIGA, G1020, IDRiD** | **3,386** | Optic Disc, Optic Cup, Fovea | **Glaucoma Detection** (REFUGE, ORIGA, G1020) |

Results are compared against paper baselines and specialist-trained CNNs to show the gap between general LVLMs and domain-specific models.

---

## How to run

- **For the Quick Demo**: Click the **Open in Colab** badge for `colab_demo.ipynb`. It automatically clones this repo and runs immediately.
- **For Full Dataset Inference**:
  1. Upload your dataset zip file(s) or folders to Google Drive at `Datasets/LMOD`.
  2. Open `colab_full_dataset_inference.ipynb` in Colab.
  3. Mount Drive and run the cells. The notebook will automatically unzip the dataset to local SSD, index all OCT and Fundus samples, run batch inference with checkpointing, and export prediction CSVs and plots to `Datasets/LMOD/results/`.

> **GPU requirement:** InternVL2-4B needs ~8 GB GPU RAM. A free **T4 GPU** (or A100/V100) is sufficient.
> Go to **Runtime → Change runtime type → T4 GPU** before running.

---

## Repository structure

```
MedVLM/
├── colab_demo.ipynb                    ← Quick interactive demo notebook (10 curated samples)
├── colab_full_dataset_inference.ipynb  ← Full dataset inference & evaluation notebook (Google Drive)
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
