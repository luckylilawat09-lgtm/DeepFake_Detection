# 🔮 DeepFake Biometric Detection & Neural Telemetry Platform

<div align="center">

[![CI Pipeline](https://github.com/luckylilawat09-lgtm/DeepFake_Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/luckylilawat09-lgtm/DeepFake_Detection/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/PyTest-20%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](backend/tests/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL%203D-black?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20140k%20Faces-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <b>A deepfake detection platform featuring an EfficientNet-B0 transfer learning classifier paired with an interactive 3D chromatic glass refraction interface, active learning telemetry, and an automated 20-test QA suite.</b>
</p>

**Group 233 · VIT Bhopal · Project Exhibition-I**  
*Team Members:* Tadiboyina Siva, Prisha Verma, Lucky Lilawat, Nanya, Jitesh Maru.  
*AI Assistance Disclosure:* Development and training execution were implemented using **Antigravity** (Google DeepMind's AI coding assistant), with full review, verification, and empirical testing conducted by the project team.

[Key Features](#-key-features) •
[System Architecture](#-system-architecture) •
[Quick Start](#-quick-start-guide) •
[Model & Training CLI](#-deep-learning-model--training-pipeline) •
[API Specification](#-api-specification) •
[Automated Tests](#-automated-testing--qa) •
[Limitations](#-limitations--generalization-scope) •
[Contributing](CONTRIBUTING.md) •
[Project Report](PROJECT_REPORT.md)

</div>

---

## 📌 Executive Summary

With the rise of generative adversarial networks (GANs)—specifically StyleGAN—detecting synthesized facial media is a key challenge for biometric security and digital content integrity.

This repository implements an **experimental facial deepfake detection baseline**:
1. **Core AI Engine:** A fine-tuned **EfficientNet-B0** binary classifier built in **PyTorch**, trained on real faces (FFHQ) versus GAN-synthesized human faces (StyleGAN) with progressive 2-stage fine-tuning and learning rate scheduling.
2. **Visual Telemetry Frontend:** An interactive 3D viewport inspired by modern WebGL aesthetics ([motionsites.ai prompt](https://motionsites.ai/?prompt=design-world)), featuring custom GLSL chromatic refraction, dynamic laser scanning, and real-time neural pipeline HUD metrics.
3. **High-Throughput API:** A modular Flask REST server with ephemeral upload sanitation, active learning human feedback logging (`/feedback`), dynamic checkpoint reloading (`/reload-model`), and real-time training progress streaming.
4. **Automated QA & CI:** Comprehensive PyTest test suite (20 passing tests) and GitHub Actions CI workflow covering Python 3.10 through 3.13.

---

## 🏗️ System Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      INTERACTIVE 3D WEBGL CLIENT                       │
 │  • Three.js 2-Pass Chromatic Refraction Cube  • Dynamic Laser Scanner  │
 │  • Real-Time Neural Telemetry HUD             • Human-in-the-Loop UI   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP / REST (Fetch API)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         FLASK REST API BACKEND                         │
 │                                                                        │
 │  ┌───────────────────────┐  ┌──────────────────────┐  ┌─────────────┐ │
 │  │    POST /predict      │  │ GET /training-status │  │   POST/GET  │ │
 │  │ Secure Upload & Infer │  │   Live Progress JSON │  │  /feedback  │ │
 │  └───────────┬───────────┘  └──────────────────────┘  └──────┬──────┘ │
 │              │ Ephemeral File Auto-Cleanup                   │        │
 └──────────────┼───────────────────────────────────────────────┼────────┘
                │ Tensor Input (224x224 RGB)                    │
                ▼                                               ▼
 ┌──────────────────────────────────────────┐    ┌──────────────────────┐
 │       EFFICIENTNET-B0 INFERENCE          │    │ ACTIVE LEARNING LOG  │
 │  • Pretrained ImageNet Backbone          │    │ Human corrections &  │
 │  • 256-d Dense Classifier Head           │    │ audit trail for      │
 │  • Softmax Probability Distribution      │    │ retraining loops     │
 └──────────────────────────────────────────┘    └──────────────────────┘
```

---

## ✨ Key Features

- **💎 3D Chromatic Glass Refraction Cube:** Powered by Three.js with custom GLSL shaders executing a 2-pass refraction pipeline (`rtBack` background depth map $\rightarrow$ inner specimen geometry $\rightarrow$ `rtFront` chromatic dispersion).
- **⚡ Holographic Laser Scanner:** Real-time sweeping laser visualization over suspect imagery with responsive biometric grid overlays.
- **🔄 Active Learning & Human Feedback:** User correction loop logged to `backend/models/feedback_log.json` to enable continuous model improvement and auditing.
- **📊 Live Neural Telemetry HUD:** Polling engine displaying live training epochs, batch loss, real-time accuracy, and pipeline milestone status.
- **🛡️ Ephemeral Upload Sanitation:** Uploaded files are immediately deleted inside `finally` blocks upon tensor conversion, preventing biometric disk leakage.
- **⚙️ CLI Training Engine:** Full-featured `train.py` CLI with argument flags, checkpoint resume, LR scheduling, and graceful `Ctrl+C` state persistence.
- **🧪 Automated QA:** 14 unit and integration tests with PyTest covering model forward pass, API response schemas, and training helper utilities.

---

## 🎨 UI & Design Foundation

The visual identity, dark chromatic aesthetic, and glassmorphic layout were engineered using the design prompt from **[motionsites.ai](https://motionsites.ai/)**:
> **Template Reference:** [`https://motionsites.ai/?prompt=design-world`](https://motionsites.ai/?prompt=design-world)

Key interface components:
- **WebGL 3D Specimen Stage:** Interactive drag-and-drop rotating cube with physics dampening.
- **Cyber Hologram Overlays:** JetBrains Mono font readouts for confidence metrics, latency, and biometric classification.
- **Confidence Speedometer & Gauge:** Real vs. Fake probability visualization with glowing status accents.

---

## 📊 Dataset & Data Engineering

The deep learning model is trained on the benchmark **140k Real and Fake Faces** collection sourced from Kaggle:
- **Dataset Link:** [Kaggle: 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Real Imagery Source:** 70,000 authentic portraits from Nvidia’s **Flickr-Faces-HQ (FFHQ)** dataset.
- **Fake Imagery Source:** 70,000 synthesized portraits produced by Nvidia’s **StyleGAN** generative adversarial network.
- **Resolution & Preprocessing:** 256×256 crops resized and normalized to 224×224 tensors using ImageNet normalization constants (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### Data Partitioning (`Data Set 1`)
| Partition | Real Faces | Fake (GAN) Faces | Total Samples | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 20,001 | 20,001 | **40,002** | Supervised parameter optimization |
| **Validation** | 6,199 | 6,161 | **12,360** | Generalization tracking & model checkpointing |
| **Test** | 2,604 | 2,623 | **5,227** | Final unbiased performance evaluation |
| **Total** | **28,804** | **28,785** | **57,589** | Balanced distribution from Kaggle archive |

*(Note: Data Sets 1 through 4 in the Kaggle archive total 230,335 images; model baseline is trained on Data Set 1).*

---

## 🧠 Deep Learning Model & Training Pipeline

### Architecture
- **Backbone:** `EfficientNet-B0` (Pretrained on ImageNet-1K)
- **Classifier Head:**
  ```text
  Input (1280 features) ──► Dropout(p=0.3) ──► Linear(1280, 256) ──► ReLU ──► Dropout(p=0.2) ──► Linear(256, 2)
  ```

### Progressive 2-Stage Training Strategy
1. **Stage 1 — Classifier Warmup (Epochs 1–2):** Feature extractor is frozen (`requires_grad = False`). Custom classifier head is trained at `lr = 1e-3`.
2. **Stage 2 — Backbone Fine-Tuning (Epochs 3–10):** Convolutional backbone is unfrozen and trained at `fine_tune_lr = 1e-5` with classifier head maintaining `lr = 1e-3` and `weight_decay = 1e-4` to adapt feature extractors to subtle GAN textures.
3. **Optimization & Scheduling:** Adam optimizer paired with `ReduceLROnPlateau(factor=0.5, patience=2)` and early stopping on validation accuracy (patience = 5).

### Empirical Performance & Evaluation Metrics

The baseline model checkpoint (`backend/models/deepfake_detector.pth`) was evaluated on the strictly held-out test split (2,623 fake, 2,604 real portraits) with standard ImageNet normalization and no data augmentation. Full evaluation logs are saved in [evaluation_results.txt](evaluation_results.txt).

| Metric | Score / Value | Evaluation Scope & Notes |
| :--- | :---: | :--- |
| **Validation Accuracy** | **`95.66%`** | Evaluated on 12,360 validation portraits at Epoch 10 (checkpoint selection metric) |
| **Validation Loss** | **`0.1191`** | Cross-Entropy Loss on validation partition |
| **Test Accuracy** | **`78.59%`** | Evaluated on **5,227** held-out test portraits (4,108 / 5,227 correct) |
| **Test Loss** | **`0.5822`** | Cross-Entropy Loss on held-out test partition |
| **Precision (Fake / Class 0)** | **`72.51%`** | Proportion of predicted fake portraits that are truly synthetic ($2,422 / 3,340$) |
| **Recall (Fake / Class 0)** | **`92.34%`** | Proportion of actual synthetic portraits correctly flagged ($2,422 / 2,623$) |
| **F1-Score (Fake / Class 0)** | **`81.23%`** | Harmonic mean of precision and recall on synthetic faces |
| **Precision (Real / Class 1)** | **`89.35%`** | Proportion of predicted real portraits that are truly authentic ($1,686 / 1,887$) |
| **Recall (Real / Class 1)** | **`64.75%`** | Proportion of actual authentic portraits correctly identified ($1,686 / 2,604$) |
| **F1-Score (Real / Class 1)** | **`75.08%`** | Harmonic mean of precision and recall on authentic faces |
| **Macro Average F1** | **`78.16%`** | Unweighted average across both classes |
| **Inference Latency** | **`33.96 ms`** $\pm$ `5.48 ms` | Per-image latency on commodity CPU ($N = 100$ iterations) |

#### Confusion Matrix ($N = 5,227$ Held-out Test Portraits):
| Ground Truth \ Predicted | Predicted Fake (0) | Predicted Real (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Fake (StyleGAN)** | **2,422** (TP) | **201** (FN) | 2,623 |
| **Actual Real (FFHQ)** | **918** (FP) | **1,686** (TN) | 2,604 |
| **Total Predicted** | **3,340** | **1,887** | **5,227** |

### CLI Training Options
The training engine (`backend/train.py`) exposes a robust command-line interface:

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--data-dir` | `str` | `"../../models"` | Root path containing the Kaggle dataset partitions |
| `--dataset` | `str` | `"Data Set 1"` | Specific dataset folder name |
| `--all-datasets` | `flag` | `False` | Concatenate all dataset partitions into a unified dataset |
| `--epochs` | `int` | `10` | Total number of epochs to train |
| `--batch-size` | `int` | `16` | Mini-batch size for DataLoader |
| `--lr` | `float` | `1e-3` | Classifier head learning rate |
| `--fine-tune-lr` | `float` | `1e-5` | Fine-tuning backbone learning rate |
| `--unfreeze-after-epoch`| `int` | `2` | Epoch after which the backbone is unfrozen |
| `--no-resume` | `flag` | `False` | Start fresh without loading previous checkpoint |
| `--save-dir` | `str` | `"models"` | Directory to store checkpoints and telemetry |
| `--model-name` | `str` | `"deepfake_detector.pth"` | Checkpoint output filename |
| `--max-batches-per-epoch`| `int` | `None` | Limit batches per epoch for rapid verification |

#### Example CLI Commands:
```bash
# Standard training run
python backend/train.py --epochs 10 --batch-size 32

# Multi-dataset fine-tuning run
python backend/train.py --all-datasets --epochs 15 --fine-tune-lr 5e-6

# Quick smoke test (verify pipeline end-to-end in 1 minute)
python backend/train.py --epochs 1 --max-batches-per-epoch 5
```

---

## ⚡ Quick Start Guide

### Prerequisites
- Python 3.9+ (Python 3.10–3.13 supported)
- WebGL-enabled web browser (Chrome, Edge, Firefox, Safari)

### 1. Clone the Repository
```bash
git clone https://github.com/luckylilawat09-lgtm/DeepFake_Detection.git
cd DeepFake_Detection
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Run Automated Tests
```bash
python -m pytest backend/tests -v
```

### 5. Launch Backend Server
```bash
python backend/app.py
```
*API will listen on `http://127.0.0.1:5000`.*

### 6. Launch Frontend UI
In a separate terminal:
```bash
cd frontend
python -m http.server 8000
```
Open your browser to: **`http://localhost:8000`**

---

## 📡 API Specification

### 1. Health Check
`GET /`
```json
{
  "status": "healthy",
  "message": "DeepFake Detection API is running",
  "model_loaded": true
}
```

### 2. Biometric Prediction
`POST /predict`
- **Content-Type:** `multipart/form-data`
- **Body:** `file`: `<image binary>` (supports PNG, JPG, JPEG, WEBP up to 16MB)

**Response:**
```json
{
  "label": "Fake",
  "confidence": 0.9421,
  "is_real": false,
  "using_trained_model": true
}
```

### 3. Live Training Telemetry
`GET /training-status`
**Response:**
```json
{
  "status": "completed",
  "epoch": 10,
  "total_epochs": 10,
  "batch": 2500,
  "total_batches": 2500,
  "accuracy": 95.66,
  "loss": 0.1191,
  "model_loaded": true,
  "model_ready": true,
  "phase": "Completed",
  "phase_idx": 4,
  "phases": [
    { "id": 1, "title": "Data Ingestion & Augmentation", "desc": "57,589 images indexed across train/val/test", "status": "completed" },
    { "id": 2, "title": "Classifier Head Convergence", "desc": "Training dense classifier layers (Epochs 1-2)", "status": "completed" },
    { "id": 3, "title": "Backbone Fine-Tuning", "desc": "Unfreezing EfficientNet-B0 backbone (Epochs 3+)", "status": "completed" },
    { "id": 4, "title": "Model Validation & Checkpoint", "desc": "Benchmark on 12,360 validation images & export .pth", "status": "completed" }
  ]
}
```

> [!NOTE]
> **Metric Distinction:** The `"accuracy"` field in `/training-status` telemetry reflects the **validation accuracy** achieved on the validation partition (95.66%) during training. Unbiased evaluation on the held-out test split yields a **test accuracy** of 78.59% (see [evaluation_results.txt](evaluation_results.txt)).

### 4. Human Feedback (Active Learning)
`POST /feedback`
- **Content-Type:** `application/json`
- **Body:**
```json
{
  "prediction": "Real",
  "confidence": 0.85,
  "is_correct": true,
  "user_label": "Real"
}
```

`GET /feedback`
**Response:**
```json
{
  "total": 38,
  "correct": 36,
  "incorrect": 2,
  "recent": [ ... ]
}
```

### 5. Hot Reload Model
`POST /reload-model`
```json
{
  "success": true,
  "model_loaded": true,
  "message": "Model reloaded successfully"
}
```

---

## 🧪 Automated Testing & QA

The project includes unit and integration tests powered by `pytest`:

```bash
python -m pytest backend/tests -v
```

### Test Coverage Summary (20 Tests):
- **`test_api.py` (10 tests):**
  - Root health check endpoint (`/`) validation
  - Prediction endpoint (`/predict`) with valid PNG/JPG tensors
  - Ephemeral file cleanup verification
  - Error handling for missing files, empty filenames, invalid extensions, corrupted buffers
  - Training status telemetry schema validation (`/training-status`)
  - Active learning human feedback submission (`POST /feedback`) and retrieval (`GET /feedback`)
  - Mock model fallback handling when weights are not loaded (`using_trained_model: false`)
  - Model hot-reloading endpoint (`POST /reload-model`)
- **`test_model.py` (5 tests):**
  - EfficientNet-B0 model initialization and custom binary head architecture
  - Single and batch tensor forward-pass output dimension tests ($1 \times 2$ and $B \times 2$)
  - Feature backbone unfreezing and differential learning rate assignment
  - Dual dropout regularizer presence and dropout rate verification ($p=0.3$, $p=0.2$)
  - Checkpoint state loading and inference verification from `models/deepfake_detector.pth`
- **`test_train_utils.py` (5 tests):**
  - Atomic training telemetry persistence (`training_progress.json`)
  - Dataset split directory discovery for direct structures
  - Exception handling for non-existent dataset paths
  - Torchvision ImageNet transform normalization bounds verification
  - CLI argument parser flags, defaults, and overrides (`train.parse_args`)

---

## 📁 Repository Structure

```text
DeepFake_Detection/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                      # GitHub Actions CI automated testing pipeline
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md               # Standardized bug reporting template
│   │   └── feature_request.md          # Feature proposal template
│   └── PULL_REQUEST_TEMPLATE.md        # Pull request checklist & review schema
│
├── backend/
│   ├── .env.example                    # Environment variable configuration guide
│   ├── app.py                          # Flask REST API server (inference & active learning)
│   ├── train.py                        # CLI PyTorch training & validation engine
│   ├── requirements.txt                # Python dependencies
│   ├── models/
│   │   ├── deepfake_detector.pth       # Trained EfficientNet-B0 weights
│   │   ├── feedback_log.json           # Active learning human-in-the-loop audit log
│   │   └── training_progress.json      # Live training telemetry progress
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_api.py                 # API integration tests
│   │   ├── test_model.py               # Deep learning model tests
│   │   └── test_train_utils.py         # Training utilities & transforms tests
│   └── uploads/
│       └── .gitkeep                    # Ephemeral upload staging directory
│
├── frontend/
│   └── index.html                      # 3D WebGL Chromatic Glass SPA interface
│
├── .gitignore                          # Git exclusion rules
├── CODE_OF_CONDUCT.md                  # Contributor Covenant Code of Conduct
├── CONTRIBUTING.md                     # Contributor guidelines & developer workflow
├── LICENSE                             # MIT Open-Source License
├── PROJECT_REPORT.md                   # Technical research & forensic analysis report
├── README.md                           # Main documentation & developer portal
├── evaluation_results.txt              # Complete empirical evaluation metrics output
└── WALKTHROUGH.md                      # Feature walkthrough and setup guide
```

---

## ⚠️ Limitations & Generalization Scope

> [!IMPORTANT]
> **Dataset & Architecture Scope:**
> The model was trained and evaluated strictly on facial portraits from the Kaggle 140k dataset (comparing NVIDIA **StyleGAN** synthesized faces against authentic **Flickr-Faces-HQ (FFHQ)** portraits).
>
> **Known Generalization Constraints:**
> - **Other DeepFake & Generative Methods:** The model may not generalize reliably to other synthetic media synthesis pipelines, such as Latent Diffusion Models (Stable Diffusion, Midjourney, FLUX, Imagen), video-level face-swapping (e.g., DeepFaceLab, SimSwap, RoOP), or facial expression reenactment / neural puppet drivers.
> - **Social Media Compression:** Heavy JPEG compression and resizing applied by social platforms act as high-frequency low-pass filters, dampening the subtle GAN convolution residuals that the classifier leverages.
> - **Domain Shift:** Unseen lighting conditions, extreme facial poses, occlusions (glasses, masks), or non-portrait imagery outside the FFHQ domain may result in lower confidence or false positives.

---

## 🔒 Security & Privacy Notice

- **Ephemeral File Handling:** Uploaded media is processed directly in RAM and immediately removed from the file system within `finally` blocks.
- **Path Traversal Protection:** Input filenames are sanitized using `werkzeug.utils.secure_filename`.
- **Resource Protection:** Upload limits are enforced (16 MB cap), and inference runs strictly inside `torch.no_grad()` contexts to prevent memory growth.

---

## 📚 References & Acknowledgments

- **UI Prompt & Concept:** [motionsites.ai/?prompt=design-world](https://motionsites.ai/?prompt=design-world)
- **Face Dataset:** [Kaggle - 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Backbone Architecture:** [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)
- **Generative Sources:** [Flickr-Faces-HQ (FFHQ)](https://github.com/NVlabs/ffhq-dataset) & [StyleGAN](https://github.com/NVlabs/stylegan) (NVIDIA Research)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).