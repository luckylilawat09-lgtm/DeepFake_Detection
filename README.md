# 🔮 DeepFake Biometric Detection & Neural Telemetry Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-000000?style=for-the-badge&logo=flask&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-WebGL%203D-black?style=for-the-badge&logo=three.js&logoColor=white)
![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle%20140k%20Faces-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<p align="center">
  <b>A state-of-the-art synthetic media detection platform featuring an EfficientNet-B0 convolutional neural network paired with an interactive 3D chromatic glass refraction interface.</b>
</p>

[Key Features](#-key-features) •
[UI & Design Origin](#-ui-frontend-architecture--prompt-credit) •
[Dataset Information](#-kaggle-dataset--data-engineering) •
[Model Architecture](#-deep-learning-model--training) •
[Quick Start](#-quick-start-guide) •
[API Reference](#-api-specification) •
[Project Report](PROJECT_REPORT.md)

</div>

---

## 📌 Executive Summary

With the exponential surge of hyper-realistic generative models (GANs, Diffusion, Face-Swapping), detecting digitally manipulated facial media has become a cornerstone of biometric security and digital authenticity.

This repository provides an **end-to-end, production-grade DeepFake detection ecosystem**:
1. **Core AI Engine:** A fine-tuned **EfficientNet-B0** binary classifier built in **PyTorch**, trained on real versus GAN-synthesized human faces.
2. **Visual Telemetry Frontend:** An interactive, cyber-aesthetic 3D viewport inspired by cutting-edge WebGL aesthetics, providing live biometric analysis and neural pipeline metrics.
3. **High-Throughput API:** A lightweight Flask server with automated upload sanitation, CORS headers, and real-time training telemetry polling.

---

## 🎨 UI / Frontend Architecture & Prompt Credit

The user interface is designed to transform complex computer vision analysis into an intuitive, visually captivating experience.

### 🌟 Design Foundation & motionsites.ai Prompt
The visual identity, dark chromatic aesthetic, and glassmorphic layout were engineered using the prompt from **[motionsites.ai](https://motionsites.ai/)**:
> **Template Prompt:** [`https://motionsites.ai/?prompt=design-world`](https://motionsites.ai/?prompt=design-world)

### ✨ Key Interface Capabilities
- **3D Chromatic Glass Refraction Cube:** Powered by Three.js with custom GLSL shaders executing a **2-pass refraction pipeline** (`rtBack` background depth map $\rightarrow$ inner specimen geometry $\rightarrow$ `rtFront` chromatic dispersion).
- **Interactive Biometric Dropzone:** Users can drag-and-drop or upload suspect portraits directly onto the 3D specimen stage.
- **Dynamic Laser Scanning Effect:** A holographic laser-sweep shader traverses the specimen during inference with real-time biometric telemetry.
- **Live Neural Pipeline HUD:** Real-time metrics streaming displaying training phase milestones, epoch progression, batch loss, and accuracy gauges.
- **Responsive Cyber Aesthetics:** Built with vanilla CSS, glassmorphism filters, Poppins typography, and JetBrains Mono code displays.

---

## 📊 Kaggle Dataset & Data Engineering

The deep learning model is trained on the benchmark **140k Real and Fake Faces** collection sourced from Kaggle:
- **Dataset Link:** [Kaggle: 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Real Imagery Source:** 70,000 authentic portraits from Nvidia’s **Flickr-Faces-HQ (FFHQ)** dataset.
- **Fake Imagery Source:** 70,000 synthesized portraits produced by Nvidia’s **StyleGAN** generative adversarial network.
- **Resolution & Formats:** High-resolution crops standard-sized to 256×256 pixels, downsampled and normalized to 224×224 for convolutional feature extraction.

### Data Partitioning (`Data Set 1`)
| Partition | Real Faces | Fake (GAN) Faces | Total Samples | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 20,001 | 20,001 | **40,002** | Primary supervised parameter optimization |
| **Validation** | 5,000 | 5,000 | **10,000** | Epoch-level generalization monitoring & checkpointing |
| **Test** | 3,793 | 3,794 | **7,587** | Final unbiased model performance evaluation |
| **Total** | **28,794** | **28,795** | **57,589** | Balanced distribution preventing class bias |

### Data Augmentation Pipeline
To generalize beyond StyleGAN artifacts and prevent overfitting on skin smoothing:
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 🧠 Deep Learning Model & Training

### Backbone Architecture
- **Base Model:** `EfficientNet-B0` (Pretrained on ImageNet-1K)
- **Classifier Head:**
  ```text
  Input (1280 features) ──► Dropout(p=0.3) ──► Linear(1280, 256) ──► ReLU ──► Dropout(p=0.2) ──► Linear(256, 2)
  ```

### Training Strategy: Progressive Two-Stage Fine-Tuning
1. **Stage 1 — Classifier Warmup (Epochs 1–2):**
   The convolutional feature extractor backbone is frozen (`requires_grad = False`). Only the custom 256-node classification head is optimized using `Adam` at a learning rate of `1e-3`.
2. **Stage 2 — Backbone Fine-Tuning (Epochs 3+):**
   The deep convolutional stages are unfrozen (`requires_grad = True`) and trained jointly at a conservative learning rate of `1e-5` with `weight_decay = 1e-4` to adapt high-level feature maps to frequency/grid artifacts unique to GAN faces.
3. **Loss Function:** Cross-Entropy Loss with Softmax.
4. **Early Stopping:** Monitored on validation loss with patience $p = 5$.

### Checkpoint Specifications
- **Checkpoint Location:** `backend/models/deepfake_detector.pth` (~17.6 MB)
- **Validation Accuracy:** **76.69%** (Single-fold validation on diverse real vs. StyleGAN subsets)

---

## 📁 Repository Structure

```text
DeepFake-Detection/
├── .gitignore                      # Python, uploads, environment, and OS exclusion rules
├── LICENSE                         # MIT Open-Source License
├── CONTRIBUTING.md                 # Contribution standards and PR guidelines
├── README.md                       # Comprehensive documentation
├── PROJECT_REPORT.md               # In-depth technical & forensic research report
│
├── backend/                        # Server and deep learning codebase
│   ├── app.py                      # Flask REST API server (inference & telemetry)
│   ├── train.py                    # PyTorch training & validation pipeline
│   ├── requirements.txt            # Python dependencies (PyTorch, Flask, PIL)
│   ├── models/
│   │   ├── .gitkeep                # Directory anchor
│   │   └── deepfake_detector.pth   # Trained EfficientNet-B0 weights (~17.6MB)
│   └── uploads/
│       └── .gitkeep                # Ephemeral upload staging (auto-sanitized)
│
└── frontend/                       # Interactive WebGL client application
    └── index.html                  # Standalone SPA with 3D glass cube & telemetry
```

---

## ⚡ Quick Start Guide

### Prerequisites
- Python 3.9+ (Python 3.10/3.11 recommended)
- `pip` package manager
- Modern WebGL-capable browser (Chrome, Edge, Brave, Firefox)

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/DeepFake-Detection.git
cd DeepFake-Detection
```

### 2. Backend Setup & Run
Create a virtual environment and launch the Flask server:
```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Start the Flask API
python app.py
```
*The server will boot on `http://localhost:5000` and automatically load `deepfake_detector.pth` if present (or cleanly fall back to demonstration mode).*

### 3. Launch Frontend Client
In a separate terminal, serve the frontend:
```bash
cd frontend
python -m http.server 8080
```
Open your browser to: **`http://localhost:8080`**

---

## 📡 API Specification

### 1. Image Classification Inference
`POST /predict`
- **Content-Type:** `multipart/form-data`
- **Body:** `file`: `<binary image data>` (Supports `.png`, `.jpg`, `.jpeg`, `.gif` up to 16MB)

**Response:**
```json
{
  "label": "Fake",
  "confidence": 0.9421,
  "is_real": false,
  "using_trained_model": true
}
```

### 2. Real-Time Telemetry & Status
`GET /training-status`
- **Content-Type:** `application/json`

**Response:**
```json
{
  "status": "training",
  "epoch": 2,
  "total_epochs": 10,
  "batch": 1250,
  "total_batches": 2500,
  "accuracy": 76.69,
  "loss": 0.4812,
  "model_loaded": true,
  "model_ready": true,
  "phases": [
    { "id": 1, "title": "Data Ingestion & Augmentation", "status": "completed" },
    { "id": 2, "title": "Classifier Head Convergence", "status": "completed" },
    { "id": 3, "title": "Backbone Fine-Tuning", "status": "completed" },
    { "id": 4, "title": "Model Validation & Checkpoint", "status": "completed" }
  ]
}
```

---

## 🔒 Security & Privacy Notice

- **Ephemeral Uploads:** Image uploads submitted via the `/predict` endpoint are immediately unlinked (`os.remove`) after tensor generation. No private biometric data is persisted on disk.
- **Sanitized Filenames:** Filenames are normalized with `werkzeug.utils.secure_filename` to prevent path traversal vulnerabilities.
- **Memory Footprint:** In-memory inference executes inside a `torch.no_grad()` context to minimize VRAM/RAM overhead.

---

## 📚 References & Acknowledgments

- **UI Prompt & Concept:** [motionsites.ai/?prompt=design-world](https://motionsites.ai/?prompt=design-world)
- **Face Dataset:** [Kaggle - 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Backbone Architecture:** [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)
- **Generative Sources:** [Flickr-Faces-HQ (FFHQ)](https://github.com/NVlabs/ffhq-dataset) & [StyleGAN](https://github.com/NVlabs/stylegan) (NVIDIA Research)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free for educational, research, and commercial applications.