# Contributing to DeepFake Detection System

Thank you for your interest in contributing to the **DeepFake Biometric Detection & Neural Pipeline**! We welcome contributions from machine learning practitioners, computer vision engineers, and frontend developers.

---

## 🛠️ Code of Conduct

This project is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior following the guidelines in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## 🚀 Getting Started

1. **Fork the Repository** to your own GitHub account.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/DeepFake_Detection.git
   cd DeepFake_Detection
   ```
3. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

---

## 📌 Development Guidelines

### Model & Training (`backend/`)
- Adhere to PyTorch best practices: maintain clean tensor operations, support both CUDA and CPU execution gracefully, and ensure deterministic data loading where applicable.
- Any changes to `train.py` should support the CLI arguments (`--data-dir`, `--epochs`, `--batch-size`, `--lr`, `--fine-tune-lr`, etc.) and preserve checkpoint compatibility.
- Keep inference latency low. Ensure preprocessing transformations match between `train.py` and `app.py`.

### API & Server (`backend/app.py`)
- Endpoints must return consistent JSON schemas.
- Always include automated file cleanup in upload handlers (`finally: os.remove(filepath)`) to prevent disk leaks.
- Maintain mock model fallbacks so frontend testing is possible without GPU hardware or pre-downloaded model checkpoints.
- Ensure feedback logging and progress tracking safely handle concurrent access and missing files.

### Frontend & 3D Shaders (`frontend/index.html`)
- Preserve the 60fps performance budget on WebGL/Canvas rendering.
- Keep shaders cleanly documented with uniform references.
- Ensure responsive scaling across desktop, tablet, and mobile viewports.

---

## 🧪 Testing Your Changes

Before submitting a Pull Request, verify that all automated unit and integration tests pass:

1. **Run the Test Suite**:
   ```bash
   python -m pytest backend/tests -v
   ```
   Ensure all 14 unit and integration tests pass without failures.

2. **Verify Backend Startup**:
   ```bash
   python backend/app.py
   ```

3. **Test Prediction & Feedback Endpoints**:
   ```bash
   # Predict
   curl -X POST -F "file=@sample.jpg" http://127.0.0.1:5000/predict

   # Health Check
   curl http://127.0.0.1:5000/

   # Telemetry Status
   curl http://127.0.0.1:5000/training-status

   # Human Feedback Loop
   curl -X POST -H "Content-Type: application/json" -d '{"prediction":"Real","confidence":0.85,"is_correct":true,"user_label":"Real"}' http://127.0.0.1:5000/feedback
   ```

4. **Verify Frontend**:
   Open `frontend/index.html` in modern browsers (Chrome, Firefox, Safari, Edge) and confirm no JavaScript errors in DevTools console.

---

## 📝 Pull Request Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Commit your changes using Conventional Commits:
   ```bash
   git commit -m "feat(pipeline): add gradient-weighted class activation mapping (Grad-CAM)"
   ```
3. Push to your fork:
   ```bash
   git push origin feat/your-feature-name
   ```
4. Open a Pull Request using the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).

Thank you for helping push forward the state of the art in synthetic media detection!
