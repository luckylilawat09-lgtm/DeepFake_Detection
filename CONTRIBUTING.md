# Contributing to DeepFake Detection System

Thank you for your interest in contributing to the **DeepFake Biometric Detection & Neural Pipeline**! We welcome contributions from machine learning practitioners, computer vision engineers, and frontend developers.

---

## 🛠️ Code of Conduct

We are dedicated to providing a welcoming, respectful, and harassment-free environment for everyone. Please treat all contributors with kindness, constructive feedback, and mutual respect.

---

## 🚀 Getting Started

1. **Fork the Repository** to your own GitHub account.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/DeepFake-Detection.git
   cd DeepFake-Detection
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
- Any changes to `train.py` should maintain backward compatibility with `deepfake_detector.pth` or detail migration strategies in the pull request.
- Keep inference latency low. Ensure preprocessing transformations match between `train.py` and `app.py`.

### API & Server (`backend/app.py`)
- Endpoints should return consistent JSON schemas.
- Always include automated file cleanup in upload handlers to prevent persistent storage of temporary biometric files.
- Maintain mock model fallbacks so frontend testing is possible without GPU hardware or pre-downloaded model checkpoints.

### Frontend & 3D Shaders (`frontend/index.html`)
- Preserve the 60fps performance budget on WebGL/Canvas rendering.
- Keep shaders cleanly documented with uniform references.
- Ensure responsive scaling across desktop, tablet, and mobile viewports.

---

## 🧪 Testing Your Changes

Before submitting a Pull Request:
1. Verify backend starts without errors:
   ```bash
   python backend/app.py
   ```
2. Test the prediction endpoint with a sample image:
   ```bash
   curl -X POST -F "file=@sample.jpg" http://localhost:5000/predict
   ```
3. Verify the frontend loads properly without JavaScript console errors in modern browsers (Chrome, Firefox, Safari, Edge).

---

## 📝 Pull Request Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with meaningful commit messages:
   ```bash
   git commit -m "feat(pipeline): add gradient-weighted class activation mapping (Grad-CAM)"
   ```
3. Push to your fork and submit a Pull Request to `main`.
4. Provide a detailed PR description describing the motivation, implementation details, and screenshots/benchmarks.

Thank you for helping push forward the state of the art in synthetic media detection!
