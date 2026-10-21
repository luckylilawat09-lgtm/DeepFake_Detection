# DeepFake Detection System — Complete Walkthrough Guide

An end-to-end walkthrough for setting up, running, testing, and understanding the **DeepFake Biometric Detection & Telemetry System**.

---

## 📑 Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Setup & Launch](#step-by-step-setup--launch)
   - [Step 1: Backend Setup](#step-1-backend-setup)
   - [Step 2: Starting the Flask Server](#step-2-starting-the-flask-server)
   - [Step 3: Launching the Frontend](#step-3-launching-the-frontend)
4. [User Interface & Detection Walkthrough](#user-interface--detection-walkthrough)
   - [3D Chromatic Glass Analyzer](#3d-chromatic-glass-analyzer)
   - [Image Upload & Scan Workflow](#image-upload--scan-workflow)
   - [Live Biometric Pipeline Phases](#live-biometric-pipeline-phases)
   - [Verdict & Active Learning Feedback Loop](#verdict--active-learning-feedback-loop)
5. [Training & Neural Weights Pipeline](#training--neural-weights-pipeline)
6. [API Endpoints Reference](#api-endpoints-reference)
7. [Automated Testing](#automated-testing)
8. [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## System Architecture Overview

The system consists of two primary layers connected via a lightweight REST API:

```
┌────────────────────────────────────────────────────────┐
│             FRONTEND (Three.js + HTML5)               │
│  - 6-Band Chromatic Dispersion Refraction Shader Cube  │
│  - Floating Draggable / Resizable Telemetry HUD        │
│  - Real-time Laser Scan & Artifact Visualization       │
└───────────────────────────▲────────────────────────────┘
                            │ HTTP / JSON (Port 5000)
┌───────────────────────────▼────────────────────────────┐
│              BACKEND (Flask + PyTorch)                 │
│  - Flask 2.3 API Server with CORS Enabled              │
│  - EfficientNet-B0 Deep Learning Model                 │
│  - Automatic Fallback to Mock Inference Engine         │
│  - Active Learning Feedback Logger                     │
└────────────────────────────────────────────────────────┘
```

- **Frontend (`frontend/index.html`)**: Interactive WebGL 3D glass cuboid with refraction optics, interactive drag controls, floating HUD dashboard, and real-time biometric progress indicators.
- **Backend (`backend/app.py`)**: Flask web service handling multipart image ingestion, neural tensor preprocessing, model inference, training metrics, and user feedback.
- **Model Engine (`backend/models/`)**: PyTorch EfficientNet-B0 architecture trained to distinguish synthetic/manipulated facial imagery from authentic captures.

---

## Prerequisites

- **Python**: Version `3.9` to `3.13`
- **Package Manager**: `pip`
- **Web Browser**: Modern browser with WebGL hardware acceleration enabled (Chrome, Edge, Firefox, Brave)
- **Optional**: VS Code with the *Live Server* extension

---

## Step-by-Step Setup & Launch

### Step 1: Backend Setup

1. **Open PowerShell or Command Prompt** and navigate to the backend folder:
   ```powershell
   cd "backend"
   ```

2. **Create a Python Virtual Environment**:
   ```powershell
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - **PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
     *(If script execution is disabled, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*
   - **Command Prompt (CMD)**:
     ```cmd
     venv\Scripts\activate.bat
     ```

4. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

5. **Initialize Environment Variables**:
   ```powershell
   Copy-Item .env.example .env
   ```

---

### Step 2: Starting the Flask Server

Run the Flask application:

```powershell
python app.py
```

You should see output similar to:
```text
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

> [!NOTE]
> If a trained `.pth` weight file is not yet found in `models/deepfake_detector.pth`, the backend will display a warning and automatically engage the high-fidelity mock inference engine so you can test all UI workflows immediately.

---

### Step 3: Launching the Frontend

You can run the frontend in any of the following ways:

#### Option A: Using Python's Built-in HTTP Server (Recommended)
Open a second terminal window:
```powershell
cd "frontend"
python -m http.server 8000
```
Then visit **[http://localhost:8000](http://localhost:8000)** in your browser.

#### Option B: Using VS Code Live Server
1. Open the project in VS Code.
2. Right-click [`frontend/index.html`](file:///c:/Users/lucky/FCC/DeepFake%20Detection/frontend/index.html).
3. Click **"Open with Live Server"**.

#### Option C: Direct File Opening
Double-click [`frontend/index.html`](file:///c:/Users/lucky/FCC/DeepFake%20Detection/frontend/index.html) in Windows File Explorer or open it in your browser.

---

## User Interface & Detection Walkthrough

### 3D Chromatic Glass Analyzer
- **Interactive Cube**: Click and drag on the 3D cube to rotate and inspect the refraction angles.
- **Controls**: Use the bottom arrows (`<` and `>`) or pagination dots to rotate the specimen chamber in 90° increments.
- **Floating Dashboard**:
  - Drag the header to position the HUD anywhere on screen.
  - Use the zoom buttons (`+`, `-`) or slider to scale the dashboard between 50% and 150%.
  - Click the **Minimize** button to collapse into a compact widget.

### Image Upload & Scan Workflow
1. **Trigger Upload**:
   - Drag & drop an image directly onto the hero section, OR
   - Click the **"UPLOAD IMAGE"** CTA button, OR
   - Click the interactive pill button on the 3D cube.
2. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`.

### Live Biometric Pipeline Phases
Once an image is loaded, the HUD switches to the **Biometric Pipeline** view and displays step-by-step progress:
- **Phase B1 (Ingestion & Normalization)**: Decodes spatial resolution and builds tensor representations.
- **Phase B2 (Artifact Scan)**: Extracts multi-scale neural texture anomalies, edge blending artifacts, and synthesis boundaries.
- **Phase B3 (Neural Backbone Propagation)**: Passes features through the EfficientNet-B0 network.
- **Phase B4 (Verdict Classification)**: Outputs final authenticity score and confidence level.

The 3D glass cuboid runs a dual-laser scan animation over the specimen hologram during inference.

### Verdict & Active Learning Feedback Loop
- **Real Image Verdict**: The cube and HUD pulse green (`#00ff88`) with label **"REAL"**.
- **DeepFake Verdict**: The cube and HUD emit a crimson alarm glow (`#ff3366`) with label **"DEEPFAKE"**.
- **Active Learning**: Click **"Yes (Accurate)"** or **"No (Incorrect)"** in the feedback panel to submit telemetry to `/feedback`. This logs user validation data to `models/feedback_log.json` for retraining datasets.

---

## Training & Neural Weights Pipeline

To train the EfficientNet-B0 neural model on a dataset:

1. Ensure the virtual environment is activated:
   ```powershell
   cd "backend"
   .\venv\Scripts\Activate.ps1
   ```

2. Run the training script:
   ```powershell
   python train.py
   ```

3. **Telemetry Tracking**:
   - `train.py` writes live progress (loss, accuracy, epoch) to `models/training_progress.json`.
   - The frontend HUD **"Training Telemetry"** tab polls `/training-status` every 2.5 seconds to visualize live training curves.
   - Upon completion, the model weights are saved to `models/deepfake_detector.pth`.

4. **Hot-Reload Weights**:
   Reload the new weights into the running server without restarting:
   ```powershell
   curl -X POST http://127.0.0.1:5000/reload-model
   ```

---

## API Endpoints Reference

| Method | Endpoint | Description | Request Body | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Health check & model status | None | `{"status": "healthy", "model_loaded": bool}` |
| `POST` | `/predict` | Classify image for deepfakes | Multipart `file` | `{"label": "Real"/"Fake", "confidence": float, "is_real": bool}` |
| `GET` | `/training-status` | Get training progress metrics | None | JSON containing loss, accuracy, epochs, phases |
| `POST` | `/feedback` | Submit classification verification | `{"prediction": str, "is_correct": bool}` | `{"success": true, "total_feedbacks": int}` |
| `GET` | `/feedback` | Retrieve aggregate feedback stats | None | Summary of correct vs incorrect feedback |
| `POST` | `/reload-model` | Reload weights into memory | None | `{"success": bool, "model_loaded": bool}` |

---

## Automated Testing

Run the test suite to verify backend routes and model inference logic:

```powershell
cd "backend"
.\venv\Scripts\Activate.ps1
pytest tests -v
```

---

## Troubleshooting & FAQs

### 1. Frontend shows "Backend Offline"
- Ensure `python app.py` is actively running in the backend terminal.
- Verify `http://127.0.0.1:5000/` returns `{"status": "healthy"}` in your browser.
- If running on a non-standard port, configure `window.API_BASE_URL` in [`frontend/index.html`](file:///c:/Users/lucky/FCC/DeepFake%20Detection/frontend/index.html).

### 2. "Execution of scripts is disabled on this system" (PowerShell)
Run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then rerun `.\venv\Scripts\Activate.ps1`.

### 3. Port 5000 is already in use
Check what process is using port 5000:
```powershell
netstat -ano | findstr :5000
```
Terminate the conflicting process or change `PORT=5001` in your `.env` file and update `app.py`.

---
*Created for the DeepFake Biometric Detection & Telemetry Platform.*
