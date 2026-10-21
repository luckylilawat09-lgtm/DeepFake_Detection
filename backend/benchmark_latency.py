"""Latency and Inference Benchmarking Script for DeepFake Detection Model.

Measures:
1. Model-only inference time (forward pass) over 100 iterations (10 warm-up).
2. End-to-end /predict API latency over 100 iterations (10 warm-up).
3. Hardware environment specifications (CPU model, GPU model, RAM, PyTorch device).
"""

import io
import json
import os
import platform
import subprocess
import sys
import time
import statistics
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn

# Add backend directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def build_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )
    return model


def get_hardware_info():
    """Query CPU model, GPU model, and RAM from system and PyTorch."""
    cpu_name = platform.processor() or "Unknown CPU"
    total_ram_gb = "Unknown RAM"
    gpu_name = "None detected"

    # Windows CIM / WMI query
    try:
        cpu_cmd = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=5
        )
        if cpu_cmd.returncode == 0 and cpu_cmd.stdout.strip():
            cpu_name = cpu_cmd.stdout.strip().splitlines()[0]
    except Exception:
        pass

    try:
        ram_cmd = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "([math]::Round((Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1GB, 2)).ToString() + ' GB'"],
            capture_output=True, text=True, timeout=5
        )
        if ram_cmd.returncode == 0 and ram_cmd.stdout.strip():
            total_ram_gb = ram_cmd.stdout.strip()
    except Exception:
        pass

    try:
        gpu_cmd = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ', '"],
            capture_output=True, text=True, timeout=5
        )
        if gpu_cmd.returncode == 0 and gpu_cmd.stdout.strip():
            gpu_name = gpu_cmd.stdout.strip()
    except Exception:
        pass

    cuda_available = torch.cuda.is_available()
    torch_device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (PyTorch build lacks CUDA or no CUDA device)"

    return {
        "cpu_model": cpu_name,
        "ram": total_ram_gb,
        "system_gpu": gpu_name,
        "cuda_available": cuda_available,
        "torch_device": torch_device_name,
        "os": f"{platform.system()} {platform.release()}"
    }


def benchmark_model_only(model, device, num_warmup=10, num_runs=100):
    """Benchmark raw model forward pass latency on a tensor."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Benchmark runs
    durations_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            durations_ms.append((t1 - t0) * 1000.0)

    return {
        "runs": num_runs,
        "warmup": num_warmup,
        "mean_ms": round(statistics.mean(durations_ms), 3),
        "std_ms": round(statistics.stdev(durations_ms), 3),
        "median_ms": round(statistics.median(durations_ms), 3),
        "min_ms": round(min(durations_ms), 3),
        "max_ms": round(max(durations_ms), 3),
        "p95_ms": round(sorted(durations_ms)[int(0.95 * len(durations_ms))], 3),
    }


def benchmark_end_to_end_predict(image_bytes, num_warmup=10, num_runs=100, api_url="http://127.0.0.1:5000/predict"):
    """Benchmark end-to-end HTTP /predict latency against the Flask server."""
    import requests

    session = requests.Session()

    # Check connection
    try:
        r = session.post(api_url, files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}, timeout=10)
        r.raise_for_status()
        live_mode = True
    except Exception as e:
        print(f"[*] Live API request failed ({e}). Falling back to in-process Flask test_client.")
        live_mode = False

    durations_ms = []

    if live_mode:
        # Warm-up
        for _ in range(num_warmup):
            session.post(api_url, files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}, timeout=10)

        # Timed runs
        for _ in range(num_runs):
            t0 = time.perf_counter()
            resp = session.post(api_url, files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}, timeout=10)
            t1 = time.perf_counter()
            durations_ms.append((t1 - t0) * 1000.0)
    else:
        import app
        app.app.config["TESTING"] = True
        client = app.app.test_client()

        # Warm-up
        for _ in range(num_warmup):
            client.post("/predict", data={"file": (io.BytesIO(image_bytes), "test.jpg")}, content_type="multipart/form-data")

        # Timed runs
        for _ in range(num_runs):
            t0 = time.perf_counter()
            resp = client.post("/predict", data={"file": (io.BytesIO(image_bytes), "test.jpg")}, content_type="multipart/form-data")
            t1 = time.perf_counter()
            durations_ms.append((t1 - t0) * 1000.0)

    return {
        "mode": "Live HTTP" if live_mode else "In-process test_client",
        "runs": num_runs,
        "warmup": num_warmup,
        "mean_ms": round(statistics.mean(durations_ms), 3),
        "std_ms": round(statistics.stdev(durations_ms), 3),
        "median_ms": round(statistics.median(durations_ms), 3),
        "min_ms": round(min(durations_ms), 3),
        "max_ms": round(max(durations_ms), 3),
        "p95_ms": round(sorted(durations_ms)[int(0.95 * len(durations_ms))], 3),
    }


def main():
    print("=" * 70)
    print(" DeepFake Detection Latency & Inference Benchmark")
    print("=" * 70)

    # 1. Hardware Info
    hw = get_hardware_info()
    print("\n[1] HARDWARE ENVIRONMENT:")
    print(f"  CPU Model        : {hw['cpu_model']}")
    print(f"  System RAM       : {hw['ram']}")
    print(f"  GPU(s) Detected  : {hw['system_gpu']}")
    print(f"  PyTorch CUDA Avail: {hw['cuda_available']}")
    print(f"  PyTorch Device   : {hw['torch_device']}")
    print(f"  Operating System : {hw['os']}")

    # 2. Load Checkpoint
    checkpoint_path = os.path.join(SCRIPT_DIR, "models", "deepfake_detector.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2] LOADING CHECKPOINT FROM: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Model loaded successfully on device: {device.type.upper()}")

    # 3. Model-Only Latency on CPU
    print("\n[3] BENCHMARKING MODEL-ONLY INFERENCE (CPU, 10 warmup + 100 runs)...")
    model_cpu = model.to("cpu")
    cpu_model_bench = benchmark_model_only(model_cpu, torch.device("cpu"), num_warmup=10, num_runs=100)
    print(f"  Mean: {cpu_model_bench['mean_ms']} ms +/- {cpu_model_bench['std_ms']} ms")
    print(f"  Median: {cpu_model_bench['median_ms']} ms | P95: {cpu_model_bench['p95_ms']} ms | Min: {cpu_model_bench['min_ms']} ms | Max: {cpu_model_bench['max_ms']} ms")

    gpu_model_bench = None
    if torch.cuda.is_available():
        print("\n[3b] BENCHMARKING MODEL-ONLY INFERENCE (GPU, 10 warmup + 100 runs)...")
        model_gpu = model.to("cuda")
        gpu_model_bench = benchmark_model_only(model_gpu, torch.device("cuda"), num_warmup=10, num_runs=100)
        print(f"  Mean: {gpu_model_bench['mean_ms']} ms +/- {gpu_model_bench['std_ms']} ms")
        print(f"  Median: {gpu_model_bench['median_ms']} ms | P95: {gpu_model_bench['p95_ms']} ms")
    else:
        print("\n[3b] GPU inference benchmark skipped: PyTorch CUDA is not available.")

    # 4. End-to-End Latency
    print("\n[4] BENCHMARKING END-TO-END /predict API (10 warmup + 100 runs)...")
    # Load real image if available, else synthetic 224x224 RGB jpeg
    sample_img_path = r"C:\Users\lucky\Downloads\archive\Data Set 1\Data Set 1\test\fake\fake_4243.jpg"
    if os.path.exists(sample_img_path):
        with open(sample_img_path, "rb") as f:
            test_img_bytes = f.read()
        print(f"  Using test dataset image: {sample_img_path} ({len(test_img_bytes)} bytes)")
    else:
        img = Image.new("RGB", (224, 224), color=(140, 180, 220))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        test_img_bytes = buf.getvalue()
        print(f"  Using generated 224x224 JPEG image ({len(test_img_bytes)} bytes)")

    e2e_bench = benchmark_end_to_end_predict(test_img_bytes, num_warmup=10, num_runs=100)
    print(f"  Mode: {e2e_bench['mode']}")
    print(f"  Mean: {e2e_bench['mean_ms']} ms +/- {e2e_bench['std_ms']} ms")
    print(f"  Median: {e2e_bench['median_ms']} ms | P95: {e2e_bench['p95_ms']} ms | Min: {e2e_bench['min_ms']} ms | Max: {e2e_bench['max_ms']} ms")

    # 5. Save JSON report
    report = {
        "hardware": hw,
        "model_only_cpu": cpu_model_bench,
        "model_only_gpu": gpu_model_bench,
        "end_to_end_predict": e2e_bench,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_json = os.path.join(SCRIPT_DIR, "models", "latency_results.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[5] Benchmark results saved to: {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
