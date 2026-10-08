import io
import json
import pytest
from PIL import Image
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import app


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client


@pytest.fixture
def sample_image_bytes():
    """Generate in-memory RGB PNG image bytes for testing."""
    img = Image.new("RGB", (224, 224), color=(120, 180, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def test_index_health_endpoint(client):
    """Verify root health check endpoint returns 200 and schema."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


def test_training_status_schema(client):
    """Verify training telemetry endpoint returns valid progress schema."""
    response = client.get("/training-status")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "epoch" in data
    assert "accuracy" in data
    assert "phases" in data
    assert isinstance(data["phases"], list)
    assert len(data["phases"]) == 4


def test_predict_no_file(client):
    """Verify error when no file is provided."""
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_empty_filename(client):
    """Verify error when file has empty filename."""
    data = {"file": (io.BytesIO(b""), "")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_invalid_extension(client):
    """Verify error when file extension is forbidden."""
    data = {"file": (io.BytesIO(b"echo 1"), "script.sh")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "extension not permitted" in response.get_json()["error"]


def test_predict_corrupt_image(client):
    """Verify error when image bytes are corrupted."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_HEADER_12345"
    data = {"file": (io.BytesIO(corrupt_bytes), "corrupt.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Invalid or corrupt image format" in response.get_json()["error"]


def test_predict_valid_image(client, sample_image_bytes):
    """Verify successful inference response with valid image."""
    data = {"file": (io.BytesIO(sample_image_bytes), "face.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_data = response.get_json()
    assert "label" in res_data
    assert res_data["label"] in ["Real", "Fake"]
    assert "confidence" in res_data
    assert 0.0 <= res_data["confidence"] <= 1.0
    assert "is_real" in res_data
    assert isinstance(res_data["is_real"], bool)
    assert "using_trained_model" in res_data


def test_reload_model_endpoint(client):
    """Verify manual reload endpoint triggers and returns status."""
    response = client.post("/reload-model")
    assert response.status_code in [200, 500]
    data = response.get_json()
    assert "success" in data
    assert "model_loaded" in data
