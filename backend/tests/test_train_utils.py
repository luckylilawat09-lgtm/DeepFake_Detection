import sys
import os
import json
import pytest
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import train


def test_atomic_write_progress():
    """Verify write_progress writes valid JSON without data corruption."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_payload = {
            "status": "training",
            "epoch": 2,
            "accuracy": 88.5,
            "loss": 0.25
        }
        train.write_progress(tmp_dir, test_payload)

        progress_file = os.path.join(tmp_dir, "training_progress.json")
        assert os.path.exists(progress_file)

        with open(progress_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["epoch"] == 2
        assert data["accuracy"] == 88.5
        assert data["status"] == "training"


def test_find_dataset_splits_nonexistent():
    """Verify FileNotFoundError is raised for non-existent root."""
    with pytest.raises(FileNotFoundError):
        train.find_dataset_splits("Z:\\nonexistent_directory_for_tests_12345")


def test_find_dataset_splits_direct_structure():
    """Verify discovery of direct {train, val} folder structures."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "train", "fake"))
        os.makedirs(os.path.join(tmp_dir, "train", "real"))
        os.makedirs(os.path.join(tmp_dir, "val", "fake"))
        os.makedirs(os.path.join(tmp_dir, "val", "real"))

        splits = train.find_dataset_splits(tmp_dir)
        assert len(splits) == 1
        assert os.path.isdir(splits[0]["train"])
        assert os.path.isdir(splits[0]["val"])
