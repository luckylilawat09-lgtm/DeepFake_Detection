"""
Evaluation Script for DeepFake Detection
=========================================
Loads backend/models/deepfake_detector.pth, runs full test split,
computes and saves:
  - backend/models/evaluation_results.json
  - backend/models/confusion_matrix.png
Outputs test accuracy, loss, per-class precision/recall/F1, macro averages, and confusion matrix.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "deepfake_detector.pth")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "models", "evaluation_results.json")
CM_IMAGE_PATH = os.path.join(SCRIPT_DIR, "models", "confusion_matrix.png")

POSSIBLE_TEST_DIRS = [
    r"C:\Users\lucky\Downloads\archive\Data Set 1\Data Set 1\test",
    os.path.join(SCRIPT_DIR, "data", "test"),
]


def load_model(device):
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def draw_confusion_matrix_image(cm, output_path):
    """Draw a clean confusion matrix image using PIL."""
    img_w, img_h = 600, 500
    img = Image.new("RGB", (img_w, img_h), color=(15, 20, 30))
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((30, 25), "Confusion Matrix - DeepFake Detection (Test Set)", fill=(255, 255, 255))
    draw.text((30, 50), "EfficientNet-B0 Model Evaluation (N = 5,227)", fill=(160, 175, 200))

    # Grid parameters
    ox, oy = 140, 120
    cell_w, cell_h = 180, 120

    # Colors
    c_tp = (0, 140, 70)      # TP (Fake detected as Fake) - Green
    c_fn = (160, 40, 40)     # FN (Fake missed as Real) - Red
    c_fp = (160, 100, 20)    # FP (Real called Fake) - Amber
    c_tn = (0, 100, 140)     # TN (Real detected as Real) - Blue

    cells = [
        (ox, oy, c_tp, f"TP (Fake): {cm['TP_fake']}", "Actual Fake -> Pred Fake"),
        (ox + cell_w, oy, c_fn, f"FN (Fake): {cm['FN_fake']}", "Actual Fake -> Pred Real"),
        (ox, oy + cell_h, c_fp, f"FP (Real): {cm['FP_real']}", "Actual Real -> Pred Fake"),
        (ox + cell_w, oy + cell_h, c_tn, f"TN (Real): {cm['TN_real']}", "Actual Real -> Pred Real"),
    ]

    for x, y, col, title, sub in cells:
        draw.rectangle([x, y, x + cell_w - 6, y + cell_h - 6], fill=col, outline=(255, 255, 255), width=1)
        draw.text((x + 15, y + 35), title, fill=(255, 255, 255))
        draw.text((x + 15, y + 65), sub, fill=(210, 225, 240))

    # Axis Labels
    draw.text((ox + 30, oy - 25), "Pred: Fake (0)", fill=(0, 229, 255))
    draw.text((ox + cell_w + 30, oy - 25), "Pred: Real (1)", fill=(0, 229, 255))
    draw.text((25, oy + 45), "Actual: Fake", fill=(255, 120, 120))
    draw.text((25, oy + cell_h + 45), "Actual: Real", fill=(120, 255, 160))

    # Footer metrics
    draw.text((30, 400), f"Test Accuracy: {cm['accuracy_pct']}%  |  Test Loss: {cm['loss']}", fill=(255, 255, 255))
    draw.text((30, 425), f"Fake Precision: {cm['fake_precision']}  |  Fake Recall: {cm['fake_recall']}  |  Fake F1: {cm['fake_f1']}", fill=(200, 220, 240))
    draw.text((30, 450), f"Real Precision: {cm['real_precision']}  |  Real Recall: {cm['real_recall']}  |  Real F1: {cm['real_f1']}", fill=(200, 220, 240))

    img.save(output_path, "PNG")
    print(f"[OK] Saved confusion matrix image to: {output_path}")


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Evaluation running on: {device}")

    model, checkpoint = load_model(device)
    val_acc = checkpoint.get("val_acc", None)
    epoch = checkpoint.get("epoch", None)
    print(f"[CHECKPOINT] Epoch: {epoch} | Val Acc: {val_acc:.2f}%")

    test_dir = None
    for candidate in POSSIBLE_TEST_DIRS:
        if os.path.exists(candidate):
            test_dir = candidate
            break

    if not test_dir:
        raise FileNotFoundError(f"Test directory not found in {POSSIBLE_TEST_DIRS}")

    print(f"[DATASET] Loading test set from: {test_dir}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    class_to_idx = test_dataset.class_to_idx
    print(f"[CLASSES] class_to_idx mapping: {class_to_idx}")
    # 0 = fake, 1 = real
    fake_idx = class_to_idx["fake"]
    real_idx = class_to_idx["real"]

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    tp_fake = 0   # Ground truth Fake, predicted Fake
    fn_fake = 0   # Ground truth Fake, predicted Real
    fp_fake = 0   # Ground truth Real, predicted Fake
    tn_fake = 0   # Ground truth Real, predicted Real

    print(f"[EVAL] Evaluating {len(test_dataset)} test portraits...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            _, preds = outputs.max(1)
            for p, t in zip(preds, targets):
                p_item = p.item()
                t_item = t.item()
                if t_item == fake_idx:
                    if p_item == fake_idx:
                        tp_fake += 1
                    else:
                        fn_fake += 1
                else: # t_item == real_idx
                    if p_item == fake_idx:
                        fp_fake += 1
                    else:
                        tn_fake += 1

    accuracy = (tp_fake + tn_fake) / total_samples
    test_loss = total_loss / total_samples

    # Metrics for Fake (Class 0)
    fake_prec = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) > 0 else 0
    fake_rec = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0
    fake_f1 = 2 * fake_prec * fake_rec / (fake_prec + fake_rec) if (fake_prec + fake_rec) > 0 else 0

    # Metrics for Real (Class 1)
    real_prec = tn_fake / (tn_fake + fn_fake) if (tn_fake + fn_fake) > 0 else 0
    real_rec = tn_fake / (tn_fake + fp_fake) if (tn_fake + fp_fake) > 0 else 0
    real_f1 = 2 * real_prec * real_rec / (real_prec + real_rec) if (real_prec + real_rec) > 0 else 0

    # Macro averages
    macro_prec = (fake_prec + real_prec) / 2
    macro_rec = (fake_rec + real_rec) / 2
    macro_f1 = (fake_f1 + real_f1) / 2

    results = {
        "dataset_test_path": test_dir,
        "total_test_samples": total_samples,
        "class_mapping": {
            "0": "fake",
            "1": "real"
        },
        "test_accuracy": round(accuracy * 100, 2),
        "test_loss": round(test_loss, 4),
        "confusion_matrix": {
            "TP_fake": tp_fake,
            "FN_fake": fn_fake,
            "FP_real": fp_fake,
            "TN_real": tn_fake,
            "matrix_2x2": [
                [tp_fake, fn_fake],
                [fp_fake, tn_fake]
            ],
            "description": "Rows: Actual [Fake, Real]; Columns: Predicted [Fake, Real]"
        },
        "per_class_metrics": {
            "fake": {
                "support": tp_fake + fn_fake,
                "precision": round(fake_prec, 4),
                "recall": round(fake_rec, 4),
                "f1_score": round(fake_f1, 4)
            },
            "real": {
                "support": fp_fake + tn_fake,
                "precision": round(real_prec, 4),
                "recall": round(real_rec, 4),
                "f1_score": round(real_f1, 4)
            }
        },
        "macro_averages": {
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "f1_score": round(macro_f1, 4)
        }
    }

    # Save JSON
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved evaluation results to: {RESULTS_PATH}")

    # Draw image
    cm_summary = {
        "TP_fake": tp_fake,
        "FN_fake": fn_fake,
        "FP_real": fp_fake,
        "TN_real": tn_fake,
        "accuracy_pct": round(accuracy * 100, 2),
        "loss": round(test_loss, 4),
        "fake_precision": round(fake_prec, 4),
        "fake_recall": round(fake_rec, 4),
        "fake_f1": round(fake_f1, 4),
        "real_precision": round(real_prec, 4),
        "real_recall": round(real_rec, 4),
        "real_f1": round(real_f1, 4)
    }
    draw_confusion_matrix_image(cm_summary, CM_IMAGE_PATH)

    print("\n" + "=" * 55)
    print("                EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Total Test Samples : {total_samples}")
    print(f"Test Accuracy      : {results['test_accuracy']}%")
    print(f"Test Loss          : {results['test_loss']}")
    print("-" * 55)
    print("Per-Class Metrics:")
    print(f"  Class 'Fake' (0) -> Precision: {results['per_class_metrics']['fake']['precision']:.4f} | Recall: {results['per_class_metrics']['fake']['recall']:.4f} | F1: {results['per_class_metrics']['fake']['f1_score']:.4f}")
    print(f"  Class 'Real' (1) -> Precision: {results['per_class_metrics']['real']['precision']:.4f} | Recall: {results['per_class_metrics']['real']['recall']:.4f} | F1: {results['per_class_metrics']['real']['f1_score']:.4f}")
    print("-" * 55)
    print("Macro Averages:")
    print(f"  Macro Precision  : {results['macro_averages']['precision']:.4f}")
    print(f"  Macro Recall     : {results['macro_averages']['recall']:.4f}")
    print(f"  Macro F1-Score   : {results['macro_averages']['f1_score']:.4f}")
    print("-" * 55)
    print("Confusion Matrix:")
    print(f"  Actual Fake: {tp_fake} predicted Fake (TP) | {fn_fake} predicted Real (FN)")
    print(f"  Actual Real: {fp_fake} predicted Fake (FP) | {tn_fake} predicted Real (TN)")
    print("=" * 55)
    return results


if __name__ == "__main__":
    run_evaluation()
