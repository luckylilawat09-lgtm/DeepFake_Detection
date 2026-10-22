"""
Evaluation Script for DeepFake Detection
=========================================
Loads backend/models/deepfake_detector.pth, runs full held-out test split,
computes and saves:
  - evaluation_results.txt (full detailed evaluation output)
  - backend/models/evaluation_results.json
  - backend/models/evaluation_summary.json
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
SUMMARY_PATH = os.path.join(SCRIPT_DIR, "models", "evaluation_summary.json")
CM_IMAGE_PATH = os.path.join(SCRIPT_DIR, "models", "confusion_matrix.png")
REPO_ROOT_TXT = os.path.join(os.path.dirname(SCRIPT_DIR), "evaluation_results.txt")
BACKEND_TXT = os.path.join(SCRIPT_DIR, "evaluation_results.txt")

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

    print(f"[CHECKPOINT] Loading checkpoint from: {os.path.abspath(MODEL_PATH)}")
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
    draw.text((30, 50), f"EfficientNet-B0 Model Evaluation (N = {cm.get('total_samples', 5227):,})", fill=(160, 175, 200))

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
    checkpoint_acc_str = f"{val_acc:.2f}%" if isinstance(val_acc, (int, float)) else str(val_acc)
    print(f"[CHECKPOINT INFO] Epoch: {epoch} | Stored Val Acc: {checkpoint_acc_str}")

    test_dir = None
    for candidate in POSSIBLE_TEST_DIRS:
        if os.path.exists(candidate):
            test_dir = candidate
            break

    if not test_dir:
        raise FileNotFoundError(f"Test directory not found in {POSSIBLE_TEST_DIRS}")

    print(f"[DATASET] Loading held-out TEST set from: {test_dir}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    total_imgs = len(test_dataset)
    batch_size = 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_batches = len(test_loader)

    class_to_idx = test_dataset.class_to_idx
    print(f"[CLASSES] class_to_idx mapping: {class_to_idx}")
    fake_idx = class_to_idx["fake"]
    real_idx = class_to_idx["real"]

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    tp_fake = 0   # Ground truth Fake, predicted Fake
    fn_fake = 0   # Ground truth Fake, predicted Real
    fp_fake = 0   # Ground truth Real, predicted Fake
    tn_fake = 0   # Ground truth Real, predicted Real

    print(f"[EVAL] Evaluating {total_imgs} held-out test portraits across {total_batches} batches (batch_size={batch_size})...")
    start_eval_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            b_size = inputs.size(0)
            total_loss += loss.item() * b_size
            total_samples += b_size

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

            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == total_batches:
                cur_acc = (tp_fake + tn_fake) / total_samples * 100
                elapsed = time.time() - start_eval_time
                print(f"  -> Batch [{batch_idx + 1:2d}/{total_batches}] | Processed: {total_samples:4d}/{total_imgs} | Current Acc: {cur_acc:.2f}% | Elapsed: {elapsed:.1f}s")

    elapsed_total = time.time() - start_eval_time
    accuracy = (tp_fake + tn_fake) / total_samples
    test_loss = total_loss / total_samples

    # Metrics for Fake (Class 0 - Positive Class in DeepFake Detection)
    fake_prec = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) > 0 else 0.0
    fake_rec = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0.0
    fake_f1 = (2 * fake_prec * fake_rec / (fake_prec + fake_rec)) if (fake_prec + fake_rec) > 0 else 0.0

    # Metrics for Real (Class 1)
    real_prec = tn_fake / (tn_fake + fn_fake) if (tn_fake + fn_fake) > 0 else 0.0
    real_rec = tn_fake / (tn_fake + fp_fake) if (tn_fake + fp_fake) > 0 else 0.0
    real_f1 = (2 * real_prec * real_rec / (real_prec + real_rec)) if (real_prec + real_rec) > 0 else 0.0

    # Macro averages
    macro_prec = (fake_prec + real_prec) / 2
    macro_rec = (fake_rec + real_rec) / 2
    macro_f1 = (fake_f1 + real_f1) / 2

    # Weighted averages
    weight_fake = (tp_fake + fn_fake) / total_samples
    weight_real = (fp_fake + tn_fake) / total_samples
    weighted_prec = fake_prec * weight_fake + real_prec * weight_real
    weighted_rec = fake_rec * weight_fake + real_rec * weight_real
    weighted_f1 = fake_f1 * weight_fake + real_f1 * weight_real

    results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_evaluated": os.path.abspath(MODEL_PATH),
        "checkpoint_filename": os.path.basename(MODEL_PATH),
        "checkpoint_epoch": epoch,
        "checkpoint_val_acc_pct": round(val_acc, 2) if isinstance(val_acc, (int, float)) else val_acc,
        "dataset_test_path": os.path.abspath(test_dir),
        "total_test_samples": total_samples,
        "total_fake_samples": tp_fake + fn_fake,
        "total_real_samples": fp_fake + tn_fake,
        "class_mapping": {
            "0": "fake",
            "1": "real"
        },
        "test_accuracy_pct": round(accuracy * 100, 2),
        "test_loss": round(test_loss, 4),
        "evaluation_duration_seconds": round(elapsed_total, 2),
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
        },
        "weighted_averages": {
            "precision": round(weighted_prec, 4),
            "recall": round(weighted_rec, 4),
            "f1_score": round(weighted_f1, 4)
        }
    }

    # Format output text for evaluation_results.txt
    text_report = f"""================================================================================
           DEEPFAKE DETECTOR EMPIRICAL EVALUATION RESULTS
================================================================================
Date & Time             : {results['evaluation_timestamp']}
Checkpoint Evaluated    : {results['checkpoint_evaluated']}
Checkpoint File         : {results['checkpoint_filename']}
Checkpoint Epoch        : {epoch}
Checkpoint Val Accuracy : {results['checkpoint_val_acc_pct']}% (Validation Split)
Dataset Evaluated       : {results['dataset_test_path']}
Held-out Test Split Only: YES (Train and Validation splits excluded)
Preprocessing           : Resize 224x224, ToTensor, Normalize ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), No Augmentation
Evaluation Device       : {device}
Evaluation Duration     : {elapsed_total:.2f} seconds

--------------------------------------------------------------------------------
OVERALL TEST METRICS
--------------------------------------------------------------------------------
Total Test Images       : {total_samples}
  - Ground Truth Fake   : {tp_fake + fn_fake} images
  - Ground Truth Real   : {fp_fake + tn_fake} images
Test Accuracy           : {results['test_accuracy_pct']}% ({tp_fake + tn_fake} / {total_samples})
Test Loss (CE)          : {results['test_loss']:.4f}

--------------------------------------------------------------------------------
CLASSIFICATION REPORT & DETAILED METRICS
--------------------------------------------------------------------------------
Class             Precision    Recall       F1-Score     Support
--------------------------------------------------------------------------------
Fake (0)          {fake_prec:.4f}       {fake_rec:.4f}       {fake_f1:.4f}       {tp_fake + fn_fake}
Real (1)          {real_prec:.4f}       {real_rec:.4f}       {real_f1:.4f}       {fp_fake + tn_fake}
--------------------------------------------------------------------------------
Macro Avg         {macro_prec:.4f}       {macro_rec:.4f}       {macro_f1:.4f}       {total_samples}
Weighted Avg      {weighted_prec:.4f}       {weighted_rec:.4f}       {weighted_f1:.4f}       {total_samples}

Positive Class 'Fake' Summary:
  - Precision : {fake_prec * 100:.2f}% (when model predicts Fake, how often it is truly Fake)
  - Recall    : {fake_rec * 100:.2f}% (fraction of true Fake images correctly detected)
  - F1-Score  : {fake_f1 * 100:.2f}% (harmonic mean of Precision and Recall)

--------------------------------------------------------------------------------
CONFUSION MATRIX
--------------------------------------------------------------------------------
                    Predicted Fake (0)    Predicted Real (1)    Total Actual
Actual Fake (0)     {tp_fake:<21} {fn_fake:<21} {tp_fake + fn_fake}
Actual Real (1)     {fp_fake:<21} {tn_fake:<21} {fp_fake + tn_fake}
Total Predicted     {tp_fake + fp_fake:<21} {fn_fake + tn_fake:<21} {total_samples}

Breakdown:
  - True Positives  (TP) : {tp_fake:5d}  [Actual Fake  -> Predicted Fake]
  - False Negatives (FN) : {fn_fake:5d}  [Actual Fake  -> Predicted Real] (Missed Fakes)
  - False Positives (FP) : {fp_fake:5d}  [Actual Real  -> Predicted Fake] (False Alarms)
  - True Negatives  (TN) : {tn_fake:5d}  [Actual Real  -> Predicted Real]
================================================================================
"""

    print("\n" + text_report)

    # Save JSON results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved evaluation results JSON to: {RESULTS_PATH}")

    # Also sync summary json
    summary_data = {
        "device": str(device),
        "checkpoint_val_acc": val_acc,
        "checkpoint_epoch": epoch,
        "checkpoint_file": os.path.basename(MODEL_PATH),
        "test_metrics": {
            "total_test_samples": total_samples,
            "test_accuracy_pct": round(accuracy * 100, 2),
            "test_loss": round(test_loss, 4),
            "confusion_matrix": {
                "true_positives_fake": tp_fake,
                "false_positives_fake": fp_fake,
                "true_negatives_real": tn_fake,
                "false_negatives_real": fn_fake
            },
            "precision": round(fake_prec, 4),
            "recall": round(fake_rec, 4),
            "f1_score": round(fake_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4)
        }
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[OK] Saved evaluation summary JSON to: {SUMMARY_PATH}")

    # Save TXT results in repo root and backend directory
    with open(REPO_ROOT_TXT, "w", encoding="utf-8") as f:
        f.write(text_report)
    print(f"[OK] Saved full evaluation results TXT to: {REPO_ROOT_TXT}")

    with open(BACKEND_TXT, "w", encoding="utf-8") as f:
        f.write(text_report)
    print(f"[OK] Saved full evaluation results TXT to: {BACKEND_TXT}")

    # Draw and save confusion matrix image
    cm_summary = {
        "TP_fake": tp_fake,
        "FN_fake": fn_fake,
        "FP_real": fp_fake,
        "TN_real": tn_fake,
        "total_samples": total_samples,
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

    return results


if __name__ == "__main__":
    run_evaluation()
