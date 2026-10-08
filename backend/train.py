"""
DeepFake Detection Model Training Script
=========================================
Uses EfficientNet-B0 (pretrained on ImageNet) with transfer learning.
Trains a binary classifier: Real vs Fake.
Supports:
  - Resuming from existing checkpoints (e.g. models/deepfake_detector.pth)
  - Differential learning rates (backbone fine-tuning)
  - Auto-discovery of datasets in archive directories (Data Set 1..4)
  - Live progress telemetry reporting for frontend dashboard
  - Command-line arguments for custom training runs

Usage:
    python train.py
    python train.py --data-dir "C:\\Users\\lucky\\Downloads\\archive" --resume --epochs 10
    python train.py --data-dir "C:\\Users\\lucky\\Downloads\\archive" --all-datasets
"""

import os
import sys
import time
import copy
import json
import glob
import uuid
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "models")

# Portable dataset root resolution (Environment -> Archive path -> local data fallback)
ENV_DATA_DIR = os.environ.get("DATASET_DIR")
if ENV_DATA_DIR and os.path.exists(ENV_DATA_DIR):
    DEFAULT_DATA_ROOT = ENV_DATA_DIR
elif os.path.exists(r"C:\Users\lucky\Downloads\archive"):
    DEFAULT_DATA_ROOT = r"C:\Users\lucky\Downloads\archive"
else:
    DEFAULT_DATA_ROOT = os.path.join(SCRIPT_DIR, "data")

# Default configuration
DEFAULT_CONFIG = {
    # Dataset paths
    "data_root": DEFAULT_DATA_ROOT,
    "dataset_name": "Data Set 1",
    "all_datasets": False,

    # Training hyperparameters
    "batch_size": 16,
    "num_epochs": 10,
    "learning_rate": 1e-3,        # For classifier head
    "fine_tune_lr": 1e-5,         # For backbone (when unfrozen)
    "weight_decay": 1e-4,
    "unfreeze_after_epoch": 2,    # Unfreeze backbone after N epochs

    # Image settings
    "image_size": 224,

    # Model saving & checkpoint
    "save_dir": DEFAULT_SAVE_DIR,
    "model_name": "deepfake_detector.pth",
    "resume": True,

    # Performance
    "num_workers": 0,             # 0 for Windows stability by default
    "pin_memory": False,
    "use_amp": False,
    "max_batches_per_epoch": None, # None for full run

    # Early stopping & epoch control
    "patience": 5,
    "start_epoch": None,
    "reset_best_val": False,
}


def get_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(min(num_cores, 8))
        print(f"[INFO] Using CPU with {torch.get_num_threads()} compute threads")
    return device


def find_dataset_splits(data_root):
    """
    Locates (train, val, test) folder triplets inside data_root.
    Handles:
      - Direct layout: data_root/{train, validation, test}
      - Archive layout: data_root/Data Set X/Data Set X/{train, validation, test}
    """
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_root}")

    # Check direct layout
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "validation") if os.path.isdir(os.path.join(data_root, "validation")) else os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        return [{
            "name": os.path.basename(data_root) or "Dataset",
            "root": data_root,
            "train": train_dir,
            "val": val_dir,
            "test": test_dir if os.path.isdir(test_dir) else None,
        }]

    # Search subdirectories for folder triplets
    splits = []
    for root, dirs, _ in os.walk(data_root):
        if "train" in dirs and ("validation" in dirs or "val" in dirs):
            val_name = "validation" if "validation" in dirs else "val"
            test_path = os.path.join(root, "test") if "test" in dirs else None
            parent = os.path.basename(os.path.dirname(root))
            curr = os.path.basename(root)
            display_name = parent if "Data Set" in parent else curr
            splits.append({
                "name": display_name,
                "root": root,
                "train": os.path.join(root, "train"),
                "val": os.path.join(root, val_name),
                "test": test_path
            })

    splits.sort(key=lambda s: s["name"])
    return splits


def get_data_loaders(config):
    """Create train, validation, and test data loaders with augmentation and label consistency check."""
    data_root = config["data_root"]
    splits = find_dataset_splits(data_root)

    if not splits:
        raise FileNotFoundError(f"Could not find any valid train/validation folders in {data_root}")

    print(f"\n[DATA] Found {len(splits)} dataset partitions in '{data_root}':")
    for s in splits:
        print(f"   * {s['name']} -> {s['train']}")

    selected_splits = []
    if config.get("all_datasets"):
        selected_splits = splits
        print(f"[DATA] Mode: Combining all {len(splits)} partitions.")
    else:
        req_name = str(config.get("dataset_name", "Data Set 1")).strip().lower()
        matched = [s for s in splits if req_name in s["name"].lower() or req_name in s.get("root", "").lower()]
        if matched:
            selected_splits = [matched[0]]
            print(f"[DATA] Selected partition: {selected_splits[0]['name']}")
        else:
            selected_splits = [splits[0]]
            print(f"[DATA] Partition '{req_name}' not matched; defaulting to {splits[0]['name']}")

    # Training transforms: augmentation + normalization
    train_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Validation/Test transforms: resize + normalize
    eval_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Build datasets with strict label consistency verification
    train_datasets = []
    val_datasets = []
    test_datasets = []
    class_to_idx = None

    for s in selected_splits:
        t_ds = datasets.ImageFolder(s["train"], transform=train_transform)
        v_ds = datasets.ImageFolder(s["val"], transform=eval_transform)

        if class_to_idx is None:
            class_to_idx = t_ds.class_to_idx
        elif t_ds.class_to_idx != class_to_idx:
            raise ValueError(
                f"Class mapping mismatch between partitions! "
                f"Expected {class_to_idx}, but partition '{s['name']}' has {t_ds.class_to_idx}. "
                "Ensure subfolder names and casing match across datasets."
            )

        train_datasets.append(t_ds)
        val_datasets.append(v_ds)
        if s.get("test") and os.path.exists(s["test"]):
            te_ds = datasets.ImageFolder(s["test"], transform=eval_transform)
            test_datasets.append(te_ds)

    final_train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    final_val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    final_test_ds = (ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]) if test_datasets else None

    print(f"\nClass mapping: {class_to_idx}")
    print(f"   Train:      {len(final_train_ds):,} images")
    print(f"   Validation: {len(final_val_ds):,} images")
    if final_test_ds:
        print(f"   Test:       {len(final_test_ds):,} images")

    # Guard drop_last to avoid empty dataloader if total images < batch_size
    should_drop_last = len(final_train_ds) > config["batch_size"]

    train_loader = DataLoader(
        final_train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"], drop_last=should_drop_last
    )
    val_loader = DataLoader(
        final_val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    test_loader = DataLoader(
        final_test_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    ) if final_test_ds else None

    return train_loader, val_loader, test_loader, class_to_idx


def build_model(num_classes=2, pretrained=True):
    """
    Build EfficientNet-B0 with a custom classifier head.
    The backbone is initially frozen for transfer learning.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze backbone initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )

    return model


def unfreeze_backbone(model, lr_backbone, optimizer):
    """Unfreeze the backbone and add its parameters to the optimizer with a lower LR."""
    print("\n>> Unfreezing backbone for Phase 3: Backbone Fine-Tuning...")
    for param in model.features.parameters():
        param.requires_grad = True

    # Check if backbone parameters are already in the optimizer
    existing_params = {id(p) for group in optimizer.param_groups for p in group['params']}
    backbone_params = [p for p in model.features.parameters() if id(p) not in existing_params]

    if backbone_params:
        optimizer.add_param_group({
            "params": backbone_params,
            "lr": lr_backbone,
        })
        print(f"   [OK] Added {len(backbone_params)} backbone tensors to optimizer (LR: {lr_backbone:.1e})")


def write_progress(save_dir, data):
    """Save progress dictionary using an atomic file write to prevent race conditions in polling."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        progress_path = os.path.join(save_dir, "training_progress.json")
        temp_path = os.path.join(save_dir, f"training_progress_{uuid.uuid4().hex[:8]}.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, progress_path)
    except Exception as e:
        print(f"[WARNING] Could not write progress file: {e}")


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, epoch, total_epochs, backbone_unfrozen, save_dir, max_batches=None):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    raw_len = len(loader)
    num_batches = max(1, raw_len if max_batches is None else min(raw_len, max_batches))

    phase_title = "Backbone Fine-Tuning" if backbone_unfrozen else "Head Convergence"
    phase_idx = 3 if backbone_unfrozen else 2

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_amp and scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress log every 20 batches or on edge batches
        if batch_idx == 0 or (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            curr_acc = 100. * correct / total if total > 0 else 0.0
            curr_loss = running_loss / total if total > 0 else 0.0
            print(f"    Batch {batch_idx+1}/{num_batches} | "
                  f"Loss: {curr_loss:.4f} | "
                  f"Acc: {curr_acc:.1f}% [{phase_title}]", flush=True)

            write_progress(save_dir, {
                "status": "training",
                "epoch": epoch,
                "total_epochs": total_epochs,
                "batch": batch_idx + 1,
                "total_batches": num_batches,
                "loss": round(curr_loss, 4),
                "accuracy": round(curr_acc, 2),
                "phase": f"Phase {phase_idx}: {phase_title}",
                "phase_idx": phase_idx,
                "backbone_unfrozen": backbone_unfrozen
            })

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = 100. * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    """Evaluate model on a dataset. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = 100. * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def train(config):
    """Full training and checkpoint resume pipeline with full state recovery."""
    print("=" * 65)
    print("  DeepFake Detection — Model Training & Fine-Tuning Pipeline")
    print("=" * 65)

    device = get_device()

    # Data loaders
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(config)

    # Save path
    os.makedirs(config["save_dir"], exist_ok=True)
    save_path = os.path.join(config["save_dir"], config["model_name"])

    # Resume check
    start_epoch = 1
    best_val_acc = 0.0
    best_model_state = None
    backbone_unfrozen = False
    saved_optimizer_state = None
    saved_scheduler_state = None

    checkpoint_loaded = False
    if config["resume"] and os.path.exists(save_path):
        print(f"\n[RESUME] Found existing checkpoint: {save_path}")
        try:
            checkpoint = torch.load(save_path, map_location=device, weights_only=True)
            model = build_model(num_classes=2, pretrained=False).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 1) + 1
            best_val_acc = checkpoint.get("val_acc", 0.0)
            best_model_state = copy.deepcopy(model.state_dict())
            saved_optimizer_state = checkpoint.get("optimizer_state_dict")
            saved_scheduler_state = checkpoint.get("scheduler_state_dict")
            checkpoint_loaded = True
            print(f"[RESUME] Loaded weights from Epoch {checkpoint.get('epoch')}. Previous Val Acc: {best_val_acc:.2f}%")
        except Exception as e:
            print(f"[RESUME WARNING] Failed to load checkpoint: {e}. Building fresh model.")

    if not checkpoint_loaded:
        print("\n[INIT] Initializing fresh EfficientNet-B0 with ImageNet pretraining...")
        model = build_model(num_classes=2, pretrained=True).to(device)

    if config.get("start_epoch") is not None:
        start_epoch = int(config["start_epoch"])
        print(f"[CONFIG] Overriding start epoch to Epoch {start_epoch}")

    if best_val_acc >= 100.0 or config.get("reset_best_val", False):
        print(f"[NOTICE] Resetting best validation baseline from {best_val_acc:.2f}% to 0.0% so new checkpoints can be saved.")
        best_val_acc = 0.0

    # Check unfreeze state
    if start_epoch > config["unfreeze_after_epoch"]:
        print(f"\n[UNFREEZE] Start epoch ({start_epoch}) > unfreeze threshold ({config['unfreeze_after_epoch']}).")
        print("           Unfreezing backbone for differential fine-tuning immediately.")
        for param in model.features.parameters():
            param.requires_grad = True
        backbone_unfrozen = True
        optimizer = optim.Adam([
            {"params": model.classifier.parameters(), "lr": config["learning_rate"]},
            {"params": model.features.parameters(), "lr": config["fine_tune_lr"]},
        ], weight_decay=config["weight_decay"])
    else:
        backbone_unfrozen = False
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

    # Restore optimizer state if compatible
    if saved_optimizer_state and not backbone_unfrozen:
        try:
            optimizer.load_state_dict(saved_optimizer_state)
            print("[RESUME] Restored optimizer state successfully.")
        except Exception as oe:
            print(f"[RESUME NOTE] Optimizer state skipped due to layout changes: {oe}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture: EfficientNet-B0")
    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable params: {trainable_params:,} ({'Backbone + Head' if backbone_unfrozen else 'Head Only'})")

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    if saved_scheduler_state:
        try:
            scheduler.load_state_dict(saved_scheduler_state)
            print("[RESUME] Restored scheduler state successfully.")
        except Exception:
            pass

    scaler = GradScaler() if config["use_amp"] else None

    total_epochs = config["num_epochs"]
    if start_epoch > total_epochs:
        total_epochs = start_epoch + 3
        print(f"[INFO] Adjusting total epochs to {total_epochs} to allow continued training.")

    patience_counter = 0
    actual_last_epoch = start_epoch
    print(f"\n>> Starting training from Epoch {start_epoch} through {total_epochs}...\n")

    for epoch in range(start_epoch, total_epochs + 1):
        actual_last_epoch = epoch
        epoch_start = time.time()

        # Unfreeze backbone after N epochs
        if epoch > config["unfreeze_after_epoch"] and not backbone_unfrozen:
            unfreeze_backbone(model, config["fine_tune_lr"], optimizer)
            backbone_unfrozen = True

        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=config["use_amp"],
            epoch=epoch,
            total_epochs=total_epochs,
            backbone_unfrozen=backbone_unfrozen,
            save_dir=config["save_dir"],
            max_batches=config.get("max_batches_per_epoch")
        )

        # Validate
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=config.get("max_batches_per_epoch")
        )

        scheduler.step(val_loss)
        elapsed = time.time() - epoch_start
        print(f"\nEpoch [{epoch}/{total_epochs}] ({elapsed:.0f}s) | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%")

        # Save checkpoint if accuracy improved
        is_best = val_acc > best_val_acc
        if is_best or best_model_state is None:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "class_to_idx": class_to_idx,
                "val_acc": best_val_acc,
                "epoch": epoch,
                "image_size": config["image_size"],
            }, save_path)
            print(f"  [SAVED] Checkpoint updated! (Val Acc: {best_val_acc:.2f}%) -> {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  [INFO] Val Acc ({val_acc:.2f}%) did not exceed best ({best_val_acc:.2f}%). Patience: {patience_counter}/{config['patience']}")
            if patience_counter >= config["patience"]:
                print(f"\n[STOP] Early stopping triggered after {epoch} epochs.")
                break

        # Write telemetry after each epoch
        write_progress(config["save_dir"], {
            "status": "in_progress",
            "epoch": epoch,
            "total_epochs": total_epochs,
            "batch": len(train_loader),
            "total_batches": len(train_loader),
            "loss": round(train_loss, 4),
            "accuracy": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "best_val_acc": round(best_val_acc, 2),
            "phase": "Backbone Fine-Tuning" if backbone_unfrozen else "Classifier Head Convergence",
            "phase_idx": 3 if backbone_unfrozen else 2,
            "backbone_unfrozen": backbone_unfrozen,
            "model_ready": True
        })

    # Final Test
    if test_loader and best_model_state is not None:
        print("\n" + "=" * 65)
        print("  Final Evaluation on Test Set")
        print("=" * 65)
        model.load_state_dict(best_model_state)
        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=config.get("max_batches_per_epoch")
        )
        print(f"\n>> Test Accuracy: {test_acc:.2f}%")
        print(f"   Test Loss:     {test_loss:.4f}")
        print(f"   Best Val Acc:  {best_val_acc:.2f}%")
        print(f"\n>> Final model saved to: {os.path.abspath(save_path)}")

    # Mark training complete in telemetry with actual stopped epoch
    write_progress(config["save_dir"], {
        "status": "completed",
        "epoch": actual_last_epoch,
        "total_epochs": total_epochs,
        "batch": len(train_loader),
        "total_batches": len(train_loader),
        "accuracy": round(best_val_acc, 2),
        "best_val_acc": round(best_val_acc, 2),
        "phase": "Completed",
        "phase_idx": 4,
        "model_ready": True
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Train / Continue Training DeepFake Detection EfficientNet-B0 Model")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_CONFIG["data_root"],
                        help="Root directory containing dataset")
    parser.add_argument("--dataset", type=str, default="Data Set 1",
                        help="Specific partition to use if multiple found (default: 'Data Set 1')")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Combine all partitions found into a single large dataset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"],
                        help="Total epochs to train")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Classifier head learning rate")
    parser.add_argument("--fine-tune-lr", type=float, default=DEFAULT_CONFIG["fine_tune_lr"],
                        help="Backbone learning rate when fine-tuning")
    parser.add_argument("--unfreeze-after-epoch", type=int, default=DEFAULT_CONFIG["unfreeze_after_epoch"],
                        help="Epoch after which backbone is unfrozen (default: 2)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resuming from existing checkpoint and start fresh")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_CONFIG["save_dir"],
                        help="Directory to save model checkpoint")
    parser.add_argument("--model-name", type=str, default=DEFAULT_CONFIG["model_name"],
                        help="Filename for model checkpoint")
    parser.add_argument("--start-epoch", type=int, default=None,
                        help="Explicit epoch to start training from (e.g. 8)")
    parser.add_argument("--reset-best-val", action="store_true",
                        help="Reset best validation accuracy to 0.0 to track new records")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None,
                        help="Debug flag to limit batches per epoch for quick verification")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data_root"] = args.data_dir
    config["dataset_name"] = args.dataset
    config["all_datasets"] = args.all_datasets
    config["num_epochs"] = args.epochs
    config["start_epoch"] = args.start_epoch
    config["reset_best_val"] = args.reset_best_val
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["fine_tune_lr"] = args.fine_tune_lr
    config["unfreeze_after_epoch"] = args.unfreeze_after_epoch
    config["resume"] = not args.no_resume
    config["save_dir"] = args.save_dir
    config["model_name"] = args.model_name
    config["max_batches_per_epoch"] = args.max_batches_per_epoch

    try:
        train(config)
    except KeyboardInterrupt:
        print("\n\n[PAUSED] Training paused by user (Ctrl+C). Current checkpoint and progress preserved.")
        progress_path = os.path.join(config["save_dir"], "training_progress.json")
        if os.path.exists(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                pdata["status"] = "paused"
                write_progress(config["save_dir"], pdata)
            except Exception:
                pass
        sys.exit(0)
