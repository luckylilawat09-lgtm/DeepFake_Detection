"""
DeepFake Detection Model Training Script
=========================================
Uses EfficientNet-B0 (pretrained on ImageNet) with transfer learning.
Trains a binary classifier: Real vs Fake.

Usage:
    python train.py
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast

import sys
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # Dataset paths
    "data_root": r"C:\Users\lucky\Downloads\archive\Data Set 1\Data Set 1",

    # Training hyperparameters
    "batch_size": 16,
    "num_epochs": 10,
    "learning_rate": 1e-3,        # For the classifier head
    "fine_tune_lr": 1e-5,         # For the backbone (when unfrozen)
    "weight_decay": 1e-4,
    "unfreeze_after_epoch": 2,    # Unfreeze backbone after N epochs

    # Image settings
    "image_size": 224,

    # Model saving
    "save_dir": "models",
    "model_name": "deepfake_detector.pth",

    # Performance
    "num_workers": 0,             # 0 for Windows CPU stability
    "pin_memory": False,          # Disabled for CPU
    "use_amp": False,             # Disabled for CPU training (no CUDA)

    # Early stopping
    "patience": 5,
}


def get_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[WARNING] No GPU detected, using CPU (training will be slow)")
    return device


def get_data_loaders(config):
    """Create train, validation, and test data loaders with augmentation."""

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

    # Validation/Test transforms: only resize + normalize
    eval_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets using ImageFolder (expects subfolders: fake/, real/)
    train_dataset = datasets.ImageFolder(
        os.path.join(config["data_root"], "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(config["data_root"], "validation"),
        transform=eval_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(config["data_root"], "test"),
        transform=eval_transform
    )

    # Print class mapping
    print(f"\nClass mapping: {train_dataset.class_to_idx}")
    print(f"   Train:      {len(train_dataset):,} images")
    print(f"   Validation: {len(val_dataset):,} images")
    print(f"   Test:       {len(test_dataset):,} images")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"], drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def build_model(num_classes=2):
    """
    Build EfficientNet-B0 with a custom classifier head.
    The backbone is initially frozen for transfer learning.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

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
    print("\n>> Unfreezing backbone for fine-tuning...")
    for param in model.features.parameters():
        param.requires_grad = True

    # Add backbone params to optimizer with lower learning rate
    optimizer.add_param_group({
        "params": model.features.parameters(),
        "lr": lr_backbone,
    })


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_amp:
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

        # Progress update every 25 batches (and batch 1)
        if batch_idx == 0 or (batch_idx + 1) % 25 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100. * correct / total:.1f}%", flush=True)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train(config):
    """Full training pipeline."""
    print("=" * 60)
    print("  DeepFake Detection — Model Training")
    print("=" * 60)

    device = get_device()

    # Data
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(config)

    # Model
    model = build_model(num_classes=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EfficientNet-B0")
    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable params: {trainable_params:,} (head only)")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = GradScaler() if config["use_amp"] else None

    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    backbone_unfrozen = False

    os.makedirs(config["save_dir"], exist_ok=True)
    save_path = os.path.join(config["save_dir"], config["model_name"])

    print(f"\n>> Starting training for {config['num_epochs']} epochs...\n")

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()

        # Unfreeze backbone after N epochs
        if epoch > config["unfreeze_after_epoch"] and not backbone_unfrozen:
            unfreeze_backbone(model, config["fine_tune_lr"], optimizer)
            backbone_unfrozen = True

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config["use_amp"]
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch}/{config['num_epochs']}] ({elapsed:.0f}s) | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_model_state,
                "class_to_idx": class_to_idx,
                "val_acc": best_val_acc,
                "epoch": epoch,
                "image_size": config["image_size"],
            }, save_path)
            print(f"  [SAVED] New best model! (Val Acc: {best_val_acc:.1f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n[STOP] Early stopping triggered after {epoch} epochs (no improvement for {config['patience']} epochs)")
                break

    # Load best model and test
    print("\n" + "=" * 60)
    print("  Final Evaluation on Test Set")
    print("=" * 60)

    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n>> Test Accuracy: {test_acc:.2f}%")
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"   Best Val Acc:  {best_val_acc:.2f}%")
    print(f"\n>> Model saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    train(CONFIG)
