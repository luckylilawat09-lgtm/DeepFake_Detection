import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import train


def test_build_model_architecture():
    """Verify build_model builds EfficientNet-B0 with custom 2-class head."""
    model = train.build_model(num_classes=2, pretrained=False)
    assert isinstance(model, nn.Module)

    # Check that backbone is frozen initially
    for param in model.features.parameters():
        assert param.requires_grad is False

    # Check classifier head structure
    classifier = model.classifier
    assert isinstance(classifier, nn.Sequential)
    # Final layer must have out_features == 2
    final_layer = classifier[-1]
    assert isinstance(final_layer, nn.Linear)
    assert final_layer.out_features == 2


def test_model_forward_pass():
    """Verify forward pass tensor shape on a dummy batch."""
    model = train.build_model(num_classes=2, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 2)


def test_unfreeze_backbone():
    """Verify unfreeze_backbone enables gradient tracking on features."""
    model = train.build_model(num_classes=2, pretrained=False)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    train.unfreeze_backbone(model, lr_backbone=1e-5, optimizer=optimizer)

    # All backbone features must now require grad
    for param in model.features.parameters():
        assert param.requires_grad is True

    # Optimizer should now have 2 param groups (classifier + backbone)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] == 1e-5
