
import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def create_efficientnet(num_classes=9):
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    return model


if __name__ == "__main__":
    model = create_efficientnet()
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.inference_mode():
        result = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.shape}")

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    weights = EfficientNet_B0_Weights.DEFAULT
    print(f"\nImage preprocessing:\n{weights.transforms()}")
