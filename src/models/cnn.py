import torch.nn as nn
import torch


class WasteCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    print("Testing model")
    model = WasteCNN()
    x = torch.randn(4, 3, 224, 224)

    result = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.shape}")
    print(model)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Total parameters: {total_params}")