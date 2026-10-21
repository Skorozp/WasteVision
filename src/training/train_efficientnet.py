import torch
import torch.nn as nn
from src.models.efficientnet import create_efficientnet
from src.data.dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    model_type="efficientnet"
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")

model = create_efficientnet().to(device)

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=0.001
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Trainable parameters: {trainable_params}")

epochs = 30

criterion = nn.CrossEntropyLoss()

best_eff_loss = float("inf")
best_val_accuracy = 0.0

for epoch in range(epochs):
    model.train()
    model.features.eval()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    average_loss = total_loss / len(train_loader.dataset)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {average_loss:.4f}")

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)
            val_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total * 100
    val_loss = val_loss / len(val_loader.dataset)

    if val_loss < best_eff_loss:
        best_eff_loss = val_loss

        torch.save(
            model.state_dict(),
            "efficientnet_b0_best_30.pth"
        )

        print(
            f"Best model saved! "
            f"Epoch: {epoch + 1} | "
            f"Val Loss: {best_eff_loss:.4f} | "
            f"Accuracy: {val_accuracy:.2f}%"
        )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "efficientnet_b0_best_acc_30.pth"
        )

        print(
            f"Best Accuracy saved! "
            f"Epoch: {epoch + 1} | "
            f"Accuracy: {best_val_accuracy:.2f}%"
        )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )
