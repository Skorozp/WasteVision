import torch
import torch.nn as nn
from src.models.cnn import WasteCNN
from src.data.dataset import create_dataloaders

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")

train_loader, val_loader, test_loader = create_dataloaders()

model = WasteCNN().to(device)
print(f"Model device: {next(model.parameters()).device}")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

epochs = 30

best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
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

    with torch.no_grad():
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best.pth")
        print("Best model saved!")

    print(f"Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}%")


# python -m src.training.train

# I need at least 70% accuracy...God bless