from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from src.models.cnn import WasteCNN
from src.data.dataset import create_dataloaders

project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "wastecnn_30epochs.pth"
output_dir = project_root / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")

model = WasteCNN().to(device)

state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully!")

train_loader, val_loader, test_loader = create_dataloaders()

class_names = val_loader.dataset.dataset.classes

criterion = nn.CrossEntropyLoss()

correct = 0
total = 0
val_loss = 0.0

all_labels = []
all_predictions = []

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

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

val_accuracy = correct / total * 100
val_loss = val_loss / total

print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}%")
print(f"Correct predictions: {correct}/{total}")

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(13, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=False
)

ax.set_title("WasteCNN - Validation Confusion Matrix")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")

plt.tight_layout()

save_path = output_dir / "wastecnn_confusion_matrix.png"

plt.savefig(save_path, dpi=200, bbox_inches="tight")

print(f"\nConfusion Matrix saved to: {save_path}")

plt.show()