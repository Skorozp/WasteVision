from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from src.data.dataset import create_dataloaders
from src.models.efficientnet import create_efficientnet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "efficientnet_b0_best_acc_30.pth"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Make sure efficientnet_b0_best_acc_30.pth is in the project root."
        )

    # Create the same 9-class architecture used during training.
    model = create_efficientnet()

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("EfficientNet-B0 loaded successfully!")

    # Use the exact validation split and preprocessing from training.
    _, val_loader, _ = create_dataloaders(model_type="efficientnet")
    class_names = val_loader.dataset.dataset.classes
    class_ids = list(range(len(class_names)))

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    val_loss = total_loss / total
    val_accuracy = 100.0 * correct / total

    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{total}")

    report = classification_report(
        all_labels,
        all_predictions,
        labels=class_ids,
        target_names=class_names,
        zero_division=0,
    )
    print(f"\nClassification Report:\n{report}")

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=class_ids,
    )
    print(f"Confusion Matrix:\n{matrix}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 11))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False,
    )
    ax.set_title("EfficientNet-B0 — Validation Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()

    image_path = OUTPUT_DIR / "efficientnet_b0_confusion_matrix.png"
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    print(f"\nConfusion Matrix saved to: {image_path}")

    report_path = OUTPUT_DIR / "efficientnet_b0_classification_report.txt"
    report_path.write_text(
        f"Validation Loss: {val_loss:.4f}\n"
        f"Validation Accuracy: {val_accuracy:.2f}%\n"
        f"Correct predictions: {correct}/{total}\n\n"
        f"{report}",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()