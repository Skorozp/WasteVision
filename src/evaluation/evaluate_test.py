
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = create_efficientnet(pretrained=False)

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("EfficientNet-B0 loaded successfully!")

    _, _, test_loader = create_dataloaders(
        model_type="efficientnet"
    )

    class_names = test_loader.dataset.dataset.classes
    class_ids = list(range(len(class_names)))

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    test_loss = total_loss / len(test_loader.dataset)

    test_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=class_ids,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=class_ids,
        average="weighted",
        zero_division=0
    )

    correct = sum(
        true == predicted
        for true, predicted in zip(all_labels, all_predictions)
    )

    print("\nFINAL TEST RESULTS")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Correct predictions: {correct}/{len(all_labels)}")

    report = classification_report(
        all_labels,
        all_predictions,
        labels=class_ids,
        target_names=class_names,
        zero_division=0,
        digits=4
    )

    print(f"\nClassification Report:\n{report}")

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=class_ids
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 11))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False
    )

    ax.set_title("EfficientNet-B0 - Test Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    fig.tight_layout()

    image_path = OUTPUT_DIR / "efficientnet_b0_test_confusion_matrix.png"

    fig.savefig(
        image_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    report_path = OUTPUT_DIR / "efficientnet_b0_test_report.txt"

    report_path.write_text(
        f"Test Loss: {test_loss:.4f}\n"
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n"
        f"Correct predictions: {correct}/{len(all_labels)}\n\n"
        f"{report}",
        encoding="utf-8"
    )

    print(f"\nConfusion Matrix saved to: {image_path}")
    print(f"Test Report saved to: {report_path}")


if __name__ == "__main__":
    main()
