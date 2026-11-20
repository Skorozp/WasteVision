from pathlib import Path

import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights

from src.models.efficientnet import create_efficientnet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "efficientnet_b0_best_acc_30.pth"

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]


class WasteClassifier:
    def __init__(self):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.model = create_efficientnet(pretrained=False)

        state_dict = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = EfficientNet_B0_Weights.DEFAULT.transforms()

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        image = image.convert("RGB")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = probabilities.argmax().item()

        return {
            "class_name": CLASS_NAMES[predicted_index],
            "confidence": round(probabilities[predicted_index].item(), 4),
            "probabilities": {
                name: round(probabilities[i].item(), 4)
                for i, name in enumerate(CLASS_NAMES)
            }
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m src.inference.predict <image_path>"
        )

    classifier = WasteClassifier()

    with Image.open(sys.argv[1]) as image:
        result = classifier.predict(image)

    print(result)