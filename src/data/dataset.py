
from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import InterpolationMode

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split


project_root = Path(__file__).resolve().parents[2]

realwaste_dir = (
    project_root / "data" / "raw" / "realwaste-main" / "RealWaste"
)


def create_dataloaders(model_type="wastecnn", batch_size=32):

    if model_type == "wastecnn":

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    elif model_type == "efficientnet":

        weights = EfficientNet_B0_Weights.DEFAULT
        preprocess = weights.transforms()

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocess.mean,
                std=preprocess.std
            )
        ])

        eval_transform = preprocess

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Choose 'wastecnn' or 'efficientnet'."
        )

    train_full_dataset = ImageFolder(
        root=realwaste_dir,
        transform=train_transform
    )

    eval_full_dataset = ImageFolder(
        root=realwaste_dir,
        transform=eval_transform
    )

    assert (
        train_full_dataset.samples == eval_full_dataset.samples
    ), "Train and evaluation datasets have different sample order!"

    indices = list(range(len(eval_full_dataset)))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=eval_full_dataset.targets
    )

    temp_targets = [
        eval_full_dataset.targets[index]
        for index in temp_indices
    ]

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=42,
        stratify=temp_targets
    )

    train_dataset = Subset(
        train_full_dataset,
        train_indices
    )

    val_dataset = Subset(
        eval_full_dataset,
        val_indices
    )

    test_dataset = Subset(
        eval_full_dataset,
        test_indices
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders(
        model_type="efficientnet"
    )

    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print(f"\nImages shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Classes: {train_loader.dataset.dataset.classes}")
