
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from src.data.dataset import create_dataloaders


train_loader, val_loader, test_loader = create_dataloaders(
    model_type="efficientnet"
)

loaders = {
    "Train": train_loader,
    "Validation": val_loader,
    "Test": test_loader
}

hashes = defaultdict(lambda: defaultdict(list))

for split_name, loader in loaders.items():
    subset = loader.dataset
    dataset = subset.dataset

    for index in subset.indices:
        image_path = Path(dataset.samples[index][0])

        file_hash = sha256(
            image_path.read_bytes()
        ).hexdigest()

        hashes[file_hash][split_name].append(image_path)

duplicates = 0

for file_hash, splits in hashes.items():
    if len(splits) > 1:
        duplicates += 1

        print("\nDuplicate found:")

        for split_name, paths in splits.items():
            for path in paths:
                print(f"{split_name}: {path}")

print(f"\nCross-split duplicate groups: {duplicates}")
