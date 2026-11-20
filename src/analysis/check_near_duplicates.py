
import csv
from itertools import combinations
from pathlib import Path

import imagehash
from PIL import Image

from src.data.dataset import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "reports" / "near_duplicates.csv"

THRESHOLD = 5


def calculate_hashes(loader):
    subset = loader.dataset
    results = []

    for index in subset.indices:
        path = Path(subset.dataset.samples[index][0])

        with Image.open(path) as image:
            image_hash = imagehash.phash(image.convert("RGB"))

        results.append((path, int(str(image_hash), 16)))

    return results


def main():
    train_loader, val_loader, test_loader = create_dataloaders(
        model_type="efficientnet"
    )

    splits = {
        "Train": calculate_hashes(train_loader),
        "Validation": calculate_hashes(val_loader),
        "Test": calculate_hashes(test_loader)
    }

    for name, images in splits.items():
        print(f"{name}: {len(images)} hashes calculated")

    matches = []

    for (name_a, images_a), (name_b, images_b) in combinations(
        splits.items(), 2
    ):
        print(f"\nComparing {name_a} vs {name_b}...")

        for path_a, hash_a in images_a:
            for path_b, hash_b in images_b:
                distance = bin(hash_a ^ hash_b).count("1")

                if distance <= THRESHOLD:
                    matches.append((
                        name_a,
                        name_b,
                        str(path_a.relative_to(PROJECT_ROOT)),
                        str(path_b.relative_to(PROJECT_ROOT)),
                        distance
                    ))

    matches.sort(key=lambda item: item[4])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "split_a",
            "split_b",
            "image_a",
            "image_b",
            "phash_distance"
        ])

        writer.writerows(matches)

    print(f"\nPotential near-duplicate pairs: {len(matches)}")

    for match in matches[:20]:
        print(
            f"\nDistance: {match[4]}"
            f"\n{match[2]}"
            f"\n{match[3]}"
        )

    print(f"\nFull report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
