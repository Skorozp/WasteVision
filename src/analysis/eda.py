from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
realwaste_dir = project_root / "data" / "raw" / "realwaste-main" / "RealWaste"

if not realwaste_dir.exists():
    raise FileNotFoundError(f"Dataset directory not found: {realwaste_dir}")

class_dirs = []
class_counts = {}

for item in sorted(realwaste_dir.iterdir()):
    if item.is_dir():
        class_dirs.append(item)
        class_counts[item.name] = len(list(item.glob("*.jpg")))

total_images = sum(class_counts.values())

class_percent = {}

for class_name, count in class_counts.items():
    class_percent[class_name] = count / total_images * 100

print("Class distribution:")

for class_name, count in class_counts.items():
    percent = class_percent[class_name]
    print(
        f"{class_name}: "
        f"{count} images "
        f"({percent:.2f}%)"
    )

print()
print(f"Total images: {total_images}")

largest_class = max(class_counts, key=class_counts.get)
smallest_class = min(class_counts, key=class_counts.get)

largest_count = class_counts[largest_class]
smallest_count = class_counts[smallest_class]

imbalance_ratio = largest_count / smallest_count

print()
print(f"Largest class: {largest_class} - {largest_count}")
print(f"Smallest class: {smallest_class} - {smallest_count}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

image_sizes = set()
image_modes = set()
image_formats = set()

corrupted_images = []

for class_dir in class_dirs:
    for image_path in class_dir.glob("*.jpg"):
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            corrupted_images.append(image_path)
            continue
        with Image.open(image_path) as image:
            image_sizes.add(image.size)
            image_modes.add(image.mode)
            image_formats.add(image.format)

print()
print(f"Corrupted images: {len(corrupted_images)}")
print(f"Image sizes: {image_sizes}")
print(f"Image modes: {image_modes}")
print(f"Image formats: {image_formats}")

if corrupted_images:
    print("\nCorrupted image paths:")

    for image_path in corrupted_images:
        print(image_path)

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    class_counts.keys(),
    class_counts.values()
)

ax.set_title("RealWaste Class Distribution")
ax.set_xlabel("Waste Class")
ax.set_ylabel("Number of Images")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

random.seed(52)

fig, axes = plt.subplots(
    3,
    3,
    figsize=(10, 10)
)

axes = axes.flatten()

for i, class_dir in enumerate(class_dirs):

    image_paths = list(class_dir.glob("*.jpg"))
    random_image_path = random.choice(image_paths)

    with Image.open(random_image_path) as image:
        axes[i].imshow(image.convert("RGB"))

    axes[i].set_title(
        f"{class_dir.name}\n"
        f"({len(image_paths)} images)",
        fontsize=10
    )

    axes[i].axis("off")

plt.tight_layout()
plt.show()