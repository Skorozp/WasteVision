# WasteVision

Image-based waste classification with PyTorch and transfer learning
From a custom CNN trained from scratch to a pretrained EfficientNet-B0 - a reproducible learning project exploring how to classify real-world waste images.

Overview
WasteVision is an end-to-end computer vision project for identifying the type of waste shown in a photograph. It explores the full machine learning workflow: dataset acquisition, exploratory analysis, preprocessing, stratified splitting, model development, training, validation, and error analysis.

The project compares two approaches:
1. WasteCNN - a lightweight convolutional neural network designed and trained from scratch.
2. EfficientNet-B0 - an ImageNet-pretrained model adapted to the same nine waste categories using transfer learning.

The first milestone is a working single-image, single-label classifier. 

## Results

| Model | Approach | Best validation accuracy | Test accuracy |
|---|---|---:|---:|
| **WasteCNN** | Custom CNN trained from scratch | 60.73% | Not evaluated |
| **EfficientNet-B0** | ImageNet-pretrained, frozen feature extractor | **88.08%** | **86.96%** |

Transfer learning improved the best validation accuracy by **27.35 percentage points** compared with the custom CNN baseline.

The selected EfficientNet-B0 checkpoint achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Test images | 713 |
| Correct predictions | 620 / 713 |
| **Test accuracy** | **86.96%** |
| Macro F1-score | **0.8739** |
| Weighted F1-score | 0.8701 |
| Test loss | 0.3947 |

## Dataset
This project uses the RealWaste dataset (UCI Machine Learning Repository).
- 4,752 images
- 9 waste categories
- 3,326 training images (70%)
- 713 validation images (15%)
- 713 test images (15%)
- Stratified train/validation/test split with random_state=42

Classes (in the order used by torchvision.datasets.ImageFolder):
1. Cardboard
2. Food Organics
3. Glass
4. Metal
5. Miscellaneous Trash
6. Paper
7. Plastic
8. Textile Trash
9. Vegetation

The dataset is not redistributed in this repository. Please consult the dataset provider for its terms and attribution requirements.

## Models
1. WasteCNN - a from-scratch baseline
The custom CNN is intentionally small and interpretable:
```commandline
Input: RGB image [B, 3, 224, 224]
    ↓
Conv2d(3 → 16)  → ReLU → MaxPool2d
    ↓
Conv2d(16 → 32) → ReLU → MaxPool2d
    ↓
Conv2d(32 → 64) → ReLU → MaxPool2d
    ↓
AdaptiveAvgPool2d(1 × 1)
    ↓
Flatten
    ↓
Linear(64 → 9)
    ↓
9 logits
```
- 24,169 trainable parameters
- AdamW optimizer and cross-entropy loss
- ImageNet mean/std normalization
- Basic training augmentation: random horizontal flip
This model serves as an educational baseline and provides a reference point for measuring the impact of pretraining.

2. EfficientNet-B0 - transfer learning

EfficientNet-B0 starts with ImageNet-pretrained weights. The original classifier is replaced by a nine-class linear layer:
```commandline
Image [B, 3, 224, 224]
    ↓
EfficientNet-B0 feature extractor (frozen)
    ↓
Pretrained visual features (1,280 dimensions)
    ↓
Dropout + Linear(1280 → 9) (trainable)
    ↓
9 logits
```
- 11,529 trainable parameters during this stage
- Feature extractor frozen with requires_grad=False
- Frozen feature extractor kept in evaluation mode during classifier training to preserve BatchNorm statistics
- AdamW (lr=0.001), cross-entropy loss, 30 epochs
- Training acceleration via Apple Metal Performance Shaders (MPS) when available, with CPU fallback

## Backend API

The backend uses **FastAPI** and loads the trained EfficientNet-B0 checkpoint once during application startup. It provides two endpoints:

| Endpoint | Description |
|---|---|
| `GET /health` | Returns the service status |
| `POST /predict` | Accepts an image and returns the predicted class and scores |

The `/predict` endpoint accepts JPEG, PNG, and WebP uploads and validates image format and size. The initial implementation limits uploads to **10 MB** and rejects images with more than **16 million pixels**.

Example response from a plastic-bottle photograph:

```json
{
  "class_name": "Plastic",
  "confidence": 0.9971,
  "probabilities": {
    "Cardboard": 0.0002,
    "Food Organics": 0.0,
    "Glass": 0.0003,
    "Metal": 0.0001,
    "Miscellaneous Trash": 0.0013,
    "Paper": 0.0009,
    "Plastic": 0.9971,
    "Textile Trash": 0.0,
    "Vegetation": 0.0
  }
}
```

The `confidence` field is the model's **Softmax score**, not a calibrated probability of correctness. Returned values are rounded to four decimal places.

## Project structure
```
WasteVision/
├── src/
│   ├── analysis/
│   │   ├── eda.py
│   │   ├── check_duplicates.py
│   │   └── check_near_duplicates.py
│   ├── data/
│   │   ├── download.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── cnn.py
│   │   └── efficientnet.py
│   ├── training/
│   │   ├── train.py
│   │   └── train_efficientnet.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── evaluate_efficientnet.py
│   │   └── evaluate_test.py
│   ├── inference/
│   │   └── predict.py
│   └── api/
│       └── main.py
├── tests/
│   ├── test_api.py
│   └── test_integration.py
├── reports/
│   └── figures/
│       ├── efficientnet_b0_confusion_matrix.png
│       ├── efficientnet_b0_classification_report.txt
│       ├── efficientnet_b0_test_confusion_matrix.png
│       └── efficientnet_b0_test_report.txt
├── data/
│   └── raw/                       # Local dataset, excluded from Git
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-api.txt
├── .gitignore
└── README.md
```
Model checkpoints such as efficientnet_b0_best_acc_30.pth are generated by training and may be stored separately from the source repository.
## Getting started

### 1. Clone and set up

```bash
git clone https://github.com/Skorozp/WasteVision.git
cd WasteVision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-api.txt
```

### 2. Download and inspect RealWaste

```bash
python -m src.data.download
python -m src.data.dataset
```

### 3. Train a model

```bash
# Custom CNN baseline
python -m src.training.train

# EfficientNet-B0 transfer learning
python -m src.training.train_efficientnet
```

Training produces model checkpoints in the project root. To run predictions **without retraining**, place an existing compatible `efficientnet_b0_best_acc_30.pth` checkpoint there.

### 4. Evaluate the models

```bash
# WasteCNN validation metrics
python -m src.evaluation.evaluate

# EfficientNet-B0 validation metrics
python -m src.evaluation.evaluate_efficientnet

# Final EfficientNet-B0 test evaluation
python -m src.evaluation.evaluate_test
```

The test split should not be used to tune the model or repeatedly select checkpoints.

### 5. Classify a local image

```bash
python -m src.inference.predict sample.jpg
```

### 6. Run the FastAPI server

```bash
python -m uvicorn src.api.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to inspect and try the endpoints, or send an image from the terminal:

```bash
curl -X POST -F "file=@sample.jpg" http://127.0.0.1:8000/predict
```

### 7. Run automated tests

```bash
python -m pip install pytest httpx
python -m pytest tests/ -v
```

The project currently has **six passing tests**: five API tests using a fake classifier and one integration test using the actual checkpoint. The integration test requires the checkpoint in the project root.

For the optional perceptual-hash audit, install `ImageHash` and run:

```bash
python -m pip install ImageHash
python -m src.analysis.check_near_duplicates
```

## Docker

The API has also been built and tested locally inside a Docker container. The container uses **CPU inference**; it does not use Apple's MPS backend.

Build the image:

```bash
docker build -t wastevision-api:0.1.0 .
```

Run with the trained checkpoint mounted as a read-only file:

```bash
docker run --rm \
  --name wastevision-api \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/efficientnet_b0_best_acc_30.pth:/app/efficientnet_b0_best_acc_30.pth:ro" \
  wastevision-api:0.1.0
```

Check the service:

```bash
curl http://127.0.0.1:8000/health
curl -X POST -F "file=@sample.jpg" http://127.0.0.1:8000/predict
```

This runs the API **locally**, not on a publicly accessible server. Public HTTPS deployment is planned.

## Roadmap
- [x] Download and explore RealWaste
- [x] Create stratified train/validation/test splits
- [x] Implement and train WasteCNN from scratch
- [x] Train EfficientNet-B0 using transfer learning
- [x] Analyze validation metrics and confusion matrices
- [x] Check cross-split SHA-256 duplicates and screen for near-duplicates with pHash
- [x] Evaluate the selected model on the held-out test split
- [x] Implement single-image inference and a FastAPI backend
- [x] Add six automated backend tests
- [x] Build and verify the Dockerized API locally
- [ ] Manually review flagged near-duplicate image pairs
- [ ] Add production hardening, CI, and public HTTPS deployment
- [ ] Build and connect the mobile application (no registration required)
- [ ] Explore fine-tuning, confidence calibration, and additional data

## Why this project?
WasteVision is both a practical computer vision prototype and an exercise in building an ML system from the ground up. Rather than only calling a pretrained classifier, the project develops a small CNN baseline, measures its limitations, introduces transfer learning, and evaluates the improvement with per-class metrics.
The main takeaway so far: a strong pretrained visual representation can make a major difference when labeled data is limited.
