
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import main as api_module


class FakeWasteClassifier:
    def predict(self, image):
        return {
            "class_name": "Plastic",
            "confidence": 0.95,
            "probabilities": {
                "Cardboard": 0.0,
                "Food Organics": 0.0,
                "Glass": 0.02,
                "Metal": 0.03,
                "Miscellaneous Trash": 0.0,
                "Paper": 0.0,
                "Plastic": 0.95,
                "Textile Trash": 0.0,
                "Vegetation": 0.0
            }
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "WasteClassifier",
        FakeWasteClassifier
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_image(client):
    image = Image.new("RGB", (224, 224), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    response = client.post(
        "/predict",
        files={
            "file": (
                "sample.jpg",
                buffer.getvalue(),
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["class_name"] == "Plastic"
    assert data["confidence"] == 0.95
    assert len(data["probabilities"]) == 9


def test_predict_unsupported_format(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "document.txt",
                b"Not an image",
                "text/plain"
            )
        }
    )

    assert response.status_code == 415


def test_predict_corrupted_image(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "corrupted.jpg",
                b"This is not JPEG data",
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 400


def test_predict_large_image(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "large.jpg",
                b"x" * (10 * 1024 * 1024 + 1),
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 413
