
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.inference.predict import CLASS_NAMES


def test_real_model_prediction():
    image = Image.new("RGB", (224, 224), "white")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.jpg",
                    buffer.getvalue(),
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 200

    result = response.json()

    assert result["class_name"] in CLASS_NAMES
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) == 9

    assert set(result["probabilities"]) == set(CLASS_NAMES)

    assert sum(result["probabilities"].values()) == pytest.approx(
        1.0,
        abs=0.001
    )
