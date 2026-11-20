
from contextlib import asynccontextmanager
from io import BytesIO
from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference.predict import WasteClassifier

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: dict[str, float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = WasteClassifier()
    yield


app = FastAPI(
    title="WasteVision API",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format"
        )

    data = await file.read(10 * 1024 * 1024 + 1)

    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB"
        )

    try:
        with Image.open(BytesIO(data)) as image:
            if image.width * image.height > 16_000_000:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large"
                )

            return app.state.classifier.predict(image)

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image"
        )
