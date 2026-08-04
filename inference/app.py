"""FastAPI service for NEU surface defect classification."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile

from inference.predictor import DefectPredictor, create_predictor
from inference.schemas import HealthResponse, PredictionResponse
from training.utils import load_config, setup_logging

logger = setup_logging(__name__)

predictor: DefectPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global predictor
    try:
        model_checkpoint = os.getenv("MODEL_CHECKPOINT")
        predictor = create_predictor(model_checkpoint)
        logger.info(
            "Inference model loaded successfully from %s",
            getattr(predictor, "checkpoint_path", "unknown"),
        )
    except FileNotFoundError as exc:
        logger.error("Could not load model: %s", exc)
        predictor = None
    yield


app = FastAPI(
    title="NEU Surface Defect API",
    description="Classify steel surface defects from uploaded images.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        classes=predictor.class_names if predictor else [],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        result = predictor.predict(image_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}") from exc

    return PredictionResponse(**result)


def main() -> None:
    import uvicorn

    config = load_config()
    inference_cfg = config["inference"]
    uvicorn.run(
        "inference.app:app",
        host=inference_cfg["host"],
        port=inference_cfg["port"],
        reload=False,
    )


if __name__ == "__main__":
    main()
