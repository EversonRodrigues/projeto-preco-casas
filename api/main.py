from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.schemas import (
    HealthResponse,
    HouseFeatures,
    ModelInfoResponse,
    PredictionResponse,
)
from src.predict import load_model, predict

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "models" / "metadata.json"


def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _model_version(metadata: dict) -> str:
    if not metadata:
        return "unknown"
    return f"{metadata.get('algorithm', 'unknown')}-{metadata.get('train_date', 'unknown')}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Preço de Casas — API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        load_model()
        return HealthResponse(status="ok", model_loaded=True)
    except Exception:
        return HealthResponse(status="degraded", model_loaded=False)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    metadata = _load_metadata()
    if not metadata:
        raise HTTPException(status_code=503, detail="metadata indisponível")
    return ModelInfoResponse(**metadata)


@app.post("/predict", response_model=PredictionResponse)
def post_predict(features: HouseFeatures) -> PredictionResponse:
    payload = features.model_dump(by_alias=True)
    try:
        price = predict(payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    metadata = _load_metadata()
    return PredictionResponse(
        predicted_price=price,
        model_version=_model_version(metadata),
    )
