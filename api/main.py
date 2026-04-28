from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    HouseFeatures,
    ModelInfoResponse,
    PredictionResponse,
)
from src.predict import load_model, predict

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "models" / "metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("api")


@lru_cache(maxsize=1)
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
    try:
        load_model()
        _load_metadata()
        app.state.model_loaded = True
        logger.info("modelo e metadata carregados no startup")
    except Exception as e:
        app.state.model_loaded = False
        logger.exception("falha ao carregar modelo no startup: %s", e)
    yield


app = FastAPI(title="Preço de Casas — API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = bool(getattr(app.state, "model_loaded", False))
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
    )


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
        logger.warning("payload inválido: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
    metadata = _load_metadata()
    logger.info("predict ok: price=%.2f version=%s", price, _model_version(metadata))
    return PredictionResponse(
        predicted_price=price,
        model_version=_model_version(metadata),
    )
