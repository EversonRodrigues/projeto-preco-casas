from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.preprocessing import CATEGORICAL_COLS, FEATURE_NAMES


def _safe_id(name: str) -> str:
    if name.isidentifier():
        return name
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if safe and safe[0].isdigit():
        safe = "f_" + safe
    return safe


_CAT_SET = set(CATEGORICAL_COLS)
_fields: dict[str, Any] = {}
for _name in FEATURE_NAMES:
    _safe = _safe_id(_name)
    _type = str if _name in _CAT_SET else float
    if _safe == _name:
        _fields[_safe] = (_type, ...)
    else:
        _fields[_safe] = (_type, Field(..., alias=_name))

HouseFeatures = create_model(
    "HouseFeatures",
    __config__=ConfigDict(populate_by_name=True, protected_namespaces=()),
    **_fields,
)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_price: float
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    algorithm: str
    metrics: dict
    train_date: str
    lib_versions: dict
