import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.preprocessing import get_feature_defaults


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200_and_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_returns_200_with_metadata(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "algorithm" in body
    assert "metrics" in body
    assert "rmsle" in body["metrics"]


def test_predict_with_defaults_returns_positive_price(client):
    payload = get_feature_defaults()
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["predicted_price"], float)
    assert body["predicted_price"] > 0
    assert "model_version" in body


def test_predict_missing_field_returns_422(client):
    payload = get_feature_defaults()
    incomplete = {k: v for k, v in payload.items() if k != "OverallQual"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422
