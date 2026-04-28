import pytest
from sklearn.base import BaseEstimator

from src.preprocessing import get_feature_defaults
from src.predict import load_model, predict


def test_load_model_returns_estimator():
    model = load_model()
    assert isinstance(model, BaseEstimator)
    assert hasattr(model, "predict")


def test_predict_with_defaults_in_reasonable_range():
    price = predict(get_feature_defaults())
    assert isinstance(price, float)
    assert 50_000 <= price <= 500_000


def test_predict_missing_key_raises():
    defaults = get_feature_defaults()
    incomplete = {k: v for k, v in list(defaults.items())[:-1]}
    with pytest.raises(ValueError, match="features faltantes"):
        predict(incomplete)
