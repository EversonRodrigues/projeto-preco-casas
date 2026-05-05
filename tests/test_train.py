import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.data import load_data, split_holdout
from src.preprocessing import NUMERIC_COLS, prepare_features
from src.train import _scrub_nan, evaluate_holdout, save_model


def test_scrub_nan_replaces_nan_with_none():
    obj = {
        "a": 1,
        "b": float("nan"),
        "c": [1, float("nan"), {"d": float("nan")}],
        "e": "ok",
    }
    out = _scrub_nan(obj)
    assert out["b"] is None
    assert out["c"][1] is None
    assert out["c"][2]["d"] is None
    assert out["a"] == 1
    assert out["e"] == "ok"
    json.dumps(out, allow_nan=False)


def test_scrub_nan_preserves_non_nan():
    obj = {"x": 0.0, "y": -1.5, "z": [1, 2, 3], "k": None}
    assert _scrub_nan(obj) == obj


def test_evaluate_holdout_returns_expected_keys():
    treino, _ = load_data()
    _, hold = split_holdout(treino)
    X_h, y_h = prepare_features(hold)

    X_dev, y_dev = prepare_features(split_holdout(treino)[0])
    # Ridge não lida com strings; usar só numéricas (smoke test, não treino real)
    ridge = Ridge(alpha=1.0).fit(X_dev[NUMERIC_COLS], y_dev)

    metrics = evaluate_holdout(ridge, X_h[NUMERIC_COLS], y_h)
    assert set(metrics.keys()) == {"rmsle", "mae_usd", "r2"}
    assert metrics["rmsle"] > 0
    assert metrics["mae_usd"] > 0
    assert -1 < metrics["r2"] <= 1


def test_save_model_writes_pickle_and_metadata(tmp_path: Path):
    treino, _ = load_data()
    X, y = prepare_features(treino)
    model = Ridge(alpha=1.0).fit(X[NUMERIC_COLS], y)

    pkl_path = tmp_path / "test_model.pkl"
    save_model(
        model,
        {"algorithm": "ridge", "metrics": {"rmsle": 0.5}},
        path=pkl_path,
    )

    assert pkl_path.exists()
    metadata_path = tmp_path / "metadata.json"
    assert metadata_path.exists()
    parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert parsed["algorithm"] == "ridge"
    assert "train_date" in parsed
    assert "lib_versions" in parsed
