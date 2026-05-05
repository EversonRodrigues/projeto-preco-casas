from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.pkl"


@lru_cache(maxsize=4)
def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    return joblib.load(Path(path))


def predict(features: dict[str, Any]) -> float:
    expected = set(FEATURE_NAMES)
    received = set(features.keys())
    missing = expected - received
    if missing:
        raise ValueError(f"features faltantes: {sorted(missing)}")

    row = {c: [features[c]] for c in FEATURE_NAMES}
    X = pd.DataFrame(row, columns=FEATURE_NAMES)
    pred_log = load_model().predict(X)[0]
    return float(np.expm1(pred_log))
