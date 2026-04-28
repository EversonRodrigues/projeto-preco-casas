import json
import sys
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import requests
import streamlit as st

from src.predict import predict as _predict_local

API_URL = "http://localhost:8000"


def get_predictor() -> Callable[[dict], float]:
    def _predict(features: dict) -> float:
        try:
            r = requests.post(
                f"{API_URL}/predict", json=features, timeout=2.0
            )
            if r.status_code == 200:
                return float(r.json()["predicted_price"])
        except requests.exceptions.RequestException:
            pass
        return _predict_local(features)

    return _predict


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((_ROOT / "docs" / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads((_ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_train() -> pd.DataFrame:
    from src.data import load_data

    treino, _ = load_data()
    return treino
