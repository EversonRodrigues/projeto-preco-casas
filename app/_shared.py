import json
import os
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

API_URL = os.environ.get("PRICEAPI_URL", "http://localhost:8000")


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


class ShapUnavailable(RuntimeError):
    """Marca que SHAP não pôde ser importado/inicializado neste ambiente."""


@st.cache_resource(show_spinner=False)
def get_shap_explainer():
    """TreeExplainer cacheado. Roda no estimator interno do pipeline.

    Custo do build: ~1-2s no XGBoost com 800 árvores. Vale cachear. Em
    ambientes onde `shap`/`numba` não estão disponíveis, levanta
    `ShapUnavailable` (cacheado pelo @st.cache_resource — não tenta
    reimportar a cada chamada).
    """
    try:
        import shap
    except Exception as e:
        raise ShapUnavailable(f"shap não disponível: {e}") from e

    from src.predict import load_model

    pipeline = load_model()
    return shap.TreeExplainer(pipeline.named_steps["model"])


def shap_for_payload(payload: dict) -> tuple[list[tuple[str, float]], float]:
    """Retorna (contribuições por feature raw, base value).

    Agrega SHAP values dos one-hots de volta para o feature categórico original
    (ex: soma `Neighborhood_NridgHt`, `Neighborhood_NoRidge`, ... em
    `Neighborhood`). Numéricas passam direto.

    Levanta `ShapUnavailable` se o ambiente não suporta shap/numba.
    """
    from src.predict import load_model
    from src.preprocessing import CATEGORICAL_COLS, FEATURE_NAMES

    pipeline = load_model()
    preproc = pipeline.named_steps["preproc"]
    explainer = get_shap_explainer()

    row = pd.DataFrame([{c: payload[c] for c in FEATURE_NAMES}], columns=FEATURE_NAMES)
    Xt = preproc.transform(row)
    shap_values = explainer(Xt)

    expanded_names = list(preproc.get_feature_names_out())
    cat_set = set(CATEGORICAL_COLS)

    aggregated: dict[str, float] = {c: 0.0 for c in FEATURE_NAMES}
    for name, val in zip(expanded_names, shap_values.values[0]):
        # Para cat: o nome expandido começa com "<cat>_"; precisamos mapear de volta
        source = name
        if name not in aggregated:
            for cat in cat_set:
                if name.startswith(cat + "_"):
                    source = cat
                    break
        aggregated[source] = aggregated.get(source, 0.0) + float(val)

    contribs = sorted(aggregated.items(), key=lambda kv: abs(kv[1]), reverse=True)
    base_value = float(shap_values.base_values[0])
    return contribs, base_value
