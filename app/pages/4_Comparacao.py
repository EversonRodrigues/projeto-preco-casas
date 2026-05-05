import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app._shared import load_metrics, load_train
from src.preprocessing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    get_feature_defaults,
)

MODELS_DIR = _ROOT / "models"


@st.cache_resource(show_spinner=False)
def _load_individual(name: str):
    return joblib.load(MODELS_DIR / f"{name}_model.pkl")


@st.cache_data(show_spinner=False)
def _category_options() -> dict[str, list[str]]:
    train = load_train()
    return {
        c: sorted([str(v) for v in train[c].dropna().unique().tolist()])
        for c in CATEGORICAL_COLS
    }


def _predict_with(model_name: str, payload: dict) -> float:
    pipeline = _load_individual(model_name)
    row = pd.DataFrame([payload], columns=list(payload.keys()))
    pred_log = pipeline.predict(row)[0]
    return float(np.expm1(pred_log))


st.title("Comparação RF × XGB")
st.markdown(
    "Predição lado a lado. Ambos os modelos passaram pelo mesmo grid search e "
    "foram avaliados no mesmo holdout 20%. O **XGBoost** venceu por CV RMSLE."
)

metrics = load_metrics()

st.subheader("Métricas no holdout")
comp = pd.DataFrame(
    {
        "RandomForest": [
            metrics["comparison"]["rf"]["cv_rmsle"],
            metrics["comparison"]["rf"]["rmsle"],
            metrics["comparison"]["rf"]["mae_usd"],
            metrics["comparison"]["rf"]["r2"],
        ],
        "XGBoost": [
            metrics["comparison"]["xgb"]["cv_rmsle"],
            metrics["comparison"]["xgb"]["rmsle"],
            metrics["comparison"]["xgb"]["mae_usd"],
            metrics["comparison"]["xgb"]["r2"],
        ],
    },
    index=["CV RMSLE", "Holdout RMSLE", "Holdout MAE (USD)", "Holdout R²"],
)
st.dataframe(
    comp.style.format(
        {
            "RandomForest": "{:.4f}",
            "XGBoost": "{:.4f}",
        }
    ),
    use_container_width=True,
)

st.divider()

st.subheader("Predição lado a lado")
st.caption(
    "Ajuste algumas características-chave; o resto fica em valores típicos. "
    "Cada modelo prevê independentemente — a diferença entre os dois mostra "
    "a sensibilidade a cada feature."
)

defaults = get_feature_defaults()
cat_options = _category_options()

c1, c2, c3 = st.columns(3)
with c1:
    overall_qual = st.slider(
        "OverallQual", 1, 10, int(defaults.get("OverallQual", 5)), key="cmp_oq"
    )
    grliv = st.number_input(
        "GrLivArea (sq ft)",
        100, 6000, int(defaults.get("GrLivArea", 1500)),
        step=50,
        key="cmp_grliv",
    )
with c2:
    year_built = st.number_input(
        "YearBuilt",
        1870, 2025, int(defaults.get("YearBuilt", 1970)),
        step=1,
        key="cmp_yb",
    )
    garage_cars = st.slider(
        "GarageCars", 0, 4, int(defaults.get("GarageCars", 2)), key="cmp_gc"
    )
with c3:
    neigh_options = cat_options.get("Neighborhood", [])
    n_default = str(defaults.get("Neighborhood", neigh_options[0] if neigh_options else ""))
    neighborhood = st.selectbox(
        "Neighborhood",
        neigh_options,
        index=neigh_options.index(n_default) if n_default in neigh_options else 0,
        key="cmp_nb",
    )
    kq_options = cat_options.get("KitchenQual", [])
    kq_default = str(defaults.get("KitchenQual", kq_options[0] if kq_options else ""))
    kitchen_qual = st.selectbox(
        "KitchenQual",
        kq_options,
        index=kq_options.index(kq_default) if kq_default in kq_options else 0,
        key="cmp_kq",
    )

if st.button("Comparar predições", type="primary"):
    payload = dict(defaults)
    payload.update(
        {
            "OverallQual": float(overall_qual),
            "GrLivArea": float(grliv),
            "YearBuilt": float(year_built),
            "GarageCars": float(garage_cars),
            "Neighborhood": neighborhood,
            "KitchenQual": kitchen_qual,
        }
    )

    with st.spinner("Predizendo com ambos modelos..."):
        rf_price = _predict_with("rf", payload)
        xgb_price = _predict_with("xgb", payload)

    c_rf, c_xgb, c_diff = st.columns(3)
    c_rf.metric("RandomForest", f"US$ {rf_price:,.0f}")
    c_xgb.metric("XGBoost (vencedor)", f"US$ {xgb_price:,.0f}")
    diff = xgb_price - rf_price
    c_diff.metric(
        "Diferença (XGB − RF)",
        f"US$ {diff:+,.0f}",
        delta=f"{(diff / rf_price * 100):.1f}%",
        delta_color="normal",
    )

    fig = go.Figure(
        go.Bar(
            x=["RandomForest", "XGBoost"],
            y=[rf_price, xgb_price],
            marker_color=["#1f77b4", "#2ca02c"],
            text=[f"US$ {rf_price:,.0f}", f"US$ {xgb_price:,.0f}"],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis_title="Preço estimado (USD)",
        height=320,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)

    if abs(diff) / rf_price > 0.05:
        st.info(
            f"Os modelos divergem em mais de 5% nesta predição — pode indicar uma "
            f"combinação de features incomum no treino, onde a confiança é menor."
        )
    else:
        st.success("Modelos concordam dentro de 5% — predição mais confiável.")
