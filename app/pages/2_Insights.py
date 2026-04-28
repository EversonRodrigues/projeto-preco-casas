import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app._shared import load_metrics, load_train
from src.data import load_data, split_holdout
from src.predict import load_model
from src.preprocessing import FEATURE_NAMES, prepare_features

st.title("Insights")

st.markdown(
    "Análise exploratória, importância das features no modelo vencedor e "
    "performance no holdout."
)

metrics = load_metrics()
treino = load_train()
model = load_model()

c1, c2, c3 = st.columns(3)
c1.metric("RMSLE", f"{metrics['metrics']['rmsle']:.4f}", help="Critério Kaggle")
c2.metric("MAE (USD)", f"$ {metrics['metrics']['mae_usd']:,.0f}")
c3.metric("R²", f"{metrics['metrics']['r2']:.3f}")

st.divider()

st.subheader("1. Importância das features (top 15)")
importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES)
top15 = importances.sort_values(ascending=True).tail(15)
fig1 = px.bar(
    x=top15.values,
    y=top15.index,
    orientation="h",
    labels={"x": "Importância", "y": "Feature"},
)
fig1.update_layout(height=500, margin={"l": 0, "r": 0, "t": 20, "b": 0})
st.plotly_chart(fig1, use_container_width=True)

st.subheader("2. Distribuição de SalePrice")
fig2 = px.histogram(
    treino,
    x="SalePrice",
    nbins=50,
    labels={"SalePrice": "Preço (USD)"},
)
fig2.update_layout(height=350, margin={"l": 0, "r": 0, "t": 20, "b": 0})
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. Preço médio por OverallQual")
st.caption(
    "Substituto para 'preço por bairro' — `Neighborhood` foi descartado pelo "
    "pré-processamento herdado das Partes 1/2 (apenas colunas numéricas restaram)."
)
qual_summary = treino.groupby("OverallQual")["SalePrice"].agg(["mean", "median", "count"]).reset_index()
fig3 = px.bar(
    qual_summary,
    x="OverallQual",
    y="mean",
    labels={"mean": "Preço médio (USD)", "OverallQual": "Qualidade geral"},
    text="count",
)
fig3.update_traces(texttemplate="n=%{text}", textposition="outside")
fig3.update_layout(height=400, margin={"l": 0, "r": 0, "t": 20, "b": 0})
st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. Predito vs. Real (holdout)")
treino_dev, treino_holdout = split_holdout(treino)
X_h, y_h_log = prepare_features(treino_holdout)
y_pred = np.expm1(model.predict(X_h))
y_real = np.expm1(y_h_log).values
limit = max(y_real.max(), y_pred.max()) * 1.05
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=y_real, y=y_pred, mode="markers", name="Holdout"))
fig4.add_trace(
    go.Scatter(
        x=[0, limit],
        y=[0, limit],
        mode="lines",
        line={"dash": "dash"},
        name="y = x",
    )
)
fig4.update_layout(
    xaxis_title="Preço real (USD)",
    yaxis_title="Preço predito (USD)",
    height=450,
    margin={"l": 0, "r": 0, "t": 20, "b": 0},
)
st.plotly_chart(fig4, use_container_width=True)

st.subheader("5. Comparação RF vs XGB no holdout")
comp = pd.DataFrame(
    {
        "RandomForest": [
            metrics["comparison"]["rf"]["rmsle"],
            metrics["comparison"]["rf"]["mae_usd"],
            metrics["comparison"]["rf"]["r2"],
        ],
        "XGBoost": [
            metrics["comparison"]["xgb"]["rmsle"],
            metrics["comparison"]["xgb"]["mae_usd"],
            metrics["comparison"]["xgb"]["r2"],
        ],
    },
    index=["RMSLE", "MAE (USD)", "R²"],
)
st.dataframe(comp.style.format({"RandomForest": "{:.4f}", "XGBoost": "{:.4f}"}))

st.subheader("6. Correlação das top features com o target")
numeric = treino.select_dtypes(include=["int64", "float64"])
corr_target = (
    numeric.corr()["SalePrice"]
    .drop(["SalePrice", "Id"], errors="ignore")
    .abs()
    .sort_values(ascending=True)
    .tail(10)
)
fig6 = px.bar(
    x=corr_target.values,
    y=corr_target.index,
    orientation="h",
    labels={"x": "Correlação (Pearson, abs)", "y": "Feature"},
)
fig6.update_layout(height=400, margin={"l": 0, "r": 0, "t": 20, "b": 0})
st.plotly_chart(fig6, use_container_width=True)
