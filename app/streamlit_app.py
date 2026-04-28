import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(page_title="Preço de Casas", layout="wide")

st.title("Preço de Casas — Pipeline ML com Demo")

st.markdown(
    """
Demo de um pipeline de ML completo: dataset Ames Housing, grid search comparando
RandomForest e XGBoost, persistência do vencedor, API FastAPI e este front-end Streamlit.

**Stack:** Python · scikit-learn · XGBoost · FastAPI · Streamlit
**Modelo vencedor:** XGBoost com RMSLE 0.139 no holdout

Use o menu na lateral para navegar:

- **Predição** — estime o preço de uma casa preenchendo as características principais
- **Insights** — gráficos com EDA, performance e importância de features
- **Técnico** — pipeline, hiperparâmetros vencedores e comparação RF vs XGB
"""
)

with st.sidebar:
    st.markdown("### Sobre")
    st.caption(
        "Refatoração da série didática 'Adicionando novos algoritmos' "
        "(Partes 1-3) em pipeline produtivo."
    )
    st.caption("Design completo em `docs/PRD.md`.")
