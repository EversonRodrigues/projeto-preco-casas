import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import streamlit as st

from app._shared import get_predictor, load_train
from src.preprocessing import get_feature_defaults

st.title("Estimativa de preço")

st.markdown(
    "Preencha as características da casa abaixo. Os campos não exibidos são "
    "preenchidos automaticamente com a mediana do treino."
)

defaults = get_feature_defaults()

VISIBLE_FIELDS = [
    ("OverallQual", "Qualidade geral (1-10)", 1, 10, 1),
    ("OverallCond", "Condição geral (1-10)", 1, 10, 1),
    ("GrLivArea", "Área habitável acima do solo (sq ft)", 100, 6000, 50),
    ("LotArea", "Tamanho do lote (sq ft)", 1000, 60000, 500),
    ("YearBuilt", "Ano de construção", 1870, 2025, 1),
    ("YearRemodAdd", "Ano da última reforma", 1950, 2025, 1),
    ("TotalBsmtSF", "Área total do porão (sq ft)", 0, 5000, 50),
    ("1stFlrSF", "Área do 1º andar (sq ft)", 100, 5000, 50),
    ("2ndFlrSF", "Área do 2º andar (sq ft)", 0, 3000, 50),
    ("FullBath", "Banheiros completos", 0, 5, 1),
    ("HalfBath", "Banheiros parciais", 0, 3, 1),
    ("BedroomAbvGr", "Quartos acima do solo", 0, 8, 1),
    ("GarageCars", "Vagas de garagem", 0, 4, 1),
    ("Fireplaces", "Lareiras", 0, 4, 1),
]

col1, col2 = st.columns(2)
inputs: dict[str, float] = {}
for i, (name, label, lo, hi, step) in enumerate(VISIBLE_FIELDS):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        default_val = defaults.get(name, 0)
        if isinstance(step, int) and float(default_val).is_integer():
            inputs[name] = float(
                st.number_input(
                    label,
                    min_value=int(lo),
                    max_value=int(hi),
                    value=int(default_val),
                    step=int(step),
                    key=f"input_{name}",
                )
            )
        else:
            inputs[name] = st.number_input(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default_val),
                step=float(step),
                key=f"input_{name}",
            )

submit = st.button("Estimar preço", type="primary")

if submit:
    edits = sum(
        1
        for name in inputs
        if abs(inputs[name] - defaults[name]) > 1e-9
    )
    if edits < 5:
        st.warning(
            f"Apenas {edits} campos foram alterados em relação à mediana. "
            "A estimativa será dominada pelos defaults."
        )

    payload = {**defaults, **inputs}
    predictor = get_predictor()
    with st.spinner("Estimando..."):
        price = predictor(payload)

    st.metric("Preço estimado", f"US$ {price:,.0f}")

    treino = load_train()
    sale_dist = treino["SalePrice"].values
    pct = float((sale_dist <= price).mean() * 100)
    st.caption(
        f"Esse valor está no **percentil {pct:.0f}** da distribuição de preços "
        f"do treino (1460 imóveis em Ames, IA, 2006-2010)."
    )

    p25, p50, p75 = np.percentile(sale_dist, [25, 50, 75])
    st.caption(
        f"Referência da distribuição — P25: US$ {p25:,.0f} · "
        f"Mediana: US$ {p50:,.0f} · P75: US$ {p75:,.0f}"
    )
