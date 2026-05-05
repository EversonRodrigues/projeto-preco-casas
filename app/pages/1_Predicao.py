import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app._shared import get_predictor, load_train, shap_for_payload
from src.preprocessing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    get_feature_defaults,
)


@st.cache_data(show_spinner=False)
def _category_options() -> dict[str, list[str]]:
    train = load_train()
    return {
        c: sorted([str(v) for v in train[c].dropna().unique().tolist()])
        for c in CATEGORICAL_COLS
    }


# --- Form schema ---------------------------------------------------------
# Cada seção é (titulo, [campos]). Campos numéricos: (name, label, lo, hi, step)
# Campos categoricals: (name, label) -- options vêm do treino.

NUMERIC_FIELDS = {
    "Localização e lote": [
        ("LotArea", "Tamanho do lote (sq ft)", 1000, 60000, 500),
        ("LotFrontage", "Frente do lote (ft)", 0, 350, 5),
    ],
    "Construção": [
        ("OverallQual", "Qualidade geral (1-10)", 1, 10, 1),
        ("OverallCond", "Condição geral (1-10)", 1, 10, 1),
        ("YearBuilt", "Ano de construção", 1870, 2025, 1),
        ("YearRemodAdd", "Ano da última reforma", 1950, 2025, 1),
    ],
    "Áreas": [
        ("GrLivArea", "Área habitável acima do solo (sq ft)", 100, 6000, 50),
        ("TotalBsmtSF", "Área total do porão (sq ft)", 0, 5000, 50),
        ("1stFlrSF", "Área 1º andar (sq ft)", 100, 5000, 50),
        ("2ndFlrSF", "Área 2º andar (sq ft)", 0, 3000, 50),
    ],
    "Banheiros e quartos": [
        ("FullBath", "Banheiros completos", 0, 5, 1),
        ("HalfBath", "Banheiros parciais", 0, 3, 1),
        ("BedroomAbvGr", "Quartos acima do solo", 0, 8, 1),
        ("TotRmsAbvGrd", "Total de cômodos acima do solo", 0, 15, 1),
    ],
    "Garagem e externos": [
        ("GarageCars", "Vagas de garagem", 0, 4, 1),
        ("Fireplaces", "Lareiras", 0, 4, 1),
        ("WoodDeckSF", "Deck de madeira (sq ft)", 0, 1000, 10),
        ("OpenPorchSF", "Varanda aberta (sq ft)", 0, 600, 10),
    ],
}

CATEGORICAL_FIELDS = {
    "Localização e lote": [
        ("Neighborhood", "Bairro"),
        ("MSZoning", None),  # nem todos os cat estão no treino raw — pulado se ausente
    ],
    "Construção": [
        ("BldgType", "Tipo de construção"),
        ("HouseStyle", "Estilo da casa"),
        ("Foundation", "Tipo de fundação"),
    ],
    "Áreas": [
        ("ExterQual", "Qualidade do acabamento externo"),
        ("BsmtQual", "Qualidade do porão"),
        ("KitchenQual", "Qualidade da cozinha"),
    ],
    "Banheiros e quartos": [],
    "Garagem e externos": [
        ("GarageType", "Tipo de garagem"),
        ("Heating", "Sistema de aquecimento"),
    ],
}

# --- Render --------------------------------------------------------------

st.title("Estimativa de preço")
st.markdown(
    "Preencha as características da casa abaixo. Os campos não exibidos são "
    "preenchidos automaticamente com a mediana (numéricos) ou a moda (categoricals) do treino."
)

defaults = get_feature_defaults()
cat_options = _category_options()
inputs: dict = {}


def _render_numeric(name: str, label: str, lo, hi, step):
    default_val = defaults.get(name, 0)
    if isinstance(step, int) and float(default_val).is_integer():
        return float(
            st.number_input(
                label,
                min_value=int(lo),
                max_value=int(hi),
                value=int(default_val),
                step=int(step),
                key=f"input_{name}",
            )
        )
    return st.number_input(
        label,
        min_value=float(lo),
        max_value=float(hi),
        value=float(default_val),
        step=float(step),
        key=f"input_{name}",
    )


def _render_categorical(name: str, label: str):
    options = cat_options.get(name)
    if not options:
        return None
    default = str(defaults.get(name, options[0]))
    if default not in options:
        default = options[0]
    return st.selectbox(label, options, index=options.index(default), key=f"input_{name}")


for section, num_fields in NUMERIC_FIELDS.items():
    cat_fields = [
        (n, l) for n, l in CATEGORICAL_FIELDS.get(section, []) if n in CATEGORICAL_COLS and l
    ]
    if not num_fields and not cat_fields:
        continue
    st.subheader(section)
    col1, col2 = st.columns(2)
    items = [("num", f) for f in num_fields] + [("cat", f) for f in cat_fields]
    for i, (kind, field) in enumerate(items):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            if kind == "num":
                name, label, lo, hi, step = field
                inputs[name] = _render_numeric(name, label, lo, hi, step)
            else:
                name, label = field
                val = _render_categorical(name, label)
                if val is not None:
                    inputs[name] = val

submit = st.button("Estimar preço", type="primary")

if submit:
    edits = 0
    for name, val in inputs.items():
        d = defaults.get(name)
        if name in CATEGORICAL_COLS:
            if val != d:
                edits += 1
        else:
            if abs(float(val) - float(d)) > 1e-9:
                edits += 1
    if edits < 5:
        st.warning(
            f"Apenas {edits} campos foram alterados em relação aos valores típicos. "
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

    st.session_state["last_payload"] = payload
    st.session_state["last_price"] = price

    with st.expander("Por que esse preço? — análise SHAP", expanded=True):
        st.caption(
            "Contribuição de cada feature à predição (em log USD). "
            "Categoricals são agregadas — ex: o efeito do bairro inclui todas as "
            "dummies one-hot relacionadas. Soma das contribuições + base ≈ log(preço)."
        )
        try:
            with st.spinner("Calculando contribuições..."):
                contribs, base = shap_for_payload(payload)
            top = contribs[:12]
            labels = [name for name, _ in top][::-1]
            values = [v for _, v in top][::-1]
            colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.3f}" for v in values],
                    textposition="outside",
                )
            )
            fig.update_layout(
                xaxis_title="Contribuição (log USD)",
                yaxis_title="Feature",
                height=420,
                margin={"l": 0, "r": 20, "t": 20, "b": 0},
            )
            st.plotly_chart(fig, use_container_width=True)

            base_usd = float(np.expm1(base))
            st.caption(
                f"Verde = empurra o preço pra cima · Vermelho = pra baixo. "
                f"Base (preço médio do treino em log): **{base:.3f}** ≈ "
                f"US$ {base_usd:,.0f}."
            )
        except Exception as e:
            st.warning(f"Não foi possível gerar a análise SHAP: {e}")
