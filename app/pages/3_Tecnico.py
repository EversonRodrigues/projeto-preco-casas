import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from app._shared import load_metadata, load_metrics

st.title("Visão técnica")

metadata = load_metadata()
metrics = load_metrics()

st.subheader("Pipeline")
st.markdown(
    """
```
data/train_3_1.csv  ─┐
                     │
              src/data.load_data
                     │
              src/data.split_holdout (20% holdout)
                     │
            src/preprocessing.prepare_features
              (48 features numéricas, np.log1p no target)
                     │
            ┌────────┴────────┐
            │                 │
   src/train.run_grid_search('rf')   src/train.run_grid_search('xgb')
   (12 combos × 5 folds)             (24 combos × 5 folds)
            │                 │
            └────────┬────────┘
                     │
       Vencedor por menor RMSLE no CV
                     │
            src/train.save_model
            (models/best_model.pkl + metadata.json)
                     │
                     ├── api/main.py  (POST /predict)
                     └── app/streamlit_app.py  (consumidor)
```
"""
)

st.subheader("Modelo vencedor")
st.write(
    f"**Algoritmo:** {metadata['algorithm']}  ·  "
    f"**Treinado em:** {metadata['train_date']}"
)

st.subheader("Hiperparâmetros vencedores")
hp_winner = (
    metrics["comparison"][metadata["algorithm"]]["best_params"]
)
st.json(hp_winner)

st.subheader("Métricas no holdout")
m = metadata["metrics"]
mc1, mc2, mc3 = st.columns(3)
mc1.metric("RMSLE", f"{m['rmsle']:.4f}")
mc2.metric("MAE (USD)", f"$ {m['mae_usd']:,.0f}")
mc3.metric("R²", f"{m['r2']:.3f}")

st.subheader("Comparação RF vs XGB")
comp_rows = []
for alg in ("rf", "xgb"):
    c = metrics["comparison"][alg]
    comp_rows.append(
        {
            "algoritmo": alg,
            "cv_rmsle": c["cv_rmsle"],
            "holdout_rmsle": c["rmsle"],
            "holdout_mae": c["mae_usd"],
            "holdout_r2": c["r2"],
            "best_params": str(c["best_params"]),
        }
    )
st.dataframe(
    pd.DataFrame(comp_rows).set_index("algoritmo"),
    use_container_width=True,
)

st.subheader("Versões das libs no treino")
st.json(metadata["lib_versions"])

st.subheader("Links")
st.markdown(
    """
- Dataset: [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- PRD do projeto: `docs/PRD.md`
- Plano de implementação: `docs/plano_implementacao.md`
- Notebook refatorado: `notebooks/03_grid_search.ipynb`
"""
)
