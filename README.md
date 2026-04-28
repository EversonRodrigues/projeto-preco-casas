# Preço de Casas — Pipeline ML com Demo

> **Demo público:** https://projeto-preco-casas-rai4onxlzxzwjxvntdxmdv.streamlit.app/

Pipeline completo de Machine Learning para estimar preços de imóveis no dataset
**Ames Housing** (Kaggle "House Prices: Advanced Regression Techniques").
Refatoração da série didática "Adicionando novos algoritmos" (Partes 1-3) em
arquitetura produtiva: módulos testáveis, API FastAPI, app Streamlit multi-página.

- **Documento de design:** [`docs/PRD.md`](docs/PRD.md)
- **Plano de implementação:** [`docs/plano_implementacao.md`](docs/plano_implementacao.md)
- **Decisões e tradeoffs:** [`docs/decisions.md`](docs/decisions.md)
- **Métricas finais:** [`docs/metrics.json`](docs/metrics.json)

## Resultado

Modelo vencedor: **XGBoost**

| Métrica | Holdout 20% |
|---|---|
| RMSLE | **0.139** (alvo PRD ≤ 0.16) |
| MAE | US$ 17,195 |
| R² | 0.883 |

Comparação no mesmo holdout (random_state=42):

| Modelo | RMSLE | MAE (USD) | R² |
|---|---|---|---|
| Ridge (baseline) | ver `02_baseline.ipynb` | — | — |
| RandomForest | 0.145 | 17,307 | 0.883 |
| **XGBoost (vencedor)** | **0.139** | **17,195** | **0.883** |

## Como rodar localmente

```bash
git clone <repo>
cd Projeto_Preço_de_casas

python -m venv .venv
# Windows bash:
source .venv/Scripts/activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt

# Treina e salva models/best_model.pkl + metadata.json (~3-10 min)
python -m src.train

# Sobe a API em http://localhost:8000
uvicorn api.main:app --reload

# Em outro terminal: app Streamlit em http://localhost:8501
streamlit run app/streamlit_app.py
```

Com a API rodando, o Streamlit chama `POST /predict`. Sem a API, ele cai
automaticamente para `src.predict.predict` direto (ver decisão R-04 no PRD).

## Testes

```bash
pytest tests/ -v
```

12 testes cobrindo preprocessing, predict e API.

## Critérios de aceitação (PRD Seção 6)

| ID | Critério | Status |
|---|---|---|
| CA-01 | `prepare_features` retorna `X` shape `(N, 49)` e `y` com log-transform | ⚠️ atendido com **48 features** (não 49); ver decisão D-03 |
| CA-02 | RF e XGB grid em < 15 min cada em hardware doméstico | ✅ ambos completam em poucos minutos |
| CA-03 | RMSLE ≤ 0.16 no holdout 20% | ✅ **0.139** |
| CA-04 | `models/best_model.pkl` carrega e prediz | ✅ `tests/test_predict.py` |
| CA-05 | `POST /predict` retorna 200 em < 500ms | ✅ p50 ~7ms, max ~14ms |
| CA-06 | `POST /predict` payload inválido → 422 | ✅ `tests/test_api.py` |
| CA-07 | Streamlit carrega 3 páginas sem erro | ✅ validado via `streamlit.testing.AppTest` |
| CA-08 | Página Predição estima preço em < 2s end-to-end | ✅ AppTest sub-segundo |
| CA-09 | Página Insights exibe 6 gráficos | ✅ `app/pages/2_Insights.py` |
| CA-10 | Deploy público com link no README | ✅ [Streamlit Cloud](https://projeto-preco-casas-rai4onxlzxzwjxvntdxmdv.streamlit.app/) |

## Estrutura

```
src/                # núcleo do pipeline
├── data.py             # load_data, split_holdout
├── preprocessing.py    # prepare_features, FEATURE_NAMES (48), get_feature_defaults
├── train.py            # run_grid_search, evaluate_holdout, save_model
└── predict.py          # load_model (cacheado), predict

api/                # FastAPI
├── main.py             # /health, /model-info, /predict
└── schemas.py          # HouseFeatures (gerado dinâmico das 48 features)

app/                # Streamlit
├── streamlit_app.py    # entry
├── _shared.py          # helpers (predictor com fallback)
└── pages/
    ├── 1_Predicao.py   # form + estimativa + percentil
    ├── 2_Insights.py   # 6 gráficos Plotly
    └── 3_Tecnico.py    # hiperparams, métricas, comparação

notebooks/
├── 01_eda.ipynb                # análise exploratória
├── 02_baseline.ipynb           # Ridge baseline
├── 03_grid_search.ipynb        # refatoração consumindo src/
├── 03_grid_search_original.ipynb  # original (referência)
└── 04_avaliacao_final.ipynb    # predições, resíduos, comparação

models/             # artefatos (versionados)
├── best_model.pkl
└── metadata.json

data/               # CSVs pré-processados pelas Partes 1/2 (versionados)
├── train_3_1.csv   # 1460 × 85
└── test_3_1.csv    # 1459 × 84

tests/              # smoke tests pytest
docs/               # PRD, plano, decisions, métricas
```

Detalhamento completo da arquitetura: `docs/PRD.md` (Seção 5).

## Stack

Python 3.10+ · pandas · numpy · scikit-learn 1.5 · XGBoost 2.1 · joblib ·
FastAPI · Pydantic 2 · Streamlit · Plotly · matplotlib · seaborn · pytest

## Licença

MIT
