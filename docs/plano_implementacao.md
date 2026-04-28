# Plano de Implementação — Preço de Casas

| Campo | Valor |
|---|---|
| Documento-pai | [`docs/PRD.md`](./PRD.md) |
| Data | 2026-04-26 |
| Origem | Decomposição dos Milestones M1-M6 do PRD em tarefas executáveis |

> Este plano segue o espírito da skill `writing-plans` (não instalada localmente). Cada tarefa tem: **arquivos**, **o que muda**, **critério de feito**, **comando de verificação** quando aplicável.

---

## Pré-requisitos (executar uma vez)

### P-1. Inicializar repositório git
```bash
cd "C:/Pasta 01 - Everson/Projeto_Preço_de_casas"
git init
git add CLAUDE.md docs/PRD.md docs/plano_implementacao.md
git commit -m "chore: inicializa repo com PRD e plano de implementação"
```
**Feito quando:** `git status` retorna working tree clean.

### P-2. Criar `requirements.txt` na raiz
**Arquivo novo:** `requirements.txt`
```
pandas==2.2.*
numpy==1.26.*
scikit-learn==1.5.*
xgboost==2.1.*
joblib==1.4.*
fastapi==0.115.*
uvicorn==0.32.*
pydantic==2.9.*
streamlit==1.39.*
matplotlib==3.9.*
seaborn==0.13.*
plotly==5.24.*
requests==2.32.*
pytest==8.3.*
```
**Feito quando:** `pip install -r requirements.txt` completa sem erros em venv novo.

### P-3. Criar `.gitignore`
**Arquivo novo:** `.gitignore`
```
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/
.pytest_cache/
.DS_Store
*.egg-info/
.streamlit/secrets.toml
# dados ficam versionados (são pequenos, parte do "produto")
# models/*.pkl ficam versionados (são produto reproduzível)
```
**Feito quando:** `git status` ignora `__pycache__/` e `.ipynb_checkpoints/`.

### P-4. Criar estrutura de pastas
```bash
mkdir -p src api app/pages models notebooks tests data
touch src/__init__.py api/__init__.py
```
**Feito quando:** árvore corresponde à Seção 5 do PRD.

### P-5. Mover dados para `data/`
```bash
mv "Usando o grid_search para encontrar os melhores parâmetros para o modelo/train_3_1.csv" data/
mv "Usando o grid_search para encontrar os melhores parâmetros para o modelo/test_3_1.csv" data/
mv "Usando o grid_search para encontrar os melhores parâmetros para o modelo/Adicionando%20novos%20algoritmos%20-%20Parte%203.ipynb" notebooks/03_grid_search_original.ipynb
rmdir "Usando o grid_search para encontrar os melhores parâmetros para o modelo"
```
**Feito quando:** `data/` tem os 2 CSVs; `notebooks/` tem cópia do original como referência.

---

## M1 — Refatoração base do pipeline em módulos

> **Objetivo:** mover lógica do notebook para `src/`, deixando o notebook como consumidor.

### M1.1. `src/data.py` — carregamento e split
**Arquivo novo:** `src/data.py`

Funções públicas:
- `load_data() -> tuple[pd.DataFrame, pd.DataFrame]`: lê `data/train_3_1.csv` e `data/test_3_1.csv`
- `split_holdout(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]`: separa 20% holdout

Caminhos resolvidos via `pathlib.Path(__file__).resolve().parents[1] / "data"` para funcionar de qualquer cwd.

**Critério:** `from src.data import load_data; t, te = load_data()` retorna shapes `(1460, 85)` e `(1459, 84)`.

### M1.2. `src/preprocessing.py` — features e target
**Arquivo novo:** `src/preprocessing.py`

Funções públicas:
- `prepare_features(df: pd.DataFrame, has_target: bool = True) -> tuple[pd.DataFrame, pd.Series | None]`:
  - Seleciona apenas colunas numéricas (`int64`, `float64`)
  - Se `has_target`, separa `y = np.log1p(df["SalePrice"])` e remove `SalePrice` de X
  - Remove `Id` de X (se presente)
  - Retorna `(X, y)` com X tendo as 49 features esperadas
- `get_feature_defaults() -> dict[str, float]`: lê o treino e retorna `{coluna: mediana}` para todas as 49 features. Cacheado em memória após primeira chamada.
- `FEATURE_NAMES: list[str]`: constante com os 49 nomes em ordem fixa (importante para a API).

**Critério:** `X.shape[1] == 49` e `y.min() > 0` (log de preços positivos).

### M1.3. `src/train.py` — grid search e persistência
**Arquivo novo:** `src/train.py`

Funções públicas:
- `run_grid_search(model_type: Literal["rf", "xgb"], X: pd.DataFrame, y: pd.Series) -> tuple[BaseEstimator, dict]`:
  - Define grid conforme **PRD Seção 3.C2** (RF: 12 combos, XGB: 24 combos)
  - `KFold(n_splits=5, shuffle=True, random_state=42)`
  - `scoring="neg_root_mean_squared_error"` (RMSE no log = RMSLE)
  - Retorna `(best_estimator, cv_results_summary)`
- `evaluate_holdout(model, X_h: pd.DataFrame, y_h_log: pd.Series) -> dict`:
  - Prediz em log, converte com `np.expm1`
  - Retorna `{"rmsle": ..., "mae_usd": ..., "r2": ...}`
- `save_model(model, metadata: dict, path: str = "models/best_model.pkl") -> None`:
  - `joblib.dump(model, path)`
  - Escreve `models/metadata.json` com: algorithm, hyperparams, metrics, train_date (ISO 8601 UTC), lib_versions
- `main()`: roda grid em RF e XGB, escolhe o de menor RMSLE no CV, salva como modelo vencedor. Permite executar `python -m src.train`.

**Critério:** após `python -m src.train`, existem `models/best_model.pkl` e `models/metadata.json`.

### M1.4. `src/predict.py` — inferência
**Arquivo novo:** `src/predict.py`

Funções públicas:
- `load_model(path: str = "models/best_model.pkl") -> BaseEstimator`: carrega com `joblib.load`. Cacheado via `functools.lru_cache`.
- `predict(features: dict[str, float]) -> float`:
  - Valida que `features.keys() == set(FEATURE_NAMES)` (raise `ValueError` se faltam chaves)
  - Constrói DataFrame com colunas em ordem de `FEATURE_NAMES`
  - Prediz em log, retorna `float(np.expm1(pred))`

**Critério:** `predict({...defaults...})` retorna float entre 50_000 e 500_000 (range razoável de Ames).

### M1.5. Refatorar notebook
**Arquivo novo:** `notebooks/03_grid_search.ipynb`

Substituir células do original por chamadas a `src/`:
```python
from src.data import load_data, split_holdout
from src.preprocessing import prepare_features
from src.train import run_grid_search, evaluate_holdout, save_model
import json

treino, _ = load_data()
treino_dev, treino_holdout = split_holdout(treino)

X_dev, y_dev = prepare_features(treino_dev)
X_h, y_h = prepare_features(treino_holdout)

# Grid search
best_rf, _ = run_grid_search("rf", X_dev, y_dev)
best_xgb, _ = run_grid_search("xgb", X_dev, y_dev)

# Avaliação
metrics_rf = evaluate_holdout(best_rf, X_h, y_h)
metrics_xgb = evaluate_holdout(best_xgb, X_h, y_h)
print(metrics_rf, metrics_xgb)

# Salvar vencedor
winner, name = (best_rf, "rf") if metrics_rf["rmsle"] < metrics_xgb["rmsle"] else (best_xgb, "xgb")
save_model(winner, {"algorithm": name, "hyperparams": winner.get_params(), "metrics": metrics_rf if name == "rf" else metrics_xgb})
```

Adicionar narrativa em markdown explicando cada passo. **Não duplicar lógica** — apenas chamar.

**Critério:** notebook executa do início ao fim sem erros, gera o `.pkl`.

### M1.6. Validação de M1
```bash
python -m src.train
ls -la models/best_model.pkl models/metadata.json
```
**Feito quando:** ambos existem; `cat models/metadata.json` mostra métricas válidas.

---

## M2 — Pipeline robusto + smoke tests

### M2.1. `tests/test_preprocessing.py`
**Arquivo novo.** Testa:
- `prepare_features` em treino retorna `X.shape[1] == 49` e `y` não None
- `prepare_features` em teste (`has_target=False`) retorna `y is None`
- `get_feature_defaults()` retorna dict com 49 chaves, todos floats
- `FEATURE_NAMES` tem comprimento 49 e nenhum duplicado

**Critério:** `pytest tests/test_preprocessing.py -v` passa.

### M2.2. `tests/test_predict.py`
**Arquivo novo.** Testa:
- `load_model()` retorna estimator
- `predict(get_feature_defaults())` retorna float em range razoável (50k-500k)
- `predict({...})` com chave faltante levanta `ValueError`

**Critério:** `pytest tests/test_predict.py -v` passa.

### M2.3. Métricas finais em `docs/metrics.json`
Adicionar em `src/train.py::main()`: depois de salvar modelo, escrever `docs/metrics.json` com as métricas do holdout do vencedor + comparação RF vs XGB.

**Critério:** `docs/metrics.json` existe e contém `rmsle`, `mae_usd`, `r2`, e `comparison: {rf: {...}, xgb: {...}}`.

### M2.4. Validação CA-03 (PRD)
Verificar `rmsle` em `docs/metrics.json`.
- Se ≤ 0.16: registrar como sucesso
- Se > 0.16: documentar valor real em `docs/decisions.md` (sem bloquear)

---

## M3 — API FastAPI

### M3.1. `api/schemas.py` — Pydantic
**Arquivo novo.** Define:
- `HouseFeatures(BaseModel)` com os 49 campos (gerar via `for name in FEATURE_NAMES: ...`). Todos `float`, todos obrigatórios.
- `PredictionResponse(BaseModel)`: `predicted_price: float`, `model_version: str`
- `HealthResponse(BaseModel)`: `status: str`, `model_loaded: bool`
- `ModelInfoResponse(BaseModel)`: campos de `models/metadata.json`

**Critério:** `from api.schemas import HouseFeatures; HouseFeatures(**defaults)` valida sem erros.

### M3.2. `api/main.py` — endpoints
**Arquivo novo.** Estrutura:
```python
from fastapi import FastAPI, HTTPException
from src.predict import load_model, predict
from api.schemas import HouseFeatures, PredictionResponse, HealthResponse, ModelInfoResponse

app = FastAPI(title="Preço de Casas — API")

@app.on_event("startup")
def _warmup():
    load_model()  # carrega .pkl uma vez

@app.get("/health", response_model=HealthResponse)
def health(): ...

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(): ...

@app.post("/predict", response_model=PredictionResponse)
def post_predict(features: HouseFeatures):
    try:
        price = predict(features.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictionResponse(predicted_price=price, model_version=...)
```

**Critério:** `uvicorn api.main:app` sobe; `curl http://localhost:8000/health` retorna 200.

### M3.3. `tests/test_api.py`
**Arquivo novo.** Usa `fastapi.testclient.TestClient`. Testa:
- `GET /health` retorna 200 com `model_loaded=True`
- `POST /predict` com defaults completos retorna 200, `predicted_price` é float positivo
- `POST /predict` com payload faltando 1 campo retorna 422
- `GET /model-info` retorna 200 com algorithm e metrics

**Critério:** `pytest tests/test_api.py -v` passa.

---

## M4 — Streamlit multi-página

### M4.1. `app/streamlit_app.py` — entry e config global
**Arquivo novo.**
- `st.set_page_config(page_title="Preço de Casas", layout="wide")`
- Sidebar com descrição do projeto e link pro repo
- Streamlit detecta automaticamente as páginas em `app/pages/`

### M4.2. Helper compartilhado: `app/_shared.py`
**Arquivo novo.**
- `get_predictor()`: tenta `requests.post` na API local; se falhar (deploy sem API), usa `src.predict.predict` direto. Decisão R-04 do PRD.
- `load_metrics()`: lê `docs/metrics.json` cacheado com `@st.cache_data`
- `load_metadata()`: lê `models/metadata.json` cacheado

### M4.3. `app/pages/1_Predicao.py` — formulário cliente
**Arquivo novo.** ~15 campos visíveis (escolher pelos top features mais intuitivos):
- `OverallQual` (1-10), `GrLivArea` (m²/ft²), `YearBuilt`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `GarageCars`, `GarageArea` (se existir, senão pular), `FullBath`, `HalfBath`, `BedroomAbvGr`, `LotArea`, `Fireplaces`, `YearRemodAdd`, `OverallCond`

Outros 34 campos preenchidos com `get_feature_defaults()`. Botão "Estimar preço" → chama `get_predictor()` → mostra:
- Preço grande no topo
- Faixa: percentil onde o preço cai vs distribuição do treino (carregar uma vez via `@st.cache_data`)
- Aviso se < 5 campos foram editados pelo usuário

**Critério:** preencher só `OverallQual=7, GrLivArea=1700` retorna preço entre 100k-300k.

### M4.4. `app/pages/2_Insights.py` — dashboard negócio
**Arquivo novo.** 6 visualizações (PRD Seção 2):

1. **Importância de features** (top 15): bar chart horizontal de `model.feature_importances_`
2. **Distribuição de SalePrice**: histograma + KDE
3. **Preço médio por bairro** (top 10): para isso precisa do CSV original com `Neighborhood` (vamos carregar `data/train.csv` se disponível, ou usar o one-hot atual com agregação aproximada — documentar limitação)
4. **Scatter `y_real × y_pred`** no holdout: linha y=x de referência
5. **Cards de métricas**: RMSLE, MAE em USD, R² (do `metrics.json`)
6. **Correlação top 10 features × target**: bar chart de Pearson

Usar Plotly (interatividade) ou Matplotlib estático (mais leve). Plotly é o recomendado.

**Crítico:** o gráfico 3 (preço por bairro) só faz sentido se tivermos a coluna `Neighborhood` — checar se está no `data/train_3_1.csv` ou se foi descartada. Se foi descartada, substituir gráfico por outra coisa (ex.: preço médio por `OverallQual`) e documentar em `docs/decisions.md`.

**Critério:** página carrega em < 5s e renderiza os 6 gráficos.

### M4.5. `app/pages/3_Tecnico.py` — visão técnica
**Arquivo novo.**
- Seção "Pipeline" com diagrama da arquitetura (markdown)
- Seção "Hiperparâmetros vencedores": tabela de `metadata.json`
- Seção "Métricas finais": de `metrics.json`
- Seção "Comparação RF vs XGB": tabela com métricas de ambos
- Links: GitHub do repo, Kaggle dataset, notebook técnico
- Versão do modelo (de `metadata.train_date`)

**Critério:** todos os links funcionam; tabelas renderizam corretamente.

### M4.6. Smoke test manual
```bash
# Em terminal 1
uvicorn api.main:app

# Em terminal 2
streamlit run app/streamlit_app.py
```
Navegar pelas 3 páginas. Cada uma deve carregar sem erro. Página Predição faz uma predição com sucesso.

---

## M5 — Notebooks complementares

### M5.1. `notebooks/01_eda.ipynb`
- Carrega `data/train_3_1.csv`
- Distribuição de `SalePrice` (raw + log)
- Heatmap de correlações top 20 features × target
- Análise de NAs (provavelmente zero, dado pré-processamento)
- Distribuição de `OverallQual` vs `SalePrice` (boxplot)

**Critério:** roda sem erros, narrativa em markdown clara.

### M5.2. `notebooks/02_baseline.ipynb`
- Treina `Ridge` (alpha padrão) com `prepare_features`
- Avalia no mesmo holdout que será usado em M2 (`split_holdout(random_state=42)`)
- Reporta RMSLE — esta é a baseline contra a qual RF/XGB são comparados
- Conclusão em markdown: "RF e XGB precisam bater RMSLE de X.XX"

**Critério:** baseline RMSLE registrado no notebook.

### M5.3. `notebooks/04_avaliacao_final.ipynb`
- Carrega `models/best_model.pkl` e `docs/metrics.json`
- Re-roda predição no holdout
- Gráfico scatter `y_real × y_pred`
- Análise de resíduos
- Tabela comparativa final: Ridge baseline vs RF vs XGB
- Conclusão em markdown: qual venceu e por que

**Critério:** roda standalone sem precisar re-treinar.

---

## M6 — Documentação + Deploy

### M6.1. `README.md` na raiz
**Arquivo novo.** Estrutura mínima:
```markdown
# Preço de Casas — Pipeline de ML com Demo

[Screenshots das 3 páginas Streamlit]

[Link demo público]

## Sobre
Breve descrição (1 parágrafo). Link pro PRD em docs/PRD.md.

## Como rodar localmente
```bash
git clone <repo>
cd projeto-preco-casas
python -m venv .venv && source .venv/Scripts/activate  # Windows bash
pip install -r requirements.txt
python -m src.train  # treina e salva modelo
uvicorn api.main:app --reload  # API em :8000
streamlit run app/streamlit_app.py  # app em :8501
```

## Estrutura
[Árvore resumida apontando para PRD Seção 5]

## Stack
Python 3.10+, scikit-learn, XGBoost, FastAPI, Streamlit

## Licença
MIT
```

**Critério:** seguir as instruções no README do zero em ambiente limpo funciona ponta a ponta.

### M6.2. `docs/decisions.md` (ADRs informais)
**Arquivo novo.** Documentar decisões com tradeoffs:
- Por que log-transform no target
- Por que 35 colunas object foram descartadas (limitação herdada)
- Por que Streamlit chama `src/predict.py` direto em deploy (R-04)
- Substituições no dashboard caso `Neighborhood` não esteja disponível
- Qualquer ajuste de grid se CA-02 (15 min) for excedido

**Critério:** mínimo 4 entradas; cada uma com Contexto / Decisão / Consequências.

### M6.3. Deploy no Streamlit Cloud
1. Push do repo no GitHub (público)
2. Criar conta em share.streamlit.io
3. Conectar ao repo, branch main, file `app/streamlit_app.py`
4. Adicionar secrets se necessário (não previstos)
5. Aguardar build (~5 min)
6. Validar URL pública abrindo no navegador

**Critério:** URL pública carrega as 3 páginas em < 30s no cold start.

### M6.4. Atualizar README com link demo
Editar `README.md` adicionando o link do Streamlit Cloud no topo.

### M6.5. Validação final dos critérios CA-01 a CA-10
Checklist do PRD Seção 6. Marcar cada um como ✅ ou ❌ em `docs/metrics.json` ou em uma seção do README.

**Feito quando:** todos os 10 CAs marcados ✅ (com CA-03 documentado se ficar acima de 0.16).

---

## Sequenciamento e paralelismo

```
P-1 → P-5  (sequencial, fundação)
   ↓
M1 (sequencial: M1.1 → M1.2 → M1.3 → M1.4 → M1.5 → M1.6)
   ↓
M2 (sequencial: M2.1 → M2.2 → M2.3 → M2.4)
   ↓
   ├─ M3 (M3.1 → M3.2 → M3.3) ─┐
   └─ M4 (M4.1 → M4.2 → ...)  ─┤  podem rodar em paralelo
                                ↓
                               M5 (M5.1, M5.2, M5.3 paralelas entre si)
                                ↓
                               M6 (M6.1 → M6.2 → M6.3 → M6.4 → M6.5)
```

## Orçamento de esforço (referência)

| Milestone | Estimativa |
|---|---|
| P-1 a P-5 | 1-2h |
| M1 | 1-2 dias |
| M2 | 1 dia |
| M3 | 1 dia |
| M4 | 2 dias |
| M5 | 1 dia |
| M6 | 1 dia |
| **Total** | **~7-8 dias focados** |

---

## Como usar este plano em sessões futuras com Claude

1. Abrir Claude Code na raiz do projeto
2. Mencionar "estamos no Mx.y do `docs/plano_implementacao.md`" para retomar contexto
3. Pedir Claude para executar a tarefa específica seguindo os critérios
4. Validar com o comando indicado em "Critério:" / "Feito quando:"
5. Marcar tarefa como concluída e seguir

O `CLAUDE.md` na raiz já aponta para o PRD; este plano é o complemento operacional.
