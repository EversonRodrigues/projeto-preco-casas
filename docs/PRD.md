# PRD — Preço de Casas: Pipeline de ML com Demo Interativa

| Campo | Valor |
|---|---|
| Autor | Amanda Ribeiro (amanda.ribeiro@visionone.com.br) |
| Data | 2026-04-26 |
| Status | Aprovado pelo autor — pendente de transição para plano de implementação |
| Dataset | Ames Housing (Kaggle: House Prices — Advanced Regression Techniques) |
| Stack | Python 3.10+, scikit-learn, XGBoost, FastAPI, Streamlit, joblib |

---

## 1. Visão Geral e Objetivos

### Problema
Demonstrar, ponta a ponta, como prever preço de imóveis residenciais usando técnicas de regressão sobre o dataset *Ames Housing* (Kaggle), entregando o resultado em três interfaces diferentes para três públicos distintos.

### Objetivos (em ordem de prioridade)

1. **O1 — Pipeline reprodutível**: pré-processamento, treino e validação encapsulados em módulos Python reutilizáveis (não presos ao notebook).
2. **O2 — Comparação justa entre modelos**: RandomForest vs XGBoost com `GridSearchCV` enxuto, validação cruzada explícita e métrica em escala log (RMSLE) alinhada à competição Kaggle.
3. **O3 — Modelo servido via API**: FastAPI expondo `POST /predict` que consome o `.pkl` do melhor modelo.
4. **O4 — Web app multi-página**: Streamlit com 3 páginas (predição interativa para cliente, dashboard de insights para negócio, seção técnica com EDA/métricas para o autor).
5. **O5 — Portfólio publicável**: README claro, deploy funcional no Streamlit Cloud (ou HF Spaces) com link compartilhável.

### Não-objetivos (fora de escopo)
- Generalizar para dados brasileiros / scraping de imóveis
- Re-treino automático / pipeline contínuo (MLOps)
- Autenticação, multi-usuário, persistência de histórico
- Otimização extrema de leaderboard Kaggle (top 10%)
- Feature engineering exótica (interações, target encoding, etc.)

---

## 2. Públicos e Casos de Uso

### Persona 1 — Autor (Data Scientist / dono do projeto)
- **Onde entra**: notebooks (`notebooks/`), módulos (`src/`), API (`api/`), página técnica do Streamlit
- **Caso de uso**: rodar EDA, experimentar features, executar grid_search, comparar modelos, persistir o vencedor, expor via API
- **Saída esperada**: código limpo + métricas reproduzíveis + modelo `.pkl` versionado

### Persona 2 — Cliente (usuário final da predição)
- **Onde entra**: página "Predição" do Streamlit
- **Caso de uso**: preencher um formulário com características de uma casa (área, qualidade geral, ano de construção, bairro, etc.) e receber o preço estimado em USD
- **Saída esperada**: número de preço + leitura visual ("essa casa está em qual faixa do mercado de Ames?")

### Persona 3 — Área de Negócio (stakeholder analítico)
- **Onde entra**: página "Insights" do Streamlit
- **Caso de uso**: entender **o que move o preço** — quais features pesam mais, distribuição de preços por bairro, correlações chave, performance do modelo em segmentos
- **Saída esperada**: dashboard com 6 gráficos — importância de features (top 15), distribuição de SalePrice, preço médio por bairro (top 10), scatter `y_real × y_pred`, métricas resumidas (RMSLE, MAE, R²), correlação top features × target

### Fluxo de uso típico
```
Autor     → notebook EDA → src/train.py → models/best.pkl → testa via API
Cliente   → abre Streamlit → página "Predição" → preenche formulário → vê preço
Negócio   → abre Streamlit → página "Insights" → explora gráficos
```

---

## 3. Escopo Funcional Detalhado

### A. Camada de dados
- **A1.** Leitura dos CSVs `train_3_1.csv` (1460×85) e `test_3_1.csv` (1459×84) já existentes.
- **A2.** Função `load_data()` em `src/data.py` retornando DataFrames prontos.
- **A3.** Manter o pré-processamento já existente (drop de colunas object, one-hot de MSZoning/GarageType com `-1` como marcador de NA).

### B. Camada de pré-processamento
- **B1.** `src/preprocessing.py` com `prepare_features(df) -> X, y` que isola colunas numéricas e separa target.
- **B2.** **Log-transform no target** (`np.log1p(SalePrice)`) — alinha com a métrica oficial Kaggle (RMSLE).
- **B3.** Pipeline `sklearn` com `StandardScaler` opcional (RF/XGB não precisam, mas deixa pronto pra futuros modelos lineares).

### C. Camada de treino
- **C1.** `src/train.py` com `train_random_forest()` e `train_xgboost()`.
- **C2.** **Grid enxuto** (do original ~750 fits → ~180 fits totais com KFold=5; ~4× mais rápido):
  - **RF**: `n_estimators=[200, 500]`, `max_depth=[10, 20, None]`, `max_features=['sqrt', 0.5]` → 12 combos × 5 folds = 60 fits
  - **XGB**: `learning_rate=[0.05, 0.1]`, `max_depth=[4, 6, 10]`, `n_estimators=[300, 800]`, `colsample_bytree=[0.7, 1.0]` → 24 combos × 5 folds = 120 fits
- **C3.** `KFold(n_splits=5, shuffle=True, random_state=42)` explícito no `GridSearchCV`.
- **C4.** Métrica de scoring: `neg_root_mean_squared_error` no log-target (= RMSLE).
- **C5.** Persistência: `joblib.dump(best_model, 'models/best_model.pkl')` + `models/metadata.json` com métricas, hiperparâmetros, data, versão das libs.

### D. Camada de avaliação
- **D1.** Hold-out final (test set 20%) que NUNCA entra no grid_search — para métrica honesta.
- **D2.** Reportar: RMSLE, MAE (em USD), R², scatter `y_real × y_pred`.
- **D3.** Submissão Kaggle gerada (`submission.csv`) como artefato secundário.

### E. API (FastAPI)
- **E1.** `POST /predict` aceita JSON com **as 49 features do modelo** (todas obrigatórias). Retorna `{"predicted_price": float, "model_version": str}`. A API não tem lógica de defaults — quem chama é responsável por enviar payload completo.
- **E2.** `GET /health` e `GET /model-info` (metadados).
- **E3.** Carrega o `.pkl` na inicialização (uma vez).

### F. Web app (Streamlit, 3 páginas)
- **F1. Página Predição**: formulário com ~15 campos essenciais visíveis ao usuário. Os outros ~34 são preenchidos automaticamente pelo Streamlit usando `get_feature_defaults()` (mediana do treino) antes de enviar o JSON de 49 campos para a API. Botão "Estimar preço". Mostra preço + posição na distribuição.
- **F2. Página Insights**: 6 gráficos definidos na Seção 2.
- **F3. Página Técnica**: resumo do pipeline, hiperparâmetros vencedores, versão do modelo, link pro repo/notebook.
- **F4.** App consome a API local (`requests.post(...)`) em desenvolvimento; em deploy chama `src/predict.py` direto (ver Seção 7, R-04).

### G. Documentação
- **G1.** `README.md` na raiz com descrição, como rodar, screenshots, link demo.
- **G2.** `docs/PRD.md` (este documento).
- **G3.** Notebooks renomeados/organizados na pasta `notebooks/` mantendo a narrativa "Parte N".

### Fora do MVP
- Re-treino automatizado, CI/CD, monitoramento
- Autenticação, banco de dados, histórico de predições
- Feature engineering pesada
- Modelos além de RF + XGBoost
- Deploy multi-cloud / Docker complexo (Streamlit Cloud + uvicorn local é suficiente)

---

## 4. Arquitetura e Fluxo de Dados

### Arquitetura em camadas

```
┌──────────────────────────────────────────────────────────────┐
│  CAMADA DE APRESENTAÇÃO                                       │
│  ┌──────────────────────┐      ┌──────────────────────────┐  │
│  │  Streamlit App       │      │  Notebooks Jupyter       │  │
│  │  (3 páginas)         │      │  (EDA, treino, análise)  │  │
│  └──────────┬───────────┘      └──────────┬───────────────┘  │
│             │ HTTP                         │ import           │
└─────────────┼─────────────────────────────┼──────────────────┘
              ▼                              ▼
┌──────────────────────────────────────────────────────────────┐
│  CAMADA DE SERVIÇO                                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FastAPI  ( POST /predict, GET /health, /model-info )│   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  CAMADA DE DOMÍNIO (src/)                                     │
│  ┌────────────┐  ┌──────────────┐  ┌────────┐  ┌──────────┐ │
│  │  data.py   │→ │preprocessing │→ │train.py│→ │predict.py│ │
│  └────────────┘  └──────────────┘  └────────┘  └──────────┘ │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  CAMADA DE PERSISTÊNCIA                                       │
│  data/*.csv      models/best_model.pkl + metadata.json       │
└──────────────────────────────────────────────────────────────┘
```

### Fluxo 1 — Treino (offline, manual)
```
data/train_3_1.csv
   │
   ▼  src/data.py::load_data()
DataFrame bruto
   │
   ▼  src/preprocessing.py::prepare_features()
X, y (com log-transform em y)
   │
   ▼  src/train.py::run_grid_search(model_type='rf'|'xgb')
GridSearchCV(KFold=5, scoring=neg_RMSE) sobre log(y)
   │
   ▼  Seleção do melhor (menor RMSLE no CV)
models/best_model.pkl + models/metadata.json
   │
   ▼  src/train.py::evaluate_holdout()
Métricas finais no holdout 20% → docs/metrics.json
```

### Fluxo 2 — Predição (online)
```
Cliente preenche formulário (15 campos)
   │
   ▼  Streamlit completa campos faltantes com defaults (mediana do treino)
JSON com 49 features
   │
   ▼  POST /predict (FastAPI) — em dev local
       OU chamada direta src/predict.py — em deploy
api/main.py recebe → valida com Pydantic → chama src/predict.py::predict()
   │
   ▼  preprocessing aplicado + model.predict()
Preço em log
   │
   ▼  np.expm1() para USD
{"predicted_price": 215000.0, "model_version": "rf_v1_2026-04-26"}
   │
   ▼  Streamlit recebe → exibe valor + posição na distribuição
```

### Princípios arquiteturais
- **Single source of truth**: lógica de preprocessing vive UMA VEZ em `src/preprocessing.py`. Notebook, API e app importam de lá.
- **Modelo como artefato**: o `.pkl` é o contrato entre treino e serving.
- **API como fronteira (em dev)**: Streamlit não chama o modelo direto em desenvolvimento — sempre via API. Em deploy, chama `src/` direto pra evitar overhead de rede no mesmo container.
- **Defaults na UI, não no modelo**: quando o cliente não preenche um campo, o Streamlit injeta a mediana do treino antes de chamar a API. A API rejeita payloads incompletos.

---

## 5. Estrutura de Pastas e Componentes

```
projeto-preco-casas/
│
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml                    # opcional: ruff/black
│
├── data/
│   ├── train_3_1.csv
│   ├── test_3_1.csv
│   └── data_description.txt          # opcional: dicionário Kaggle
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_grid_search.ipynb          # refatoração do notebook atual
│   └── 04_avaliacao_final.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── best_model.pkl
│   └── metadata.json
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_Predicao.py
│       ├── 2_Insights.py
│       └── 3_Tecnico.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_predict.py
│   └── test_api.py
│
└── docs/
    ├── PRD.md
    ├── metrics.json
    └── decisions.md                  # opcional: ADRs informais
```

### Contratos dos componentes

| Componente | Função pública | Entrada | Saída |
|---|---|---|---|
| `src/data.py` | `load_data()` | — | `(treino_df, teste_df)` |
| `src/data.py` | `split_holdout(df, test_size=0.2)` | DataFrame | `(train_df, holdout_df)` |
| `src/preprocessing.py` | `prepare_features(df)` | DataFrame | `(X: DataFrame, y: Series \| None)` |
| `src/preprocessing.py` | `get_feature_defaults()` | — | `dict[feature → mediana]` |
| `src/train.py` | `run_grid_search(model_type)` | str `'rf'`/`'xgb'` | `(best_estimator, cv_results)` |
| `src/train.py` | `evaluate_holdout(model, X_h, y_h)` | modelo + holdout | `dict` de métricas |
| `src/train.py` | `save_model(model, metadata)` | modelo + dict | escreve `.pkl` + `.json` |
| `src/predict.py` | `predict(features_dict)` | `dict` features | `float` preço em USD |
| `api/main.py` | `POST /predict` | JSON `HouseFeatures` | JSON `PredictionResponse` |

### Mudanças vs. notebook atual
- O notebook atual será **refatorado** em `03_grid_search.ipynb`, mas a lógica vai pra `src/`. O notebook vira "consumidor" dos módulos.
- `joblib` substitui código solto: o modelo fica em `models/best_model.pkl` e é carregado pela API e pelo notebook 04.
- Tests mínimos (3 arquivos) — smoke tests, não TDD.

---

## 6. Critérios de Aceitação

### Critérios técnicos (objetivos)

| ID | Critério | Como verificar |
|---|---|---|
| CA-01 | `prepare_features` retorna `X` shape `(N, 49)` e `y` com log-transform aplicado | `tests/test_preprocessing.py` |
| CA-02 | `run_grid_search('rf')` e `run_grid_search('xgb')` rodam em < 15 min cada em hardware doméstico | timing manual; documentar no README |
| CA-03 | Modelo vencedor atinge **RMSLE ≤ 0.16** no holdout 20% | `docs/metrics.json` |
| CA-04 | `models/best_model.pkl` carrega com `joblib.load()` e prediz para uma linha do `test_3_1.csv` | `tests/test_predict.py` |
| CA-05 | `POST /predict` com payload válido retorna HTTP 200 + JSON em < 500ms | `tests/test_api.py` |
| CA-06 | `POST /predict` com payload inválido retorna HTTP 422 com mensagem clara | `tests/test_api.py` |
| CA-07 | `streamlit run app/streamlit_app.py` carrega as 3 páginas sem erro | smoke test manual |
| CA-08 | Página "Predição" estima preço para um exemplo manual em < 2s end-to-end | manual |
| CA-09 | Página "Insights" exibe os 6 gráficos definidos | manual |
| CA-10 | Deploy público (Streamlit Cloud OU HF Spaces) com link no README | manual |

### Critérios de qualidade

| ID | Critério |
|---|---|
| CQ-01 | README com: descrição, motivação, screenshots das 3 páginas, instruções `pip install`/`uvicorn`/`streamlit run`, link demo |
| CQ-02 | Notebooks com narrativa em markdown — alguém lendo entende o "porquê" |
| CQ-03 | Código em `src/` segue convenções: docstrings, type hints, nomes descritivos |
| CQ-04 | Zero código duplicado entre notebook ↔ API ↔ Streamlit |
| CQ-05 | `requirements.txt` com versões fixadas (==), reproduzível |

### Definição de "pronto"
Todos os CA-XX e CQ-XX cumpridos. Se CA-03 não bater 0.16, documentar o número real e seguir — é alvo, não bloqueador.

---

## 7. Riscos, Premissas e Mitigações

### Premissas

| ID | Premissa |
|---|---|
| P-01 | Os CSVs em `data/` são o estado canônico (saída das Partes 1 e 2). Não vamos reconstruir esse pré-processamento. |
| P-02 | Hardware doméstico (CPU, sem GPU) é suficiente. |
| P-03 | Streamlit Cloud (free tier) ou HF Spaces aceita o tamanho do `.pkl` (poucos MB). |
| P-04 | Ambiente Python 3.10+ funcional no Windows 11 com permissão para instalar libs. |
| P-05 | "Cliente" e "área de negócio" são personas demonstrativas — sem validação UX com usuários reais. |

### Riscos

| ID | Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|---|
| R-01 | Grid_search demora demais | Média | Médio | Grid já reduzido (CA-02 = 15 min). Plano B: 6 combos cada. |
| R-02 | `max_features='auto'` deprecated quebra em sklearn ≥1.3 | Alta | Médio | Substituir por `'sqrt'` e `0.5`. Fixar `scikit-learn` no requirements. |
| R-03 | Overfitting com `max_depth=None` no RF | Média | Médio | KFold=5 protege. Reportar gap train/holdout no `metrics.json`. |
| R-04 | Streamlit Cloud não acessa API local — em deploy, precisa estar acessível | Alta | Alto | **Decisão**: em deploy, Streamlit chama `src/predict.py` direto. Em dev local, opcionalmente via API. README documenta os 2 modos. |
| R-05 | Free tier tem limites de RAM (~1 GB) e cold start lento | Baixa | Baixo | Modelo enxuto cabe folgado. Cold start ~30s aceitável para demo. |
| R-06 | Notebook atual tem typo "Regressão Linear" mas é RF | Baixa | Baixo | Corrigir na refatoração. |
| R-07 | Features object (35 colunas string) descartadas perdem sinal valioso (ex.: `Neighborhood`, `KitchenQual`) | Alta | Alto | Aceito como dívida do MVP. Documentar em `decisions.md` como melhoria futura. |
| R-08 | Defaults da UI (mediana) podem dar predições estranhas se cliente preenche poucos campos | Média | Médio | Aviso no Streamlit: "Campos não preenchidos usaram a mediana — preencha mais para predição mais precisa". |
| R-09 | Pipeline data leakage se `prepare_features` for fitted no dataset inteiro antes do split | Média | Alto | `StandardScaler` (se usado) entra dentro de `Pipeline` do sklearn, fitted apenas no `X_train` do CV. |

### Decisões deliberadas que vão contra o "ideal"
- **Não usar features categóricas tipo string**: aceitamos a limitação herdada das partes anteriores.
- **Sem MLOps**: nada de MLflow, DVC, monitoramento. É demo.
- **Sem testes de modelo robustos**: os testes são smoke tests; não validamos drift, fairness, etc.

---

## 8. Roadmap de Implementação (macro)

> O plano detalhado virá da próxima skill (`writing-plans`). Esta seção define apenas o sequenciamento.

```
M1 — Refatoração base                         (~1-2 dias)
   ├─ Mover código do notebook atual para src/{data,preprocessing,train,predict}.py
   ├─ Criar estrutura de pastas (Seção 5)
   ├─ Adicionar log-transform no target
   ├─ Trocar max_features='auto' por valores válidos
   └─ Criar 03_grid_search.ipynb consumindo src/

M2 — Pipeline robusto + modelo persistido     (~1 dia)
   ├─ KFold=5 explícito no GridSearchCV
   ├─ Holdout 20% reservado fora do grid
   ├─ joblib.dump → models/best_model.pkl
   ├─ models/metadata.json com hiperparams + métricas
   └─ Smoke test em tests/test_preprocessing.py + test_predict.py

M3 — API FastAPI                              (~1 dia)
   ├─ api/schemas.py com HouseFeatures + PredictionResponse
   ├─ api/main.py com POST /predict, GET /health, GET /model-info
   ├─ Carregamento do .pkl no startup
   └─ tests/test_api.py

M4 — Streamlit multi-página                   (~2 dias)
   ├─ app/streamlit_app.py (entry + sidebar)
   ├─ pages/1_Predicao.py (formulário + chamada)
   ├─ pages/2_Insights.py (6 gráficos)
   ├─ pages/3_Tecnico.py (metadata + links)
   └─ Decisão R-04: predict via src/ direto em deploy

M5 — Notebooks complementares                 (~1 dia)
   ├─ 01_eda.ipynb (distribuições, NAs, correlações)
   ├─ 02_baseline.ipynb (linear como referência)
   └─ 04_avaliacao_final.ipynb (RF vs XGB no holdout, gráficos)

M6 — Documentação + Deploy                    (~1 dia)
   ├─ README.md com screenshots, instruções, link demo
   ├─ docs/decisions.md (ADRs informais)
   ├─ requirements.txt fixado
   ├─ Deploy no Streamlit Cloud
   └─ Validação dos critérios CA-01 a CA-10
```

### Dependências
- M1 é pré-requisito de tudo
- M2 desbloqueia M3 (API precisa do `.pkl`) e M4 (gráficos do dashboard precisam do modelo)
- M3 e M4 podem rodar em paralelo
- M5 e M6 ficam no final

### Estimativa total
~7-8 dias de trabalho focado (referência, não compromisso).

### Definição de "feito"
- Todos os CA-XX e CQ-XX da Seção 6 verificados
- Demo público acessível via link
- Repo no GitHub público com README completo

---

*Documento elaborado seguindo a skill `brainstorming`. Próximo passo: invocar `writing-plans` para gerar o plano detalhado de implementação.*
