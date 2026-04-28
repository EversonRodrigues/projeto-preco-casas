# Decisões de design (ADRs informais)

Cada entrada: **Contexto → Decisão → Consequências**.

---

## D-01. Log-transform no target

**Contexto.** A métrica oficial Kaggle para este dataset é **RMSLE** (raiz do erro
quadrático médio sobre `log1p`). O `SalePrice` é fortemente assimétrico à direita
(skew ≈ 1.88 no bruto), o que dificulta modelos que assumem variância homogênea.

**Decisão.** Aplicar `np.log1p` no target durante o treino e `np.expm1` na saída
(em `src/preprocessing.prepare_features` e `src/predict.predict`).

**Consequências.**
- O scoring `neg_root_mean_squared_error` no log-target equivale a otimizar RMSLE.
- A skewness cai de ~1.88 para ~0.12, melhorando o fit dos lineares (Ridge baseline).
- A predição precisa do `expm1` simétrico — qualquer caminho que pule essa
  inversão (notebook, API, Streamlit) retorna log, não USD.

---

## D-02. Apenas colunas numéricas — `Neighborhood` descartado

**Contexto.** O pré-processamento das Partes 1/2 (não versionadas neste repo)
descartou as 35 colunas tipo `object` dos CSVs originais. Sobraram 50 numéricas,
das quais retiramos `Id` e `SalePrice` para chegar nas **48 features** do modelo.
Algumas categóricas relevantes (incluindo `Neighborhood`) foram perdidas;
outras (`MSZoning`, `GarageType`) viraram one-hot e sobreviveram com prefixo.

**Decisão.** Aceitar a limitação como dívida do MVP (R-07 do PRD) e construir o
modelo apenas com as 48 numéricas. Não reintroduzir o pré-processamento das
Partes 1/2 — viola P-01 do PRD ("dados pré-processados são estado canônico").

**Consequências.**
- Página Insights do Streamlit não pode mostrar "preço médio por bairro"
  conforme o plano original — substituído por **preço médio por OverallQual**.
- Erro de predição maior do que seria possível com `Neighborhood` (proxy
  geográfico forte). Aceitável para o MVP — RMSLE 0.139 ainda bate o alvo 0.16.
- Próxima iteração: refazer o pipeline de pré-processamento incluindo
  one-hot/target encoding de `Neighborhood` e revisar o gráfico 3 da Insights.

---

## D-03. Plano errou em "49 features" — modelo tem 48

**Contexto.** O plano (`docs/plano_implementacao.md` linhas 102, 106, 217) e o
PRD (Seção E.E1) afirmam "49 features". Inspeção dos CSVs mostrou
50 numéricas − `Id` − `SalePrice` = **48**.

**Decisão.** Implementar com base na realidade (48). Não corrigir o plano nem o
PRD: a discrepância está documentada aqui e a single source of truth para o
número é `src.preprocessing.FEATURE_NAMES` (carregado no import do treino).

**Consequências.**
- `HouseFeatures` (Pydantic) tem 48 campos.
- Documentos `docs/PRD.md` e `docs/plano_implementacao.md` ficam levemente
  desatualizados — caso sejam reapresentados, atualizar com nota de errata.

---

## D-04. Streamlit chama API; cai para `src/predict` em deploy

**Contexto.** Em desenvolvimento local, ter o app consumindo a API garante que
ambos compartilhem o caminho de inferência. Em deploy (Streamlit Cloud / HF
Spaces), subir uvicorn separado é caro / não suportado nativamente.

**Decisão (R-04 do PRD).** `app/_shared.get_predictor()` tenta `requests.post` em
`localhost:8000/predict`; em caso de falha (timeout/conexão), faz fallback para
`src.predict.predict` direto (mesma função que a API usa internamente).

**Consequências.**
- Mesma lógica de inferência nos dois caminhos — sem divergência.
- Em deploy, latência menor (sem hop HTTP); em dev, exercitamos o caminho de
  produção (HTTP).
- Trade-off: o usuário em dev pode não perceber se a API caiu (Streamlit segue
  funcionando via fallback). Solução: a página Predição pode mostrar qual
  caminho foi usado — não implementado no MVP.

---

## D-05. NaN scrub no `metadata.json`

**Contexto.** `XGBRegressor.get_params()` inclui `"missing": np.nan` por padrão.
`json.dumps` escreve `NaN` literal — válido para `json.loads` do Python, mas
inválido pela RFC 8259 e quebra parsers de outras linguagens (e da página
Técnica do Streamlit, que já leu `metadata.json` direto).

**Decisão.** `src/train._scrub_nan` substitui `float('nan')` por `null`
recursivamente antes de serializar. `json.dumps(..., allow_nan=False)` agora
levanta erro se algum NaN escapar.

**Consequências.**
- Semanticamente, `"missing": null` significa "modelo sem sentinela de missing
  configurada" — equivalente ao default do XGBoost.
- Defesa em profundidade: parsers estritos (HF Spaces, ferramentas externas)
  não quebram.

---

## D-06. Cache em-memória de `_TRAIN_DF`

**Contexto.** `src/preprocessing` precisa do treino para construir
`FEATURE_NAMES` e `_FEATURE_DEFAULTS`. Carregar a cada chamada teria overhead.

**Decisão.** Carregar o treino uma vez no import (módulo-level) e cachear como
`_TRAIN_DF`. `get_feature_defaults` retorna **cópia** do dict para evitar que
mutação externa contamine o cache.

**Consequências.**
- Side-effect de I/O no import — testes ficam ~50 ms mais lentos.
- Testes precisam que `data/train_3_1.csv` esteja presente em qualquer pytest
  (incluindo testes da API). Aceitável — esses CSVs estão versionados.

---

## D-07. Critério de seleção do vencedor: CV, não holdout

**Contexto.** `main()` em `src/train.py` precisa escolher entre RF e XGB para
salvar o `.pkl` final. Plano M1.3 diz "menor RMSLE no CV"; M1.5 (snippet de
notebook) usa holdout. Decisão de desempate.

**Decisão.** Usar **CV RMSLE** (`summary["best_score_rmsle_cv"]`). O holdout é
reservado para a métrica final de reportagem — usá-lo para selecionar modelo
introduz vazamento de informação para a decisão.

**Consequências.**
- Seleção alinhada com a lógica do `GridSearchCV` (que já otimiza CV).
- Como ambos modelos foram avaliados no MESMO split, a métrica de holdout
  reportada ainda é honesta (não foi usada para escolher).
