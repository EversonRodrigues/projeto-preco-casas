# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status do projeto

**Pré-implementação.** O projeto está na fase de planejamento — não há código-fonte Python organizado ainda. O que existe:

- `docs/PRD.md` — **fonte da verdade do design**. Descreve toda a arquitetura planejada, escopo, critérios de aceitação, riscos. Sempre leia este documento antes de propor mudanças.
- `Usando o grid_search .../Adicionando%20novos%20algoritmos%20-%20Parte%203.ipynb` — único notebook existente. É a "Parte 3" de uma série didática; faz grid_search em RandomForest e XGBoost. Será **refatorado** em `notebooks/03_grid_search.ipynb` + módulos `src/` durante a implementação (Milestone M1 do PRD).
- `Usando o grid_search .../{train_3_1.csv, test_3_1.csv}` — datasets já pré-processados pelas Partes 1 e 2 (não incluídas). 1460 linhas treino × 85 cols, 1459 linhas teste × 84 cols. Tratar como **estado canônico** dos dados (premissa P-01 do PRD); não reconstruir o pré-processamento.

A estrutura final planejada (`src/`, `api/`, `app/`, `models/`, `tests/`) ainda não existe. O PRD detalha tudo na Seção 5.

## Decisões-chave do PRD que afetam todo o desenvolvimento

- **Stack**: Python 3.10+, scikit-learn, XGBoost, FastAPI, Streamlit, joblib. Tudo Python, deploy alvo é Streamlit Cloud ou HF Spaces.
- **Algoritmos**: apenas RandomForest e XGBoost (não adicionar outros sem revisar o PRD).
- **Target em log**: aplicar `np.log1p(SalePrice)` no treino e `np.expm1` na saída — alinha com a métrica oficial Kaggle (RMSLE).
- **Apenas colunas numéricas**: as 35 colunas tipo `object` foram descartadas pelo pré-processamento herdado das Partes 1/2 (risco R-07 aceito como dívida do MVP). Não reintroduzir sem revisar o PRD.
- **Single source of truth para preprocessing**: viverá em `src/preprocessing.py` e será importado por notebook, API e Streamlit. Nunca duplicar essa lógica.
- **Dois modos de inferência**: dev local usa `POST /predict` (FastAPI); deploy chama `src/predict.py` direto (decisão R-04 do PRD, motivada por limitações do Streamlit Cloud).

## Pegadinhas conhecidas do notebook atual

Ao refatorar, corrigir:
- `max_features='auto'` está deprecated em `scikit-learn ≥ 1.3` — usar `'sqrt'` ou valor float.
- Comentário "vamos usar a Regressão Linear" em uma célula é typo: o código usa RF.
- `train_test_split` simples sem KFold explícito; sem holdout reservado fora do grid.
- Sem persistência do modelo (`joblib.dump`) — bloqueador para API/Streamlit.
- Grid pesado (~750 fits): o PRD reduz para ~180 fits (Seção 3.C2).

## Ambiente e convenções de caminho

- Plataforma: **Windows 11**, shell bash via Claude Code (use sintaxe Unix nas chamadas Bash, ex.: `/dev/null` em vez de `NUL`).
- A raiz do projeto e várias subpastas têm **espaços e caractere `ç`** (`Projeto_Preço_de_casas`, `Usando o grid_search...`). Sempre cite caminhos entre aspas duplas em comandos shell.
- O notebook tem `%20` no nome do arquivo (não é URL-encoding renderizado — é parte literal do nome). Manter ao referenciar.
- **Não é um repositório git.** Não rodar `git` antes de inicializar (`git init`) — confirmar com a usuária antes.

## Pasta `Skill/` — não é código do projeto

A pasta `Skill/brainstorming/` contém uma cópia local da skill `brainstorming` da Anthropic (SKILL.md + visual-companion + scripts auxiliares). É **material de referência**, não código-fonte do projeto. Não modificar nem versionar como parte do produto. A skill foi usada para gerar `docs/PRD.md`.

A skill terminal `writing-plans` referenciada no fim do `SKILL.md` **não está instalada** neste ambiente — ao avançar para o plano de implementação, ou se cria o plano manualmente seguindo o espírito da skill, ou a usuária instala a skill primeiro.

## Comandos

Ainda não há comandos de build/test/lint definidos (estrutura não existe). Após M1-M2 do PRD, espera-se:

```bash
# Treino (gera models/best_model.pkl)
python -m src.train

# API local
uvicorn api.main:app --reload

# Streamlit
streamlit run app/streamlit_app.py

# Tests (smoke)
pytest tests/
```

Atualizar esta seção conforme os módulos forem criados.
