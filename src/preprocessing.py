from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.data import load_data

_TRAIN_DF, _ = load_data()

CATEGORICAL_COLS: list[str] = _TRAIN_DF.select_dtypes(include=["object"]).columns.tolist()

NUMERIC_COLS: list[str] = [
    c
    for c in _TRAIN_DF.select_dtypes(include=["int64", "float64"]).columns
    if c not in ("Id", "SalePrice")
]

FEATURE_NAMES: list[str] = NUMERIC_COLS + CATEGORICAL_COLS

_NUMERIC_DEFAULTS: dict[str, float] = (
    _TRAIN_DF[NUMERIC_COLS].median().to_dict()
)
_CATEGORICAL_DEFAULTS: dict[str, str] = {
    c: str(_TRAIN_DF[c].mode(dropna=False).iloc[0]) for c in CATEGORICAL_COLS
}
_FEATURE_DEFAULTS: dict[str, float | str] = {
    **_NUMERIC_DEFAULTS,
    **_CATEGORICAL_DEFAULTS,
}


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: OneHot nos 35 categoricals, passthrough nos 49 numéricos.

    `handle_unknown='ignore'` permite que o modelo receba categorias não vistas
    no treino (ex: bairro novo) sem quebrar — todas as dummies dessa categoria
    ficam em zero. `sparse_output=False` deixa pandas/numpy compatível com
    XGBoost sem extra densificação.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("num", "passthrough", NUMERIC_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_features(
    df: pd.DataFrame, has_target: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    X = df[FEATURE_NAMES].copy()
    if has_target:
        y = np.log1p(df["SalePrice"])
        return X, y
    return X, None


def get_feature_defaults() -> dict[str, float | str]:
    """Mediana (numéricas) e moda (categoricals) no treino completo.

    Uso: preencher campos não-visíveis do form Streamlit. NÃO usar como
    imputador de treino — inclui o holdout. Ver D-06 em docs/decisions.md.
    """
    return dict(_FEATURE_DEFAULTS)
