import numpy as np
import pandas as pd

from src.data import load_data

_TRAIN_DF, _ = load_data()

FEATURE_NAMES: list[str] = [
    c
    for c in _TRAIN_DF.select_dtypes(include=["int64", "float64"]).columns
    if c not in ("Id", "SalePrice")
]

_FEATURE_DEFAULTS: dict[str, float] = (
    _TRAIN_DF[FEATURE_NAMES].median().to_dict()
)


def prepare_features(
    df: pd.DataFrame, has_target: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    X = df[FEATURE_NAMES].copy()
    if has_target:
        y = np.log1p(df["SalePrice"])
        return X, y
    return X, None


def get_feature_defaults() -> dict[str, float]:
    return dict(_FEATURE_DEFAULTS)
