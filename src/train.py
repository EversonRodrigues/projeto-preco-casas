from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.data import load_data, split_holdout
from src.preprocessing import build_preprocessor, prepare_features

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

_RF_GRID = {
    "model__n_estimators": [200, 500],
    "model__max_depth": [10, 20, None],
    "model__max_features": ["sqrt", 0.5],
}

_XGB_GRID = {
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [4, 6, 10],
    "model__n_estimators": [300, 800],
    "model__colsample_bytree": [0.7, 1.0],
}


def _make_estimator(model_type: Literal["rf", "xgb"]):
    if model_type == "rf":
        return RandomForestRegressor(random_state=42, n_jobs=1)
    if model_type == "xgb":
        return xgb.XGBRegressor(
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    raise ValueError(f"model_type inválido: {model_type}")


def _make_pipeline(model_type: Literal["rf", "xgb"]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preproc", build_preprocessor()),
            ("model", _make_estimator(model_type)),
        ]
    )


def run_grid_search(
    model_type: Literal["rf", "xgb"],
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[Any, dict]:
    grid = _RF_GRID if model_type == "rf" else _XGB_GRID
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        _make_pipeline(model_type),
        param_grid=grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X, y)

    summary = {
        "best_score_rmsle_cv": float(-gs.best_score_),
        "best_params": gs.best_params_,
        "n_combos": len(gs.cv_results_["mean_test_score"]),
    }
    return gs.best_estimator_, summary


def evaluate_holdout(
    model, X_h: pd.DataFrame, y_h_log: pd.Series
) -> dict:
    pred_log = model.predict(X_h)
    pred = np.expm1(pred_log)
    actual = np.expm1(y_h_log)

    rmsle = float(np.sqrt(mean_squared_error(y_h_log, pred_log)))
    mae_usd = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))

    return {"rmsle": rmsle, "mae_usd": mae_usd, "r2": r2}


def _scrub_nan(obj):
    if isinstance(obj, dict):
        return {k: _scrub_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nan(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def save_model(
    model,
    metadata: dict,
    path: str | Path = "models/best_model.pkl",
) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)

    enriched = {
        **metadata,
        "train_date": datetime.now(timezone.utc).isoformat(),
        "lib_versions": {
            "sklearn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    metadata_path = p.parent / "metadata.json"
    metadata_path.write_text(
        json.dumps(_scrub_nan(enriched), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    treino, _ = load_data()
    treino_dev, treino_holdout = split_holdout(treino)

    X_dev, y_dev = prepare_features(treino_dev)
    X_h, y_h = prepare_features(treino_holdout)

    print(">> Grid RF...")
    best_rf, summary_rf = run_grid_search("rf", X_dev, y_dev)
    metrics_rf = evaluate_holdout(best_rf, X_h, y_h)
    print(f"   RF CV RMSLE = {summary_rf['best_score_rmsle_cv']:.4f}")
    print(f"   RF holdout: {metrics_rf}")

    print(">> Grid XGB...")
    best_xgb, summary_xgb = run_grid_search("xgb", X_dev, y_dev)
    metrics_xgb = evaluate_holdout(best_xgb, X_h, y_h)
    print(f"   XGB CV RMSLE = {summary_xgb['best_score_rmsle_cv']:.4f}")
    print(f"   XGB holdout: {metrics_xgb}")

    rf_cv = summary_rf["best_score_rmsle_cv"]
    xgb_cv = summary_xgb["best_score_rmsle_cv"]
    if rf_cv < xgb_cv:
        winner_name, winner_model, winner_metrics = "rf", best_rf, metrics_rf
    else:
        winner_name, winner_model, winner_metrics = "xgb", best_xgb, metrics_xgb

    def _strip_prefix(d: dict) -> dict:
        return {k.replace("model__", ""): v for k, v in d.items()}

    n_features_in = winner_model.named_steps["preproc"].n_features_in_
    n_features_out = winner_model.named_steps["preproc"].transform(X_dev.head(1)).shape[1]

    metadata = {
        "algorithm": winner_name,
        "hyperparams": _strip_prefix(
            winner_model.named_steps["model"].get_params()
        ),
        "metrics": winner_metrics,
        "n_features_pre": int(n_features_in),
        "n_features_post_onehot": int(n_features_out),
        "comparison": {
            "rf": {
                **metrics_rf,
                "cv_rmsle": rf_cv,
                "best_params": _strip_prefix(summary_rf["best_params"]),
            },
            "xgb": {
                **metrics_xgb,
                "cv_rmsle": xgb_cv,
                "best_params": _strip_prefix(summary_xgb["best_params"]),
            },
        },
    }
    save_model(winner_model, metadata)
    print(f">> Vencedor: {winner_name}. Salvo em models/best_model.pkl")

    metrics_doc = {
        "winner": winner_name,
        "metrics": winner_metrics,
        "comparison": metadata["comparison"],
    }
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "metrics.json").write_text(
        json.dumps(_scrub_nan(metrics_doc), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    print(">> docs/metrics.json atualizado")


if __name__ == "__main__":
    main()
