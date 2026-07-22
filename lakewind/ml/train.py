"""LightGBM / XGBoost quantile MOS trainer (Spec §7.1).

V1 model:
- One model per target (target_u, target_v) per quantile.
- Quantile objective (10/50/90) gives calibrated uncertainty for free.
- `expected_error_kn` = half the predicted 90th-10th interval width.
- Tree-based, supports NaN natively (Spec §6 missing-data policy).

Backend selection (configurable via settings.yaml `model.backend`):
- `lightgbm` (default; CPU only via pip)
- `xgboost_gpu` — uses XGBoost with `device='cuda'` for RTX 3070 GPU
  acceleration. Significantly faster on large datasets. The user has an
  RTX 3070 and "training time is not a problem" — but GPU still helps for
  walk-forward backtests with hundreds of windows.

Spec §11 Phase 1: feature engineering + single quantile MOS model + walk-forward
backtest.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.features.build import build_features_for

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"


@dataclass
class TrainingResult:
    model_version: str
    feature_set_version: str
    backend: str
    trained_at: datetime
    n_samples: int
    n_features: int
    quantiles: list[float]
    metrics: dict[str, float]
    model_paths: dict[str, Path]


def _build_dataset(
    point_id: str | None,
    start: datetime,
    end: datetime,
    reference_forecast_model: str = "icon_eu",
) -> pd.DataFrame:
    """Materialize a training dataset by calling the shared feature builder.

    For every (point, valid_time) sample where we have BOTH a forecast and an
    observation, build the feature vector and the (target_u, target_v) target.
    """
    s = load_settings()
    # Train only on operational points (exclude aux points like zurich/milano_linate
    # which are used as feature inputs only, not prediction targets).
    op_ids = s.operational_point_ids or [p.id for p in s.virtual_points]
    points = [point_id] if point_id else op_ids

    rows: list[dict[str, Any]] = []
    cur = start
    while cur < end:
        for pid in points:
            try:
                fr = build_features_for(pid, cur, reference_forecast_model=reference_forecast_model)
            except Exception as exc:
                logger.debug("Feature build failed for %s @ %s: %s", pid, cur, exc)
                continue
            if fr is None or fr.target_u is None or fr.target_v is None:
                continue
            row = {"point_id": pid, "valid_time": cur, **fr.feature_vector}
            row["target_u"] = fr.target_u
            row["target_v"] = fr.target_v
            rows.append(row)
        cur = cur + timedelta(hours=1)

    df = pd.DataFrame(rows)
    return df


def _feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop non-feature columns and return (X, feature_names)."""
    drop_cols = {"point_id", "valid_time", "target_u", "target_v"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, feature_cols


# --- LightGBM backend ---


def _train_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    quantile: float,
    params: dict[str, Any],
) -> tuple[Any, float]:
    """Train one LightGBM quantile model. Returns (model, in_sample_mae)."""
    import lightgbm as lgb

    p = dict(params)
    p["objective"] = "quantile"
    p["metric"] = "quantile"
    p["alpha"] = quantile
    p["verbose"] = -1
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    model = lgb.train(p, dtrain, num_boost_round=p.pop("num_iterations", 500))
    pred = model.predict(X)
    mae = float(np.mean(np.abs(pred - y)))
    return model, mae


def _save_lightgbm(model: Any, path: Path) -> None:
    model.save_model(str(path))


def _load_lightgbm(path: Path) -> Any:
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


def _predict_lightgbm(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


# --- XGBoost GPU backend ---


def _train_xgboost_gpu(
    X: pd.DataFrame,
    y: np.ndarray,
    quantile: float,
    params: dict[str, Any],
) -> tuple[Any, float]:
    """Train one XGBoost quantile model on GPU (falls back to CPU if no GPU).

    Spec note: user has RTX 3070, so device='cuda' is the default.
    """
    import xgboost as xgb

    p = dict(params)
    n_estimators = p.pop("num_iterations", 500)
    # Map LightGBM-style params to XGBoost equivalents
    xgb_params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "tree_method": "hist",
        "device": "cuda",  # RTX 3070 — auto-falls back to CPU if no GPU
        "objective": "reg:quantileerror",
        "quantile_alpha": quantile,
        "learning_rate": p.get("learning_rate", 0.05),
        "max_leaves": p.get("num_leaves", 63),
        "subsample": p.get("bagging_fraction", 0.9),
        "colsample_bytree": p.get("feature_fraction", 0.9),
        "min_child_weight": p.get("min_data_in_leaf", 30),
        "verbosity": 0,
    }
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y, verbose=False)
    pred = model.predict(X)
    mae = float(np.mean(np.abs(pred - y)))
    return model, mae


def _save_xgboost(model: Any, path: Path) -> None:
    # XGBoost can save as JSON but we use pickle for the wrapper
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def _load_xgboost(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _predict_xgboost(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


# --- Backend dispatch ---


def _get_backend() -> str:
    s = load_settings()
    backend = getattr(s.model, "backend", "lightgbm")
    return backend


def _train_one(backend: str, X, y, q, params):
    if backend == "xgboost_gpu":
        return _train_xgboost_gpu(X, y, q, params)
    return _train_lightgbm(X, y, q, params)


def _save_one(backend: str, model, path):
    if backend == "xgboost_gpu":
        _save_xgboost(model, path)
    else:
        _save_lightgbm(model, path)


def _load_one(backend: str, path):
    if backend == "xgboost_gpu":
        return _load_xgboost(path)
    return _load_lightgbm(path)


def _predict_one(backend: str, model, X):
    if backend == "xgboost_gpu":
        return _predict_xgboost(model, X)
    return _predict_lightgbm(model, X)


def train(
    *,
    point_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    reference_forecast_model: str = "icon_eu",
    model_version: str | None = None,
    backend: str | None = None,
) -> TrainingResult | None:
    """Train the quantile MOS model on stored history."""
    s = load_settings()
    backend = backend or _get_backend()
    end = end or datetime.utcnow()
    start = start or (end - timedelta(days=s.model.walk_forward.train_window_days))

    df = _build_dataset(point_id, start, end, reference_forecast_model=reference_forecast_model)
    if len(df) < s.model.walk_forward.min_train_samples:
        logger.warning(
            "Not enough training samples: %d (min %d). Skipping train.",
            len(df),
            s.model.walk_forward.min_train_samples,
        )
        return None

    X, feature_cols = _feature_matrix(df)
    y_u = df["target_u"].values
    y_v = df["target_v"].values

    # V5: Feature count vs sample count warning (Claude audit: overfitting risk)
    n_features = len(feature_cols)
    n_samples = len(df)
    if n_features > n_samples / 5:
        logger.warning(
            "⚠ OVERFITTING RISK: %d features vs %d samples (ratio 1:%.1f, "
            "recommended max 1:5). Consider running feature selection or "
            "collecting more data before trusting this model.",
            n_features, n_samples, n_samples / n_features,
        )
        # Auto-prune: keep only top features by variance (drop constant + low-variance)
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        X_array = X.fillna(0).values
        selector.fit(X_array)
        kept_mask = selector.get_support()
        kept_cols = [c for c, keep in zip(feature_cols, kept_mask) if keep]
        dropped = len(feature_cols) - len(kept_cols)
        if dropped > 0:
            logger.info("Auto-pruned %d low-variance features (%d → %d)",
                        dropped, len(feature_cols), len(kept_cols))
            X = X[kept_cols]
            feature_cols = kept_cols

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mv = model_version or f"mos_v1_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    metrics: dict[str, float] = {}
    model_paths: dict[str, Path] = {}

    # Train one model per (target, quantile)
    params = s.model.lgbm_params.model_dump()
    for target_name, y in [("u", y_u), ("v", y_v)]:
        for q in s.model.quantiles:
            model, in_sample_mae = _train_one(backend, X, y, q, params)
            metrics[f"{target_name}_q{q}_insample_mae"] = in_sample_mae
            ext = ".json" if backend == "lightgbm" else ".pkl"
            path = MODELS_DIR / f"{mv}_{target_name}_q{int(q*100):02d}{ext}"
            _save_one(backend, model, path)
            model_paths[f"{target_name}_q{int(q*100):02d}"] = path
            logger.info(
                "Trained %s q=%.2f in-sample MAE=%.3f -> %s [%s]",
                target_name, q, in_sample_mae, path, backend,
            )

    # Persist feature columns + backend metadata
    feature_path = MODELS_DIR / f"{mv}_features.json"
    feature_path.write_text(json.dumps({
        "features": feature_cols,
        "backend": backend,
    }, indent=2))
    model_paths["features"] = feature_path

    # Register in DB
    access.register_model(
        model_version=mv,
        trained_at=datetime.utcnow(),
        feature_set_version=s.model.feature_set_version,
        training_start=start.date(),
        training_end=end.date(),
        backtest_mae_kn=metrics.get("u_q0.5_insample_mae", 0.0),
        backtest_dir_error_deg=None,
        promoted=False,
        git_commit="",
        notes=f"backend={backend}; in-sample metrics: {metrics}",
    )

    return TrainingResult(
        model_version=mv,
        feature_set_version=s.model.feature_set_version,
        backend=backend,
        trained_at=datetime.utcnow(),
        n_samples=len(df),
        n_features=len(feature_cols),
        quantiles=list(s.model.quantiles),
        metrics=metrics,
        model_paths=model_paths,
    )


# Public re-exports for infer.py
def load_model_bundle(model_version: str) -> dict[str, Any]:
    """Load the per-(target, quantile) models + feature column list.

    Cached in-process to avoid re-reading from disk on every predict() call.
    """
    if model_version in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[model_version]
    bundle: dict[str, Any] = {}
    # Read backend from saved feature metadata first, fallback to settings
    feat_path = MODELS_DIR / f"{model_version}_features.json"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature list missing: {feat_path}")
    feat_meta = json.loads(feat_path.read_text())
    actual_backend = feat_meta.get("backend", _get_backend()) if isinstance(feat_meta, dict) else _get_backend()
    for target in ("u", "v"):
        for q in load_settings().model.quantiles:
            key = f"{target}_q{int(q*100):02d}"
            ext = ".json" if actual_backend == "lightgbm" else ".pkl"
            path = MODELS_DIR / f"{model_version}_{target}_q{int(q*100):02d}{ext}"
            if not path.exists():
                raise FileNotFoundError(f"Model artifact missing: {path}")
            bundle[key] = _load_one(actual_backend, path)
    bundle["features"] = feat_meta["features"] if isinstance(feat_meta, dict) else feat_meta
    bundle["backend"] = actual_backend
    _BUNDLE_CACHE[model_version] = bundle
    return bundle


def predict_with_bundle(bundle: dict[str, Any], X: pd.DataFrame, target: str, q: float) -> np.ndarray:
    """Predict using the loaded bundle."""
    key = f"{target}_q{int(q*100):02d}"
    backend = bundle.get("backend", "lightgbm")
    return _predict_one(backend, bundle[key], X)


_BUNDLE_CACHE: dict[str, dict[str, Any]] = {}


__all__ = ["train", "TrainingResult", "MODELS_DIR", "load_model_bundle", "predict_with_bundle"]
