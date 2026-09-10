"""LightGBM / XGBoost quantile MOS trainer (Spec §7.1).

V1 model:
- One model per target (target_u, target_v) per quantile.
- Quantile objective (10/50/90) gives calibrated uncertainty for free.
- `expected_error_kn` = half the predicted 90th-10th interval width.
- Tree-based, supports NaN natively (Spec §6 missing-data policy).

Backend selection (configurable via settings.yaml `model.backend`):
- `lightgbm` (default; CPU only via pip)
- `xgboost_gpu` — uses XGBoost with `device='cuda'` when a usable GPU exists
  (real CUDA detection with CPU fallback, V6.6 fix).

Phase 3 training hardening (this file):
1. TIME-ORDERED validation split (`model.validation_fraction`, default 15%):
   the last fraction of samples by valid_time is held out. Never random —
   random splits leak autocorrelated weather regimes into the validation set
   and overstate skill.
2. EARLY STOPPING on the validation pinball loss (`early_stopping_rounds`,
   `max_boost_rounds`): replaces the blind fixed-500-rounds training that
   could neither under- nor over-fit adaptively.
3. TWO-PHASE FEATURE SELECTION (`model.feature_selection`): gain-based
   shortlist → validation-set permutation pruning (train side only; the val
   split never touches fitting — pruning uses it as an honest probe).
4. HETEROGENEOUS ENSEMBLE (`model.ensemble`): trains BOTH backends per
   (target, quantile) and averages predictions — variance reduction across
   two different implementations' inductive biases.
5. `train(dataset=...)` accepts a prebuilt DataFrame so tuning/evaluation
   loops materialize features ONCE (the expensive part is I/O, not boosting).

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
from lakewind.utils.timeutil import utcnow
from lakewind.utils.wind import WindVector

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"

# Backend tags used in artifact filenames and features.json
_BACKEND_TAG = {"lightgbm": "lgb", "xgboost_gpu": "xgb"}


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
            # Deep Audit R2/R8: hierarchy metadata travels with every sample
            # (target-quality weight + observed speed for the windy upweight).
            row["target_weight"] = fr.meta.get("target_weight")
            row["obs_speed_kn"] = fr.meta.get("obs_speed_kn")
            row["obs_source"] = fr.meta.get("obs_source")
            rows.append(row)
        cur = cur + timedelta(hours=1)

    df = pd.DataFrame(rows)
    return df


def _feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop non-feature columns and return (X, feature_names)."""
    drop_cols = {
        "point_id", "valid_time", "target_u", "target_v",
        "target_weight", "obs_speed_kn", "obs_source",  # Deep Audit R2/R8 meta
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, feature_cols


def compute_sample_weights(
    df: pd.DataFrame,
    *,
    half_life_days: float = 0.0,
    windy_upweight: float = 1.0,
    windy_threshold_kn: float = 8.0,
    reference_time: pd.Timestamp | datetime | None = None,
) -> np.ndarray:
    """Combined per-sample training weights (Deep Audit R2 + R8).

    product of three factors (1.0 whenever the corresponding input is absent,
    so callers with legacy datasets get unchanged behaviour):
      - target quality (R2): station 1.0 / intermediate reanalysis ~0.6 /
        ERA5 ~0.4, times the observation's own confidence — computed in the
        feature builder and carried in the ``target_weight`` column.
      - recency (R8): exponential 0.5**(age_days / half_life) so recent
        regimes dominate without erasing the seasonal prior.
      - windy upweight (R8): ``windy_upweight`` for samples whose observed
        speed >= ``windy_threshold_kn`` — the business metric is decision
        precision at sailing thresholds, the wind distribution is calm-heavy.
    """
    w = np.ones(len(df), dtype=float)
    if "target_weight" in df.columns:
        tw = pd.to_numeric(df["target_weight"], errors="coerce").to_numpy(dtype=float)
        tw = np.where(np.isfinite(tw), tw, 1.0)
        w = w * np.clip(tw, 0.0, 1.0)
    if "obs_speed_kn" in df.columns and windy_upweight and windy_upweight > 1.0:
        sp = pd.to_numeric(df["obs_speed_kn"], errors="coerce").to_numpy(dtype=float)
        sp = np.where(np.isfinite(sp), sp, 0.0)
        w = w * np.where(sp >= windy_threshold_kn, float(windy_upweight), 1.0)
    if "valid_time" in df.columns and half_life_days and half_life_days > 0:
        vt = pd.to_datetime(df["valid_time"])
        ref = pd.Timestamp(reference_time) if reference_time is not None else vt.max()
        age_days = (ref - vt).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        age_days = np.clip(np.where(np.isfinite(age_days), age_days, 0.0), 0.0, None)
        w = w * np.power(0.5, age_days / float(half_life_days))
    return w


def _time_ordered_split(
    df: pd.DataFrame, validation_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split into (train, val) by valid_time — the LAST slice becomes val.

    Phase 3 anti-leakage rule: the validation set must be strictly after the
    training set in time. All points sharing an hour stay on the same side.
    Returns (train, None) when the fraction or the data is too small to
    produce a meaningful validation slice.
    """
    if validation_fraction <= 0 or len(df) < 100:
        return df, None
    df_sorted = df.sort_values("valid_time").reset_index(drop=True)
    cut = int(len(df_sorted) * (1.0 - validation_fraction))
    # Guardrails: at least 50 val samples and at least 100 train samples
    if cut < 100 or (len(df_sorted) - cut) < 50:
        return df, None
    return df_sorted.iloc[:cut], df_sorted.iloc[cut:]


# --- LightGBM backend ---


def _train_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    quantile: float,
    params: dict[str, Any],
    *,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_rounds: int = 0,
    max_rounds: int | None = None,
    sample_weight: np.ndarray | None = None,
    sample_weight_val: np.ndarray | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train one LightGBM quantile model.

    With a validation set: early stopping on the validation pinball loss and
    `predict` automatically uses `best_iteration`. Returns (model, info).
    Deep Audit R2/R8: optional per-sample weights (target quality x recency
    x windy upweight) applied to BOTH train and validation so early stopping
    optimizes the same weighted objective the model is trained on.
    """
    import lightgbm as lgb

    p = dict(params)
    p["objective"] = "quantile"
    p["metric"] = "quantile"
    p["alpha"] = quantile
    p["verbose"] = -1
    num_rounds = max_rounds or p.pop("num_iterations", 500)
    dtrain = lgb.Dataset(X, label=y, weight=sample_weight, free_raw_data=False)
    info: dict[str, float] = {}
    if X_val is not None and y_val is not None and early_stopping_rounds > 0:
        dval = lgb.Dataset(X_val, label=y_val, weight=sample_weight_val, reference=dtrain)
        model = lgb.train(
            p,
            dtrain,
            num_boost_round=num_rounds,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        info["best_iteration"] = float(model.best_iteration or num_rounds)
        val_pred = model.predict(X_val)
        info["val_mae"] = float(np.mean(np.abs(val_pred - y_val)))
    else:
        model = lgb.train(p, dtrain, num_boost_round=num_rounds)
    pred = model.predict(X)
    info["insample_mae"] = float(np.mean(np.abs(pred - y)))
    return model, info


def _save_lightgbm(model: Any, path: Path) -> None:
    model.save_model(str(path))


def _load_lightgbm(path: Path) -> Any:
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


def _predict_lightgbm(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


# --- XGBoost GPU backend ---


def _cuda_available() -> bool:
    """True if XGBoost was built with CUDA AND a GPU device is usable.

    V6.6 FIX: the code set device='cuda' unconditionally, claiming XGBoost
    "auto-falls back to CPU" — it does not: training raises on CUDA-less
    machines (e.g. the T420 Docker deployment), killing every retrain.
    """
    try:
        import xgboost as xgb

        info = xgb.build_info()
        if not info.get("USE_CUDA"):
            return False
        # Also verify a device is actually visible via nvidia-smi (cheap check)
        import shutil
        import subprocess

        if shutil.which("nvidia-smi") is None:
            return False
        res = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
        return res.returncode == 0 and "GPU" in res.stdout
    except Exception:
        return False


def _train_xgboost_gpu(
    X: pd.DataFrame,
    y: np.ndarray,
    quantile: float,
    params: dict[str, Any],
    *,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_rounds: int = 0,
    max_rounds: int | None = None,
    sample_weight: np.ndarray | None = None,
    sample_weight_val: np.ndarray | None = None,
) -> tuple[Any, dict[str, float]]:
    """Train one XGBoost quantile model on GPU (falls back to CPU if no GPU).

    Phase 3: early stopping via eval_set when a validation split is provided.
    """
    import xgboost as xgb

    p = dict(params)
    n_estimators = max_rounds or p.pop("num_iterations", 500)
    use_gpu = _cuda_available()
    if not use_gpu:
        logger.info("XGBoost: no usable CUDA device — training on CPU")
    # Map LightGBM-style params to XGBoost equivalents
    xgb_params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "objective": "reg:quantileerror",
        "quantile_alpha": quantile,
        "learning_rate": p.get("learning_rate", 0.05),
        "max_leaves": p.get("num_leaves", 63),
        "subsample": p.get("bagging_fraction", 0.9),
        "colsample_bytree": p.get("feature_fraction", 0.9),
        "min_child_weight": p.get("min_data_in_leaf", 30),
        "reg_alpha": p.get("lambda_l1", 0.0),
        "reg_lambda": p.get("lambda_l2", 0.0),
        "max_depth": p.get("max_depth", 0) if p.get("max_depth", -1) > 0 else 0,
        "verbosity": 0,
    }
    info: dict[str, float] = {}
    model = xgb.XGBRegressor(**xgb_params)
    if X_val is not None and y_val is not None and early_stopping_rounds > 0:
        # sklearn API: early_stopping_rounds is a constructor param in 2.x
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        try:
            model.fit(
                X, y,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[sample_weight_val] if sample_weight_val is not None else None,
                verbose=False,
            )
            info["best_iteration"] = float(getattr(model, "best_iteration", n_estimators) or n_estimators)
        except TypeError:
            # Older sklearn wrapper without eval_set/sample-weight support:
            # fall back to fixed-rounds training.
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X, y, sample_weight=sample_weight, verbose=False)
    else:
        model.fit(X, y, sample_weight=sample_weight, verbose=False)
    pred = model.predict(X)
    info["insample_mae"] = float(np.mean(np.abs(pred - y)))
    if X_val is not None and y_val is not None and "best_iteration" in info:
        val_pred = model.predict(X_val)
        info["val_mae"] = float(np.mean(np.abs(val_pred - np.asarray(y_val))))
    return model, info


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


def _train_one(backend: str, X, y, q, params, *, X_val=None, y_val=None,
               early_stopping_rounds: int = 0, max_rounds: int | None = None,
               sample_weight=None, sample_weight_val=None):
    if backend == "xgboost_gpu":
        return _train_xgboost_gpu(
            X, y, q, params, X_val=X_val, y_val=y_val,
            early_stopping_rounds=early_stopping_rounds, max_rounds=max_rounds,
            sample_weight=sample_weight, sample_weight_val=sample_weight_val,
        )
    return _train_lightgbm(
        X, y, q, params, X_val=X_val, y_val=y_val,
        early_stopping_rounds=early_stopping_rounds, max_rounds=max_rounds,
        sample_weight=sample_weight, sample_weight_val=sample_weight_val,
    )


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


# --- Phase 3: two-phase feature selection ---


def _select_features(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    top_k: int,
    quantile: float = 0.5,
) -> list[str]:
    """Gain shortlist → validation permutation pruning (train side only).

    Stage 1 trains a quick reference model and keeps the top-K features by
    gain. Stage 2 permutes each shortlisted feature on the VALIDATION set and
    drops those whose shuffling does not degrade the metric (i.e. features
    the model does not genuinely use — noise suppliers).
    """
    from sklearn.inspection import permutation_importance as _perm

    cols = list(X_tr.columns)
    if len(cols) <= top_k:
        shortlist = cols
    else:
        ref_model, _ = _train_lightgbm(
            X_tr, y_tr, quantile, {**params, "learning_rate": max(params.get("learning_rate", 0.05), 0.05)},
            max_rounds=300,
        )
        gain = ref_model.feature_importance(importance_type="gain")
        order = np.argsort(-gain)
        shortlist = [cols[i] for i in order[:top_k]]
        # keep original column order for downstream determinism
        shortlist = [c for c in cols if c in set(shortlist)]

    if len(shortlist) <= 4:
        return shortlist

    try:
        quick = _train_lightgbm(
            X_tr[shortlist], y_tr, quantile,
            {**params, "learning_rate": max(params.get("learning_rate", 0.05), 0.05)},
            X_val=X_val[shortlist], y_val=y_val,
            early_stopping_rounds=50, max_rounds=600,
        )[0]
        perm = _perm(
            quick, X_val[shortlist], y_val,
            scoring="neg_mean_absolute_error", n_repeats=3, random_state=7,
        )
        keep = [c for c, imp in zip(shortlist, perm.importances_mean, strict=False) if imp > 0]
        return keep or shortlist
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Permutation pruning failed (%s) — keeping gain shortlist", exc)
        return shortlist


def train(
    *,
    point_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    reference_forecast_model: str = "icon_eu",
    model_version: str | None = None,
    backend: str | None = None,
    dataset: pd.DataFrame | None = None,
    disable_models: list[str] | None = None,
) -> TrainingResult | None:
    """Train the quantile MOS model on stored history.

    Phase 3: `dataset` allows callers (tuner, experiment harness, auto
    retraining) to pass a prebuilt feature DataFrame — the expensive part is
    materializing features, not boosting. Time-ordered validation split +
    early stopping + optional feature selection + optional heterogeneous
    ensemble, all driven by settings.yaml.

    Deep Audit R14 (3.8): `disable_models` drops every fc_<model>_* column
    before fitting — the mechanism for the walk-forward ablation that decides
    with data whether a member (e.g. gfs_seamless, the audit's weakest-link
    candidate) earns its quota. Compare registry metrics of the ablated run
    against the full run via experiment_attempts/upgrade gate.
    """
    s = load_settings()
    backend = backend or _get_backend()
    end = end or utcnow()
    # Deep Audit R8: production retraining uses the LONG window (12-18 months)
    # from model.train_window_days; the 60-day walk_forward window is the
    # evaluation protocol and no longer silently caps deployed models.
    production_window_days = int(
        getattr(s.model, "train_window_days", 0) or s.model.walk_forward.train_window_days
    )
    start = start or (end - timedelta(days=production_window_days))

    df = dataset if dataset is not None else _build_dataset(
        point_id, start, end, reference_forecast_model=reference_forecast_model
    )
    if disable_models:
        drop = [
            c for c in df.columns
            if any(c.startswith(f"fc_{m}_") or c == f"fc_{m}" for m in disable_models)
        ]
        if drop:
            logger.info(
                "Ablation: dropping %d columns for disabled models %s", len(drop), disable_models
            )
            df = df.drop(columns=drop)
    if len(df) < s.model.walk_forward.min_train_samples:
        logger.warning(
            "Not enough training samples: %d (min %d). Skipping train.",
            len(df),
            s.model.walk_forward.min_train_samples,
        )
        return None

    df = df.sort_values("valid_time").reset_index(drop=True)
    X_all, feature_cols = _feature_matrix(df)
    y_u = df["target_u"].values
    y_v = df["target_v"].values

    # --- Phase 3: time-ordered validation split ---
    val_fraction = float(getattr(s.model, "validation_fraction", 0.15) or 0.0)
    es_rounds = int(getattr(s.model, "early_stopping_rounds", 150))
    max_rounds = int(getattr(s.model, "max_boost_rounds", 3000))
    df_tr, df_val = _time_ordered_split(df, val_fraction)
    # Deep Audit R2/R8: sample weights (target quality x recency x windy).
    # Reference time for recency is the end of the available history so the
    # newest samples weigh ~1.0 regardless of when training is invoked.
    half_life = float(getattr(s.model, "recency_half_life_days", 0.0) or 0.0)
    windy_up = float(getattr(s.model, "windy_sample_upweight", 1.0) or 1.0)
    windy_thr = float(getattr(s.model, "windy_threshold_kn", 8.0) or 8.0)
    weight_ref = df["valid_time"].max()
    if df_val is None:
        X_tr, y_tr_u, y_tr_v = X_all, y_u, y_v
        feature_cols_tr = list(feature_cols)
        X_val = y_val_u = y_val_v = None
        sw_tr_u = sw_tr_v = compute_sample_weights(
            df, half_life_days=half_life, windy_upweight=windy_up,
            windy_threshold_kn=windy_thr, reference_time=weight_ref,
        )
        sw_val_u = sw_val_v = None
        logger.info("Validation split disabled/too small — fixed-rounds training on all data")
    else:
        X_tr, feature_cols_tr = _feature_matrix(df_tr)
        X_val, _ = _feature_matrix(df_val)
        y_tr_u, y_tr_v = df_tr["target_u"].values, df_tr["target_v"].values
        y_val_u, y_val_v = df_val["target_u"].values, df_val["target_v"].values
        sw_tr_u = sw_tr_v = compute_sample_weights(
            df_tr, half_life_days=half_life, windy_upweight=windy_up,
            windy_threshold_kn=windy_thr, reference_time=weight_ref,
        )
        sw_val_u = sw_val_v = compute_sample_weights(
            df_val, half_life_days=0.0, windy_upweight=windy_up,
            windy_threshold_kn=windy_thr, reference_time=weight_ref,
        )
        logger.info(
            "Time-ordered split: %d train / %d val (val ends %s); weighted objective "
            "(half-life %.0fd, windy %.1fx >= %.0fkn)",
            len(df_tr), len(df_val), df_val["valid_time"].max(),
            half_life, windy_up, windy_thr,
        )

    # --- Phase 3: two-phase feature selection (train side only) ---
    selected_cols: list[str] | None = None
    if getattr(s.model, "feature_selection", False) and X_val is not None:
        params_sel = s.model.lgbm_params.model_dump()
        try:
            sel_u = _select_features(
                X_tr, y_tr_u, X_val, y_val_u, params_sel,
                top_k=int(getattr(s.model, "feature_selection_top_k", 120)),
            )
            sel_v = _select_features(
                X_tr, y_tr_v, X_val, y_val_v, params_sel,
                top_k=int(getattr(s.model, "feature_selection_top_k", 120)),
            )
            # Union of per-target survivors (targets share the input schema)
            selected_cols = [c for c in feature_cols_tr if c in set(sel_u) | set(sel_v)]
            logger.info(
                "Feature selection: %d → %d features (u:%d survivors, v:%d survivors)",
                len(feature_cols_tr), len(selected_cols), len(sel_u), len(sel_v),
            )
        except Exception as exc:
            logger.warning("Feature selection failed (%s) — training on all features", exc)
            selected_cols = None
    if selected_cols is not None:
        X_tr = X_tr[selected_cols]
        X_val = X_val[selected_cols] if X_val is not None else None
    else:
        selected_cols = list(feature_cols_tr)
    if X_val is not None:
        X_val = X_val[selected_cols]

    # V5: Feature count vs sample count warning (Claude audit: overfitting risk)
    n_features = len(selected_cols)
    n_samples = len(df)
    if n_features > n_samples / 5:
        logger.warning(
            "⚠ OVERFITTING RISK: %d features vs %d samples (ratio 1:%.1f, "
            "recommended max 1:5). Enable model.feature_selection or collect "
            "more data before trusting this model.",
            n_features, n_samples, n_samples / n_features,
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mv = model_version or f"mos_v1_{utcnow().strftime('%Y%m%d_%H%M%S')}"

    # --- Phase 3: heterogeneous ensemble membership ---
    members: list[str] = [backend]
    if getattr(s.model, "ensemble", False):
        other = "lightgbm" if backend == "xgboost_gpu" else "xgboost_gpu"
        if other not in members:
            members.append(other)
        logger.info("Ensemble training enabled — members: %s", members)

    metrics: dict[str, float] = {}
    model_paths: dict[str, Path] = {}

    # Train one model per (target, quantile, member)
    params = s.model.lgbm_params.model_dump()
    for target_name, y_tr, y_val, sw_tr, sw_val in (
        ("u", y_tr_u, y_val_u, sw_tr_u, sw_val_u),
        ("v", y_tr_v, y_val_v, sw_tr_v, sw_val_v),
    ):
        for q in s.model.quantiles:
            q_key = f"{target_name}_q{int(q*100):02d}"
            for member in members:
                model, info = _train_one(
                    member, X_tr, y_tr, q, params,
                    X_val=X_val, y_val=y_val,
                    early_stopping_rounds=es_rounds if X_val is not None else 0,
                    max_rounds=max_rounds,
                    sample_weight=sw_tr, sample_weight_val=sw_val,
                )
                for k, v in info.items():
                    metrics[f"{q_key}_{k}"] = v
                if member == members[0]:
                    ext = ".json" if member == "lightgbm" else ".pkl"
                    path = MODELS_DIR / f"{mv}_{q_key}{ext}"
                else:
                    tag = _BACKEND_TAG[member]
                    ext = ".json" if member == "lightgbm" else ".pkl"
                    path = MODELS_DIR / f"{mv}_{q_key}_{tag}{ext}"
                _save_one(member, model, path)
                model_paths[f"{q_key}{'_' + _BACKEND_TAG[member] if member != members[0] else ''}"] = path
                logger.info(
                    "Trained %s [%s] q=%.2f insample MAE=%.3f val MAE=%s -> %s",
                    q_key, member, q,
                    info.get("insample_mae", float("nan")),
                    f"{info['val_mae']:.3f}" if "val_mae" in info else "n/a",
                    path,
                )

    # Persist feature columns + backend/ensemble metadata
    feature_path = MODELS_DIR / f"{mv}_features.json"
    feature_path.write_text(json.dumps({
        "features": selected_cols,
        "backend": backend,
        "ensemble": members,
        "feature_set_version": s.model.feature_set_version,
        "validation_fraction": val_fraction,
        "early_stopping_rounds": es_rounds,
    }, indent=2))
    model_paths["features"] = feature_path

    # Register in DB — with SPEED-SPACE metrics (Deep Audit R9).
    # The registry previously stored the BIAS-u validation MAE, which is not
    # comparable across model versions and not the product metric. Reconstruct
    # observed and predicted speeds on the validation slice and register the
    # speed-space MAE + circular direction error.
    speed_mae: float | None = None
    dir_err: float | None = None
    if X_val is not None and df_val is not None:
        try:
            bundle = load_model_bundle(mv)
            bu = predict_with_bundle(bundle, X_val, "u", 0.5)
            bv = predict_with_bundle(bundle, X_val, "v", 0.5)
            ref_col_speed = f"fc_{reference_forecast_model}_speed"
            ref_col_dir = f"fc_{reference_forecast_model}_dir"
            if ref_col_speed in X_val.columns and ref_col_dir in X_val.columns:
                speed_errs: list[float] = []
                dir_errs: list[float] = []
                # Deep Audit R14 (5.5): signed direction residual per regime —
                # Breva/Tivano/Foehn have distinct direction-error signatures;
                # the artifact lets serving apply a small regime-conditional
                # rotation (only where enough samples justify it).
                regime_dir_residuals: dict[str, list[float]] = {}
                for i in range(len(X_val)):
                    rs = X_val[ref_col_speed].iloc[i]
                    rd = X_val[ref_col_dir].iloc[i]
                    if pd.isna(rs) or pd.isna(rd):
                        continue
                    ru, rv = WindVector(float(rs), float(rd)).to_uv()
                    # observed = reference + true bias; predicted = reference + predicted bias
                    ou, ov = ru + float(df_val["target_u"].iloc[i]), rv + float(df_val["target_v"].iloc[i])
                    pu, pv = ru + float(bu[i]), rv + float(bv[i])
                    o_vec, p_vec = WindVector.from_uv(ou, ov), WindVector.from_uv(pu, pv)
                    speed_errs.append(abs(o_vec.speed_kn - p_vec.speed_kn))
                    signed = (p_vec.direction_deg - o_vec.direction_deg + 180.0) % 360.0 - 180.0
                    dir_errs.append(abs(signed))
                    active = next(
                        (
                            lbl
                            for lbl in ("breva", "tivano", "foehn", "storm", "calm")
                            if f"regime_{lbl}" in X_val.columns
                            and int(X_val[f"regime_{lbl}"].iloc[i]) == 1
                        ),
                        None,
                    )
                    if active is not None:
                        regime_dir_residuals.setdefault(active, []).append(signed)
                if speed_errs:
                    speed_mae = float(np.mean(speed_errs))
                    dir_err = float(np.mean(dir_errs))
                    # persist the per-regime rotation artifact (min 50 samples)
                    artifact = {
                        lbl: {
                            "residual_deg": round(float(np.mean(vals)), 3),
                            "n": len(vals),
                        }
                        for lbl, vals in regime_dir_residuals.items()
                        if len(vals) >= 50
                    }
                    if artifact:
                        art_path = MODELS_DIR / f"{mv}_regime_dir.json"
                        art_path.write_text(json.dumps(artifact, indent=2))
                        logger.info("Regime direction artifact: %s", artifact)
        except Exception as exc:
            logger.debug("Speed-space registry metrics skipped: %s", exc)

    metrics.get("u_q50_val_mae")
    access.register_model(
        model_version=mv,
        trained_at=utcnow(),
        feature_set_version=s.model.feature_set_version,
        training_start=start.date(),
        training_end=end.date(),
        backtest_mae_kn=round(speed_mae, 4) if speed_mae is not None else 0.0,
        backtest_dir_error_deg=round(dir_err, 3) if dir_err is not None else None,
        promoted=False,
        git_commit="",
        notes=(
            f"backend={backend}; ensemble={members}; "
            f"features={n_features}; samples={n_samples}; "
            f"val_split={val_fraction}; "
            + (f"speed_space_val_mae_kn={speed_mae:.4f}; " if speed_mae is not None else "")
            + f"bias_space_val_metrics: { {k: round(v, 4) for k, v in metrics.items() if 'val_' in k} }"
        ),
    )

    return TrainingResult(
        model_version=mv,
        feature_set_version=s.model.feature_set_version,
        backend=backend,
        trained_at=utcnow(),
        n_samples=n_samples,
        n_features=n_features,
        quantiles=list(s.model.quantiles),
        metrics=metrics,
        model_paths=model_paths,
    )


# Public re-exports for infer.py
def load_model_bundle(model_version: str) -> dict[str, Any]:
    """Load the per-(target, quantile) models + feature column list.

    Phase 3: understands the `ensemble` metadata — bundle[key] is a LIST of
    member models when more than one backend was trained, else a single model
    (backward compatible with pre-Phase-3 artifacts). Cached in-process.
    """
    if model_version in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[model_version]
    bundle: dict[str, Any] = {}
    feat_path = MODELS_DIR / f"{model_version}_features.json"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature list missing: {feat_path}")
    feat_meta = json.loads(feat_path.read_text())
    actual_backend = feat_meta.get("backend", _get_backend()) if isinstance(feat_meta, dict) else _get_backend()
    members = feat_meta.get("ensemble") or [actual_backend]
    if isinstance(members, str):
        members = [members]
    members[0]
    for target in ("u", "v"):
        for q in load_settings().model.quantiles:
            key = f"{target}_q{int(q*100):02d}"
            loaded: list[Any] = []
            for mi, member in enumerate(members):
                ext = ".json" if member == "lightgbm" else ".pkl"
                if mi == 0:
                    path = MODELS_DIR / f"{model_version}_{key}{ext}"
                else:
                    path = MODELS_DIR / f"{model_version}_{key}_{_BACKEND_TAG[member]}{ext}"
                if not path.exists():
                    raise FileNotFoundError(f"Model artifact missing: {path}")
                loaded.append(_load_one(member, path))
            bundle[key] = loaded if len(loaded) > 1 else loaded[0]
    bundle["features"] = feat_meta["features"] if isinstance(feat_meta, dict) else feat_meta
    bundle["backend"] = actual_backend
    bundle["ensemble_members"] = members
    _BUNDLE_CACHE[model_version] = bundle
    return bundle


def predict_with_bundle(bundle: dict[str, Any], X: pd.DataFrame, target: str, q: float) -> np.ndarray:
    """Predict using the loaded bundle.

    Ensemble bundles average member predictions (equal weights — both members
    are validated before promotion, and weighted blends are only justified by
    the walk-forward evidence collected per deployment).
    """
    key = f"{target}_q{int(q*100):02d}"
    model_or_list = bundle[key]
    if isinstance(model_or_list, list):
        backend_of = bundle.get("ensemble_members") or ["lightgbm"]
        preds = [_predict_one(backend_of[i], m, X) for i, m in enumerate(model_or_list)]
        return np.mean(preds, axis=0)
    backend = bundle.get("backend", "lightgbm")
    return _predict_one(backend, model_or_list, X)


_BUNDLE_CACHE: dict[str, dict[str, Any]] = {}


__all__ = [
    "train", "TrainingResult", "MODELS_DIR", "load_model_bundle", "predict_with_bundle",
    "_time_ordered_split", "_select_features",
]
