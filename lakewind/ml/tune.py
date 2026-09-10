"""Phase 3: hyperparameter tuning via Optuna TPE (tree-structured Parzen
estimator) over the LightGBM parameter space.

Design decisions (precision-first):

1. OBJECTIVE = mean validation pinball (quantile) loss across both targets
   and all configured quantiles. This is the loss the production models
   actually minimize; averaging u/v and q10/50/90 prevents the search from
   trading calibrated tails for a slightly better median.

2. SPLIT = time-ordered, last `validation_fraction` of the window. The tuner
   never sees the backtest's test periods — tuning on the eventual test set
   would be selection leakage (the classic rolling-origin mistake).

3. SAMPLE-EFFICIENT: the dataset is materialized ONCE by the caller and
   passed in (train(dataset=...) is the expensive I/O part). 40 trials ×
   6 quantile-models × ~2-5 s ≈ 10-15 min on a laptop CPU.

4. OVERFIT GUARDS: every trial trains with early stopping on its own
   validation slice; the search space keeps min_data_in_leaf >= 10 and
   feature_fraction <= 1.0 so no trial can memorize.

Usage:
    from lakewind.ml.tune import tune_lgbm_params
    best = tune_lgbm_params(dataset=df, n_trials=40)

CLI:
    lakewind tune --days 120 --trials 40
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from lakewind.config import load_settings
from lakewind.ml.train import _feature_matrix, _time_ordered_split, _train_lightgbm

logger = logging.getLogger(__name__)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss — the metric quantile models minimize."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def _quantile_objective(
    trial: Any,
    X_tr: pd.DataFrame,
    y_tr: dict[str, np.ndarray],
    X_val: pd.DataFrame,
    y_val: dict[str, np.ndarray],
    quantiles: list[float],
    max_rounds: int,
    early_stopping_rounds: int,
) -> float:
    """Train u+v quantile models with trial params, return mean val pinball."""
    params: dict[str, Any] = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 150),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
        "verbose": -1,
        # num_iterations is popped inside _train_lightgbm; keep max_rounds as cap
        "num_iterations": max_rounds,
    }

    losses: list[float] = []
    for target, y_tr_arr in y_tr.items():
        y_val_arr = y_val[target]
        for q in quantiles:
            model, _info = _train_lightgbm(
                X_tr, y_tr_arr, q, params,
                X_val=X_val, y_val=y_val_arr,
                early_stopping_rounds=early_stopping_rounds,
                max_rounds=max_rounds,
            )
            pred = model.predict(X_val)
            losses.append(_pinball_loss(y_val_arr, pred, q))
    return float(np.mean(losses))


def tune_lgbm_params(
    dataset: pd.DataFrame,
    *,
    n_trials: int = 40,
    validation_fraction: float | None = None,
    max_rounds: int | None = None,
    early_stopping_rounds: int | None = None,
    seed: int = 42,
    storage_path: str | None = None,
) -> dict[str, Any]:
    """Run the TPE search on a prebuilt dataset; return the best param dict.

    The returned dict uses LightGBM-native key names and drops search-only
    keys; it is directly mergeable into settings.yaml `model.lgbm_params`.

    `storage_path` (SQLite) makes the study RESUMABLE — killed runs continue
    from completed trials instead of restarting the search. The caller may
    pass a smaller max_rounds/patience budget than production: tuning ranks
    configurations; final models train with the full budget.
    """
    import optuna

    s = load_settings()
    val_fraction = validation_fraction if validation_fraction is not None else s.model.validation_fraction
    rounds = max_rounds if max_rounds is not None else s.model.max_boost_rounds
    esr = early_stopping_rounds if early_stopping_rounds is not None else s.model.early_stopping_rounds

    df = dataset.dropna(subset=["target_u", "target_v"]).reset_index(drop=True)
    if len(df) < 500:
        raise ValueError(
            f"Tuning needs >= 500 samples, got {len(df)}. Backfill more history first."
        )
    df_tr, df_val = _time_ordered_split(df, val_fraction)
    if df_val is None:
        raise ValueError(
            "Dataset too small for a time-ordered validation split — backfill more history."
        )
    X_tr, _ = _feature_matrix(df_tr)
    X_val, _ = _feature_matrix(df_val)
    y_tr = {"u": df_tr["target_u"].values, "v": df_tr["target_v"].values}
    y_val = {"u": df_val["target_u"].values, "v": df_val["target_v"].values}
    print(f"Tune split: {len(X_tr)} train / {len(X_val)} val samples, "
          f"{X_tr.shape[1]} features, budget={rounds} rounds (patience {esr})")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    if storage_path:
        storage = f"sqlite:///{storage_path}"
        study = optuna.create_study(direction="minimize", sampler=sampler,
                                    study_name="lakewind_lgbm", storage=storage,
                                    load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler,
                                    study_name="lakewind_lgbm")

    remaining = max(0, n_trials - len(study.trials))
    if remaining == 0:
        print(f"Study already has {len(study.trials)} trials — nothing to run")
    def objective(trial: optuna.trial.Trial) -> float:
        return _quantile_objective(
            trial, X_tr, y_tr, X_val, y_val,
            quantiles=list(s.model.quantiles),
            max_rounds=rounds,
            early_stopping_rounds=esr,
        )
    study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    best = dict(study.best_params)
    logger.info("Tuning done: %d trials, best val pinball %.5f", len(study.trials), study.best_value)
    logger.info("Best params: %s", best)
    return best


def tune_from_db(
    *,
    days: int = 120,
    n_trials: int = 40,
    start: Any = None,
    end: Any = None,
    storage_path: str | None = None,
) -> dict[str, Any]:
    """Materialize the dataset from the DB and tune (CLI entry point)."""
    from datetime import timedelta

    from lakewind.ml.train import _build_dataset
    from lakewind.utils.timeutil import utcnow

    end_dt = end or utcnow()
    start_dt = start or (end_dt - timedelta(days=days))
    logger.info("Materializing tuning dataset %s → %s", start_dt.date(), end_dt.date())
    df = _build_dataset(None, start_dt, end_dt)
    return tune_lgbm_params(df, n_trials=n_trials, storage_path=storage_path)


__all__ = ["tune_lgbm_params", "tune_from_db", "_pinball_loss"]
