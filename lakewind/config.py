"""LakeWind configuration loader.

Spec §10: "pydantic-settings for config (no module reads settings.yaml directly)".

Loads `settings.yaml` into typed pydantic models and merges `.env` secrets.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- YAML-backed config models (Spec sections referenced inline) ---


class ProjectConfig(BaseModel):
    name: str
    version: str
    timezone: str = "Europe/Rome"


class OperatingArea(BaseModel):
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class VirtualPoint(BaseModel):
    id: str
    lat: float
    lon: float


class BackfillConfig(BaseModel):
    chunk_days: int = 90
    delay_seconds: float = 1.0
    default_days: int = 365
    # Deep Audit R3 (V8): multi-level backfill vars are OFF by default —
    # archive availability is unverified and requesting unavailable vars
    # triggers the silent window-clamping bug (see historical_backfill.py).
    multi_level_vars: bool = False


class OpenMeteoConfig(BaseModel):
    base_url: str
    ensemble_url: str
    historical_url: str
    historical_forecast_url: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    previous_runs_url: str = "https://previous-runs-api.open-meteo.com/v1/forecast"
    models: list[str]
    ensemble_models: list[str] = Field(default_factory=list)
    hourly_vars: list[str]
    wind_speed_unit: str = "kn"
    timezone: str = "auto"
    forecast_days: int = 7
    backfill: BackfillConfig = Field(default_factory=BackfillConfig)


class GeoPoint(BaseModel):
    lat: float
    lon: float


class PressureGradientConfig(BaseModel):
    zurich: GeoPoint
    milano_linate: GeoPoint
    foehn_likely_hpa: float = 8.0
    foehn_strong_hpa: float = 12.0


class TimeWindow(BaseModel):
    start: str  # "HH:MM"
    end: str


class LocalWindsConfig(BaseModel):
    tivano_die_window: TimeWindow
    breva_build_window: TimeWindow


class DomasoConfig(BaseModel):
    url: str
    fallback_url: str


class CmlConfig(BaseModel):
    url: str


class ArpaConfig(BaseModel):
    base_url: str
    sensor_dataset: str
    station_dataset: str
    bbox_padding_deg: float = 0.15
    app_token_env: str = "ARPA_APP_TOKEN"


class DiyBuoyConfig(BaseModel):
    enabled: bool = False
    ingestion_url: str
    source_id: str = "diy_buoy"


class LgbmParams(BaseModel):
    objective: str = "quantile"
    metric: str = "quantile"
    num_leaves: int = 63
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    min_data_in_leaf: int = 30
    verbose: int = -1
    num_iterations: int = 500
    # Phase 3: regularization depth (used by tuning search too)
    max_depth: int = -1          # -1 = unlimited (LightGBM default)
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    min_gain_to_split: float = 0.0


class UpgradeGate(BaseModel):
    min_mae_improvement_kn: float = 0.2
    min_dir_improvement_deg: float = 5.0


class WalkForwardConfig(BaseModel):
    train_window_days: int = 60
    test_window_days: int = 14
    step_days: int = 7
    min_train_samples: int = 200


class TargetQualityConfig(BaseModel):
    """Deep Audit R2: ground-truth hierarchy training weights.

    station: real anemometers (ARPA/Domaso/DIY/Netatmo) — full trust.
    intermediate_reanalysis: CERRA-class regional reanalysis.
    era5: global reanalysis surrogate — the audit's measured 100%-of-targets
    defect means ERA5 must be DEMOTED, not removed (it still teaches
    synoptic structure that a thin station ledger cannot).
    """

    station_weight: float = 1.0
    intermediate_reanalysis_weight: float = 0.6
    era5_weight: float = 0.4


class ModelConfig(BaseModel):
    feature_set_version: str
    backend: str = "lightgbm"  # "lightgbm" or "xgboost_gpu"
    target_u: str
    target_v: str
    quantiles: list[float]
    lgbm_params: LgbmParams
    upgrade_gate: UpgradeGate
    walk_forward: WalkForwardConfig
    # --- Phase 3: training hardening ---
    # Time-ordered validation slice (last fraction of samples) used for early
    # stopping and honest out-of-sample metrics. Never a random split.
    validation_fraction: float = 0.15
    # Early stopping patience + cap on boosting rounds (early stopping wins).
    early_stopping_rounds: int = 150
    max_boost_rounds: int = 3000
    # Heterogeneous ensemble: train the secondary backend per (target, q)
    # and average predictions (variance reduction across inductive biases).
    ensemble: bool = False
    # Two-phase feature selection: gain-based shortlist + validation-set
    # permutation pruning, applied on the TRAIN side only (leakage-free).
    feature_selection: bool = False
    feature_selection_top_k: int = 120
    # Terrain channeling: valley axis azimuth (deg FROM north) for the
    # along/cross-valley wind decomposition. Per-point overrides for other
    # basins (Phase 6 multi-spot): {point_id: axis_deg}.
    valley_axis_deg: float = 10.0
    valley_axis_overrides: dict[str, float] = Field(default_factory=dict)
    # Deep Audit R10: auxiliary points (Zurich, Milano, ...) that feed the
    # Foehn gradient and thermal-contrast features read this deterministic
    # model's forecast instead of an unordered latest-run pick.
    aux_reference_model: str = "icon_eu"
    # --- Deep Audit R2: ground-truth hierarchy weights ---
    target_quality: TargetQualityConfig = Field(default_factory=TargetQualityConfig)
    # --- Deep Audit R8: production training regime ---
    # Production retraining window. 12-18 months keeps seasonal competence
    # (the former 60-day default made the deployed model seasonally blind);
    # walk_forward.train_window_days stays 60 — that is the EVALUATION
    # protocol, not the production regime.
    train_window_days: int = 548
    # Exponential recency weighting half-life (days). Recent regimes dominate
    # without erasing the seasonal prior.
    recency_half_life_days: float = 90.0
    # Windy-sample upweight: the business metric is decision precision at
    # sailing-relevant thresholds while the wind distribution is heavily
    # calm-imbalanced.
    windy_sample_upweight: float = 2.5
    windy_threshold_kn: float = 8.0


class SuccessCriteria(BaseModel):
    mae_reduction_vs_raw_nwp_pct: float = 15.0
    mae_reduction_vs_persistence_pct: float = 25.0
    dir_error_reduction_pct: float = 20.0
    confidence_interval_target_pct: float = 75.0
    decision_precision_pct: float = 80.0


class PipelineConfig(BaseModel):
    target_runtime_seconds: int = 10
    degrade_confidence_per_missing_nwp: float = 5.0
    degrade_confidence_per_missing_station: float = 3.0
    # Phase 2: prediction horizons generated every cycle. Hourly 0-24 covers
    # /today (25 rows), /sailing (11-16 local) and the map offsets without
    # resorting to on-demand inference (which is 20-50x more expensive per
    # request than reading a stored row).
    horizons: list[int] = Field(default_factory=lambda: list(range(0, 25)))
    # Phase 2: pre-render heatmap/trend artifacts after every predict cycle.
    precompute_maps: bool = True


class ApiConfig(BaseModel):
    """Phase 2: internal FastAPI service consumed by the web dashboard.

    The Node web-ui no longer opens the DuckDB file itself (that caused
    cross-process lock contention with the Python writer). It proxies to
    this API, which shares the in-process caches.
    """

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


class CacheConfig(BaseModel):
    """Phase 2 cache TTLs (seconds unless noted).

    projection_ttl: in-memory copy of the latest prediction batch — bounded
    by the 30-min predict cycle; 5 min keeps data at most ~5 min stale while
    absorbing arbitrary request bursts.
    prediction_ttl: on-demand (fallback) predictions live shorter, they are
    not persisted.
    map_max_age_minutes: a pre-rendered map older than this is considered
    stale and we fall back to rendering (2 predict cycles + margin).
    """

    projection_ttl: float = 300.0
    prediction_ttl: float = 300.0
    on_demand_ttl: float = 300.0
    map_max_age_minutes: float = 100.0
    render_semaphore: int = 2


class TelegramConfig(BaseModel):
    enabled: bool = True
    token_env: str = "TELEGRAM_BOT_TOKEN"
    allowed_user_ids: list[int] = Field(default_factory=list)


class StreamlitConfig(BaseModel):
    port: int = 8501
    title: str = "LakeWind — Dongo-Dervio"


class ScheduleConfig(BaseModel):
    collectors_nwp_minutes: int = 30
    collectors_stations_minutes: int = 10
    predict_minutes: int = 30
    backtest_cron: str = "0 4 * * *"


class DbConfig(BaseModel):
    path: str = "data/lakewind.duckdb"
    features_table: str = "features"
    forecast_table: str = "forecast_runs"
    observations_table: str = "observations"
    predictions_table: str = "predictions"
    model_registry_table: str = "model_registry"
    sailing_log_table: str = "sailing_log"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class Settings(BaseModel):
    project: ProjectConfig
    operating_area: OperatingArea
    virtual_points: list[VirtualPoint]
    operational_point_ids: list[str] = Field(default_factory=list)
    open_meteo: OpenMeteoConfig
    pressure_gradient: PressureGradientConfig
    local_winds: LocalWindsConfig
    domaso: DomasoConfig
    cml: CmlConfig
    arpa_lombardia: ArpaConfig
    diy_buoy: DiyBuoyConfig
    model: ModelConfig
    success_criteria: SuccessCriteria
    pipeline: PipelineConfig
    api: ApiConfig = Field(default_factory=ApiConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    telegram: TelegramConfig
    streamlit: StreamlitConfig
    schedule: ScheduleConfig
    db: DbConfig
    logging: LoggingConfig


# --- Secrets from .env ---


class Secrets(BaseSettings):
    """Secrets only (Spec §10). Loaded from .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: SecretStr = SecretStr("")
    arpa_app_token: SecretStr = SecretStr("")


@lru_cache(maxsize=1)
def _project_root() -> Path:
    """Walk up from this file to find the directory containing settings.yaml.

    Phase 3: lru_cache'd — profiling showed ~16 uncached calls per feature
    sample (every get_db_path() re-walked the tree with 14k stat calls/150
    samples). The root never changes within a process.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "settings.yaml").exists():
            return parent
    return Path.cwd()


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load settings.yaml + .env into typed objects."""
    root = _project_root()
    yaml_path = root / "settings.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"settings.yaml not found at {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def load_secrets() -> Secrets:
    """Load .env secrets. Tolerates missing .env (returns empty secrets)."""
    root = _project_root()
    env_path = root / ".env"
    if env_path.exists():
        # pydantic-settings reads env file directly
        return Secrets(_env_file=env_path)  # type: ignore[call-arg]
    return Secrets()


def get_db_path() -> Path:
    """Absolute path to the DuckDB file (Spec §5)."""
    root = _project_root()
    p = root / load_settings().db.path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_virtual_point(point_id: str) -> VirtualPoint:
    for vp in load_settings().virtual_points:
        if vp.id == point_id:
            return vp
    raise KeyError(f"Unknown virtual point id: {point_id}")


def reset_caches() -> None:
    """Used by tests when settings/secrets need reloading."""
    load_settings.cache_clear()
    load_secrets.cache_clear()


__all__ = [
    "Settings",
    "Secrets",
    "load_settings",
    "load_secrets",
    "get_db_path",
    "get_virtual_point",
    "reset_caches",
    "VirtualPoint",
    "GeoPoint",
    "ApiConfig",
    "CacheConfig",
]
