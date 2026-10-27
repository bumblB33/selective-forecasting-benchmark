from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import pandas as pd

from core.models import _cleanup_memory, _heartbeat, derive_available_exog
from . import config

logger = logging.getLogger(__name__)


def _build_statsforecast_models() -> list:
    """Instantiate interval StatsForecast models with deferred imports."""
    from statsforecast.models import MSTL, AutoARIMA, SeasonalNaive

    trend_forecaster_factories = {
        "AutoARIMA": lambda: AutoARIMA(),
    }

    def _resolve_trend_forecaster(value):
        if isinstance(value, str):
            factory = trend_forecaster_factories.get(value)
            if factory is None:
                raise ValueError(
                    f"Unknown trend_forecaster string '{value}'. " "Pass an instantiated model object instead."
                )
            return factory()
        return value

    model_factories = {
        "SeasonalNaive": lambda cfg: SeasonalNaive(season_length=cfg["season_length"]),
        "MSTL": lambda cfg: MSTL(
            season_length=cfg["season_length"],
            **(
                {"trend_forecaster": _resolve_trend_forecaster(cfg["trend_forecaster"])}
                if "trend_forecaster" in cfg
                else {}
            ),
        ),
    }

    models = []
    for model_name, model_cfg in config.STATSFORECAST_MODELS.items():
        if model_name in model_factories:
            models.append(model_factories[model_name](model_cfg))
    return models


def _build_lag_transforms(model_name: str) -> dict:
    """Lag transforms: long-run level + same-slot trailing-week average."""
    from mlforecast.lag_transforms import ExpandingMean, RollingMean

    lag_transforms = {
        1: [ExpandingMean()],
        config.SEASON_LENGTH_DAILY: [RollingMean(window_size=7)],
    }
    if model_name == "LightGBM_Accuracy":
        lag_transforms[config.SEASON_LENGTH_WEEKLY] = [RollingMean(window_size=4)]
    return lag_transforms


def _build_mlforecast_model(model_name: str, model_cfg: dict):
    """LightGBM regressor built from MLForecast config (after stripping framework keys)."""
    from lightgbm import LGBMRegressor

    lgbm_kwargs = {
        key: value for key, value in model_cfg.items() if key not in ("lags", "date_features", "num_threads")
    }
    return LGBMRegressor(**lgbm_kwargs, verbose=-1, random_state=config.RANDOM_SEED)


def run_statsforecast(training_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train and forecast interval StatsForecast models with no exogenous features —
    isolating the library's native multi-seasonal structure as the sub-daily baseline.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    models = _build_statsforecast_models()
    if not models:
        return {}

    core_train = training_df[[config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS, config.NIXTLA_Y]].copy()

    sf = StatsForecast(
        models=models,
        freq=config.FREQUENCY,
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=config.SEASON_LENGTH_DAILY),
    )

    logger.info(
        "StatsForecast: %d models, %d locations, %d training rows",
        len(models),
        core_train[config.NIXTLA_UNIQUE_ID].nunique(),
        len(core_train),
    )
    started_at = time.time()

    with warnings.catch_warnings(), _heartbeat("StatsForecast forecast"):
        warnings.filterwarnings("ignore")
        predictions = sf.forecast(
            df=core_train,
            h=config.FORECAST_HORIZON,
        ).reset_index(drop=True)

    logger.info("StatsForecast completed in %.1fs", round(time.time() - started_at, 1))
    del sf
    _cleanup_memory()

    meta_columns = {config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS}
    results = {}
    for column in [c for c in predictions.columns if c not in meta_columns]:
        results[column] = (
            predictions[[config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS, column]]
            .rename(columns={column: config.NIXTLA_Y})
            .reset_index(drop=True)
        )
    return results


def _filter_short_series_for_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop series shorter than ``(n_windows + 2) * h`` for CV only.

    Heuristic driven by MSTL's need for at least two full weekly cycles at
    ``season_length=672``; more conservative than a pure lag-based threshold.
    Dropped series remain eligible for the main forecast.
    """
    min_length = (config.CV_NUM_WINDOWS + 2) * config.FORECAST_HORIZON
    lengths = df.groupby(config.NIXTLA_UNIQUE_ID)[config.NIXTLA_DS].count()
    eligible = lengths[lengths >= min_length].index
    dropped = len(lengths) - len(eligible)
    if dropped:
        logger.warning(
            "CV: dropping %d/%d series with fewer than %d intervals " "(h=%d, n_windows=%d, step_size=%d)",
            dropped,
            len(lengths),
            min_length,
            config.FORECAST_HORIZON,
            config.CV_NUM_WINDOWS,
            config.CV_STEP_SIZE,
        )
    return df[df[config.NIXTLA_UNIQUE_ID].isin(eligible)].copy()


def run_statsforecast_cv(training_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-validation predictions: ``[unique_id, ds, cutoff, <model>, ...]``."""
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    models = _build_statsforecast_models()
    if not models:
        return pd.DataFrame()

    core_train = _filter_short_series_for_cv(training_df[[config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS, config.NIXTLA_Y]])
    if core_train.empty:
        return pd.DataFrame()

    sf = StatsForecast(
        models=models,
        freq=config.FREQUENCY,
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=config.SEASON_LENGTH_DAILY),
    )

    logger.info(
        "StatsForecast CV: %d windows, h=%d, %d models, %d locations",
        config.CV_NUM_WINDOWS,
        config.FORECAST_HORIZON,
        len(models),
        core_train[config.NIXTLA_UNIQUE_ID].nunique(),
    )

    with warnings.catch_warnings(), _heartbeat("StatsForecast CV"):
        warnings.filterwarnings("ignore")
        cv_results = sf.cross_validation(
            df=core_train,
            h=config.FORECAST_HORIZON,
            n_windows=config.CV_NUM_WINDOWS,
            step_size=config.CV_STEP_SIZE,
        ).reset_index(drop=True)

    del sf
    _cleanup_memory()
    return cv_results


def run_mlforecast(
    training_df: pd.DataFrame,
    fourier_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Train and forecast all MLForecast (LightGBM) interval models.

    Only exog columns actually present in the training frame are wired in,
    so feature engineering can change without updating every model config.
    """
    from mlforecast import MLForecast

    all_exog_columns = list(config.FUTURE_EXOGENOUS_FEATURES) + (fourier_columns or [])
    available_exog = derive_available_exog(training_df, all_exog_columns, config.MLFORECAST_MODELS)

    results: dict[str, pd.DataFrame] = {}

    for model_name, model_cfg in config.MLFORECAST_MODELS.items():
        core_columns = [
            config.NIXTLA_UNIQUE_ID,
            config.NIXTLA_DS,
            config.NIXTLA_Y,
        ] + available_exog
        core_train = training_df[[c for c in core_columns if c in training_df.columns]].copy()

        model = _build_mlforecast_model(model_name, model_cfg)
        fcst = MLForecast(
            models={model_name: model},
            freq=config.FREQUENCY,
            lags=model_cfg["lags"],
            lag_transforms=_build_lag_transforms(model_name),
            date_features=model_cfg.get("date_features", []),
            num_threads=model_cfg.get("num_threads", -1),
        )

        with _heartbeat(f"{model_name} fit"):
            fcst.fit(core_train, static_features=[])

        x_df = _build_aligned_mlforecast_xdf(fcst, available_exog)
        predictions = fcst.predict(h=config.FORECAST_HORIZON, X_df=x_df)

        results[model_name] = (
            predictions[[config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS, model_name]]
            .rename(columns={model_name: config.NIXTLA_Y})
            .reset_index(drop=True)
        )
        del fcst
        _cleanup_memory()

    return results


def _build_aligned_mlforecast_xdf(
    fitted_fcst: Any,
    exog_columns: list[str],
) -> pd.DataFrame | None:
    """X_df aligned to MLForecast's per-series expected future timestamps.

    Fourier terms are recomputed per timestamp rather than merged from a
    global table, so series with earlier endpoints get correct values
    instead of NaN.
    """
    if not exog_columns:
        return None

    from .preprocessing import add_calendar_features, fourier_terms

    expected_df = fitted_fcst.make_future_dataframe(h=config.FORECAST_HORIZON)
    expected_df = add_calendar_features(expected_df)

    fourier_cols_needed = [c for c in exog_columns if c.startswith("fourier_")]
    if fourier_cols_needed:
        future_dates = pd.DatetimeIndex(expected_df[config.NIXTLA_DS])
        daily_terms = fourier_terms(
            future_dates,
            config.SEASON_LENGTH_DAILY,
            config.FOURIER_DAILY_K,
            "fourier_daily",
        )
        weekly_terms = fourier_terms(
            future_dates,
            config.SEASON_LENGTH_WEEKLY,
            config.FOURIER_WEEKLY_K,
            "fourier_weekly",
        )
        for col, vals in {**daily_terms, **weekly_terms}.items():
            if col in fourier_cols_needed:
                expected_df[col] = vals

    for column in exog_columns:
        if column in expected_df.columns:
            expected_df[column] = expected_df[column].fillna(0)

    available = [config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS] + [
        column for column in exog_columns if column in expected_df.columns
    ]
    return expected_df[available].copy()


def run_mlforecast_cv(
    training_df: pd.DataFrame,
    fourier_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Cross-validation predictions for all MLForecast interval models in long Nixtla shape."""
    from mlforecast import MLForecast

    all_exog_columns = list(config.FUTURE_EXOGENOUS_FEATURES) + (fourier_columns or [])
    available_exog = derive_available_exog(training_df, all_exog_columns, config.MLFORECAST_MODELS)

    all_cv_parts: list[pd.DataFrame] = []

    for model_name, model_cfg in config.MLFORECAST_MODELS.items():
        core_columns = [
            config.NIXTLA_UNIQUE_ID,
            config.NIXTLA_DS,
            config.NIXTLA_Y,
        ] + available_exog
        core_train = _filter_short_series_for_cv(training_df[[c for c in core_columns if c in training_df.columns]])
        if core_train.empty:
            continue

        model = _build_mlforecast_model(model_name, model_cfg)
        fcst = MLForecast(
            models={model_name: model},
            freq=config.FREQUENCY,
            lags=model_cfg["lags"],
            lag_transforms=_build_lag_transforms(model_name),
            date_features=model_cfg.get("date_features", []),
            num_threads=model_cfg.get("num_threads", -1),
        )

        start = time.perf_counter()
        cv_result = fcst.cross_validation(
            df=core_train,
            h=config.FORECAST_HORIZON,
            n_windows=config.CV_NUM_WINDOWS,
            step_size=config.CV_STEP_SIZE,
            static_features=[],
        ).reset_index(drop=True)
        logger.info("%s CV completed in %.1fs", model_name, time.perf_counter() - start)

        all_cv_parts.append(cv_result)
        del fcst
        _cleanup_memory()

    if not all_cv_parts:
        return pd.DataFrame()

    merged_cv_results = all_cv_parts[0]
    for cv_df in all_cv_parts[1:]:
        model_column = [
            c
            for c in cv_df.columns
            if c
            not in (
                config.NIXTLA_UNIQUE_ID,
                config.NIXTLA_DS,
                "cutoff",
                config.NIXTLA_Y,
            )
        ][0]
        merged_cv_results = merged_cv_results.merge(
            cv_df[[config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS, "cutoff", model_column]],
            on=[config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS, "cutoff"],
            how="outer",
        )
    return merged_cv_results
