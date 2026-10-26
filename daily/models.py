from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import pandas as pd

from core.models import _cleanup_memory, _heartbeat, derive_available_exog
from core.preprocessing import add_calendar_features
from . import config

logger = logging.getLogger(__name__)


def _build_statsforecast_models() -> list:
    """Instantiate configured StatsForecast models with deferred imports."""
    from statsforecast.models import (
        MSTL,
        AutoARIMA,
        AutoETS,
        DynamicOptimizedTheta,
        SeasonalNaive,
    )

    def _make_mstl(cfg: dict) -> MSTL:
        kwargs: dict = {"season_length": cfg["season_length"]}
        if cfg.get("trend_forecaster") == "AutoARIMA":
            kwargs["trend_forecaster"] = AutoARIMA()
        return MSTL(**kwargs)

    model_factories = {
        "SeasonalNaive_D": lambda cfg: SeasonalNaive(
            season_length=cfg["season_length"]
        ),
        "MSTL_D": _make_mstl,
        "AutoETS_D": lambda cfg: AutoETS(
            season_length=cfg["season_length"],
            model=cfg.get("model", "ZZZ"),
        ),
        "AutoARIMA_D": lambda cfg: AutoARIMA(season_length=cfg["season_length"]),
        "DOT_D": lambda cfg: DynamicOptimizedTheta(season_length=cfg["season_length"]),
    }

    models = []
    for model_name, model_cfg in config.STATSFORECAST_MODELS.items():
        if model_name in model_factories:
            models.append(model_factories[model_name](model_cfg))
    return models


def _build_lag_transforms(model_name: str) -> dict:
    """Lag transforms encoding a long-run level plus same-weekday seasonal signal."""
    from mlforecast.lag_transforms import (
        ExpandingMean,
        RollingMean,
        SeasonalRollingMean,
    )

    lag_transforms = {
        1: [ExpandingMean()],
        config.SEASON_LENGTH_WEEKLY: [
            SeasonalRollingMean(
                season_length=config.SEASON_LENGTH_WEEKLY, window_size=8
            )
        ],
    }
    if model_name == "LightGBM_D_Accuracy":
        lag_transforms[28] = [RollingMean(window_size=4)]
    return lag_transforms


def _build_mlforecast_model(model_name: str, model_cfg: dict):
    """LightGBM regressor built from MLForecast config values."""
    from lightgbm import LGBMRegressor

    lgbm_kwargs = {
        key: value
        for key, value in model_cfg.items()
        if key not in ("lags", "date_features", "num_threads")
    }
    return LGBMRegressor(**lgbm_kwargs, verbose=-1, random_state=config.RANDOM_SEED)


def run_statsforecast(
    training_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Train and forecast all StatsForecast models; returns ``{model: nixtla_df}``."""
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    models = _build_statsforecast_models()
    if not models:
        return {}

    core_train = training_df[
        [config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS, config.NIXTLA_Y]
    ].copy()

    sf = StatsForecast(
        models=models,
        freq=config.FREQUENCY,
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=config.SEASON_LENGTH_WEEKLY),
    )

    logger.info(
        "StatsForecast: training %d models on %d locations",
        len(models),
        core_train[config.NIXTLA_UNIQUE_ID].nunique(),
    )
    started_at = time.time()

    with warnings.catch_warnings(), _heartbeat("StatsForecast forecast"):
        warnings.filterwarnings("ignore")
        predictions = sf.forecast(
            df=core_train,
            h=config.FORECAST_HORIZON,
        ).reset_index(drop=True)

    elapsed = round(time.time() - started_at, 1)
    logger.info("StatsForecast completed in %.1fs", elapsed)
    del sf
    _cleanup_memory()

    meta_columns = {config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS}
    model_columns = [
        column for column in predictions.columns if column not in meta_columns
    ]

    results = {}
    for column in model_columns:
        results[column] = (
            predictions[[config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS, column]]
            .rename(columns={column: config.NIXTLA_Y})
            .reset_index(drop=True)
        )
    return results


def _filter_short_series_for_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop series too short for the configured CV settings.

    Threshold is ``max_lag + (n_windows - 1) * step_size + h``, derived from
    configured lags so it tracks any change to ``MLFORECAST_MODELS``.
    """
    max_lag = max(max(cfg["lags"]) for cfg in config.MLFORECAST_MODELS.values())
    min_length = (
        max_lag
        + (config.CV_NUM_WINDOWS - 1) * config.CV_STEP_SIZE
        + config.FORECAST_HORIZON
    )
    lengths = df.groupby(config.NIXTLA_UNIQUE_ID)[config.NIXTLA_DS].count()
    eligible = lengths[lengths >= min_length].index
    dropped = len(lengths) - len(eligible)
    if dropped:
        logger.warning(
            "CV: dropping %d/%d series with fewer than %d days "
            "(max_lag=%d, n_windows=%d, step_size=%d, h=%d)",
            dropped,
            len(lengths),
            min_length,
            max_lag,
            config.CV_NUM_WINDOWS,
            config.CV_STEP_SIZE,
            config.FORECAST_HORIZON,
        )
    return df[df[config.NIXTLA_UNIQUE_ID].isin(eligible)].copy()


def run_statsforecast_cv(
    training_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-validation predictions: ``[unique_id, ds, cutoff, <model>, ...]``."""
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    models = _build_statsforecast_models()
    if not models:
        return pd.DataFrame()

    core_train = _filter_short_series_for_cv(
        training_df[[config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS, config.NIXTLA_Y]]
    )
    if core_train.empty:
        return pd.DataFrame()

    sf = StatsForecast(
        models=models,
        freq=config.FREQUENCY,
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=config.SEASON_LENGTH_WEEKLY),
    )

    logger.info(
        "StatsForecast CV: %d windows, h=%d, %d models, %d locations",
        config.CV_NUM_WINDOWS,
        config.FORECAST_HORIZON,
        len(models),
        core_train[config.NIXTLA_UNIQUE_ID].nunique(),
    )
    start = time.time()

    with warnings.catch_warnings(), _heartbeat("StatsForecast CV"):
        warnings.filterwarnings("ignore")
        cv_results = sf.cross_validation(
            df=core_train,
            h=config.FORECAST_HORIZON,
            n_windows=config.CV_NUM_WINDOWS,
            step_size=config.CV_STEP_SIZE,
        ).reset_index(drop=True)

    logger.info("StatsForecast CV completed in %.1fs", round(time.time() - start, 1))
    del sf
    _cleanup_memory()
    return cv_results


def run_mlforecast(
    training_df: pd.DataFrame,
    future_exog_df: pd.DataFrame | None = None,
    fourier_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Train and forecast all MLForecast (LightGBM) models.

    Only exog columns actually present in the training frame are wired in, so
    feature engineering can change without updating every model config in lockstep.
    """
    from mlforecast import MLForecast

    all_exog_columns = list(config.FUTURE_EXOGENOUS_FEATURES) + (fourier_columns or [])
    available_exog = derive_available_exog(
        training_df, all_exog_columns, config.MLFORECAST_MODELS
    )

    results: dict[str, pd.DataFrame] = {}

    for model_name, model_cfg in config.MLFORECAST_MODELS.items():
        logger.info("MLForecast: training %s", model_name)
        started_at = time.time()

        core_columns = [
            config.NIXTLA_UNIQUE_ID,
            config.NIXTLA_DS,
            config.NIXTLA_Y,
        ] + available_exog
        core_train = training_df[
            [c for c in core_columns if c in training_df.columns]
        ].copy()

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

        x_df = _build_aligned_mlforecast_xdf(fcst, future_exog_df, available_exog)
        predictions = fcst.predict(h=config.FORECAST_HORIZON, X_df=x_df)

        elapsed = round(time.time() - started_at, 1)
        logger.info("%s completed in %.1fs", model_name, elapsed)

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
    future_exog_df: pd.DataFrame | None,
    exog_columns: list[str],
) -> pd.DataFrame | None:
    """X_df aligned to MLForecast's per-series expected future timestamps.

    MLForecast generates each series' future grid from its last observed date.
    When endpoints differ across series, a uniform future grid would mismatch;
    this rebuilds calendar and Fourier features on the exact expected timestamps.
    """
    if not exog_columns:
        return None

    expected_df = fitted_fcst.make_future_dataframe(h=config.FORECAST_HORIZON)
    expected_df = add_calendar_features(expected_df)

    if future_exog_df is not None:
        missing_columns = [
            column
            for column in exog_columns
            if column not in expected_df.columns and column in future_exog_df.columns
        ]
        if missing_columns:
            # All series share one future date grid (enforced by fill_missing_dates),
            # so merging on ds is safe. If endpoints ever diverge, switch to a
            # per-timestamp build like the interval pipeline.
            deduplicated_future_exog = future_exog_df[
                [config.NIXTLA_DS] + missing_columns
            ].drop_duplicates(subset=[config.NIXTLA_DS])
            expected_df = expected_df.merge(
                deduplicated_future_exog,
                on=config.NIXTLA_DS,
                how="left",
            )

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
    """Cross-validation predictions for all MLForecast models in long Nixtla shape."""
    from mlforecast import MLForecast

    all_exog_columns = list(config.FUTURE_EXOGENOUS_FEATURES) + (fourier_columns or [])
    available_exog = derive_available_exog(
        training_df, all_exog_columns, config.MLFORECAST_MODELS
    )

    all_cv_parts: list[pd.DataFrame] = []

    for model_name, model_cfg in config.MLFORECAST_MODELS.items():
        logger.info("MLForecast CV: %s", model_name)
        started_at = time.time()

        core_columns = [
            config.NIXTLA_UNIQUE_ID,
            config.NIXTLA_DS,
            config.NIXTLA_Y,
        ] + available_exog
        core_train = _filter_short_series_for_cv(
            training_df[[c for c in core_columns if c in training_df.columns]]
        )
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

        cv_result = fcst.cross_validation(
            df=core_train,
            h=config.FORECAST_HORIZON,
            n_windows=config.CV_NUM_WINDOWS,
            step_size=config.CV_STEP_SIZE,
            static_features=[],
        ).reset_index(drop=True)

        all_cv_parts.append(cv_result)
        logger.info(
            "%s CV completed in %.1fs", model_name, round(time.time() - started_at, 1)
        )
        del fcst
        _cleanup_memory()

    if not all_cv_parts:
        return pd.DataFrame()

    merged_cv_results = all_cv_parts[0]
    for cv_df in all_cv_parts[1:]:
        model_column = [
            column
            for column in cv_df.columns
            if column
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
