from __future__ import annotations

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "SeasonalNaive_D": "Seasonal Naive (Daily)",
    "MSTL_D": "MSTL (Daily)",
    "AutoETS_D": "AutoETS (Daily)",
    "AutoARIMA_D": "AutoARIMA (Daily)",
    "DOT_D": "Dynamic Optimised Theta (Daily)",
    "LightGBM_D_Fast": "LightGBM Fast (Daily)",
    "LightGBM_D_Standard": "LightGBM Standard (Daily)",
    "LightGBM_D_Accuracy": "LightGBM Accuracy (Daily)",
    "SeasonalNaive": "Seasonal Naive (Interval)",
    "MSTL": "MSTL (Interval)",
    "AutoETS": "AutoETS (Interval)",
    "LightGBM_Fast": "LightGBM Fast (Interval)",
    "LightGBM_Standard": "LightGBM Standard (Interval)",
    "LightGBM_Accuracy": "LightGBM Accuracy (Interval)",
    "Baseline_Production": "Baseline Production",
    "Baseline_Adjusted_Draft": "Baseline Adj. Draft",
    "Baseline_Adjusted_Published": "Baseline Adj. Published",
    "Baseline_ManagerOnly_Draft": "Baseline Manager Draft",
    "Baseline_ManagerOnly_Published": "Baseline Manager Published",
    "Baseline_Production_Strict": "Baseline Production (Strict)",
    "Baseline_Adjusted_Draft_Strict": "Baseline Adj. Draft (Strict)",
}

MODEL_FAMILIES: dict[str, str] = {
    "SeasonalNaive_D": "StatsForecast",
    "MSTL_D": "StatsForecast",
    "AutoETS_D": "StatsForecast",
    "AutoARIMA_D": "StatsForecast",
    "DOT_D": "StatsForecast",
    "LightGBM_D_Fast": "MLForecast",
    "LightGBM_D_Standard": "MLForecast",
    "LightGBM_D_Accuracy": "MLForecast",
    "SeasonalNaive": "StatsForecast",
    "MSTL": "StatsForecast",
    "AutoETS": "StatsForecast",
    "LightGBM_Fast": "MLForecast",
    "LightGBM_Standard": "MLForecast",
    "LightGBM_Accuracy": "MLForecast",
    "Baseline_Production": "Baseline",
    "Baseline_Adjusted_Draft": "Baseline",
    "Baseline_Adjusted_Published": "Baseline",
    "Baseline_ManagerOnly_Draft": "Baseline",
    "Baseline_ManagerOnly_Published": "Baseline",
    "Baseline_Production_Strict": "Baseline",
    "Baseline_Adjusted_Draft_Strict": "Baseline",
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "SeasonalNaive_D": "Repeats the value from the same weekday in the previous week.",
    "MSTL_D": "Multi-Seasonal Trend decomposition with weekly seasonality. Fitted with ETS.",
    "AutoETS_D": "Automatic exponential smoothing with additive/multiplicative selection.",
    "AutoARIMA_D": "Automatic ARIMA with weekly seasonality; AIC-selected order.",
    "DOT_D": "Dynamic Optimised Theta; ensemble of Theta models with optimal damping.",
    "LightGBM_D_Fast": "LightGBM with shallow trees; lowest latency, reduced accuracy.",
    "LightGBM_D_Standard": "LightGBM with balanced tree depth and learning rate.",
    "LightGBM_D_Accuracy": "LightGBM with deeper trees and slow learning rate for maximum accuracy.",
    "SeasonalNaive": "Repeats the corresponding 15-min slot from the previous week.",
    "MSTL": "Multi-Seasonal Trend decomposition with daily (96-slot) and weekly (672-slot) seasons.",
    "AutoETS": "Automatic ETS with 96-slot daily seasonality.",
    "LightGBM_Fast": "LightGBM with Fourier features; low-latency interval model.",
    "LightGBM_Standard": "LightGBM with Fourier features; balanced depth interval model.",
    "LightGBM_Accuracy": "LightGBM with Fourier features; high-accuracy interval model.",
    "Baseline_Production": "Baseline algorithmic projection (zero-projection rows excluded).",
    "Baseline_Adjusted_Draft": "Baseline projection plus manager draft adjustment.",
    "Baseline_Adjusted_Published": "Baseline projection plus published manager adjustment.",
    "Baseline_ManagerOnly_Draft": "Rows where the manager manually authored a draft projection.",
    "Baseline_ManagerOnly_Published": "Rows where the manager manually authored a published projection.",
    "Baseline_Production_Strict": "The same Baseline forecast as Baseline Production, but graded on more days, including days where Baseline produced a $0 projection. On an operating day, a $0 forecast is almost always wrong. The standard Baseline Production skips those days when calculating accuracy; this version includes them. The lower accuracy score reflects a broader grading scope, not a different or weaker forecast.",
    "Baseline_Adjusted_Draft_Strict": "The same forecast as Baseline Adj. Draft, graded on all days including $0-projection days. See Baseline Production (Strict); the same caveat applies. Higher error reflects more difficult days being counted, not a change in how the forecast is built.",
}

METRIC_DEFINITIONS: dict[str, str] = {
    "Mean Absolute Error (MAE)": (
        "The average dollar gap between the forecast and actual sales. "
        "If MAE = $150, the forecast was off by about $150 per location per day on average, "
        "sometimes over and sometimes under. Lower is better."
    ),
    "Mean Absolute Percentage Error (MAPE)": (
        "The average percentage gap between the forecast and actual sales, measured day by day. "
        "A MAPE of 12% means forecasts were off by roughly 12% on a typical day. "
        "Skipped on days with $0 in actual sales to avoid distorting the average. "
        "Can be inflated on very low-volume days, so use alongside MAE for the full picture. "
        "Lower is better."
    ),
    "Weighted Absolute Percentage Error (WAPE)": (
        "Total forecast error as a share of total actual sales across the evaluation period. "
        "A WAPE of 10% means that, in aggregate, the forecast missed by 10 cents for every dollar of real sales. "
        "High-revenue days naturally carry more weight, so a single bad day at a busy location matters more "
        "than several small misses at quiet ones. More reliable than MAPE when sales volumes vary widely. "
        "Lower is better."
    ),
    "Root Mean Squared Error (RMSE)": (
        "Similar to MAE, but large individual misses are penalized more heavily. "
        "A forecast that is off by $400 once will push RMSE higher than four separate $100 misses, "
        "which makes it a useful signal for catching occasional large errors that MAE might understate. "
        "Expressed in dollars. Lower is better."
    ),
}

PALETTE: dict[str, str] = {
    "StatsForecast": "#1565c0",
    "MLForecast": "#2e7d32",
    "Baseline": "#e65100",
    "Unknown": "#95a5a6",
}

# StatsForecast → blues/teals, MLForecast → greens, Baseline → warm tones.
# Daily and interval variants share a shade so the family stays visually
# coherent across reports.
MODEL_COLORS: dict[str, str] = {
    "SeasonalNaive_D": "#1565c0",
    "MSTL_D": "#0288d1",
    "AutoETS_D": "#0097a7",
    "AutoARIMA_D": "#5c6bc0",
    "DOT_D": "#42a5f5",
    "SeasonalNaive": "#1565c0",
    "MSTL": "#0288d1",
    "AutoETS": "#0097a7",
    "LightGBM_D_Fast": "#1b5e20",
    "LightGBM_D_Standard": "#2e7d32",
    "LightGBM_D_Accuracy": "#558b2f",
    "LightGBM_Fast": "#1b5e20",
    "LightGBM_Standard": "#2e7d32",
    "LightGBM_Accuracy": "#558b2f",
    "Baseline_Production": "#bf360c",
    "Baseline_Production_Strict": "#d84315",
    "Baseline_Adjusted_Draft": "#e65100",
    "Baseline_Adjusted_Draft_Strict": "#fb8c00",
    "Baseline_Adjusted_Published": "#f57c00",
    "Baseline_ManagerOnly_Draft": "#ff8f00",
    "Baseline_ManagerOnly_Published": "#ffa000",
}

# Line styles are a secondary visual cue for grayscale prints and
# colour-vision-difference readers.
MODEL_LINESTYLES: dict[str, str] = {
    "SeasonalNaive_D": "-",
    "MSTL_D": "--",
    "AutoETS_D": "-.",
    "AutoARIMA_D": ":",
    "DOT_D": "-",
    "LightGBM_D_Fast": "-",
    "LightGBM_D_Standard": "--",
    "LightGBM_D_Accuracy": "-.",
    "SeasonalNaive": "-",
    "MSTL": "--",
    "AutoETS": "-.",
    "LightGBM_Fast": "-",
    "LightGBM_Standard": "--",
    "LightGBM_Accuracy": "-.",
}

# ≥5% improvement → green; 0–5% → amber; <0 (regression) → unshaded.
VERDICT_GREEN = 0.05
VERDICT_AMBER = 0.0

CV_METRIC_NOTE = (
    "Cross-validation WAPE and MAE are computed as a single global aggregate across "
    "all locations and all CV windows combined. This is structurally different from "
    "the hold-out metrics above, which are computed per-location first and then "
    "averaged. CV WAPE tends to be lower than hold-out WAPE for the same model "
    "because high-revenue locations dominate the global aggregate. Do not compare "
    "CV and hold-out numbers directly."
)
