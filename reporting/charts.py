from __future__ import annotations

import base64
import io
import platform as _platform

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from .constants import (MODEL_COLORS, MODEL_DISPLAY_NAMES, MODEL_FAMILIES,
                            MODEL_LINESTYLES, PALETTE)
except ImportError:
    MODEL_COLORS = {}
    MODEL_DISPLAY_NAMES = {}
    MODEL_FAMILIES = {}
    MODEL_LINESTYLES = {}
    PALETTE = {
        "StatsForecast": "#1565c0",
        "MLForecast": "#2e7d32",
        "Baseline": "#e65100",
        "Unknown": "#95a5a6",
    }


# Qualitative palette for line charts, chosen for perceptual distance.
# Avoids orange (reserved for Baseline elsewhere) and near-black (Actual Sales).
_LINE_CHART_PALETTE = [
    "#8e24aa",  # vivid purple
    "#00838f",  # dark cyan
    "#f9a825",  # amber / gold
    "#388e3c",  # medium green
    "#1565c0",  # deep blue
    "#6d4c41",  # warm brown
    "#00695c",  # dark teal
    "#5e35b1",  # indigo
]

# Unpadded strftime directives: %-X on Linux/macOS, %#X on Windows.
_DAY_FMT = "%-d" if _platform.system() != "Windows" else "%#d"
_MONTH_FMT = "%-m" if _platform.system() != "Windows" else "%#m"
_HOUR_FMT = "%-I" if _platform.system() != "Windows" else "%#I"


def _fig_to_base64(fig) -> str:
    """Render a figure to a base64 PNG string for inline HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _model_label(name: str) -> str:
    """Short axis label — strips the '(Daily)'/'(Interval)' suffix."""
    label = MODEL_DISPLAY_NAMES.get(name, name)
    return label.replace(" (Daily)", "").replace(" (Interval)", "")


def _family_color(name: str) -> str:
    """Palette colour for the model's family."""
    family = MODEL_FAMILIES.get(name, "Unknown")
    return PALETTE.get(family, PALETTE.get("Unknown", "#95a5a6"))


def _model_color(name: str) -> str:
    """Per-model colour, falling back to the family colour."""
    return MODEL_COLORS.get(name) or _family_color(name)


def _model_linestyle(name: str) -> str:
    """Per-model line style (secondary visual cue beyond colour)."""
    return MODEL_LINESTYLES.get(name, "-")


def _family_legend_handles():
    """Patch handles for the family colour legend used in bar charts."""
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=PALETTE.get("Baseline", "#e65100"), label="Baseline (current system)"),
        Patch(
            facecolor=PALETTE.get("StatsForecast", "#1565c0"),
            label="Statistical models",
        ),
        Patch(
            facecolor=PALETTE.get("MLForecast", "#2e7d32"),
            label="Machine learning models",
        ),
    ]


def win_loss_bar_chart(win_loss_df: pd.DataFrame, baseline_model: str) -> str:
    """Horizontal bar of per-model win rate vs. Baseline; bars coloured by family."""
    if not HAS_MATPLOTLIB or win_loss_df.empty:
        return ""

    df = win_loss_df.sort_values("win_rate", ascending=True).copy()
    df["label"] = df["model"].apply(_model_label)
    colors = [_model_color(m) for m in df["model"]]

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(df) * 0.65)))
    bars = ax.barh(
        df["label"],
        df["win_rate"] * 100,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        height=0.6,
    )

    ax.axvline(50, color="#555", linestyle="--", linewidth=1.0, alpha=0.6)
    _baseline_display = _model_label(baseline_model)
    ax.text(
        51,
        ax.get_ylim()[1],
        f"{_baseline_display} (break-even)",
        fontsize=7.5,
        color="#555",
        va="top",
    )

    ax.set_xlabel("Locations where model is more accurate than Baseline (%)", fontsize=10)
    ax.set_title(
        "How Often Each Model Beats Baseline\n(% of evaluated locations)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_xlim(0, 108)

    for bar, (_, row) in zip(bars, df.iterrows()):
        w = bar.get_width()
        ax.text(
            min(w + 1.5, 106),
            bar.get_y() + bar.get_height() / 2,
            f"{w:.0f}%  ({int(row['wins'])}/{int(row['n_locations'])} locations)",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333",
        )

    ax.legend(
        handles=_family_legend_handles(),
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        title="Color key",
        title_fontsize=8,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return _fig_to_base64(fig)


def aggregate_metric_bar_chart(
    aggregate_metrics_dict: dict[str, pd.DataFrame],
    metric: str = "MAE",
    baseline_model: str | None = None,
) -> str:
    """Horizontal bar of mean metric per model, with a dashed line at the baseline value."""
    if not HAS_MATPLOTLIB or not aggregate_metrics_dict:
        return ""

    rows = []
    for model_name, agg_df in aggregate_metrics_dict.items():
        row = agg_df[agg_df["metric"] == metric]
        if not row.empty:
            rows.append({"model": model_name, "mean": float(row["mean"].iloc[0])})
    if not rows:
        return ""

    df = pd.DataFrame(rows).sort_values("mean")
    df["label"] = df["model"].apply(_model_label)
    colors = [_model_color(m) for m in df["model"]]

    baseline_val = None
    if baseline_model:
        brow = df[df["model"] == baseline_model]
        if not brow.empty:
            baseline_val = float(brow["mean"].iloc[0])

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(df) * 0.65)))
    bars = ax.barh(
        df["label"],
        df["mean"],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        height=0.6,
    )

    legend_handles = _family_legend_handles()
    if baseline_val is not None:
        baseline_label = (
            f"Baseline ({metric}: ${baseline_val:,.2f})"
            if metric in ("MAE", "RMSE")
            else f"Baseline ({metric}: {baseline_val:.3f})"
        )
        ax.axvline(
            baseline_val,
            color=PALETTE.get("Baseline", "#e65100"),
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
        )
        from matplotlib.lines import Line2D

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=PALETTE.get("Baseline", "#e65100"),
                linestyle="--",
                linewidth=1.4,
                label=baseline_label,
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        title="Color key",
        title_fontsize=8,
    )

    dollar = metric in ("MAE", "RMSE")
    ax.set_xlabel(
        f"Average forecast error, {metric} "
        f"{'($, lower is more accurate)' if dollar else '(lower is more accurate)'}",
        fontsize=10,
    )
    ax.set_title(
        f"Forecast Accuracy by Model: {metric}", fontsize=11, fontweight="bold", pad=10
    )

    for bar, (_, row) in zip(bars, df.iterrows()):
        w = bar.get_width()
        lbl = f"${w:,.2f}" if dollar else f"{w:.3f}"
        ax.text(
            w * 1.005,
            bar.get_y() + bar.get_height() / 2,
            lbl,
            va="center",
            ha="left",
            fontsize=8,
            color="#333",
        )

    ax.set_xlim(0, df["mean"].max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return _fig_to_base64(fig)


def location_forecast_line_chart(
    actuals: pd.DataFrame,
    predictions_dict: dict[str, pd.DataFrame],
    unique_id: str,
    unique_id_col: str = "unique_id",
    ds_col: str = "ds",
    y_col: str = "y",
    baseline_model: str | None = None,
    max_models: int = 5,
) -> str:
    """Daily line chart of actuals vs. forecasts for one location."""
    if not HAS_MATPLOTLIB:
        return ""

    actual_loc = actuals[actuals[unique_id_col] == unique_id].sort_values(ds_col)
    if actual_loc.empty:
        return ""

    uid_parts = unique_id.split("_", 1)
    friendly_id = (
        f"Location ...{uid_parts[1][:12]}" if len(uid_parts) > 1 else unique_id[:20]
    )

    d0 = pd.Timestamp(actual_loc[ds_col].min()).strftime(f"%b {_DAY_FMT}")
    d1 = pd.Timestamp(actual_loc[ds_col].max()).strftime(f"%b {_DAY_FMT}, %Y")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        actual_loc[ds_col],
        actual_loc[y_col],
        label="Actual Sales",
        color="#2c3e50",
        linewidth=2.0,
        zorder=5,
    )

    color_cycle = list(_LINE_CHART_PALETTE)
    for i, model_name in enumerate(list(predictions_dict.keys())[:max_models]):
        pred_df = predictions_dict[model_name]
        pred_loc = pred_df[pred_df[unique_id_col] == unique_id].sort_values(ds_col)
        if pred_loc.empty:
            continue
        ax.plot(
            pred_loc[ds_col],
            pred_loc[y_col],
            label=_model_label(model_name),
            color=color_cycle[i % len(color_cycle)],
            linestyle=_model_linestyle(model_name),
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_title(
        f"{friendly_id}: Forecast vs Actual ({d0} to {d1})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Date / Time", fontsize=9)
    ax.set_ylabel("Net Sales ($)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def location_forecast_line_chart_interval(
    actuals: pd.DataFrame,
    predictions_dict: dict[str, pd.DataFrame],
    unique_id: str,
    unique_id_col: str = "unique_id",
    ds_col: str = "ds",
    y_col: str = "y",
    baseline_model: str | None = None,
    max_models: int = 5,
    date_filter: "pd.Timestamp | None" = None,
    location_label: str | None = None,
) -> str:
    """15-minute interval line chart for one location.

    Passing ``date_filter`` plots a single day (96 intervals, hourly ticks);
    otherwise plots the full evaluation week with day-boundary ticks.
    """
    if not HAS_MATPLOTLIB:
        return ""

    actual_loc = actuals[actuals[unique_id_col] == unique_id].sort_values(ds_col).copy()
    if actual_loc.empty:
        return ""

    actual_loc[ds_col] = pd.to_datetime(actual_loc[ds_col])

    if date_filter is not None:
        target_date = pd.Timestamp(date_filter).date()
        actual_loc = actual_loc[actual_loc[ds_col].dt.date == target_date]
        if actual_loc.empty:
            return ""

    uid_parts = unique_id.split("_", 1)
    friendly_id = (
        f"Location ...{uid_parts[1][:12]}" if len(uid_parts) > 1 else unique_id[:20]
    )
    title_loc = (
        f"{location_label}: {friendly_id}" if location_label else friendly_id
    )

    if date_filter is not None:
        title_date = pd.Timestamp(date_filter).strftime(f"%A, %b {_DAY_FMT}, %Y")
        chart_title = f"{title_loc}\n15-Min Forecast vs Actual ({title_date})"
    else:
        d0 = actual_loc[ds_col].min().strftime(f"%b {_DAY_FMT}")
        d1 = actual_loc[ds_col].max().strftime(f"%b {_DAY_FMT}, %Y")
        chart_title = f"{title_loc}: 15-Min Forecast vs Actual ({d0} to {d1})"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        actual_loc[ds_col],
        actual_loc[y_col],
        label="Actual Sales",
        color="#2c3e50",
        linewidth=2.0,
        zorder=5,
    )

    color_cycle = list(_LINE_CHART_PALETTE)
    for i, model_name in enumerate(list(predictions_dict.keys())[:max_models]):
        pred_df = predictions_dict[model_name]
        pred_loc = (
            pred_df[pred_df[unique_id_col] == unique_id].sort_values(ds_col).copy()
        )
        if pred_loc.empty:
            continue
        pred_loc[ds_col] = pd.to_datetime(pred_loc[ds_col])
        if date_filter is not None:
            pred_loc = pred_loc[pred_loc[ds_col].dt.date == target_date]
        if pred_loc.empty:
            continue
        ax.plot(
            pred_loc[ds_col],
            pred_loc[y_col],
            label=_model_label(model_name),
            color=color_cycle[i % len(color_cycle)],
            linestyle=_model_linestyle(model_name),
            linewidth=1.5,
            alpha=0.9,
        )

    if date_filter is not None:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(f"{_HOUR_FMT} %p"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
        ax.tick_params(axis="x", which="minor", length=2, color="#ccc")
        ax.set_xlabel("Time of Day", fontsize=9)
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(f"{_MONTH_FMT}/{_DAY_FMT}"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
        ax.tick_params(axis="x", which="minor", length=3, color="#bbb")
        for day_ts in pd.date_range(
            actual_loc[ds_col].min().normalize(), actual_loc[ds_col].max(), freq="D"
        ):
            ax.axvline(day_ts, color="#ddd", linewidth=0.7, zorder=0)
        ax.set_xlabel("Date", fontsize=9)

    ax.tick_params(axis="x", which="major", labelsize=8, rotation=0)
    ax.set_title(chart_title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Sales ($)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def win_loss_donut_chart(win_loss_df: pd.DataFrame, model_name: str) -> str:
    """Donut chart of one model's win/loss share vs. the baseline."""
    if not HAS_MATPLOTLIB or win_loss_df.empty:
        return ""

    row = win_loss_df[win_loss_df["model"] == model_name]
    if row.empty:
        return ""

    row = row.iloc[0]
    wins = int(row["wins"])
    n_locations = int(row["n_locations"])
    not_wins = n_locations - wins

    sizes = [wins, not_wins]
    labels = [
        f"Beats Baseline\n({wins} locations)",
        f"Does Not Beat Baseline\n({not_wins} locations)",
    ]
    colors = ["#27ae60", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.72,
        wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 2.5},
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    ax.legend(
        wedges,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
        fontsize=9,
        framealpha=0.85,
    )
    ax.set_title(
        f"{_model_label(model_name)}\nWin / Loss vs Baseline ({n_locations} locations)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    return _fig_to_base64(fig)


def cv_metric_chart(cv_metrics_df: pd.DataFrame, metric: str = "MAE") -> str:
    """Horizontal bar of cross-validation metric values per model."""
    if not HAS_MATPLOTLIB or cv_metrics_df.empty or metric not in cv_metrics_df.columns:
        return ""

    df = cv_metrics_df[["model", metric]].sort_values(metric).copy()
    df["label"] = df["model"].apply(_model_label)
    colors = [_model_color(m) for m in df["model"]]

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(df) * 0.65)))
    bars = ax.barh(
        df["label"],
        df[metric],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        height=0.6,
    )

    dollar = metric in ("MAE", "RMSE")
    for bar, (_, row) in zip(bars, df.iterrows()):
        w = bar.get_width()
        lbl = f"${w:,.2f}" if dollar else f"{w:.3f}"
        ax.text(
            w * 1.005,
            bar.get_y() + bar.get_height() / 2,
            lbl,
            va="center",
            ha="left",
            fontsize=8,
            color="#333",
        )

    ax.set_xlabel(
        f"Cross-validation {metric} "
        f"{'($, lower is more consistent)' if dollar else '(lower is more consistent)'}",
        fontsize=10,
    )
    ax.set_title(
        f"Model Stability Across Time: {metric}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.legend(
        handles=_family_legend_handles(),
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        title="Color key",
        title_fontsize=8,
    )
    ax.set_xlim(0, df[metric].max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return _fig_to_base64(fig)
