from __future__ import annotations

import html
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from . import charts
from .constants import (
    CV_METRIC_NOTE,
    METRIC_DEFINITIONS,
    MODEL_DESCRIPTIONS,
    MODEL_DISPLAY_NAMES,
    MODEL_FAMILIES,
    VERDICT_AMBER,
    VERDICT_GREEN,
)

_CSS_PATH = Path(__file__).parent / "report.css"

logger = logging.getLogger(__name__)

_CSS_FALLBACK = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 24px; color: #2c3e50; }
  h1 { font-size: 1.8rem; border-bottom: 2px solid #3498db; padding-bottom: 8px; }
  h2 { font-size: 1.3rem; margin-top: 32px; color: #2980b9; }
  h3 { font-size: 1.1rem; color: #555; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9rem; }
  th { background: #3498db; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 6px 12px; border-bottom: 1px solid #ecf0f1; }
  tr:nth-child(even) { background: #f8f9fa; }
  tr.baseline { background: #eaf4fb; font-weight: bold; }
  tr.best { background: #eafaf1; }
  .metric-good { color: #27ae60; font-weight: bold; }
  .metric-bad  { color: #e74c3c; }
  .summary-box { background: #f0f7ff; border-left: 4px solid #3498db;
                 padding: 16px 20px; margin: 16px 0; border-radius: 4px; }
  .chart-img { max-width: 100%; margin: 12px 0; border-radius: 4px;
               box-shadow: 0 1px 4px rgba(0,0,0,0.12); }
  .section { margin-bottom: 40px; }
  .tag-better { color: #27ae60; font-weight: bold; }
  .tag-worse  { color: #e74c3c; }
  .tag-neutral { color: #7f8c8d; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 16px; margin: 20px 0; }
  .kpi-card { background: #f0f7ff; border: 1px solid #bcd6f0; border-radius: 8px;
              padding: 16px 20px; text-align: center; }
  .kpi-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
               letter-spacing: 0.06em; color: #5d6d7e; margin-bottom: 6px; }
  .kpi-value { font-size: 1.6rem; font-weight: 700; color: #1a252f; }
  .kpi-subtitle { font-size: 0.78rem; color: #7f8c8d; margin-top: 4px; }
  .text-muted { color: #7f8c8d; }
</style>
"""


@dataclass
class ReportContext:
    """Metadata for a benchmark report.

    ``baseline_model=None`` suppresses every baseline-comparison section,
    used for metrics without a production baseline (e.g. gross sales).
    """

    interval: str
    n_locations: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    baseline_model: Optional[str]
    train_history_weeks: Optional[int] = None
    cv_num_windows: Optional[int] = None
    n_train_rows: Optional[int] = None
    location_metadata: Optional[pd.DataFrame] = None
    metric_label: str = "Net Sales"
    extra_notes: list[str] = field(default_factory=list)


def load_css() -> str:
    """Read the sibling ``report.css`` or fall back to the embedded stylesheet."""
    if _CSS_PATH.exists():
        return f"<style>\n{_CSS_PATH.read_text(encoding='utf-8')}\n</style>"
    return _CSS_FALLBACK


def kpi_card(label: str, value: str, subtitle: str = "") -> str:
    """Render one KPI card with ARIA landmark attributes."""
    subtitle_html = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ""
    return (
        f'<div class="kpi-card" role="region" aria-label="{label}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{subtitle_html}"
        f"</div>"
    )


def _df_to_html(
    df: pd.DataFrame,
    highlight_column: str | None = None,
    baseline_value: float | None = None,
    float_format: str = ".4f",
    highlight_eligibility_check=None,
    column_label_mapping: dict[str, str] | None = None,
    per_column_format: dict[str, str] | None = None,
) -> str:
    """Render a DataFrame as an HTML table with optional per-cell highlighting.

    Cells in ``highlight_column`` are coloured green when ≥1% below
    ``baseline_value`` and red when ≥1% above it. ``per_column_format``
    overrides ``float_format`` for specific columns (e.g. dollar formats).
    """
    rows_html = []
    columns = list(df.columns)
    header_labels = column_label_mapping or {}
    header = "".join(f"<th>{header_labels.get(col, col)}</th>" for col in columns)

    for _, row in df.iterrows():
        cells = []
        row_class = ""
        for col in columns:
            value = row[col]
            cell_class = ""
            eligible = highlight_eligibility_check is None or highlight_eligibility_check(row)
            if col == highlight_column and baseline_value is not None and isinstance(value, float) and eligible:
                if value < baseline_value * 0.99:
                    cell_class = "metric-good"
                elif value > baseline_value * 1.01:
                    cell_class = "metric-bad"
            if isinstance(value, float):
                col_fmt = (per_column_format or {}).get(col, float_format)
                formatted = f"{value:{col_fmt}}"
            elif isinstance(value, bool):
                formatted = "Yes" if value else "No"
            else:
                formatted = html.escape(str(value))
            cells.append(f'<td class="{cell_class}">{formatted}</td>')
        rows_html.append(f'<tr class="{row_class}">{"".join(cells)}</tr>')

    return f"""<table>
  <thead><tr>{header}</tr></thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>"""


def _display_name(model_name: str) -> str:
    """Human-readable model label, with the model id as fallback."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def _get_agg_metric(
    aggregate_metrics_dict: dict[str, pd.DataFrame],
    model_name: str,
    metric: str = "MAE",
) -> float | None:
    """Mean value of ``metric`` for ``model_name``, or None."""
    if model_name not in aggregate_metrics_dict:
        return None
    df = aggregate_metrics_dict[model_name]
    row = df[df["metric"] == metric]
    return float(row["mean"].iloc[0]) if not row.empty else None


def _get_significance(
    statistical_tests_df: pd.DataFrame,
    model_name: str,
) -> tuple[bool, str] | None:
    """``(is_significant, direction)`` from the test table, or None."""
    if statistical_tests_df.empty:
        return None
    row = statistical_tests_df[statistical_tests_df["model"] == model_name]
    if row.empty:
        return None
    return bool(row["significant"].iloc[0]), str(row["direction"].iloc[0])


def _best_nixtla_model_by_mae(
    aggregate_metrics_dict: dict[str, pd.DataFrame],
) -> str:
    """Non-baseline model with the lowest mean MAE — fallback when there is no
    baseline to compute win/loss against. Returns an empty string if none qualify.
    """
    best_name = ""
    best_mae: float | None = None
    for model, df in aggregate_metrics_dict.items():
        if model.startswith("Baseline_"):
            continue
        row = df[df["metric"] == "MAE"]
        if row.empty:
            continue
        mae = float(row["mean"].iloc[0])
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_name = model
    return best_name


def _verdict_class(improvement_pct: float) -> str:
    """CSS class for the verdict callout."""
    if improvement_pct >= VERDICT_GREEN * 100:
        return "verdict-green"
    if improvement_pct >= VERDICT_AMBER * 100:
        return "verdict-amber"
    return "verdict-red"


def _verdict_label(improvement_pct: float) -> str:
    """Short recommendation string keyed off improvement %."""
    if improvement_pct >= VERDICT_GREEN * 100:
        return "Recommended"
    if improvement_pct >= VERDICT_AMBER * 100:
        return "Needs More Evidence"
    return "Not Recommended"


def _img_tag(base64_encoded_png: str, extra_style: str = "") -> str:
    """Inline ``<img>`` tag from a base64 PNG; empty string when the PNG is empty."""
    if not base64_encoded_png:
        return ""
    style_attr = f' style="{extra_style}"' if extra_style else ""
    return f'<img class="chart-img" src="data:image/png;base64,{base64_encoded_png}"{style_attr} />'


def _pick_interval_sample_locations(
    all_location_metrics: dict[str, pd.DataFrame],
    actuals: pd.DataFrame,
    best_model: str,
    baseline_model: str,
    unique_id_col: str = "unique_id",
    ds_col: str = "ds",
    y_col: str = "y",
) -> list[tuple[str, str, pd.Timestamp]]:
    """Best / typical / most-challenging locations plus a representative day for each.

    The representative day is the one whose total volume is closest to the
    location's median across the test period.
    """
    if best_model not in all_location_metrics or baseline_model not in all_location_metrics:
        return []

    model_df = all_location_metrics[best_model][[unique_id_col, "MAE"]].copy()
    baseline_df = all_location_metrics[baseline_model][[unique_id_col, "MAE"]].rename(columns={"MAE": "baseline_MAE"})
    merged = model_df.merge(baseline_df, on=unique_id_col, how="inner")
    merged = merged[merged["baseline_MAE"] > 0].copy()
    if merged.empty:
        return []

    merged["improvement"] = (merged["baseline_MAE"] - merged["MAE"]) / merged["baseline_MAE"]
    merged = merged.sort_values("improvement").reset_index(drop=True)

    candidates = [
        (merged.iloc[-1][unique_id_col], "Best-Performing Location"),
        (merged.iloc[len(merged) // 2][unique_id_col], "Typical Location"),
        (merged.iloc[0][unique_id_col], "Most Challenging Location"),
    ]

    actuals_copy = actuals.copy()
    actuals_copy[ds_col] = pd.to_datetime(actuals_copy[ds_col])

    result = []
    for uid, label in candidates:
        loc_actuals = actuals_copy[actuals_copy[unique_id_col] == uid]
        if loc_actuals.empty:
            continue
        daily_totals = loc_actuals.groupby(loc_actuals[ds_col].dt.date)[y_col].sum()
        if daily_totals.empty:
            continue
        median_vol = daily_totals.median()
        rep_date = daily_totals.index[(daily_totals - median_vol).abs().argmin()]
        result.append((uid, label, pd.Timestamp(rep_date)))

    return result


def _pick_daily_sample_locations(
    all_location_metrics: dict[str, pd.DataFrame],
    best_model: str,
    baseline_model: str,
    unique_id_col: str = "unique_id",
) -> list[tuple[str, str]]:
    """Top, median, and bottom locations ranked by ``best_model`` vs baseline MAE improvement."""
    if best_model not in all_location_metrics or baseline_model not in all_location_metrics:
        return []

    model_df = all_location_metrics[best_model][[unique_id_col, "MAE"]].copy()
    baseline_df = all_location_metrics[baseline_model][[unique_id_col, "MAE"]].rename(columns={"MAE": "baseline_MAE"})
    merged = model_df.merge(baseline_df, on=unique_id_col, how="inner")
    merged = merged[merged["baseline_MAE"] > 0].copy()
    if merged.empty:
        return []

    merged["improvement"] = (merged["baseline_MAE"] - merged["MAE"]) / merged["baseline_MAE"]
    merged = merged.sort_values("improvement").reset_index(drop=True)

    return [
        (merged.iloc[-1][unique_id_col], "Best-Performing Location"),
        (merged.iloc[len(merged) // 2][unique_id_col], "Typical Location"),
        (merged.iloc[0][unique_id_col], "Most Challenging Location"),
    ]


def _section_reporting_summary(
    best_model: str,
    best_win_rate: float,
    improvement_pct: float | None,
    significant: bool,
    context: ReportContext,
) -> str:
    interval_label = "daily" if context.interval == "daily" else "15-minute interval"
    history_note = (
        f". Training history: <strong>{context.train_history_weeks} weeks</strong>"
        if context.train_history_weeks
        else ""
    )
    eval_context = f"""<p class="text-muted">
  Evaluated on <strong>{context.n_locations:,} locations</strong>.
  {interval_label.capitalize()} forecasts.
  Test period: <strong>{context.test_start.date()} to {context.test_end.date()}</strong>
  {history_note}
</p>"""

    if context.baseline_model is None:
        return f"""<div class="summary-box">
  <strong>Nixtla Model Evaluation: No Production Baseline</strong><br>
  No production baseline forecast is available for <strong>{context.metric_label}</strong>.
  The best-performing Nixtla model by MAE is <strong>{_display_name(best_model)}</strong>.
  Win/loss comparisons and statistical significance tests are not applicable.
</div>
{eval_context}"""

    verdict_cls = _verdict_class(improvement_pct or 0.0)
    label = _verdict_label(improvement_pct or 0.0)
    improvement_str = f" ({improvement_pct:+.1f}% lower average forecast error)" if improvement_pct is not None else ""
    sig_note = (
        " This result is <strong>statistically reliable</strong>; the improvement is " "unlikely to be due to chance."
        if significant
        else " This result has <strong>not yet been confirmed as statistically reliable</strong>. "
        "More data across additional weeks may be needed to draw a firm conclusion."
    )
    return f"""<div class="{verdict_cls}">
  <strong>Overall Verdict: {label}</strong><br>
  The best-performing model (<strong>{_display_name(best_model)}</strong>) produces more accurate
  forecasts than Baseline Production at
  <strong>{best_win_rate:.0%} of evaluated locations</strong>{improvement_str}.{sig_note}
</div>
{eval_context}"""


def _section_kpi_cards(
    best_model: str,
    best_win_rate: float,
    best_mae: float | None,
    baseline_mae: float | None,
    context: ReportContext,
) -> str:
    if best_mae is not None and baseline_mae is not None and baseline_mae > 0:
        improvement_pct = (baseline_mae - best_mae) / baseline_mae * 100
        improvement_card = kpi_card(
            "MAE Improvement",
            f"{improvement_pct:+.1f}%",
            f"${baseline_mae:,.2f} to ${best_mae:,.2f}",
        )
    else:
        improvement_card = kpi_card("MAE Improvement", "N/A", "insufficient data")

    cards = [
        kpi_card("Best Win Rate", f"{best_win_rate:.0%}", _display_name(best_model)),
        improvement_card,
        kpi_card(
            "Locations Evaluated",
            f"{context.n_locations:,}",
            f"{context.test_start.date()} to {context.test_end.date()}",
        ),
        kpi_card(
            "Forecast Granularity",
            "Daily" if context.interval == "daily" else "15-min",
            "1-week evaluation horizon",
        ),
    ]
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def _section_business_impact(
    best_mae: float | None,
    baseline_mae: float | None,
    n_locations: int,
    context: ReportContext,
) -> str:
    if best_mae is None or baseline_mae is None or baseline_mae <= 0:
        return ""
    improvement_per_unit = baseline_mae - best_mae
    if improvement_per_unit <= 0:
        return ""

    operating_days = 325
    if context.interval == "daily":
        annual_impact = improvement_per_unit * n_locations * operating_days
        formula_html = (
            f"Based on a ${improvement_per_unit:,.2f} reduction in average daily forecast error "
            f"across {n_locations:,} locations, over {operating_days} operating days per year."
        )
    else:
        intervals_per_day = 48  # ~12 revenue-generating hours per operating day
        annual_impact = improvement_per_unit * intervals_per_day * n_locations * operating_days
        formula_html = (
            f"Based on a ${improvement_per_unit:,.2f} reduction in average forecast error per "
            f"15-minute interval, across {n_locations:,} locations, over {operating_days} "
            f"operating days per year."
        )

    return f"""<div class="business-impact-box">
  <div class="kpi-label">Estimated Annual Forecasting Improvement</div>
  <div class="business-impact-headline">${annual_impact:,.0f}</div>
  <div class="business-impact-formula">{formula_html}</div>
  <div class="business-impact-disclaimer">
    Illustrative estimate based on applying the observed forecast error reduction uniformly
    across all evaluated locations and a typical operating year. Actual business impact will
    vary by location and depends on how improved forecast accuracy is translated into
    labour scheduling, inventory, and operational decisions at each site.
  </div>
</div>"""


def _section_failure_mode(
    win_loss_df: pd.DataFrame,
    best_model: str,
    donut_base64_chart: str,
) -> str:
    """Donut + per-outcome table showing where the best model wins vs. baseline."""
    if win_loss_df.empty:
        return ""

    row = win_loss_df[win_loss_df["model"] == best_model]
    if row.empty:
        return ""

    row = row.iloc[0]
    wins = int(row["wins"])
    n = int(row["n_locations"])
    not_wins = n - wins

    donut_html = _img_tag(donut_base64_chart, "max-width:360px")

    return f"""<div style="display:flex;gap:32px;align-items:flex-start;flex-wrap:wrap">
  <div>{donut_html}</div>
  <div style="flex:1;min-width:220px">
    <p>
      <strong>{_display_name(best_model)}</strong> forecasts more accurately than
      Baseline Production
      at <strong>{wins} of {n} locations</strong>. Accuracy is measured by average forecast
      error (MAE): how far off, in dollars, the forecast is from actual sales on a typical day.
    </p>
    <table style="width:auto">
      <thead>
        <tr><th>Outcome</th><th>Locations</th><th>Share</th></tr>
      </thead>
      <tbody>
        <tr>
          <td class="tag-better">More accurate than Baseline</td>
          <td>{wins}</td><td>{wins / n:.0%}</td>
        </tr>
        <tr>
          <td class="tag-worse">Baseline is more accurate</td>
          <td>{not_wins}</td><td>{not_wins / n:.0%}</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>"""


def _section_model_highlights(
    nixtla_win_loss_sorted: pd.DataFrame,
    aggregate_metrics_dict: dict[str, pd.DataFrame],
    statistical_tests_df: pd.DataFrame,
    context: ReportContext,
) -> str:
    """Highlight cards for the top three non-baseline models by win rate."""
    if nixtla_win_loss_sorted.empty:
        return "<p>No model results available.</p>"

    family_colors = {
        "StatsForecast": "#3498db",
        "MLForecast": "#27ae60",
        "Baseline": "#e67e22",
        "Unknown": "#95a5a6",
    }
    baseline_mae = _get_agg_metric(aggregate_metrics_dict, context.baseline_model, "MAE")

    cards_html = []
    for _, win_loss_row in nixtla_win_loss_sorted.head(3).iterrows():
        model = win_loss_row["model"]
        display = _display_name(model)
        win_rate = float(win_loss_row["win_rate"])
        mae = _get_agg_metric(aggregate_metrics_dict, model, "MAE")
        wape = _get_agg_metric(aggregate_metrics_dict, model, "WAPE")
        sig_result = _get_significance(statistical_tests_df, model)
        family = MODEL_FAMILIES.get(model, "Unknown")
        card_color = family_colors.get(family, "#95a5a6")

        items = [f"<li>More accurate than Baseline at <strong>{win_rate:.0%} of locations</strong></li>"]

        if mae is not None and baseline_mae is not None and baseline_mae > 0:
            pct = (baseline_mae - mae) / baseline_mae * 100
            cls = "metric-good" if pct > 0 else "metric-bad"
            items.append(
                f'<li>Avg. forecast error: <span class="{cls}">${mae:,.2f} per '
                f"{'day' if context.interval == 'daily' else '15-min interval'} "
                f"({pct:+.1f}% vs Baseline)</span></li>"
            )

        if wape is not None:
            items.append(f"<li>Error as % of sales (WAPE): {wape:.1%}</li>")

        if sig_result is not None and sig_result[0]:
            items.append("<li>Improvement is statistically reliable</li>")

        desc = MODEL_DESCRIPTIONS.get(model, "")
        if desc:
            items.append(f'<li style="color:#7f8c8d;font-size:0.82rem;margin-top:6px">{desc}</li>')

        cards_html.append(
            f'<div class="highlight-card" style="--card-color:{card_color}">'
            f"<h4>{display}</h4>"
            f"<ul>{''.join(items)}</ul>"
            f"</div>"
        )

    return f'<div class="highlights-grid">{"".join(cards_html)}</div>'


def _section_summary_table(
    win_loss_df: pd.DataFrame,
    aggregate_metrics_dict: dict[str, pd.DataFrame],
    statistical_tests_df: pd.DataFrame,
    context: ReportContext,
) -> str:
    """One row per model: MAE, vs baseline, win rate, significance, verdict."""
    if win_loss_df.empty:
        return "<p>No model results available.</p>"

    baseline_mae = _get_agg_metric(aggregate_metrics_dict, context.baseline_model, "MAE")

    header = (
        "<tr>"
        "<th>Model</th>"
        "<th>Avg. Forecast Error</th>"
        "<th>vs Baseline Production</th>"
        "<th>Locations Where More Accurate</th>"
        "<th>Result Is Reliable?</th>"
        "<th>Recommendation</th>"
        "</tr>"
    )

    rows_html = []
    for _, win_loss_row in win_loss_df.iterrows():
        model = win_loss_row["model"]
        display = _display_name(model)
        mae = _get_agg_metric(aggregate_metrics_dict, model, "MAE")
        win_rate = float(win_loss_row["win_rate"])
        sig_result = _get_significance(statistical_tests_df, model)
        is_baseline = model == context.baseline_model

        mae_str = f"${mae:,.2f}" if mae is not None else "n/a"

        if is_baseline:
            vs_baseline_str = "<em>baseline</em>"
            vs_baseline_cls = ""
            verdict_str = "n/a"
        elif mae is not None and baseline_mae is not None and baseline_mae > 0:
            pct = (baseline_mae - mae) / baseline_mae * 100
            vs_baseline_cls = "metric-good" if pct > 0 else "metric-bad"
            vs_baseline_str = f"{pct:+.1f}%"
            verdict_str = _verdict_label(pct)
        else:
            vs_baseline_str = "n/a"
            vs_baseline_cls = ""
            verdict_str = "n/a"

        win_cls = "metric-good" if win_rate > 0.5 else ("metric-bad" if win_rate < 0.5 else "")

        if is_baseline:
            sig_str = "n/a"
        elif sig_result is not None:
            sig_str = (
                '<span class="tag-better">Yes</span>' if sig_result[0] else '<span class="tag-neutral">Not yet</span>'
            )
        else:
            sig_str = "n/a"

        row_class = "baseline" if is_baseline else ""
        rows_html.append(
            f'<tr class="{row_class}">'
            f"<td>{display}</td>"
            f"<td>{mae_str}</td>"
            f'<td class="{vs_baseline_cls}">{vs_baseline_str}</td>'
            f'<td class="{win_cls}">{win_rate:.0%}</td>'
            f"<td>{sig_str}</td>"
            f"<td>{verdict_str}</td>"
            f"</tr>"
        )

    return f"""<table>
  <thead>{header}</thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>"""


def _section_appendix_cv(
    cv_metrics_df: pd.DataFrame,
    cv_base64_chart: str,
    context: ReportContext,
) -> str:
    """CV metrics section — confirms held-out results aren't specific to the test week."""
    if cv_metrics_df.empty:
        return ""
    cv_table = _df_to_html(cv_metrics_df, per_column_format={"MAE": ".2f", "RMSE": ".2f"})
    cv_windows_label = context.cv_num_windows or "?"
    return f"""<h3>Consistency Check: Cross-Validation ({cv_windows_label} windows)</h3>
<p class="text-muted">
  To confirm results aren't a fluke of the test week, each model was also evaluated on
  {context.cv_num_windows or "multiple"} earlier held-out periods within the training data.
  Lower average forecast error (MAE) across these windows indicates the model performs
  reliably over time, not just on the specific week shown above.
</p>
<p class="text-muted"><strong>Note:</strong> {CV_METRIC_NOTE}</p>
{_img_tag(cv_base64_chart)}
{cv_table}"""


def _section_appendix_sig(
    statistical_tests_df: pd.DataFrame,
    context: ReportContext,
) -> str:
    if statistical_tests_df.empty:
        return "<p>Insufficient data for significance testing " "(requires at least 10 common locations).</p>"
    return f"""<h3>Statistical Reliability (vs {_display_name(context.baseline_model)})</h3>
<p class="text-muted">
  A Wilcoxon signed-rank test was run to determine whether each model's accuracy
  improvement over Baseline could plausibly be due to random variation in the data.
  A "Yes" result (p&nbsp;&lt;&nbsp;0.05) means the improvement is statistically reliable;
  we can be reasonably confident it reflects a genuine difference, not noise.
  Requires at least 10 locations with results for both the model and Baseline.
</p>
{_df_to_html(statistical_tests_df)}"""


def _section_appendix_model_descriptions(models_present: list[str]) -> str:
    rows = []
    for model in models_present:
        desc = MODEL_DESCRIPTIONS.get(model)
        if desc:
            rows.append(f"<tr><td><strong>{_display_name(model)}</strong></td><td>{desc}</td></tr>")
    if not rows:
        return ""
    return f"""<table>
  <thead><tr><th>Model</th><th>Description</th></tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>"""


def _section_appendix_glossary() -> str:
    rows = "".join(f"<tr><td><strong>{m}</strong></td><td>{d}</td></tr>" for m, d in METRIC_DEFINITIONS.items())
    return f"""<h3>Metric Glossary</h3>
<table>
  <thead><tr><th>Metric</th><th>Definition</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _section_next_steps(
    improvement_pct: float | None,
    is_significant: bool,
    best_model: str,
) -> str:
    """Plain-language next-step bullets, branched on improvement magnitude and significance."""
    display = _display_name(best_model)

    if improvement_pct is not None and improvement_pct >= VERDICT_GREEN * 100 and is_significant:
        items = [
            f"<li><strong>Consider a limited live pilot</strong> using <strong>{display}</strong> "
            f"at a representative sample of locations. The accuracy improvement is meaningful and "
            f"has been confirmed as statistically reliable.</li>",
            "<li><strong>Define success criteria before launching.</strong> For example, "
            "average forecast error must remain below a target threshold at a set share of pilot "
            "locations after four to eight weeks of live use.</li>",
            "<li><strong>Schedule a follow-up evaluation</strong> after the pilot to confirm the "
            "accuracy gain holds in production before committing to a broader rollout.</li>",
        ]
    elif improvement_pct is not None and improvement_pct > 0:
        items = [
            "<li><strong>Extend the evaluation period.</strong> Run the benchmark across "
            "additional weeks to build confidence that the accuracy improvement is consistent "
            "over time, not specific to this test window.</li>",
            "<li><strong>Explore location-level patterns.</strong> Identify whether certain "
            "location types or sales volumes show consistently stronger improvements, which could "
            "support a more targeted pilot.</li>",
            "<li><strong>Hold off on broad deployment</strong> until the accuracy advantage is "
            "confirmed as statistically reliable.</li>",
        ]
    else:
        items = [
            "<li><strong>Continue using Baseline Production</strong> as the primary forecast. "
            "No alternative model currently demonstrates a meaningful and consistent "
            "accuracy advantage.</li>",
            "<li><strong>Re-evaluate in a future period.</strong> Results may improve as "
            "additional data becomes available or models are retrained with updated history.</li>",
            "<li><strong>No deployment action is recommended</strong> at this time.</li>",
        ]

    return f"<ul>{''.join(items)}</ul>"


def build_report(
    context: ReportContext,
    all_location_metrics: dict[str, pd.DataFrame],
    aggregate_metrics_dict: dict[str, pd.DataFrame],
    win_loss_df: pd.DataFrame,
    statistical_tests_df: pd.DataFrame,
    cv_metrics_df: pd.DataFrame,
    actuals: pd.DataFrame,
    predictions_dict: dict[str, pd.DataFrame],
    sample_location_ids: list[str] | None = None,
    output_path: Path | None = None,
) -> str:
    """Assemble the self-contained HTML benchmark report.

    Executive summary first, technical detail later. Charts are inlined as
    base64 PNGs so the file has no external dependencies.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    interval_label = "Daily" if context.interval == "daily" else "15-Minute Interval"

    # Empty string is a safe sentinel: every downstream guard checks
    # `model not in all_location_metrics`, so baseline-dependent sections
    # collapse to empty output.
    effective_baseline_model: str = context.baseline_model or ""

    sorted_win_loss = win_loss_df.sort_values("win_rate", ascending=False) if not win_loss_df.empty else win_loss_df
    nixtla_win_loss_sorted = (
        sorted_win_loss[~sorted_win_loss["model"].str.startswith("Baseline_")]
        if not sorted_win_loss.empty
        else sorted_win_loss
    )

    best_model = (
        nixtla_win_loss_sorted.iloc[0]["model"]
        if not nixtla_win_loss_sorted.empty
        else _best_nixtla_model_by_mae(aggregate_metrics_dict)
    )
    best_win_rate = float(nixtla_win_loss_sorted.iloc[0]["win_rate"]) if not nixtla_win_loss_sorted.empty else 0.0
    best_mae = _get_agg_metric(aggregate_metrics_dict, best_model, "MAE")
    baseline_mae = _get_agg_metric(aggregate_metrics_dict, effective_baseline_model, "MAE")

    improvement_pct: float | None = None
    if best_mae is not None and baseline_mae is not None and baseline_mae > 0:
        improvement_pct = (baseline_mae - best_mae) / baseline_mae * 100

    sig_result = _get_significance(statistical_tests_df, best_model)
    is_significant = bool(sig_result[0]) if sig_result is not None else False

    win_loss_chart_base64 = charts.win_loss_bar_chart(sorted_win_loss, effective_baseline_model)
    mae_chart_base64 = charts.aggregate_metric_bar_chart(
        aggregate_metrics_dict,
        metric="MAE",
        baseline_model=effective_baseline_model,
    )
    cv_chart_base64 = charts.cv_metric_chart(cv_metrics_df, metric="MAE")
    donut_chart_base64 = charts.win_loss_donut_chart(win_loss_df, best_model)

    nixtla_predictions = {name: df for name, df in predictions_dict.items() if not name.startswith("Baseline_")}
    location_charts_html = ""
    if context.interval == "15min":
        interval_samples = _pick_interval_sample_locations(
            all_location_metrics,
            actuals,
            best_model,
            effective_baseline_model,
        )
        for uid, loc_label, rep_day in interval_samples:
            base64_chart = charts.location_forecast_line_chart_interval(
                actuals,
                nixtla_predictions,
                uid,
                baseline_model=effective_baseline_model,
                date_filter=rep_day,
                location_label=loc_label,
            )
            if base64_chart:
                location_charts_html += f'<h4 style="margin:20px 0 4px;color:#2c3e50">{loc_label}</h4>' + _img_tag(
                    base64_chart
                )
    else:
        daily_samples = _pick_daily_sample_locations(
            all_location_metrics,
            best_model,
            effective_baseline_model,
        )
        if not daily_samples and sample_location_ids:
            daily_samples = [(uid, "") for uid in sample_location_ids[:3]]
        for uid, loc_label in daily_samples:
            base64_chart = charts.location_forecast_line_chart(
                actuals,
                nixtla_predictions,
                uid,
                baseline_model=effective_baseline_model,
            )
            if base64_chart:
                if loc_label:
                    location_charts_html += f'<h4 style="margin:20px 0 4px;color:#2c3e50">{loc_label}</h4>' + _img_tag(
                        base64_chart
                    )
                else:
                    location_charts_html += _img_tag(base64_chart)

    n_models_tested = (
        len(nixtla_win_loss_sorted)
        if not nixtla_win_loss_sorted.empty
        else sum(1 for k in aggregate_metrics_dict if not k.startswith("Baseline_"))
    )
    all_beat = int((nixtla_win_loss_sorted["win_rate"] > 0.5).sum()) if not nixtla_win_loss_sorted.empty else 0
    granularity_label = "day" if context.interval == "daily" else "15-minute interval"
    if improvement_pct is not None and improvement_pct > 0:
        sig_item = (
            "This improvement has been confirmed as statistically reliable; "
            "the accuracy gain is unlikely to be due to random variation in the data."
            if is_significant
            else "This improvement has not yet been confirmed as statistically reliable. "
            "Evaluating additional weeks of data would strengthen confidence in the finding."
        )
        mae_item = (
            f"The best-performing model, <strong>{_display_name(best_model)}</strong>, "
            f"reduces average forecast error by <strong>{improvement_pct:.1f}%</strong>, "
            f"from ${baseline_mae:,.2f} to ${best_mae:,.2f} per {granularity_label}, per location."
            if best_mae is not None and baseline_mae is not None
            else ""
        )
        findings_html = (
            f"<ul>"
            f"<li><strong>{all_beat} of {n_models_tested}</strong> alternative models outperform "
            f"Baseline Production at a majority of evaluated locations.</li>"
            + (f"<li>{mae_item}</li>" if mae_item else "")
            + f"<li>{sig_item}</li>"
            f"</ul>"
        )
    elif context.baseline_model is None:
        mae_note = (
            f"The best-performing model, <strong>{_display_name(best_model)}</strong>, "
            f"achieves a mean MAE of <strong>${best_mae:,.2f}</strong> "
            f"per {granularity_label}, per location."
            if best_mae is not None
            else ""
        )
        findings_html = (
            f"<ul>"
            f"<li><strong>{n_models_tested}</strong> Nixtla models were evaluated on "
            f"<strong>{context.metric_label}</strong> forecasting accuracy.</li>"
            + (f"<li>{mae_note}</li>" if mae_note else "")
            + "<li>No production baseline forecast is available for this metric; "
            "win/loss comparisons and statistical tests are not applicable.</li>"
            "</ul>"
        )
    else:
        findings_html = (
            f"<ul>"
            f"<li>Of {n_models_tested} models evaluated, <strong>{all_beat}</strong> produce lower "
            f"forecast errors than Baseline Production at a majority of locations.</li>"
            f"<li>No model demonstrates a consistent, meaningful accuracy advantage over "
            f"Baseline Production at this time.</li>"
            "<li>Evaluating additional weeks of data is recommended before drawing "
            "firm conclusions.</li>"
            f"</ul>"
        )

    business_impact_html = _section_business_impact(best_mae, baseline_mae, context.n_locations, context)

    all_models_present = list(win_loss_df["model"].unique()) if not win_loss_df.empty else []
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Forecast Benchmark: {interval_label}: {context.metric_label}: {timestamp}</title>
  {load_css()}
</head>
<body>

<h1>Forecast Benchmark Report: {interval_label}: {context.metric_label}</h1>
<p class="text-muted">Generated {timestamp}</p>
<p>
  This report evaluates whether Nixtla's forecasting models can produce more accurate
  {interval_label.lower()} sales forecasts than the current Baseline system.
  It covers <strong>{context.n_locations:,} locations</strong> over the period
  <strong>{context.test_start.date()} to {context.test_end.date()}</strong>.
  Accuracy is measured in dollars: specifically, how far off the forecast is from
  actual sales on an average {granularity_label}. The goal is to determine whether an
  alternative model is ready to replace or supplement Baseline.
</p>

<!-- 1. REPORTING SUMMARY -->
<div class="section">
  {
        _section_reporting_summary(
            best_model, best_win_rate, improvement_pct, is_significant, context
        )
    }
</div>

<!-- 2. KPI CARDS -->
<div class="section">
  {_section_kpi_cards(best_model, best_win_rate, best_mae, baseline_mae, context)}
</div>

<!-- 3. BUSINESS IMPACT -->
{
        "<div class='section'>" + business_impact_html + "</div>"
        if business_impact_html
        else ""
    }

<!-- 4. KEY FINDINGS -->
<div class="section">
  <h2>Key Findings</h2>
  {findings_html}
</div>

<!-- 5. FAILURE MODE ANALYSIS -->
<div class="section">
  <h2>Where Is the New Approach More Accurate?</h2>
  {_section_failure_mode(win_loss_df, best_model, donut_chart_base64)}
</div>

<!-- 6. MODEL HIGHLIGHTS -->
<div class="section">
  <h2>Top Performing Models</h2>
  {
        _section_model_highlights(
            nixtla_win_loss_sorted,
            aggregate_metrics_dict,
            statistical_tests_df,
            context,
        )
    }
</div>

<!-- 7. HOW EACH MODEL WORKS -->
<div class="section">
  <h2>How Each Model Works</h2>
  {_section_appendix_model_descriptions(all_models_present)}
</div>

<!-- 8. SUMMARY TABLE -->
<div class="section">
  <h2>All Models: Accuracy Summary</h2>
  <p class="text-muted">
    Ranked by share of locations where each model outperforms Baseline Production.
    <span class="metric-good">Green</span> = more accurate than Baseline &nbsp;
    <span class="metric-bad">Red</span> = less accurate than Baseline.
    Forecast error (MAE) is in dollars per {granularity_label} per location.
  </p>
  <div class="summary-box">
    <strong>A note on the "Strict" rows:</strong> Baseline Production (Strict) and
    Baseline Adj. Draft (Strict) use the exact same forecasts as their standard counterparts;
    nothing about the forecast has changed. The difference is what days get graded. The standard
    versions skip days where Baseline produced a $0 projection. The Strict versions include those days.
    Because a $0 forecast on an operating day is almost always a large miss, including those days
    raises the error score. The accuracy gap is not a sign that one forecast is better than the
    other; it shows what happens when the grading covers more difficult cases.
  </div>
  {
        _section_summary_table(
            sorted_win_loss, aggregate_metrics_dict, statistical_tests_df, context
        )
    }
</div>

<!-- 9. RECOMMENDED NEXT STEPS -->
<div class="section">
  <h2>Recommended Next Steps</h2>
  {_section_next_steps(improvement_pct, is_significant, best_model)}
</div>

<!-- 10. EVIDENCE -->
<div class="section">
  <h2>Supporting Evidence</h2>
  <h3>How Often Each Model Outperforms Baseline</h3>
  <p class="text-muted">Share of evaluated locations where each model produces a lower forecast
  error than Baseline Production. The dashed line at 50 percent is the break-even point; above
  it means the model is more accurate at a majority of locations.</p>
  {_img_tag(win_loss_chart_base64)}

  <h3>Average Forecast Error by Model</h3>
  <p class="text-muted">Lower is more accurate. Each bar shows the average dollar error per
  {
        granularity_label
    } across all evaluated locations. The orange dashed line marks the Baseline
  Production baseline.</p>
  {_img_tag(mae_chart_base64)}

  {
        (
            "<h3>Sample Location Forecasts: Single Day View</h3>"
            "<p class='text-muted'>Forecast vs actual 15-minute sales for a best-performing, "
            "typical, and most-challenging location. Each chart shows one representative day "
            "(the day closest to that location's median daily sales volume during "
            "the test week).</p>"
            + location_charts_html
        )
        if location_charts_html and context.interval == "15min"
        else (
            "<h3>Sample Location Forecasts: Best, Typical &amp; Most Challenging</h3>"
            "<p class='text-muted'>Forecast vs actual daily sales for a best-performing, typical, "
            "and most-challenging location during the evaluation period.</p>"
            + location_charts_html
        )
        if location_charts_html
        else ""
    }
</div>

<!-- 11. TECHNICAL APPENDIX -->
<div class="section">
  <h2>Technical Appendix</h2>
    {_section_appendix_glossary()}
  {_section_appendix_cv(cv_metrics_df, cv_chart_base64, context)}
  {_section_appendix_sig(statistical_tests_df, context)}

  {
        "<h3>Assumptions &amp; Data Notes</h3><ul>"
        + "".join(f"<li>{n}</li>" for n in context.extra_notes)
        + "</ul>"
        if context.extra_notes
        else ""
    }
</div>

</body>
</html>"""

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_out, encoding="utf-8")
        logger.info("Report written to %s", output_path)

    return html_out
