"""Results section of the research paper.

Builds the event study table, regime comparison table, backtest
performance table and embedded figures.  Everything is pulled dynamically
from ``data/processed/`` CSVs and ``outputs/figures/`` PNGs so the paper
auto-updates whenever the analysis pipeline reruns.

Inputs:
    ``event_study_stats.csv``, ``reversion_summary.csv``,
    ``regime_stats.csv``, ``regime_anova.csv``, ``backtest_metrics.csv``
    and the figure PNGs.

Outputs:
    :func:`build` returns ``list[Flowable]`` consumed by
    ``paper/generate_paper.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable, Image, KeepTogether, Paragraph, Spacer, Table, TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

logger = config.get_logger(__name__)

_TABLE_STYLE = TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
    ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
])


def _missing(styles: dict, name: str, producer: str) -> list[Flowable]:
    """Return a placeholder paragraph for a missing input.

    Args:
        styles: Shared style dict.
        name: Human-readable name of the missing artifact.
        producer: Command that produces it.

    Returns:
        Single-paragraph flowable list.
    """
    logger.warning("Results section: %s missing (run `%s`)", name, producer)
    return [Paragraph(f"[{name} unavailable — run `{producer}`]", styles["body"])]


def _figure(path: Path, caption: str, styles: dict,
            width: float = 6.0) -> list[Flowable]:
    """Embed a figure PNG with a numbered caption, keeping them together.

    Args:
        path: Figure file path.
        caption: Caption text.
        styles: Shared style dict.
        width: Display width in inches (height scales proportionally).

    Returns:
        Flowable list, or a placeholder when the file is missing.
    """
    if not path.exists():
        return _missing(styles, f"Figure ({path.name})", "the analysis pipeline")
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        aspect = im.height / im.width
    image = Image(str(path), width=width * inch, height=width * aspect * inch)
    return [KeepTogether([image, Paragraph(caption, styles["caption"]),
                          Spacer(1, 10)])]


def _event_study_table(styles: dict) -> list[Flowable]:
    """Build the pooled event study results table.

    Args:
        styles: Shared style dict.

    Returns:
        Flowable list with caption and table.
    """
    path = config.DATA_PROCESSED_DIR / "event_study_stats.csv"
    if not path.exists():
        return _missing(styles, "Event study table", "python analysis/reversion.py")
    stats = pd.read_csv(path)
    pooled = stats[stats["ticker"] == "ALL"].copy()
    rows = [["Thresh.", "Dir.", "Horizon", "N", "Mean ret (%)",
             "95% CI (%)", "t-stat", "p-value"]]
    for _, r in pooled.sort_values(["threshold", "direction", "horizon"]).iterrows():
        rows.append([
            f"{r['threshold']:.1f}σ", r["direction"], f"{int(r['horizon'])}d",
            f"{int(r['n_events'])}", f"{r['mean_fwd_ret_pct']:+.2f}",
            f"[{r['ci_low_pct']:+.2f}, {r['ci_high_pct']:+.2f}]",
            f"{r['t_stat']:.2f}", f"{r['p_value']:.3f}",
        ])
    table = Table(rows, hAlign="CENTER", repeatRows=1)
    table.setStyle(_TABLE_STYLE)
    return [
        Paragraph(
            "Table 2. Pooled forward returns after dislocation events. "
            "Means are across all five ETFs; confidence intervals are from "
            f"{config.BOOTSTRAP_SAMPLES:,} bootstrap resamples; t-statistics "
            "test the mean against zero.", styles["caption"]),
        table,
        Spacer(1, 10),
    ]


def _reversion_summary_table(styles: dict) -> list[Flowable]:
    """Build the reversion diagnostics table.

    Args:
        styles: Shared style dict.

    Returns:
        Flowable list with caption and table.
    """
    path = config.DATA_PROCESSED_DIR / "reversion_summary.csv"
    if not path.exists():
        return _missing(styles, "Reversion summary table",
                        "python analysis/reversion.py")
    summary = pd.read_csv(path)
    rows = [["Thresh.", "Dir.", "Events", "Median rev. (days)",
             "Reverted ≤10d (%)", "Avg peak disloc. (%)"]]
    for _, r in summary.iterrows():
        rows.append([
            f"{r['threshold']:.1f}σ", r["direction"], f"{int(r['n_events'])}",
            f"{r['median_reversion_days']:.0f}",
            f"{r['pct_reverted_within_10d']:.0f}",
            f"{r['avg_peak_dislocation_pct']:.3f}",
        ])
    table = Table(rows, hAlign="CENTER", repeatRows=1)
    table.setStyle(_TABLE_STYLE)
    return [
        Paragraph("Table 3. Reversion diagnostics by threshold and "
                  "direction.", styles["caption"]),
        table,
        Spacer(1, 10),
    ]


def _regime_table(styles: dict) -> list[Flowable]:
    """Build the regime comparison table with ANOVA results.

    Args:
        styles: Shared style dict.

    Returns:
        Flowable list with caption, table and ANOVA note.
    """
    stats_path = config.DATA_PROCESSED_DIR / "regime_stats.csv"
    anova_path = config.DATA_PROCESSED_DIR / "regime_anova.csv"
    if not stats_path.exists():
        return _missing(styles, "Regime table", "python analysis/regime.py")
    regime_stats = pd.read_csv(stats_path)
    rows = [["Regime", "Events", "Avg rev. (days)", "Avg magn. (%)",
             "Hit rate (%)", "Avg 5d ret (%)"]]
    for _, r in regime_stats.iterrows():
        rows.append([
            f"{r['regime_type']} {r['regime']}", f"{int(r['n_events'])}",
            f"{r['avg_reversion_days']:.1f}", f"{r['avg_magnitude_pct']:.3f}",
            f"{r['hit_rate_pct']:.0f}", f"{r['avg_fwd_ret_5d_pct']:+.2f}",
        ])
    table = Table(rows, hAlign="CENTER", repeatRows=1)
    table.setStyle(_TABLE_STYLE)
    flowables = [
        Paragraph(
            f"Table 4. Reversion behavior by regime, events at the "
            f"{config.ALERT_THRESHOLD:.1f}σ threshold. Hit rate is the share "
            "of events whose 5-day forward return has the mean-reverting "
            "sign.", styles["caption"]),
        table,
        Spacer(1, 6),
    ]
    if anova_path.exists():
        anova = pd.read_csv(anova_path)
        notes = "; ".join(
            f"{r['regime_type']} regimes: F = {r['f_stat']:.2f}, "
            f"p = {r['p_value']:.3f}" for _, r in anova.iterrows()
        )
        flowables.append(Paragraph(
            f"One-way ANOVA of 5-day forward returns — {notes}.",
            styles["caption"]))
    flowables.append(Spacer(1, 10))
    return flowables


def _backtest_table(styles: dict) -> list[Flowable]:
    """Build the backtest performance table.

    Args:
        styles: Shared style dict.

    Returns:
        Flowable list with caption and table.
    """
    path = config.DATA_PROCESSED_DIR / "backtest_metrics.csv"
    if not path.exists():
        return _missing(styles, "Backtest table", "python backtest/strategy.py")
    metrics = pd.read_csv(path)
    rows = [["Window", "Portfolio", "Ann. ret (%)", "Sharpe", "Sortino",
             "Calmar", "Max DD (%)", "Hit (%)", "PF", "Avg hold (d)",
             "Turnover (×/yr)"]]

    def fmt(value: float, spec: str = ".2f") -> str:
        return "—" if pd.isna(value) else format(value, spec)

    for _, r in metrics.iterrows():
        rows.append([
            r["window"], r["label"], fmt(r["ann_return_pct"]),
            fmt(r["sharpe"]), fmt(r["sortino"]), fmt(r["calmar"]),
            fmt(r["max_drawdown_pct"]), fmt(r["hit_rate_pct"], ".0f"),
            fmt(r["profit_factor"]), fmt(r["avg_holding_days"], ".1f"),
            fmt(r["ann_turnover_x"], ".1f"),
        ])
    table = Table(rows, hAlign="CENTER", repeatRows=1)
    table.setStyle(_TABLE_STYLE)
    return [
        Paragraph(
            "Table 5. Backtest performance, net of "
            f"{config.TRANSACTION_COST_BPS:.0f} bp one-way transaction "
            "costs. The walk-forward split trains on the first "
            f"{config.TRAIN_YEARS} years and tests on the remainder; PF is "
            "the profit factor.", styles["caption"]),
        table,
        Spacer(1, 10),
    ]


def build(styles: dict) -> list[Flowable]:
    """Build the results section flowables.

    Args:
        styles: Shared paragraph style dict from ``generate_paper.py``.

    Returns:
        List of ReportLab flowables.
    """
    figures = config.FIGURES_DIR
    flowables: list[Flowable] = [
        Paragraph("4. Results", styles["heading"]),

        Paragraph("4.1 Event study", styles["subheading"]),
        Paragraph(
            "Table 2 reports pooled forward returns after dislocation "
            "events at each threshold, and Table 3 summarizes how quickly "
            "the z-score itself normalizes. Figure 3 plots the average "
            "event-time path of the z-score with 95% bands. Under the "
            "simulated AR(1) NAV process, z-scores revert toward zero "
            "within days—as designed—while forward <i>price</i> returns "
            "carry no systematic drift, providing a clean placebo check of "
            "the inference machinery: with a real NAV feed, genuine "
            "dislocation premia would surface in exactly these tables.",
            styles["body"],
        ),
        *_event_study_table(styles),
        *_reversion_summary_table(styles),

        Paragraph("4.2 Regime analysis", styles["subheading"]),
        Paragraph(
            "Table 4 conditions the same events on the volatility and "
            "credit environment prevailing on the event day, and the "
            "ANOVA note tests whether 5-day forward returns differ "
            "across regimes.",
            styles["body"],
        ),
        *_regime_table(styles),

        Paragraph("4.3 Backtest", styles["subheading"]),
        Paragraph(
            "Table 5 reports performance of the long-discount / "
            "short-premium rule against the equal weight buy-and-hold "
            "benchmark across the full sample and the walk-forward "
            "windows. Figures 6–8 show cumulative growth, drawdowns and "
            "the monthly return profile.",
            styles["body"],
        ),
        *_backtest_table(styles),

        Paragraph("4.4 Figures", styles["subheading"]),
        *_figure(figures / "premium_discount_timeseries.png",
                 "Figure 1. Premium/discount to NAV by ETF; shaded bands "
                 "mark the March 2020, September 2022 and October 2022 "
                 "stress episodes.", styles),
        *_figure(figures / "zscore_timeseries.png",
                 f"Figure 2. {config.ZSCORE_WINDOW}-day rolling z-scores "
                 "with ±1.5/2.0/2.5σ thresholds.", styles),
        *_figure(figures / "event_study_paths.png",
                 "Figure 3. Average z-score paths around dislocation events "
                 "with 95% bands.", styles),
        *_figure(figures / "forward_returns.png",
                 "Figure 4. Pooled mean forward returns with 95% bootstrap "
                 "confidence intervals.", styles),
        *_figure(figures / "regime_heatmap.png",
                 "Figure 5. Regime-conditional reversion statistics.",
                 styles),
        *_figure(figures / "backtest_cumulative.png",
                 "Figure 6. Cumulative growth of $1: strategy vs. equal "
                 "weight benchmark.", styles),
        *_figure(figures / "backtest_drawdown.png",
                 "Figure 7. Drawdowns of the strategy and benchmark.",
                 styles),
        *_figure(figures / "monthly_returns_heatmap.png",
                 "Figure 8. Strategy monthly returns.", styles),
    ]
    return flowables
