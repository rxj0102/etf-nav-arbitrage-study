"""Premium/discount spread construction and dislocation flagging.

Loads the raw price/NAV CSVs produced by ``data/fetch.py`` and computes, for
each ETF in the universe:

* ``premium_pct``: the premium/discount, ``(Price - NAV) / NAV * 100``.
* ``zscore``: a 63-day rolling z-score of the premium.
* Dislocation event flags at the +/-1.5, +/-2.0 and +/-2.5 sigma thresholds
  (all three retained for sensitivity analysis).
* A ``stress_episode`` annotation column marking the March 2020 COVID crash
  and the September/October 2022 rates stress windows.

Inputs:
    ``data/raw/prices_<TICKER>.csv`` files from ``data/fetch.py``.

Outputs:
    * ``data/processed/spreads.csv`` -- long-format panel with one row per
      (date, ticker) and columns ``close, nav, premium_pct, zscore,
      ret_1d, flag_1.5, flag_2.0, flag_2.5, stress_episode``.
    * 300 DPI figures in ``outputs/figures/``:
      ``premium_discount_timeseries.png``, ``zscore_timeseries.png``,
      ``dislocation_events.png``.

Usage:
    python analysis/spread.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

logger = config.get_logger(__name__)

FIGURE_DPI = 300


def load_raw_prices(ticker: str) -> pd.DataFrame:
    """Load one ETF's raw price/NAV CSV from ``data/raw/``.

    Args:
        ticker: ETF ticker symbol.

    Returns:
        DataFrame indexed by ``Date`` with at least ``Close`` and ``NAV``
        columns.

    Raises:
        FileNotFoundError: If the raw CSV does not exist (run
            ``python data/fetch.py`` first).
    """
    path = config.DATA_RAW_DIR / f"prices_{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python data/fetch.py` first."
        )
    frame = pd.read_csv(path, index_col="Date", parse_dates=True)
    return frame


def compute_premium_discount(prices: pd.DataFrame) -> pd.Series:
    """Compute the percentage premium/discount of price to NAV.

    Args:
        prices: DataFrame containing ``Close`` and ``NAV`` columns.

    Returns:
        Series of ``(Close - NAV) / NAV * 100`` values.
    """
    return (prices["Close"] - prices["NAV"]) / prices["NAV"] * 100.0


def rolling_zscore(series: pd.Series, window: int = config.ZSCORE_WINDOW) -> pd.Series:
    """Compute a rolling z-score of a series.

    Args:
        series: Input series (the premium/discount in percent).
        window: Rolling window length in trading days.

    Returns:
        Series of ``(x - rolling_mean) / rolling_std`` values; the first
        ``window - 1`` observations are NaN.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def flag_dislocations(zscore: pd.Series, threshold: float) -> pd.Series:
    """Flag observations where |z-score| meets or exceeds a threshold.

    Args:
        zscore: Rolling z-score series.
        threshold: Sigma threshold, e.g. ``2.0``.

    Returns:
        Integer series: ``-1`` for discount dislocations
        (``z <= -threshold``), ``+1`` for premium dislocations
        (``z >= +threshold``), ``0`` otherwise.
    """
    flags = pd.Series(0, index=zscore.index, dtype=int)
    flags[zscore <= -threshold] = -1
    flags[zscore >= threshold] = 1
    return flags


def annotate_stress_episodes(index: pd.DatetimeIndex) -> pd.Series:
    """Label each date with the stress episode it falls inside, if any.

    Episodes are defined in ``config.STRESS_EPISODES`` (March 2020 COVID
    crash, September 2022 gilt/LDI stress, October 2022 rates vol spike).

    Args:
        index: Datetime index of the panel.

    Returns:
        String series with the episode name, or ``""`` outside episodes.
    """
    labels = pd.Series("", index=index, dtype=object)
    for name, (start, end) in config.STRESS_EPISODES.items():
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        labels[mask] = name
    return labels


def build_spread_panel() -> pd.DataFrame:
    """Assemble the long-format spread panel for all ETFs.

    Returns:
        DataFrame with columns ``date, ticker, close, nav, premium_pct,
        zscore, ret_1d, flag_1.5, flag_2.0, flag_2.5, stress_episode``.
    """
    frames = []
    for ticker in config.ETFS:
        prices = load_raw_prices(ticker)
        premium = compute_premium_discount(prices)
        zscore = rolling_zscore(premium)
        part = pd.DataFrame(
            {
                "date": prices.index,
                "ticker": ticker,
                "close": prices["Close"].to_numpy(),
                "nav": prices["NAV"].to_numpy(),
                "premium_pct": premium.to_numpy(),
                "zscore": zscore.to_numpy(),
                "ret_1d": prices["Close"].pct_change().to_numpy(),
            }
        )
        for threshold in config.DISLOCATION_THRESHOLDS:
            part[f"flag_{threshold}"] = flag_dislocations(zscore, threshold).to_numpy()
        part["stress_episode"] = annotate_stress_episodes(prices.index).to_numpy()
        frames.append(part)
        n_events = int((part[f"flag_{config.ALERT_THRESHOLD}"] != 0).sum())
        logger.info("%s: %d obs, %d days beyond +/-%.1f sigma",
                    ticker, len(part), n_events, config.ALERT_THRESHOLD)
    panel = pd.concat(frames, ignore_index=True)
    return panel


def _shade_stress(ax: plt.Axes) -> None:
    """Shade the configured stress episodes on a time series axis.

    Args:
        ax: Matplotlib axes whose x-axis is a date axis.

    Returns:
        None.
    """
    for name, (start, end) in config.STRESS_EPISODES.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="crimson", alpha=0.12, zorder=0)


def plot_premium_timeseries(panel: pd.DataFrame) -> Path:
    """Plot the premium/discount time series for every ETF.

    Args:
        panel: Long-format spread panel from :func:`build_spread_panel`.

    Returns:
        Path of the saved PNG figure.
    """
    fig, axes = plt.subplots(len(config.ETFS), 1, figsize=(11, 12), sharex=True)
    for ax, ticker in zip(axes, config.ETFS):
        sub = panel[panel["ticker"] == ticker]
        ax.plot(sub["date"], sub["premium_pct"], lw=0.7, color="navy")
        ax.axhline(0, color="gray", lw=0.6)
        _shade_stress(ax)
        ax.set_ylabel(f"{ticker}\n(%)")
    axes[0].set_title("ETF premium/discount to NAV (shaded: stress episodes)")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    out = config.FIGURES_DIR / "premium_discount_timeseries.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def plot_zscore_timeseries(panel: pd.DataFrame) -> Path:
    """Plot the rolling z-score per ETF with threshold guide lines.

    Args:
        panel: Long-format spread panel.

    Returns:
        Path of the saved PNG figure.
    """
    fig, axes = plt.subplots(len(config.ETFS), 1, figsize=(11, 12), sharex=True)
    for ax, ticker in zip(axes, config.ETFS):
        sub = panel[panel["ticker"] == ticker]
        ax.plot(sub["date"], sub["zscore"], lw=0.7, color="darkgreen")
        for threshold in config.DISLOCATION_THRESHOLDS:
            ax.axhline(threshold, color="firebrick", lw=0.5, ls="--", alpha=0.6)
            ax.axhline(-threshold, color="firebrick", lw=0.5, ls="--", alpha=0.6)
        _shade_stress(ax)
        ax.set_ylabel(f"{ticker}\nz-score")
    axes[0].set_title(
        f"{config.ZSCORE_WINDOW}-day rolling z-score of premium/discount "
        "(dashed: +/-1.5, 2.0, 2.5 sigma)"
    )
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    out = config.FIGURES_DIR / "zscore_timeseries.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def plot_dislocation_events(panel: pd.DataFrame) -> Path:
    """Plot dislocation event counts per ETF and threshold.

    Args:
        panel: Long-format spread panel.

    Returns:
        Path of the saved PNG figure.
    """
    rows = []
    for ticker in config.ETFS:
        sub = panel[panel["ticker"] == ticker]
        for threshold in config.DISLOCATION_THRESHOLDS:
            flags = sub[f"flag_{threshold}"]
            rows.append({"ticker": ticker, "threshold": f"{threshold}σ",
                         "direction": "discount", "days": int((flags == -1).sum())})
            rows.append({"ticker": ticker, "threshold": f"{threshold}σ",
                         "direction": "premium", "days": int((flags == 1).sum())})
    counts = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, direction in zip(axes, ["discount", "premium"]):
        sns.barplot(
            data=counts[counts["direction"] == direction],
            x="ticker", y="days", hue="threshold", ax=ax,
            palette="rocket",
        )
        ax.set_title(f"Days in {direction} dislocation")
        ax.set_xlabel("")
        ax.set_ylabel("Trading days flagged")
    fig.suptitle("Dislocation frequency by ETF and threshold")
    fig.tight_layout()
    out = config.FIGURES_DIR / "dislocation_events.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def main() -> pd.DataFrame:
    """Run the full spread analysis step.

    Returns:
        The long-format spread panel (also written to
        ``data/processed/spreads.csv``).
    """
    config.ensure_directories()
    logger.info("=== Spread analysis started ===")
    panel = build_spread_panel()

    out_path = config.DATA_PROCESSED_DIR / "spreads.csv"
    panel.to_csv(out_path, index=False)
    logger.info("Saved processed panel %s (%d rows)", out_path, len(panel))

    plot_premium_timeseries(panel)
    plot_zscore_timeseries(panel)
    plot_dislocation_events(panel)
    logger.info("=== Spread analysis complete ===")
    return panel


if __name__ == "__main__":
    main()
