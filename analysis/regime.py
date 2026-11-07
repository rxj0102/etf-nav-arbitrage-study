"""Regime-conditional analysis of premium/discount mean reversion.

Classifies every trading day into volatility and credit regimes and asks
whether dislocation reversion behaves differently across them:

* **VIX regimes**: low (VIX <= 15), medium (15 < VIX <= 25), high (VIX > 25).
* **OAS regimes**: tight vs. wide high-yield credit spreads, split at the
  sample median of the FRED ``BAMLH0A0HYM2`` series.

For every regime the module computes, over dislocation events at the default
threshold (``config.ALERT_THRESHOLD``):

* Reversion speed: average days for the z-score to cross zero (capped at 20).
* Magnitude: average absolute premium/discount at the event.
* Hit rate: share of events whose 5-day forward price return has the
  mean-reverting sign (positive after discounts, negative after premiums).

One-way ANOVA (``scipy.stats.f_oneway``) tests whether 5-day forward returns
differ significantly across regimes.

Inputs:
    ``data/processed/spreads.csv``, ``data/raw/vix.csv``,
    ``data/raw/oas.csv``.

Outputs:
    * ``data/processed/regime_stats.csv`` -- per-regime statistics.
    * ``data/processed/regime_anova.csv`` -- ANOVA F-statistics/p-values.
    * ``outputs/figures/regime_heatmap.png`` and
      ``outputs/figures/regime_boxplots.png`` at 300 DPI.

Usage:
    python analysis/regime.py
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
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from analysis.reversion import find_events

logger = config.get_logger(__name__)

FIGURE_DPI = 300
VIX_ORDER = ["low", "medium", "high"]
OAS_ORDER = ["tight", "wide"]


def classify_vix(vix: pd.Series) -> pd.Series:
    """Classify VIX levels into low/medium/high regimes.

    Args:
        vix: Daily VIX close series.

    Returns:
        String series with values ``"low"`` (VIX <= 15), ``"medium"``
        (15 < VIX <= 25) or ``"high"`` (VIX > 25).
    """
    regime = pd.Series("medium", index=vix.index, dtype=object)
    regime[vix <= config.VIX_LOW_MAX] = "low"
    regime[vix > config.VIX_HIGH_MIN] = "high"
    return regime


def classify_oas(oas: pd.Series) -> pd.Series:
    """Classify high-yield OAS into tight/wide regimes via a median split.

    Args:
        oas: Daily HY OAS series.

    Returns:
        String series with values ``"tight"`` (below sample median) or
        ``"wide"`` (at or above the median).
    """
    median = oas.median()
    regime = pd.Series("tight", index=oas.index, dtype=object)
    regime[oas >= median] = "wide"
    logger.info("HY OAS median split at %.2f", median)
    return regime


def load_inputs() -> pd.DataFrame:
    """Load spreads, VIX and OAS, merge regimes onto the spread panel.

    Returns:
        Spread panel with extra ``vix, vix_regime, hy_oas, oas_regime``
        columns (regimes forward-filled onto trading days).

    Raises:
        FileNotFoundError: If any required input CSV is missing.
    """
    spreads_path = config.DATA_PROCESSED_DIR / "spreads.csv"
    vix_path = config.DATA_RAW_DIR / "vix.csv"
    oas_path = config.DATA_RAW_DIR / "oas.csv"
    for path, producer in [
        (spreads_path, "python analysis/spread.py"),
        (vix_path, "python data/fetch.py"),
        (oas_path, "python data/fetch.py"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `{producer}` first.")

    panel = pd.read_csv(spreads_path, parse_dates=["date"])
    vix = pd.read_csv(vix_path, index_col="Date", parse_dates=True)["VIX"]
    oas = pd.read_csv(oas_path, index_col="Date", parse_dates=True)["HY_OAS"]

    market = pd.DataFrame({"vix": vix}).join(pd.DataFrame({"hy_oas": oas}), how="outer")
    market = market.ffill().dropna()
    market["vix_regime"] = classify_vix(market["vix"])
    market["oas_regime"] = classify_oas(market["hy_oas"])

    merged = panel.merge(
        market, left_on="date", right_index=True, how="left",
    )
    merged[["vix", "hy_oas"]] = merged[["vix", "hy_oas"]].ffill()
    merged["vix_regime"] = merged["vix_regime"].ffill()
    merged["oas_regime"] = merged["oas_regime"].ffill()
    return merged


def _event_records(panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Build one row per dislocation event with regime and outcome columns.

    Args:
        panel: Merged spread/regime panel from :func:`load_inputs`.
        threshold: Sigma threshold used to define events.

    Returns:
        DataFrame with one row per event and columns ``ticker, date,
        direction, vix_regime, oas_regime, magnitude_pct, days_to_zero,
        fwd_ret_5d, hit``.
    """
    records: list[dict] = []
    for ticker, sub in panel.groupby("ticker"):
        sub = sub.reset_index(drop=True)
        z = sub["zscore"]
        close = sub["close"].to_numpy()
        premium = sub["premium_pct"].to_numpy()
        for direction, dir_name in [(-1, "discount"), (1, "premium")]:
            for i in find_events(z, threshold, direction):
                horizon_end = min(i + config.REVERSION_HORIZON, len(sub) - 1)
                window = z.to_numpy()[i: horizon_end + 1]
                if direction == -1:
                    crossed = np.where(window >= 0)[0]
                else:
                    crossed = np.where(window <= 0)[0]
                days_to_zero = float(crossed[0]) if len(crossed) else float(
                    config.REVERSION_HORIZON)
                fwd5 = (close[i + 5] / close[i] - 1.0) if i + 5 < len(sub) else np.nan
                hit = np.nan
                if np.isfinite(fwd5):
                    hit = float(fwd5 > 0) if direction == -1 else float(fwd5 < 0)
                records.append({
                    "ticker": ticker,
                    "date": sub["date"].iloc[i],
                    "direction": dir_name,
                    "vix_regime": sub["vix_regime"].iloc[i],
                    "oas_regime": sub["oas_regime"].iloc[i],
                    "magnitude_pct": abs(premium[i]),
                    "days_to_zero": days_to_zero,
                    "fwd_ret_5d": fwd5,
                    "hit": hit,
                })
    events = pd.DataFrame(records)
    logger.info("Identified %d events at |z| > %.1f for regime analysis",
                len(events), threshold)
    return events


def regime_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reversion statistics per regime.

    Args:
        events: Event-level frame from :func:`_event_records`.

    Returns:
        DataFrame with one row per (regime_type, regime) and columns
        ``n_events, avg_reversion_days, avg_magnitude_pct, hit_rate_pct,
        avg_fwd_ret_5d_pct``.
    """
    rows = []
    for regime_type, order in [("vix", VIX_ORDER), ("oas", OAS_ORDER)]:
        column = f"{regime_type}_regime"
        for regime in order:
            sub = events[events[column] == regime]
            rows.append({
                "regime_type": regime_type.upper(),
                "regime": regime,
                "n_events": len(sub),
                "avg_reversion_days": float(sub["days_to_zero"].mean()) if len(sub) else float("nan"),
                "avg_magnitude_pct": float(sub["magnitude_pct"].mean()) if len(sub) else float("nan"),
                "hit_rate_pct": float(sub["hit"].mean() * 100) if sub["hit"].notna().any() else float("nan"),
                "avg_fwd_ret_5d_pct": float(sub["fwd_ret_5d"].mean() * 100) if sub["fwd_ret_5d"].notna().any() else float("nan"),
            })
    return pd.DataFrame(rows)


def regime_anova(events: pd.DataFrame) -> pd.DataFrame:
    """Run one-way ANOVA of 5-day forward returns across regimes.

    Args:
        events: Event-level frame from :func:`_event_records`.

    Returns:
        DataFrame with one row per regime type containing the F-statistic,
        p-value and per-group sample sizes.
    """
    rows = []
    for regime_type, order in [("vix", VIX_ORDER), ("oas", OAS_ORDER)]:
        column = f"{regime_type}_regime"
        groups = [
            events.loc[(events[column] == regime) & events["fwd_ret_5d"].notna(),
                       "fwd_ret_5d"].to_numpy()
            for regime in order
        ]
        valid = [g for g in groups if len(g) >= 2]
        if len(valid) >= 2:
            f_stat, p_val = scipy_stats.f_oneway(*valid)
        else:
            f_stat, p_val = float("nan"), float("nan")
        rows.append({
            "regime_type": regime_type.upper(),
            "groups": ", ".join(order),
            "group_sizes": ", ".join(str(len(g)) for g in groups),
            "f_stat": float(f_stat),
            "p_value": float(p_val),
        })
        logger.info("ANOVA across %s regimes: F=%.3f, p=%.4f",
                    regime_type.upper(), f_stat, p_val)
    return pd.DataFrame(rows)


def plot_regime_heatmap(stats_df: pd.DataFrame) -> Path:
    """Plot a heatmap of regime-conditional reversion statistics.

    Args:
        stats_df: Per-regime statistics from :func:`regime_statistics`.

    Returns:
        Path of the saved PNG figure.
    """
    display = stats_df.copy()
    display["label"] = display["regime_type"] + " / " + display["regime"]
    metrics = {
        "avg_reversion_days": "Avg reversion (days)",
        "avg_magnitude_pct": "Avg magnitude (%)",
        "hit_rate_pct": "Hit rate (%)",
        "avg_fwd_ret_5d_pct": "Avg 5d fwd ret (%)",
    }
    matrix = display.set_index("label")[list(metrics)].rename(columns=metrics)
    # Normalize each column for color scale while annotating raw values.
    normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(normalized, annot=matrix.round(2), fmt="", cmap="viridis",
                cbar=False, ax=ax)
    ax.set_title("Reversion behavior by volatility and credit regime")
    fig.tight_layout()
    out = config.FIGURES_DIR / "regime_heatmap.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def plot_regime_boxplots(events: pd.DataFrame) -> Path:
    """Plot box plots of 5-day forward returns by regime.

    Args:
        events: Event-level frame from :func:`_event_records`.

    Returns:
        Path of the saved PNG figure.
    """
    plot_df = events.dropna(subset=["fwd_ret_5d"]).copy()
    plot_df["fwd_ret_5d_pct"] = plot_df["fwd_ret_5d"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.boxplot(data=plot_df, x="vix_regime", y="fwd_ret_5d_pct",
                order=VIX_ORDER, hue="vix_regime", legend=False,
                palette="Blues", ax=axes[0])
    axes[0].set_title("By VIX regime")
    axes[0].set_xlabel("VIX regime")
    axes[0].set_ylabel("5-day forward return (%)")
    sns.boxplot(data=plot_df, x="oas_regime", y="fwd_ret_5d_pct",
                order=OAS_ORDER, hue="oas_regime", legend=False,
                palette="Oranges", ax=axes[1])
    axes[1].set_title("By HY OAS regime")
    axes[1].set_xlabel("OAS regime")
    axes[1].set_ylabel("")
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.6)
    fig.suptitle("Forward returns after dislocations, by regime")
    fig.tight_layout()
    out = config.FIGURES_DIR / "regime_boxplots.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def main() -> pd.DataFrame:
    """Run the full regime analysis step.

    Returns:
        The per-regime statistics frame (also written to
        ``data/processed/regime_stats.csv``).
    """
    config.ensure_directories()
    logger.info("=== Regime analysis started ===")
    panel = load_inputs()
    events = _event_records(panel, config.ALERT_THRESHOLD)

    events_path = config.DATA_PROCESSED_DIR / "regime_events.csv"
    events.to_csv(events_path, index=False)
    logger.info("Saved %s (%d rows)", events_path, len(events))

    stats_df = regime_statistics(events)
    stats_path = config.DATA_PROCESSED_DIR / "regime_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info("Saved %s (%d rows)", stats_path, len(stats_df))

    anova_df = regime_anova(events)
    anova_path = config.DATA_PROCESSED_DIR / "regime_anova.csv"
    anova_df.to_csv(anova_path, index=False)
    logger.info("Saved %s (%d rows)", anova_path, len(anova_df))

    plot_regime_heatmap(stats_df)
    plot_regime_boxplots(events)
    logger.info("=== Regime analysis complete ===")
    return stats_df


if __name__ == "__main__":
    main()
