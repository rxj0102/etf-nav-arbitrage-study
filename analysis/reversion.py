"""Event study of mean reversion after premium/discount dislocations.

For every threshold in ``config.DISLOCATION_THRESHOLDS`` and every ETF, this
module identifies *threshold-crossing events* (the first day the rolling
z-score moves beyond the threshold, with a cooldown so overlapping episodes
are counted once), then measures what happens next:

* Forward price returns at 1, 3, 5, 10 and 20 trading-day horizons, with
  95% bootstrap confidence intervals (1000 resamples) and one-sample
  t-tests against zero.
* Discount events (``z < -threshold``) and premium events
  (``z > +threshold``) are analyzed separately, since the trade direction
  differs.
* Reversion diagnostics per (threshold, direction): median time for the
  z-score to cross zero, the share of events fully reverted within 10 days,
  and the average peak dislocation magnitude reached during the episode.

Inputs:
    ``data/processed/spreads.csv`` from ``analysis/spread.py``.

Outputs:
    * ``data/processed/event_study_stats.csv`` -- forward return statistics
      per (ticker incl. ``ALL``, threshold, direction, horizon).
    * ``data/processed/reversion_summary.csv`` -- reversion-time summary per
      (threshold, direction).
    * ``data/processed/event_paths.csv`` -- average z-score path around
      events with bootstrap bands (consumed by the dashboard and paper).
    * ``outputs/figures/event_study_paths.png`` and
      ``outputs/figures/forward_returns.png`` at 300 DPI.

Usage:
    python analysis/reversion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

logger = config.get_logger(__name__)

FIGURE_DPI = 300
RNG = np.random.default_rng(123)


def load_spreads() -> pd.DataFrame:
    """Load the processed spread panel.

    Returns:
        Long-format spread panel with parsed ``date`` column.

    Raises:
        FileNotFoundError: If ``spreads.csv`` is missing (run
            ``python analysis/spread.py`` first).
    """
    path = config.DATA_PROCESSED_DIR / "spreads.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python analysis/spread.py` first."
        )
    panel = pd.read_csv(path, parse_dates=["date"])
    return panel


def find_events(
    zscore: pd.Series, threshold: float, direction: int,
    cooldown: int = config.EVENT_COOLDOWN_DAYS,
) -> list[int]:
    """Locate threshold-crossing events in a z-score series.

    An event is the first observation at which the z-score moves beyond the
    threshold in the given direction, having been inside the band on the
    previous day.  After an event, no new event is recorded for ``cooldown``
    trading days so overlapping episodes are not double counted.

    Args:
        zscore: Z-score series (positionally indexed within one ETF).
        threshold: Sigma threshold, e.g. ``2.0``.
        direction: ``-1`` for discount events (``z < -threshold``),
            ``+1`` for premium events (``z > +threshold``).
        cooldown: Minimum spacing between consecutive events, in rows.

    Returns:
        List of integer positions (iloc indices) of event days.
    """
    z = zscore.to_numpy()
    events: list[int] = []
    last_event = -10**9
    for i in range(1, len(z)):
        if np.isnan(z[i]) or np.isnan(z[i - 1]):
            continue
        beyond_now = z[i] < -threshold if direction == -1 else z[i] > threshold
        beyond_prev = z[i - 1] < -threshold if direction == -1 else z[i - 1] > threshold
        if beyond_now and not beyond_prev and i - last_event > cooldown:
            events.append(i)
            last_event = i
    return events


def bootstrap_ci(
    values: np.ndarray,
    n_samples: int = config.BOOTSTRAP_SAMPLES,
    ci: float = config.BOOTSTRAP_CI,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for the mean of ``values``.

    Args:
        values: Sample of observations (NaNs must already be removed).
        n_samples: Number of bootstrap resamples.
        ci: Confidence level, e.g. ``0.95``.

    Returns:
        Tuple ``(lower, upper)`` of the CI bounds; ``(nan, nan)`` when
        fewer than two observations are available.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    means = np.empty(n_samples)
    n = len(values)
    for b in range(n_samples):
        means[b] = values[RNG.integers(0, n, n)].mean()
    alpha = (1 - ci) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def _forward_returns(close: np.ndarray, event_idx: list[int], horizon: int) -> np.ndarray:
    """Compute forward simple returns from event days.

    Args:
        close: Close price array for one ETF.
        event_idx: Positions of event days.
        horizon: Forward horizon in trading days.

    Returns:
        Array of forward returns; events too close to the sample end are
        dropped.
    """
    rets = []
    for i in event_idx:
        if i + horizon < len(close):
            rets.append(close[i + horizon] / close[i] - 1.0)
    return np.asarray(rets)


def _reversion_diagnostics(
    z: np.ndarray, premium: np.ndarray, event_idx: list[int], direction: int,
) -> tuple[list[float], list[bool], list[float]]:
    """Measure reversion time, full reversion within 10d, and peak magnitude.

    Args:
        z: Z-score array for one ETF.
        premium: Premium/discount (%) array for the same ETF.
        event_idx: Positions of event days.
        direction: Event direction (-1 discount, +1 premium).

    Returns:
        Tuple of three lists, one entry per event:
        ``(days_to_zero, reverted_within_10d, peak_dislocation_pct)``.
        ``days_to_zero`` is NaN when the z-score does not cross zero within
        ``config.REVERSION_HORIZON`` days.
    """
    days_to_zero: list[float] = []
    reverted_10d: list[bool] = []
    peak_mag: list[float] = []
    for i in event_idx:
        end = min(i + config.REVERSION_HORIZON, len(z) - 1)
        window = z[i: end + 1]
        # Days until the z-score first crosses zero (sign opposite the event).
        if direction == -1:
            crossed = np.where(window >= 0)[0]
        else:
            crossed = np.where(window <= 0)[0]
        days = float(crossed[0]) if len(crossed) else float("nan")
        days_to_zero.append(days)
        reverted_10d.append(bool(len(crossed)) and crossed[0] <= config.FULL_REVERSION_DAYS)
        # Peak dislocation: most extreme premium reached during the episode.
        seg = premium[i: end + 1]
        peak = float(np.nanmin(seg)) if direction == -1 else float(np.nanmax(seg))
        peak_mag.append(abs(peak))
    return days_to_zero, reverted_10d, peak_mag


def run_event_study(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full event study across ETFs, thresholds and directions.

    Args:
        panel: Long-format spread panel from ``analysis/spread.py``.

    Returns:
        Tuple ``(stats_df, summary_df, paths_df)``:

        * ``stats_df``: forward return statistics per (ticker incl. pooled
          ``ALL``, threshold, direction, horizon) with bootstrap CIs,
          t-stats and p-values.
        * ``summary_df``: reversion diagnostics per (threshold, direction).
        * ``paths_df``: average z-score path around events (days -5..+20)
          with bootstrap bands, per (threshold, direction).
    """
    stat_rows: list[dict] = []
    summary_rows: list[dict] = []
    path_rows: list[dict] = []

    by_ticker = {t: g.reset_index(drop=True) for t, g in panel.groupby("ticker")}

    for threshold in config.DISLOCATION_THRESHOLDS:
        for direction, dir_name in [(-1, "discount"), (1, "premium")]:
            pooled: dict[int, list[float]] = {h: [] for h in config.FORWARD_HORIZONS}
            all_days, all_rev10, all_peak = [], [], []
            pooled_paths: list[np.ndarray] = []

            for ticker, sub in by_ticker.items():
                z = sub["zscore"]
                close = sub["close"].to_numpy()
                premium = sub["premium_pct"].to_numpy()
                events = find_events(z, threshold, direction)

                for horizon in config.FORWARD_HORIZONS:
                    fwd = _forward_returns(close, events, horizon)
                    pooled[horizon].extend(fwd.tolist())
                    if len(fwd) >= 2:
                        t_stat, p_val = stats.ttest_1samp(fwd, 0.0)
                        lo, hi = bootstrap_ci(fwd)
                    else:
                        t_stat, p_val, lo, hi = (float("nan"),) * 4
                    stat_rows.append({
                        "ticker": ticker, "threshold": threshold,
                        "direction": dir_name, "horizon": horizon,
                        "n_events": len(fwd),
                        "mean_fwd_ret_pct": float(np.mean(fwd) * 100) if len(fwd) else float("nan"),
                        "median_fwd_ret_pct": float(np.median(fwd) * 100) if len(fwd) else float("nan"),
                        "ci_low_pct": lo * 100 if np.isfinite(lo) else float("nan"),
                        "ci_high_pct": hi * 100 if np.isfinite(hi) else float("nan"),
                        "t_stat": float(t_stat), "p_value": float(p_val),
                    })

                days, rev10, peak = _reversion_diagnostics(
                    z.to_numpy(), premium, events, direction)
                all_days.extend(days)
                all_rev10.extend(rev10)
                all_peak.extend(peak)

                # Z-score paths from 5 days before to 20 days after.
                zarr = z.to_numpy()
                for i in events:
                    if i - 5 >= 0 and i + config.REVERSION_HORIZON < len(zarr):
                        pooled_paths.append(zarr[i - 5: i + config.REVERSION_HORIZON + 1])

            # Pooled forward return stats across all ETFs.
            for horizon in config.FORWARD_HORIZONS:
                fwd = np.asarray(pooled[horizon])
                if len(fwd) >= 2:
                    t_stat, p_val = stats.ttest_1samp(fwd, 0.0)
                    lo, hi = bootstrap_ci(fwd)
                else:
                    t_stat, p_val, lo, hi = (float("nan"),) * 4
                stat_rows.append({
                    "ticker": "ALL", "threshold": threshold,
                    "direction": dir_name, "horizon": horizon,
                    "n_events": len(fwd),
                    "mean_fwd_ret_pct": float(np.mean(fwd) * 100) if len(fwd) else float("nan"),
                    "median_fwd_ret_pct": float(np.median(fwd) * 100) if len(fwd) else float("nan"),
                    "ci_low_pct": lo * 100 if np.isfinite(lo) else float("nan"),
                    "ci_high_pct": hi * 100 if np.isfinite(hi) else float("nan"),
                    "t_stat": float(t_stat), "p_value": float(p_val),
                })

            finite_days = np.asarray([d for d in all_days if np.isfinite(d)])
            summary_rows.append({
                "threshold": threshold, "direction": dir_name,
                "n_events": len(all_days),
                "median_reversion_days": float(np.median(finite_days)) if len(finite_days) else float("nan"),
                "pct_reverted_within_10d": float(np.mean(all_rev10) * 100) if all_days else float("nan"),
                "avg_peak_dislocation_pct": float(np.mean(all_peak)) if all_peak else float("nan"),
            })

            if pooled_paths:
                paths = np.vstack(pooled_paths)
                mean_path = paths.mean(axis=0)
                lo_path = np.quantile(paths, 0.025, axis=0)
                hi_path = np.quantile(paths, 0.975, axis=0)
                for offset, (m, lo, hi) in enumerate(zip(mean_path, lo_path, hi_path)):
                    path_rows.append({
                        "threshold": threshold, "direction": dir_name,
                        "day": offset - 5, "mean_z": float(m),
                        "ci_low": float(lo), "ci_high": float(hi),
                        "n_events": len(pooled_paths),
                    })
            logger.info(
                "threshold=%.1f %s: %d events, median reversion %.1f days, "
                "%.0f%% reverted within %dd",
                threshold, dir_name, len(all_days),
                summary_rows[-1]["median_reversion_days"],
                summary_rows[-1]["pct_reverted_within_10d"],
                config.FULL_REVERSION_DAYS,
            )

    return pd.DataFrame(stat_rows), pd.DataFrame(summary_rows), pd.DataFrame(path_rows)


def plot_event_paths(paths: pd.DataFrame) -> Path:
    """Plot average z-score reversion paths with bootstrap bands.

    Args:
        paths: Path frame from :func:`run_event_study`.

    Returns:
        Path of the saved PNG figure.
    """
    thresholds = sorted(paths["threshold"].unique())
    fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 4.5), sharey=True)
    if len(thresholds) == 1:
        axes = [axes]
    colors = {"discount": "tab:blue", "premium": "tab:red"}
    for ax, threshold in zip(axes, thresholds):
        for direction in ["discount", "premium"]:
            sub = paths[(paths["threshold"] == threshold)
                        & (paths["direction"] == direction)]
            if sub.empty:
                continue
            ax.plot(sub["day"], sub["mean_z"], color=colors[direction],
                    label=f"{direction} (n={int(sub['n_events'].iloc[0])})")
            ax.fill_between(sub["day"], sub["ci_low"], sub["ci_high"],
                            color=colors[direction], alpha=0.15)
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6, ls="--")
        ax.set_title(f"|z| > {threshold}σ events")
        ax.set_xlabel("Days from event")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Average z-score")
    fig.suptitle("Mean reversion paths after dislocation events (95% bands)")
    fig.tight_layout()
    out = config.FIGURES_DIR / "event_study_paths.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def plot_forward_returns(stats_df: pd.DataFrame) -> Path:
    """Plot pooled mean forward returns with bootstrap CIs per horizon.

    Args:
        stats_df: Statistics frame from :func:`run_event_study`.

    Returns:
        Path of the saved PNG figure.
    """
    pooled = stats_df[stats_df["ticker"] == "ALL"]
    thresholds = sorted(pooled["threshold"].unique())
    fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 4.5), sharey=True)
    if len(thresholds) == 1:
        axes = [axes]
    offsets = {"discount": -0.15, "premium": 0.15}
    colors = {"discount": "tab:blue", "premium": "tab:red"}
    for ax, threshold in zip(axes, thresholds):
        for direction in ["discount", "premium"]:
            sub = pooled[(pooled["threshold"] == threshold)
                         & (pooled["direction"] == direction)]
            x = np.arange(len(sub)) + offsets[direction]
            yerr = np.vstack([
                sub["mean_fwd_ret_pct"] - sub["ci_low_pct"],
                sub["ci_high_pct"] - sub["mean_fwd_ret_pct"],
            ])
            ax.errorbar(x, sub["mean_fwd_ret_pct"], yerr=yerr, fmt="o",
                        capsize=4, color=colors[direction], label=direction)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xticks(np.arange(len(config.FORWARD_HORIZONS)))
        ax.set_xticklabels([str(h) for h in config.FORWARD_HORIZONS])
        ax.set_xlabel("Horizon (days)")
        ax.set_title(f"|z| > {threshold}σ")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean forward return (%)")
    fig.suptitle("Pooled forward returns after dislocations (95% bootstrap CIs)")
    fig.tight_layout()
    out = config.FIGURES_DIR / "forward_returns.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def main() -> pd.DataFrame:
    """Run the full event study step.

    Returns:
        The forward return statistics frame (also written to
        ``data/processed/event_study_stats.csv``).
    """
    config.ensure_directories()
    logger.info("=== Event study started ===")
    panel = load_spreads()
    stats_df, summary_df, paths_df = run_event_study(panel)

    stats_path = config.DATA_PROCESSED_DIR / "event_study_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info("Saved %s (%d rows)", stats_path, len(stats_df))

    summary_path = config.DATA_PROCESSED_DIR / "reversion_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved %s (%d rows)", summary_path, len(summary_df))

    paths_path = config.DATA_PROCESSED_DIR / "event_paths.csv"
    paths_df.to_csv(paths_path, index=False)
    logger.info("Saved %s (%d rows)", paths_path, len(paths_df))

    plot_event_paths(paths_df)
    plot_forward_returns(stats_df)
    logger.info("=== Event study complete ===")
    return stats_df


if __name__ == "__main__":
    main()
