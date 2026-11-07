"""Mean reversion backtest on ETF premium/discount dislocations.

Strategy rules
--------------
* **Entry:** go long an ETF when its premium z-score drops below -2 (deep
  discount); go short when the z-score rises above +2 (rich premium).
* **Exit:** close when the z-score crosses zero, or after a 20-trading-day
  time stop, whichever comes first.
* **Sizing:** inverse 21-day rolling volatility, normalized so the average
  position is 1.0x, and capped at 2x the average position size.
* **Costs:** 5 basis points (one way) charged on every unit of turnover.

The strategy trades each ETF independently; portfolio return is the average
of per-ETF position returns (cash earns zero).  The benchmark is a
buy-and-hold equal weight portfolio of all five ETFs.

Walk-forward validation splits the sample into the first three years
("train") and the final two ("test"); metrics are reported for the full
sample and both sub-periods.  No parameters are re-fit on the train window
-- the split demonstrates out-of-sample stability of the fixed rule set.

Inputs:
    ``data/processed/spreads.csv`` from ``analysis/spread.py``.

Outputs:
    * ``data/processed/trades.csv`` -- full trade log.
    * ``data/processed/backtest_metrics.csv`` -- performance metrics for
      strategy and benchmark across full/train/test windows.
    * ``data/processed/backtest_returns.csv`` -- daily strategy and
      benchmark returns.
    * 300 DPI figures: ``backtest_cumulative.png``,
      ``backtest_drawdown.png``, ``monthly_returns_heatmap.png``.

Usage:
    python backtest/strategy.py
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


def load_spreads() -> pd.DataFrame:
    """Load the processed spread panel.

    Returns:
        Long-format spread panel with parsed ``date`` column.

    Raises:
        FileNotFoundError: If ``spreads.csv`` is missing.
    """
    path = config.DATA_PROCESSED_DIR / "spreads.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python analysis/spread.py` first."
        )
    return pd.read_csv(path, parse_dates=["date"])


def generate_positions(sub: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Generate daily positions and the trade log for one ETF.

    Implements the entry/exit state machine: enter at +/-`ENTRY_ZSCORE`,
    exit on a zero crossing of the z-score or after ``MAX_HOLDING_DAYS``.
    Position size is inverse 21-day rolling volatility, normalized to an
    average of 1.0 and capped at ``MAX_POSITION_MULTIPLE``.

    Args:
        sub: Spread panel rows for a single ETF, sorted by date, with
            ``date, close, zscore, ret_1d`` columns.

    Returns:
        Tuple ``(positions, trades)`` where ``positions`` is a signed daily
        position series (set on the decision day, applied to the next day's
        return by the caller) and ``trades`` is a list of trade dicts.
    """
    sub = sub.reset_index(drop=True)
    z = sub["zscore"].to_numpy()
    close = sub["close"].to_numpy()
    dates = sub["date"]

    vol = sub["ret_1d"].rolling(config.VOL_WINDOW).std()
    inv_vol = 1.0 / vol.replace(0, np.nan)
    size = (inv_vol / inv_vol.mean()).clip(upper=config.MAX_POSITION_MULTIPLE)
    size = size.fillna(0.0).to_numpy()

    positions = np.zeros(len(sub))
    trades: list[dict] = []
    state = 0  # 0 flat, +1 long, -1 short
    entry_i = -1
    entry_size = 0.0

    for i in range(len(sub)):
        if np.isnan(z[i]):
            continue
        if state == 0:
            if z[i] < -config.ENTRY_ZSCORE and size[i] > 0:
                state, entry_i, entry_size = 1, i, size[i]
            elif z[i] > config.ENTRY_ZSCORE and size[i] > 0:
                state, entry_i, entry_size = -1, i, size[i]
        else:
            held = i - entry_i
            crossed_zero = (state == 1 and z[i] >= 0) or (state == -1 and z[i] <= 0)
            if crossed_zero or held >= config.MAX_HOLDING_DAYS:
                gross_ret = state * (close[i] / close[entry_i] - 1.0)
                cost = 2 * config.TRANSACTION_COST_BPS / 1e4 * entry_size
                trades.append({
                    "ticker": sub["ticker"].iloc[0],
                    "direction": "long" if state == 1 else "short",
                    "entry_date": dates.iloc[entry_i],
                    "exit_date": dates.iloc[i],
                    "entry_zscore": float(z[entry_i]),
                    "exit_zscore": float(z[i]),
                    "holding_days": held,
                    "position_size": float(entry_size),
                    "gross_return_pct": float(gross_ret * 100),
                    "net_return_pct": float((gross_ret * entry_size - cost) * 100),
                    "exit_reason": "zero_cross" if crossed_zero else "time_stop",
                })
                state = 0
        positions[i] = state * (entry_size if state != 0 else 0.0)

    return pd.Series(positions, index=sub.index), trades


def backtest_portfolio(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the backtest across all ETFs and assemble portfolio returns.

    Positions decided at the close of day *t* earn day *t+1*'s return
    (one-day implementation lag, no look-ahead).  Transaction costs of
    ``TRANSACTION_COST_BPS`` are charged on each day's absolute change in
    position (turnover).

    Args:
        panel: Long-format spread panel.

    Returns:
        Tuple ``(returns, trades)``: a daily frame with ``strategy``,
        ``benchmark`` and ``turnover`` columns indexed by date, and the
        full trade log.
    """
    per_etf_strategy: dict[str, pd.Series] = {}
    per_etf_benchmark: dict[str, pd.Series] = {}
    per_etf_turnover: dict[str, pd.Series] = {}
    all_trades: list[dict] = []

    for ticker, sub in panel.groupby("ticker"):
        sub = sub.sort_values("date").reset_index(drop=True)
        positions, trades = generate_positions(sub)
        all_trades.extend(trades)

        ret = sub["ret_1d"].fillna(0.0)
        lagged = positions.shift(1).fillna(0.0)
        turnover = positions.diff().abs().fillna(0.0)
        cost = turnover * config.TRANSACTION_COST_BPS / 1e4
        strat = lagged * ret - cost

        idx = pd.DatetimeIndex(sub["date"])
        per_etf_strategy[ticker] = pd.Series(strat.to_numpy(), index=idx)
        per_etf_benchmark[ticker] = pd.Series(ret.to_numpy(), index=idx)
        per_etf_turnover[ticker] = pd.Series(turnover.to_numpy(), index=idx)
        logger.info("%s: %d trades, in-market %.0f%% of days", ticker,
                    len(trades), (lagged != 0).mean() * 100)

    strategy = pd.DataFrame(per_etf_strategy).mean(axis=1)
    benchmark = pd.DataFrame(per_etf_benchmark).mean(axis=1)
    turnover = pd.DataFrame(per_etf_turnover).mean(axis=1)

    returns = pd.DataFrame({
        "strategy": strategy, "benchmark": benchmark, "turnover": turnover,
    }).dropna()
    returns.index.name = "date"

    trades_df = pd.DataFrame(all_trades).sort_values("entry_date").reset_index(drop=True)
    return returns, trades_df


def compute_metrics(returns: pd.Series, trades: pd.DataFrame | None,
                    turnover: pd.Series | None, label: str) -> dict:
    """Compute the full performance metric set for a return series.

    Args:
        returns: Daily simple returns.
        trades: Trade log restricted to the same window, or ``None`` for
            the benchmark (trade-based metrics become NaN).
        turnover: Daily turnover series for the same window, or ``None``.
        label: Name recorded in the output row.

    Returns:
        Dict of metrics: annualized return/vol, Sharpe, Sortino, Calmar,
        max drawdown, hit rate, profit factor, average holding period and
        annualized turnover.
    """
    tdpy = config.TRADING_DAYS_PER_YEAR
    n = len(returns)
    cumulative = (1 + returns).prod()
    years = n / tdpy
    ann_return = cumulative ** (1 / years) - 1 if years > 0 and cumulative > 0 else float("nan")
    ann_vol = returns.std() * np.sqrt(tdpy)
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(tdpy) if len(downside) else float("nan")
    sortino = ann_return / downside_vol if downside_vol and downside_vol > 0 else float("nan")

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else float("nan")

    if trades is not None and len(trades):
        wins = trades[trades["net_return_pct"] > 0]["net_return_pct"]
        losses = trades[trades["net_return_pct"] <= 0]["net_return_pct"]
        hit_rate = len(wins) / len(trades) * 100
        profit_factor = (wins.sum() / abs(losses.sum())
                         if len(losses) and losses.sum() != 0 else float("nan"))
        avg_holding = trades["holding_days"].mean()
        n_trades = len(trades)
    else:
        hit_rate = profit_factor = avg_holding = n_trades = float("nan")

    ann_turnover = (turnover.mean() * tdpy) if turnover is not None else float("nan")

    return {
        "label": label,
        "ann_return_pct": ann_return * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown_pct": max_dd * 100,
        "hit_rate_pct": hit_rate,
        "profit_factor": profit_factor,
        "avg_holding_days": avg_holding,
        "n_trades": n_trades,
        "ann_turnover_x": ann_turnover,
    }


def walk_forward_windows(index: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    """Split the sample into full / train / test windows.

    The train window covers the first ``config.TRAIN_YEARS`` years; the test
    window is everything after.

    Args:
        index: Full daily date index of the backtest.

    Returns:
        Dict mapping window label to the dates it contains.
    """
    split_date = index[0] + pd.DateOffset(years=config.TRAIN_YEARS)
    return {
        "full": index,
        "train": index[index < split_date],
        "test": index[index >= split_date],
    }


def build_metric_table(returns: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for strategy and benchmark across all windows.

    Args:
        returns: Daily return frame with ``strategy/benchmark/turnover``.
        trades: Full trade log.

    Returns:
        Metric table with one row per (window, portfolio).
    """
    rows = []
    windows = walk_forward_windows(returns.index)
    for window_name, dates in windows.items():
        window_returns = returns.loc[returns.index.isin(dates)]
        window_trades = trades[
            (trades["entry_date"] >= dates[0]) & (trades["entry_date"] <= dates[-1])
        ] if len(dates) else trades.iloc[0:0]
        rows.append({
            "window": window_name,
            **compute_metrics(window_returns["strategy"], window_trades,
                              window_returns["turnover"], "strategy"),
        })
        rows.append({
            "window": window_name,
            **compute_metrics(window_returns["benchmark"], None, None, "benchmark"),
        })
    return pd.DataFrame(rows)


def plot_cumulative(returns: pd.DataFrame) -> Path:
    """Plot cumulative strategy vs. benchmark returns with the WF split.

    Args:
        returns: Daily return frame.

    Returns:
        Path of the saved PNG figure.
    """
    equity = (1 + returns[["strategy", "benchmark"]]).cumprod()
    split_date = returns.index[0] + pd.DateOffset(years=config.TRAIN_YEARS)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(equity.index, equity["strategy"], label="Mean reversion strategy",
            color="navy")
    ax.plot(equity.index, equity["benchmark"], label="Equal weight buy & hold",
            color="darkorange")
    ax.axvline(split_date, color="gray", ls="--", lw=1,
               label="Walk-forward split (train | test)")
    ax.set_title("Cumulative growth of $1")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    out = config.FIGURES_DIR / "backtest_cumulative.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def plot_drawdown(returns: pd.DataFrame) -> Path:
    """Plot strategy and benchmark drawdowns.

    Args:
        returns: Daily return frame.

    Returns:
        Path of the saved PNG figure.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for column, color in [("strategy", "navy"), ("benchmark", "darkorange")]:
        equity = (1 + returns[column]).cumprod()
        drawdown = (equity / equity.cummax() - 1) * 100
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.4, color=color,
                        label=column)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    out = config.FIGURES_DIR / "backtest_drawdown.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def plot_monthly_heatmap(returns: pd.DataFrame) -> Path:
    """Plot a year x month heatmap of strategy returns.

    Args:
        returns: Daily return frame.

    Returns:
        Path of the saved PNG figure.
    """
    monthly = (1 + returns["strategy"]).resample("ME").prod() - 1
    table = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.to_numpy() * 100,
    }).pivot(index="year", columns="month", values="ret")
    table.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][: len(table.columns)] if len(table.columns) == 12 else table.columns
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(table, annot=True, fmt=".1f", center=0, cmap="RdYlGn",
                cbar_kws={"label": "Monthly return (%)"}, ax=ax)
    ax.set_title("Strategy monthly returns (%)")
    fig.tight_layout()
    out = config.FIGURES_DIR / "monthly_returns_heatmap.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def main() -> pd.DataFrame:
    """Run the full backtest step.

    Returns:
        The metric table (also written to
        ``data/processed/backtest_metrics.csv``).
    """
    config.ensure_directories()
    logger.info("=== Backtest started ===")
    panel = load_spreads()
    returns, trades = backtest_portfolio(panel)

    trades_path = config.DATA_PROCESSED_DIR / "trades.csv"
    trades.to_csv(trades_path, index=False)
    logger.info("Saved trade log %s (%d trades)", trades_path, len(trades))

    returns_path = config.DATA_PROCESSED_DIR / "backtest_returns.csv"
    returns.to_csv(returns_path, index_label="date")
    logger.info("Saved daily returns %s (%d rows)", returns_path, len(returns))

    metrics = build_metric_table(returns, trades)
    metrics_path = config.DATA_PROCESSED_DIR / "backtest_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    logger.info("Saved metrics %s", metrics_path)
    for _, row in metrics[metrics["label"] == "strategy"].iterrows():
        logger.info("[%s] strategy: ann ret %.2f%%, Sharpe %.2f, maxDD %.2f%%",
                    row["window"], row["ann_return_pct"], row["sharpe"],
                    row["max_drawdown_pct"])

    plot_cumulative(returns)
    plot_drawdown(returns)
    plot_monthly_heatmap(returns)
    logger.info("=== Backtest complete ===")
    return metrics


if __name__ == "__main__":
    main()
