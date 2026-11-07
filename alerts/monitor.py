"""Live dislocation monitor and alert scheduler.

Fetches the latest prices on demand via yfinance, recomputes each ETF's
rolling premium z-score over the most recent 63 trading days, and compares
it against the configured threshold (default 2.0 sigma).  New crossings are
sent to the configured channels via :mod:`alerts.notifier`.

Deduplication: an ETF that already triggered an alert within the last
``config.ALERT_DEDUP_DAYS`` (3) days is not re-alerted; the check is still
logged with action ``deduplicated``.

Every check (alerted or not) is appended to ``alerts/alert_log.csv`` with
``timestamp, etf, zscore, premium_pct, threshold, action``.

Scheduling: APScheduler runs :func:`run_check` daily at 4:15 PM US/Eastern.
``pandas_market_calendars`` (NYSE calendar) gates execution so non-trading
days are skipped (and logged as such).

NAV note: like ``data/fetch.py``, the monitor simulates NAV with the same
seeded AR(1) premium process when a live NAV feed is unavailable, so live
z-scores are methodologically consistent with the study.  Wire in a real
NAV source for production monitoring.

Inputs:
    Network access to Yahoo Finance; channel credentials via environment
    variables (see ``.env.example``).

Outputs:
    ``alerts/alert_log.csv`` plus email/Slack side effects.

Usage:
    python alerts/monitor.py            # start the daily scheduler loop
    python alerts/monitor.py --once     # run a single check now and exit
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from alerts.notifier import Alert, send_alert
from data.fetch import fetch_etf_prices, simulate_nav

logger = config.get_logger(__name__)

LOG_COLUMNS = ["timestamp", "etf", "zscore", "premium_pct", "threshold", "action"]


def is_trading_day(when: datetime) -> bool:
    """Check whether a date is an NYSE trading day.

    Args:
        when: Datetime to check (only the date part matters).

    Returns:
        ``True`` if the NYSE is open on that date.  Falls back to a
        weekday check (with a warning) if the calendar lookup fails.
    """
    try:
        import pandas_market_calendars as mcal

        calendar = mcal.get_calendar(config.EXCHANGE_CALENDAR)
        day = pd.Timestamp(when.date())
        schedule = calendar.schedule(start_date=day, end_date=day)
        return not schedule.empty
    except Exception as exc:  # noqa: BLE001 -- degrade to weekday heuristic
        logger.warning("Market calendar lookup failed (%s); falling back to "
                       "weekday check", exc)
        return when.weekday() < 5


def load_alert_log() -> pd.DataFrame:
    """Load the alert log CSV, returning an empty frame if absent.

    Returns:
        DataFrame with the columns in ``LOG_COLUMNS`` and parsed
        timestamps.
    """
    if config.ALERT_LOG_PATH.exists():
        log = pd.read_csv(config.ALERT_LOG_PATH, parse_dates=["timestamp"])
        return log
    return pd.DataFrame(columns=LOG_COLUMNS)


def append_alert_log(rows: list[dict]) -> None:
    """Append check results to ``alerts/alert_log.csv``.

    Args:
        rows: List of dicts with the keys in ``LOG_COLUMNS``.

    Returns:
        None.
    """
    if not rows:
        return
    config.ensure_directories()
    log = load_alert_log()
    log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True)
    log.to_csv(config.ALERT_LOG_PATH, index=False)
    logger.info("Appended %d rows to %s", len(rows), config.ALERT_LOG_PATH)


def recently_alerted(log: pd.DataFrame, ticker: str, now: datetime) -> bool:
    """Check the dedup window: was this ETF alerted in the last N days?

    Args:
        log: Existing alert log frame.
        ticker: ETF ticker symbol.
        now: Current timestamp.

    Returns:
        ``True`` when an ``alerted`` action for this ETF exists within the
        last ``config.ALERT_DEDUP_DAYS`` days.
    """
    if log.empty:
        return False
    cutoff = pd.Timestamp(now) - timedelta(days=config.ALERT_DEDUP_DAYS)
    recent = log[(log["etf"] == ticker)
                 & (log["action"] == "alerted")
                 & (pd.to_datetime(log["timestamp"]) >= cutoff)]
    return not recent.empty


def latest_zscore(ticker: str) -> tuple[float, float, pd.Series]:
    """Fetch fresh data and compute the current premium z-score for one ETF.

    Downloads roughly six months of daily closes (enough to fill the 63-day
    rolling window), simulates NAV with the study's seeded AR(1) process,
    and computes the rolling z-score of the premium over the most recent
    ``config.ZSCORE_WINDOW`` trading days.

    Args:
        ticker: ETF ticker symbol.

    Returns:
        Tuple ``(zscore, premium_pct, recent_zscores)`` for the latest
        observation, where ``recent_zscores`` is the trailing z-score
        series used for the inline alert chart.

    Raises:
        RuntimeError: If not enough observations are available to fill the
            rolling window.
    """
    end = datetime.now()
    start = end - timedelta(days=320)  # ~ 220 trading days of buffer
    prices = fetch_etf_prices(ticker, start, end)
    if len(prices) < config.ZSCORE_WINDOW + 5:
        raise RuntimeError(
            f"Only {len(prices)} observations for {ticker}; need at least "
            f"{config.ZSCORE_WINDOW + 5}"
        )
    nav = simulate_nav(prices["Close"], ticker)
    premium = (prices["Close"] - nav) / nav * 100.0
    mean = premium.rolling(config.ZSCORE_WINDOW).mean()
    std = premium.rolling(config.ZSCORE_WINDOW).std()
    zscore = (premium - mean) / std
    return (
        float(zscore.iloc[-1]),
        float(premium.iloc[-1]),
        zscore.tail(config.ZSCORE_WINDOW).dropna(),
    )


def run_check(force: bool = False) -> list[dict]:
    """Run one monitoring pass across the whole ETF universe.

    Args:
        force: When ``True``, run even on non-trading days (used by
            ``--once`` for manual checks).

    Returns:
        The list of log rows written for this pass.
    """
    now = datetime.now()
    if not force and not is_trading_day(now):
        logger.info("%s is not a trading day; skipping check", now.date())
        return []

    logger.info("=== Monitoring pass started (threshold %.1f sigma) ===",
                config.ALERT_THRESHOLD)
    log = load_alert_log()
    rows: list[dict] = []

    for ticker in config.ETFS:
        try:
            zscore, premium_pct, recent = latest_zscore(ticker)
        except Exception as exc:  # noqa: BLE001 -- one ETF must not stop the rest
            logger.error("Check failed for %s: %s", ticker, exc)
            rows.append({"timestamp": now, "etf": ticker, "zscore": float("nan"),
                         "premium_pct": float("nan"),
                         "threshold": config.ALERT_THRESHOLD,
                         "action": f"error: {exc}"})
            continue

        crossed = abs(zscore) >= config.ALERT_THRESHOLD
        if not crossed:
            action = "no_crossing"
        elif recently_alerted(log, ticker, now):
            action = "deduplicated"
            logger.info("%s crossed (z=%.2f) but was alerted within the last "
                        "%d days; suppressing", ticker, zscore,
                        config.ALERT_DEDUP_DAYS)
        else:
            direction = "discount" if zscore < 0 else "premium"
            results = send_alert(Alert(
                ticker=ticker, zscore=zscore, premium_pct=premium_pct,
                threshold=config.ALERT_THRESHOLD, direction=direction,
                recent_zscores=recent,
            ))
            # Record plain "alerted" so dedup matching is channel-agnostic.
            action = "alerted"
            logger.info("ALERT %s: z=%.2f premium=%.3f%% channels=%s",
                        ticker, zscore, premium_pct, results)

        rows.append({"timestamp": now, "etf": ticker, "zscore": zscore,
                     "premium_pct": premium_pct,
                     "threshold": config.ALERT_THRESHOLD, "action": action})
        logger.info("%s: z=%.2f premium=%.3f%% -> %s",
                    ticker, zscore, premium_pct, action)

    append_alert_log(rows)
    logger.info("=== Monitoring pass complete ===")
    return rows


def start_scheduler() -> None:
    """Start the blocking APScheduler loop.

    Schedules :func:`run_check` every weekday at
    ``config.MONITOR_HOUR:config.MONITOR_MINUTE`` US/Eastern; the trading
    day gate inside :func:`run_check` skips holidays.  Blocks until
    interrupted (Ctrl+C).

    Returns:
        None.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone=config.MONITOR_TIMEZONE)
    scheduler.add_job(
        run_check,
        CronTrigger(day_of_week="mon-fri", hour=config.MONITOR_HOUR,
                    minute=config.MONITOR_MINUTE,
                    timezone=config.MONITOR_TIMEZONE),
        id="daily_dislocation_check",
        name="Daily ETF dislocation check",
    )
    logger.info("Scheduler started: daily check at %02d:%02d %s "
                "(trading days only). Ctrl+C to stop.",
                config.MONITOR_HOUR, config.MONITOR_MINUTE,
                config.MONITOR_TIMEZONE)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


def main() -> None:
    """CLI entry point: ``--once`` for a single pass, default scheduler loop.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="ETF dislocation monitor")
    parser.add_argument("--once", action="store_true",
                        help="run a single check immediately and exit "
                             "(ignores the trading-day gate)")
    args = parser.parse_args()
    config.ensure_directories()
    if args.once:
        run_check(force=True)
    else:
        start_scheduler()


if __name__ == "__main__":
    main()
