"""Data acquisition for the ETF NAV arbitrage study.

Downloads five years of daily OHLCV data for the fixed income ETF universe
(HYG, LQD, JNK, TLT, AGG) and the VIX index via ``yfinance``, and ICE BofA
credit option-adjusted spreads (high yield ``BAMLH0A0HYM2`` and investment
grade ``BAMLC0A0CM``) from FRED via ``fredapi``.

NAV simulation
--------------
Official end-of-day NAVs are not available through free APIs, so this module
*simulates* NAV as the ETF close price adjusted by a mean-reverting noise
term.  The simulated premium fraction ``p_t`` follows an AR(1) process::

    p_t = phi * p_{t-1} + eps_t,   eps_t ~ N(0, sigma^2)

with ``phi = 0.85`` and ``sigma = 0.003`` (see ``config.NAV_AR1_PHI`` /
``config.NAV_AR1_SIGMA``), and ``NAV_t = Close_t / (1 + p_t)``.  This
calibration reproduces the persistence and scale of observed fixed income
ETF premiums.  The simulation is seeded so the entire study is reproducible.
**All downstream results therefore demonstrate the methodology rather than
live market dislocations** -- swap in a real NAV feed (e.g. from the issuer
or a market data vendor) for production use.  This caveat is repeated in the
README and in the paper's data section.

Offline fallback
----------------
If yfinance or FRED are unreachable (no network, no API key), the module
falls back to internally generated synthetic price/VIX/OAS series so the
full pipeline still runs end to end.  Every fallback is logged loudly.

Inputs:
    Network access to Yahoo Finance and (optionally) a ``FRED_API_KEY``
    environment variable.

Outputs:
    CSV files in ``data/raw/`` with a ``Date`` datetime index:
    ``prices_<TICKER>.csv`` (OHLCV + simulated ``NAV``), ``vix.csv``
    (column ``VIX``) and ``oas.csv`` (columns ``HY_OAS``, ``IG_OAS``).

Usage:
    python data/fetch.py
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

logger = config.get_logger(__name__)


def _stable_seed(*parts: object) -> int:
    """Derive a process-independent RNG seed from arbitrary parts.

    Python's builtin ``hash`` is salted per process, so it must never be
    used for reproducible simulations; this helper hashes the string form
    of the parts with MD5 instead.

    Args:
        *parts: Values identifying the stream (e.g. ticker, date).

    Returns:
        A 32-bit integer seed.
    """
    digest = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _date_range() -> tuple[datetime, datetime]:
    """Return the (start, end) datetimes covering the sample period.

    Returns:
        Tuple of start and end ``datetime`` objects spanning
        ``config.LOOKBACK_YEARS`` years up to today.
    """
    end = datetime.now()
    start = end - timedelta(days=int(config.LOOKBACK_YEARS * 365.25) + 7)
    return start, end


def _synthetic_business_index(start: datetime, end: datetime) -> pd.DatetimeIndex:
    """Build a business-day index used by the offline fallbacks.

    Args:
        start: First date of the index.
        end: Last date of the index.

    Returns:
        ``pd.DatetimeIndex`` of business days named ``Date``.
    """
    idx = pd.bdate_range(start=start.date(), end=end.date(), name="Date")
    return idx


def _synthetic_prices(ticker: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate a synthetic OHLCV frame when Yahoo Finance is unreachable.

    Prices follow a geometric random walk with mild drift and
    ticker-dependent volatility; volume is lognormal.  Used only as an
    offline fallback and clearly logged when active.

    Args:
        ticker: ETF ticker symbol (seeds the RNG for reproducibility).
        index: Datetime index for the generated series.

    Returns:
        DataFrame with columns ``Open, High, Low, Close, Volume`` indexed
        by ``index``.
    """
    rng = np.random.default_rng(_stable_seed("synthetic-prices", ticker))
    n = len(index)
    base = {"HYG": 78.0, "LQD": 110.0, "JNK": 95.0, "TLT": 100.0, "AGG": 100.0}.get(
        ticker, 100.0
    )
    vol = {"HYG": 0.006, "LQD": 0.006, "JNK": 0.006, "TLT": 0.010, "AGG": 0.004}.get(
        ticker, 0.006
    )
    rets = rng.normal(0.00005, vol, n)
    close = base * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0, vol / 2, n))
    frame = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, vol / 3, n)),
            "High": close * (1 + spread),
            "Low": close * (1 - spread),
            "Close": close,
            "Volume": np.round(rng.lognormal(15, 0.4, n)),
        },
        index=index,
    )
    return frame


def fetch_etf_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch daily OHLCV history for one ETF via yfinance.

    Falls back to :func:`_synthetic_prices` if the download fails or returns
    an empty frame, so the pipeline remains runnable offline.

    Args:
        ticker: ETF ticker symbol, e.g. ``"HYG"``.
        start: Start of the sample period.
        end: End of the sample period.

    Returns:
        DataFrame with columns ``Open, High, Low, Close, Volume`` and a
        datetime index named ``Date``.
    """
    logger.info("Fetching OHLCV for %s from Yahoo Finance ...", ticker)
    try:
        import yfinance as yf

        frame = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
        if frame is None or frame.empty:
            raise RuntimeError(f"yfinance returned no rows for {ticker}")
        frame = frame[["Open", "High", "Low", "Close", "Volume"]].copy()
        # Yahoo can return a NaN close for the current, incomplete session.
        frame = frame.dropna(subset=["Close"])
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame.index.name = "Date"
        logger.info("Fetched %d rows for %s (%s to %s)", len(frame), ticker,
                    frame.index[0].date(), frame.index[-1].date())
        return frame
    except Exception as exc:  # noqa: BLE001 -- any failure triggers fallback
        logger.warning("Live fetch failed for %s (%s); using SYNTHETIC prices.",
                       ticker, exc)
        return _synthetic_prices(ticker, _synthetic_business_index(start, end))


def fetch_vix(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch daily VIX closes via yfinance, with a synthetic fallback.

    Args:
        start: Start of the sample period.
        end: End of the sample period.

    Returns:
        DataFrame with a single ``VIX`` column and datetime index ``Date``.
    """
    logger.info("Fetching VIX (%s) from Yahoo Finance ...", config.VIX_TICKER)
    try:
        import yfinance as yf

        frame = yf.download(
            config.VIX_TICKER,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
        if frame is None or frame.empty:
            raise RuntimeError("yfinance returned no rows for VIX")
        vix = frame[["Close"]].rename(columns={"Close": "VIX"})
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        vix.index.name = "Date"
        logger.info("Fetched %d VIX observations", len(vix))
        return vix
    except Exception as exc:  # noqa: BLE001
        logger.warning("VIX fetch failed (%s); using SYNTHETIC VIX.", exc)
        idx = _synthetic_business_index(start, end)
        rng = np.random.default_rng(7)
        # Mean-reverting log-VIX around ~18 with occasional spikes.
        log_vix = np.empty(len(idx))
        log_vix[0] = np.log(18.0)
        for i in range(1, len(idx)):
            shock = rng.normal(0, 0.05) + (rng.random() < 0.01) * rng.normal(0.4, 0.1)
            log_vix[i] = 0.98 * log_vix[i - 1] + 0.02 * np.log(18.0) + shock
        return pd.DataFrame({"VIX": np.exp(log_vix)}, index=idx)


def fetch_oas(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch HY and IG credit OAS series from FRED via fredapi.

    Requires the ``FRED_API_KEY`` environment variable.  If the key is
    missing or the request fails, generates synthetic mean-reverting OAS
    series so downstream regime analysis still runs.

    Args:
        start: Start of the sample period.
        end: End of the sample period.

    Returns:
        DataFrame with columns ``HY_OAS`` and ``IG_OAS`` (percentage
        points) and datetime index ``Date``.
    """
    logger.info("Fetching credit OAS series from FRED (%s, %s) ...",
                config.FRED_HY_OAS_SERIES, config.FRED_IG_OAS_SERIES)
    try:
        if not config.FRED_API_KEY:
            raise RuntimeError("FRED_API_KEY not set")
        from fredapi import Fred

        fred = Fred(api_key=config.FRED_API_KEY)
        hy = fred.get_series(config.FRED_HY_OAS_SERIES,
                             observation_start=start, observation_end=end)
        ig = fred.get_series(config.FRED_IG_OAS_SERIES,
                             observation_start=start, observation_end=end)
        oas = pd.DataFrame({"HY_OAS": hy, "IG_OAS": ig})
        oas.index = pd.to_datetime(oas.index)
        oas.index.name = "Date"
        oas = oas.dropna(how="all")
        logger.info("Fetched %d OAS observations from FRED", len(oas))
        return oas
    except Exception as exc:  # noqa: BLE001
        logger.warning("FRED fetch failed (%s); using SYNTHETIC OAS series.", exc)
        idx = _synthetic_business_index(start, end)
        rng = np.random.default_rng(11)
        n = len(idx)
        hy = np.empty(n)
        ig = np.empty(n)
        hy[0], ig[0] = 4.0, 1.2
        for i in range(1, n):
            common = rng.normal(0, 0.04)
            hy[i] = max(2.0, 0.99 * hy[i - 1] + 0.01 * 4.0 + common + rng.normal(0, 0.03))
            ig[i] = max(0.6, 0.99 * ig[i - 1] + 0.01 * 1.2 + 0.3 * common + rng.normal(0, 0.01))
        return pd.DataFrame({"HY_OAS": hy, "IG_OAS": ig}, index=idx)


def simulate_nav(close: pd.Series, ticker: str) -> pd.Series:
    """Simulate a NAV series from close prices with AR(1) premium noise.

    The simulated premium fraction follows
    ``p_t = phi * p_{t-1} + eps_t`` with ``eps_t ~ N(0, sigma^2)``, and the
    NAV is recovered as ``NAV_t = Close_t / (1 + p_t)``, so that the observed
    premium/discount ``(Price - NAV) / NAV`` equals ``p_t`` exactly.

    Each day's innovation is seeded from ``(config.NAV_SIM_SEED, ticker,
    date)`` via a stable hash, so the premium path for a given calendar date
    is identical across runs *and across fetch windows* (the influence of
    the initial condition decays as ``phi**t``).  This keeps the live
    monitor's z-scores consistent with the historical study.

    **Important:** this is a *simulation* used because official NAVs are not
    freely available; see the module docstring and README.

    Args:
        close: Daily close prices indexed by date.
        ticker: ETF ticker (part of the per-date innovation seed).

    Returns:
        Simulated NAV series aligned to ``close``.
    """
    n = len(close)
    sigma = config.NAV_AR1_SIGMA
    phi = config.NAV_AR1_PHI
    eps = np.array([
        np.random.default_rng(
            _stable_seed(config.NAV_SIM_SEED, ticker, pd.Timestamp(day).date())
        ).normal(0, sigma)
        for day in close.index
    ])
    premium = np.empty(n)
    # Scale the first innovation to the stationary AR(1) variance.
    premium[0] = eps[0] / np.sqrt(1 - phi**2)
    for i in range(1, n):
        premium[i] = phi * premium[i - 1] + eps[i]
    nav = close.to_numpy(dtype=float) / (1.0 + premium)
    logger.info(
        "Simulated NAV for %s: AR(1) phi=%.2f sigma=%.4f, premium std=%.3f%%",
        ticker, phi, sigma, premium.std() * 100,
    )
    return pd.Series(nav, index=close.index, name="NAV")


def main() -> None:
    """Run the full data acquisition step and write CSVs to ``data/raw/``.

    Returns:
        None.  Side effect: writes ``prices_<TICKER>.csv`` for each ETF,
        ``vix.csv`` and ``oas.csv``.
    """
    config.ensure_directories()
    start, end = _date_range()
    logger.info("=== Data fetch started: sample %s to %s ===",
                start.date(), end.date())

    for ticker in config.ETFS:
        prices = fetch_etf_prices(ticker, start, end)
        prices["NAV"] = simulate_nav(prices["Close"], ticker)
        out_path = config.DATA_RAW_DIR / f"prices_{ticker}.csv"
        prices.to_csv(out_path, index_label="Date")
        logger.info("Saved %s (%d rows)", out_path, len(prices))

    vix = fetch_vix(start, end)
    vix_path = config.DATA_RAW_DIR / "vix.csv"
    vix.to_csv(vix_path, index_label="Date")
    logger.info("Saved %s (%d rows)", vix_path, len(vix))

    oas = fetch_oas(start, end)
    oas_path = config.DATA_RAW_DIR / "oas.csv"
    oas.to_csv(oas_path, index_label="Date")
    logger.info("Saved %s (%d rows)", oas_path, len(oas))

    logger.info("=== Data fetch complete ===")


if __name__ == "__main__":
    main()
