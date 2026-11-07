"""Central configuration for the ETF NAV arbitrage study.

This module defines every shared setting used across the project:

* Universe of fixed income ETFs and auxiliary market data tickers.
* Analysis parameters (z-score window, dislocation thresholds, event-study
  horizons, bootstrap settings).
* Backtest parameters (entry/exit rules, position sizing, transaction costs,
  walk-forward split).
* Live alerting configuration (channels, schedule, deduplication window).
* Filesystem layout built with :mod:`pathlib` -- no hardcoded path strings
  are used anywhere else in the codebase.

All credentials (SMTP, Slack webhook, FRED API key) are loaded from
environment variables via ``python-dotenv``.  Nothing secret is ever stored
in source control; see ``.env.example`` for the variable names.

Inputs:
    Environment variables, optionally loaded from a ``.env`` file located in
    the project root.

Outputs:
    Module-level constants imported by every other module, plus the
    :func:`ensure_directories` and :func:`get_logger` helpers.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Filesystem layout (pathlib throughout -- no hardcoded path strings)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PAPER_OUTPUT_DIR = OUTPUTS_DIR / "paper"
ALERTS_DIR = PROJECT_ROOT / "alerts"
ALERT_LOG_PATH = ALERTS_DIR / "alert_log.csv"

# Load environment variables from .env in the project root (if present).
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Universe and data sources
# ---------------------------------------------------------------------------
ETFS = ["HYG", "LQD", "JNK", "TLT", "AGG"]

ETF_DESCRIPTIONS = {
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "JNK": "SPDR Bloomberg High Yield Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
}

VIX_TICKER = "^VIX"

# FRED series identifiers for ICE BofA option-adjusted spreads.
FRED_HY_OAS_SERIES = "BAMLH0A0HYM2"  # US High Yield Master II OAS
FRED_IG_OAS_SERIES = "BAMLC0A0CM"    # US Corporate Master (IG) OAS
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Sample length in years of daily data.
LOOKBACK_YEARS = 5

# ---------------------------------------------------------------------------
# NAV simulation (used when a live NAV feed is unavailable)
# ---------------------------------------------------------------------------
# The simulated premium follows an AR(1) process:
#     p_t = NAV_AR1_PHI * p_{t-1} + eps_t,   eps_t ~ N(0, NAV_AR1_SIGMA^2)
# and NAV_t = Price_t / (1 + p_t).
NAV_AR1_PHI = 0.85
NAV_AR1_SIGMA = 0.003
NAV_SIM_SEED = 42  # deterministic so the study is reproducible

# ---------------------------------------------------------------------------
# Spread / dislocation analysis
# ---------------------------------------------------------------------------
ZSCORE_WINDOW = 63  # trading days (~one quarter)
DISLOCATION_THRESHOLDS = [1.5, 2.0, 2.5]  # sigma levels for sensitivity study

# Stress episodes annotated on charts and in the paper.
STRESS_EPISODES = {
    "COVID-19 crash (Mar 2020)": ("2020-03-01", "2020-03-31"),
    "Gilt/LDI stress (Sep 2022)": ("2022-09-15", "2022-09-30"),
    "Rates vol spike (Oct 2022)": ("2022-10-01", "2022-10-31"),
}

# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------
FORWARD_HORIZONS = [1, 3, 5, 10, 20]  # trading days
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_CI = 0.95
EVENT_COOLDOWN_DAYS = 5     # minimum spacing between events per ETF/threshold
REVERSION_HORIZON = 20      # max days tracked when measuring reversion time
FULL_REVERSION_DAYS = 10    # window used for "% fully reverted" statistic

# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
VIX_LOW_MAX = 15.0    # VIX <= 15        -> "low"
VIX_HIGH_MIN = 25.0   # VIX >  25        -> "high"; otherwise "medium"

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
ENTRY_ZSCORE = 2.0          # |z| entry trigger
MAX_HOLDING_DAYS = 20       # time stop
VOL_WINDOW = 21             # rolling window for inverse-vol sizing
MAX_POSITION_MULTIPLE = 2.0  # cap relative to average position size
TRANSACTION_COST_BPS = 5.0  # one-way cost per trade, basis points
TRAIN_YEARS = 3             # walk-forward: first 3 years train, last 2 test
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Live alerting
# ---------------------------------------------------------------------------
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "2.0"))
ALERT_DEDUP_DAYS = 3  # do not re-alert the same ETF within this many days

# Channels can be toggled independently.
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() in ("1", "true", "yes")
SLACK_ENABLED = os.getenv("SLACK_ENABLED", "false").lower() in ("1", "true", "yes")

# SMTP settings (Gmail-compatible).  All values come from the environment.
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", SMTP_USER)
ALERT_EMAIL_TO = [
    addr.strip()
    for addr in os.getenv("ALERT_EMAIL_TO", "").split(",")
    if addr.strip()
]

# Slack incoming webhook URL.
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# Link embedded in alert messages pointing at the Streamlit dashboard.
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://localhost:8501")

# Monitoring schedule: daily at market close + 15 minutes, US/Eastern.
MONITOR_HOUR = 16
MONITOR_MINUTE = 15
MONITOR_TIMEZONE = "America/New_York"
EXCHANGE_CALENDAR = "NYSE"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def ensure_directories() -> None:
    """Create every output directory the project writes to.

    Idempotent: safe to call from any module at import or run time.

    Returns:
        None.
    """
    for directory in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        FIGURES_DIR,
        PAPER_OUTPUT_DIR,
        ALERTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    Configures the root handler once (stream handler to stderr with a
    timestamped format) and returns a child logger.  Used by every module in
    place of ``print``.

    Args:
        name: Logger name, conventionally the module's ``__name__``.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return logging.getLogger(name)
