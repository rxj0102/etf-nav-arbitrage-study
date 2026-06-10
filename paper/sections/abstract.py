"""Abstract section of the research paper.

Builds the abstract as a list of ReportLab flowables.  Key statistics
(event counts, hit rate, Sharpe ratio) are pulled dynamically from
``data/processed/`` so the abstract updates automatically whenever the
analysis is rerun.

Inputs:
    ``data/processed/reversion_summary.csv`` and
    ``data/processed/backtest_metrics.csv`` (optional -- placeholders are
    used when missing).

Outputs:
    :func:`build` returns ``list[Flowable]`` consumed by
    ``paper/generate_paper.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.platypus import Flowable, Paragraph, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

logger = config.get_logger(__name__)


def _key_numbers() -> dict:
    """Load headline numbers from the processed outputs.

    Returns:
        Dict with ``n_events``, ``pct_reverted``, ``median_days``,
        ``sharpe`` and ``ann_ret`` keys; string placeholders are
        substituted when the CSVs are unavailable.
    """
    numbers = {"n_events": "[N]", "pct_reverted": "[X]", "median_days": "[D]",
               "sharpe": "[S]", "ann_ret": "[R]"}
    try:
        summary = pd.read_csv(config.DATA_PROCESSED_DIR / "reversion_summary.csv")
        at_threshold = summary[summary["threshold"] == config.ALERT_THRESHOLD]
        numbers["n_events"] = str(int(at_threshold["n_events"].sum()))
        numbers["pct_reverted"] = f"{at_threshold['pct_reverted_within_10d'].mean():.0f}"
        numbers["median_days"] = f"{at_threshold['median_reversion_days'].median():.0f}"
    except Exception as exc:  # noqa: BLE001 -- placeholders are acceptable
        logger.warning("Abstract: reversion summary unavailable (%s)", exc)
    try:
        metrics = pd.read_csv(config.DATA_PROCESSED_DIR / "backtest_metrics.csv")
        row = metrics[(metrics["window"] == "full")
                      & (metrics["label"] == "strategy")].iloc[0]
        numbers["sharpe"] = f"{row['sharpe']:.2f}"
        numbers["ann_ret"] = f"{row['ann_return_pct']:.1f}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Abstract: backtest metrics unavailable (%s)", exc)
    return numbers


def build(styles: dict) -> list[Flowable]:
    """Build the abstract flowables.

    Args:
        styles: Shared paragraph style dict from ``generate_paper.py``.

    Returns:
        List of ReportLab flowables.
    """
    n = _key_numbers()
    text = (
        "We study mean reversion in the premium/discount of five large fixed "
        "income exchange-traded funds (HYG, LQD, JNK, TLT, AGG) over a "
        "five-year daily sample. We hypothesize that dislocations between "
        "ETF price and net asset value (NAV), normalized as 63-day rolling "
        "z-scores, revert predictably as authorized participants arbitrage "
        f"the basis. Across {n['n_events']} dislocation events at the "
        "2.0-sigma threshold, premiums and discounts revert toward zero with "
        f"a median reversion time of {n['median_days']} trading days, and "
        f"{n['pct_reverted']}% of events fully revert within ten days. "
        "Reversion is fastest, but dislocations largest, in high-volatility "
        "and wide-credit-spread regimes. A simple long-discount/short-premium "
        "rule with inverse-volatility sizing and 5 bp costs earns an "
        f"annualized {n['ann_ret']}% with a Sharpe ratio of {n['sharpe']} "
        "in walk-forward evaluation. NAV series are simulated with a "
        "calibrated AR(1) premium process where live NAVs are unavailable, "
        "so results demonstrate methodology rather than live arbitrage "
        "profits."
    )
    return [
        Paragraph("Abstract", styles["heading"]),
        Paragraph(text, styles["abstract"]),
        Spacer(1, 12),
    ]
