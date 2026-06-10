"""Methodology section of the research paper.

Documents the premium/discount construction, z-score normalization, event
study design, regime classification and backtest design (including
transaction costs and the walk-forward split).  Parameter values are read
from ``config.py`` so the text always matches the code.

Outputs:
    :func:`build` returns ``list[Flowable]`` consumed by
    ``paper/generate_paper.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.platypus import Flowable, Paragraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config


def build(styles: dict) -> list[Flowable]:
    """Build the methodology flowables.

    Args:
        styles: Shared paragraph style dict from ``generate_paper.py``.

    Returns:
        List of ReportLab flowables.
    """
    thresholds = ", ".join(f"{t}" for t in config.DISLOCATION_THRESHOLDS)
    horizons = ", ".join(str(h) for h in config.FORWARD_HORIZONS)
    return [
        Paragraph("3. Methodology", styles["heading"]),

        Paragraph("3.1 Premium/discount construction", styles["subheading"]),
        Paragraph(
            "For each fund i and day t the premium/discount is "
            "P<sub>i,t</sub> = (Price<sub>i,t</sub> − NAV<sub>i,t</sub>) / "
            "NAV<sub>i,t</sub> × 100, expressed in percent. Positive values "
            "indicate the ETF trades rich to its portfolio (a premium); "
            "negative values indicate a discount.",
            styles["body"],
        ),

        Paragraph("3.2 Z-score normalization", styles["subheading"]),
        Paragraph(
            "Premium levels are not comparable across funds—high yield "
            "ETFs habitually trade at small premiums while Treasury ETFs "
            "hug NAV—so we normalize each series with a rolling z-score: "
            "z<sub>i,t</sub> = (P<sub>i,t</sub> − μ<sub>i,t</sub>) / "
            f"σ<sub>i,t</sub>, where μ and σ are {config.ZSCORE_WINDOW}-day "
            "(one quarter) rolling means and standard deviations. The "
            "rolling window adapts to slow-moving structural premiums and "
            "isolates abnormal dislocations.",
            styles["body"],
        ),

        Paragraph("3.3 Event study design", styles["subheading"]),
        Paragraph(
            f"Dislocation events are defined at the ±{thresholds} sigma "
            "thresholds (all three retained as a sensitivity check). An "
            "event occurs on the first day the z-score moves beyond a "
            "threshold from inside the band, with a "
            f"{config.EVENT_COOLDOWN_DAYS}-day cooldown so a single "
            "prolonged episode is counted once. Discount events "
            "(z &lt; −k) and premium events (z &gt; +k) are analyzed "
            f"separately. For each event we record forward returns at {horizons} "
            "trading-day horizons; inference uses one-sample t-tests "
            "against zero and 95% confidence intervals from "
            f"{config.BOOTSTRAP_SAMPLES:,} bootstrap resamples of the "
            "event-level returns. We additionally report the median time "
            "for the z-score to cross zero, the share of events fully "
            f"reverted within {config.FULL_REVERSION_DAYS} days, and the "
            "average peak dislocation reached during each episode.",
            styles["body"],
        ),

        Paragraph("3.4 Regime classification", styles["subheading"]),
        Paragraph(
            "Each event is assigned to a volatility regime by the VIX "
            f"level on the event day—low (VIX ≤ {config.VIX_LOW_MAX:.0f}), "
            f"medium ({config.VIX_LOW_MAX:.0f} &lt; VIX ≤ "
            f"{config.VIX_HIGH_MIN:.0f}) and high (VIX &gt; "
            f"{config.VIX_HIGH_MIN:.0f})—and to a credit regime by a "
            "median split of the high yield OAS (tight vs. wide). Within "
            "each regime we compute reversion speed, dislocation "
            "magnitude and the hit rate of the mean-reversion sign at the "
            "five-day horizon, and we test equality of forward returns "
            "across regimes with one-way ANOVA.",
            styles["body"],
        ),

        Paragraph("3.5 Backtest design", styles["subheading"]),
        Paragraph(
            "The trading rule is deliberately simple. Enter long when "
            f"z &lt; −{config.ENTRY_ZSCORE:.0f}; enter short when z &gt; "
            f"+{config.ENTRY_ZSCORE:.0f}; exit when the z-score crosses "
            f"zero or after {config.MAX_HOLDING_DAYS} trading days. "
            "Positions are sized by inverse "
            f"{config.VOL_WINDOW}-day rolling volatility, normalized to a "
            "unit average and capped at "
            f"{config.MAX_POSITION_MULTIPLE:.0f}× the average position. "
            "Signals decided at the close of day t are implemented at the "
            "close of day t+1 (no look-ahead), and "
            f"{config.TRANSACTION_COST_BPS:.0f} basis points of one-way "
            "transaction costs are charged on all turnover. The portfolio "
            "equally weights the five per-ETF strategies; the benchmark "
            "is buy-and-hold equal weight across the same funds. "
            f"Walk-forward validation reserves the first {config.TRAIN_YEARS} "
            "years as the design window and the final two years as a "
            "pseudo out-of-sample window; no parameters are re-optimized, "
            "so the split measures the stability of the fixed rule.",
            styles["body"],
        ),
    ]
