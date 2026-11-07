"""Introduction section of the research paper.

Covers the ETF arbitrage mechanism, the role of authorized participants,
the motivation for studying fixed income ETF dislocations, and the
literature context (Petajisto 2017; Ben-David, Franzoni and Moussawi 2018,
among others).

Outputs:
    :func:`build` returns ``list[Flowable]`` consumed by
    ``paper/generate_paper.py``.
"""

from __future__ import annotations

from reportlab.platypus import Flowable, Paragraph

PARAGRAPHS = [
    (
        "Exchange-traded funds are open-ended vehicles whose share count "
        "adjusts through an in-kind creation and redemption mechanism. When "
        "an ETF trades rich to the value of its underlying portfolio, "
        "authorized participants (APs) can deliver the underlying basket to "
        "the sponsor, receive newly created ETF shares, and sell them at the "
        "premium; when the fund trades at a discount, APs redeem shares for "
        "the basket and capture the difference. This arbitrage channel is "
        "what tethers an ETF's market price to its net asset value (NAV). "
        "For equity ETFs the tether is tight: baskets are liquid, hedging is "
        "cheap, and premiums rarely exceed a few basis points."
    ),
    (
        "Fixed income ETFs are different. Their underlying bonds trade "
        "over-the-counter, often by appointment, and many constituents do "
        "not trade at all on a given day. NAVs are struck from matrix prices "
        "and dealer marks that lag traded levels, while the ETF itself "
        "trades continuously on exchange. In stressed markets the ETF price "
        "becomes the de facto price discovery vehicle and can deviate from "
        "stale NAV by several percent, as in March 2020 when investment "
        "grade and high yield ETFs traded at discounts not seen since 2008 "
        "and the Federal Reserve ultimately intervened in ETF markets "
        "directly. These dislocations raise a natural question: are they "
        "noise to be faded, or information to be respected?"
    ),
    (
        "This paper studies that question for five of the largest U.S. "
        "fixed income ETFs: HYG and JNK (high yield credit), LQD "
        "(investment grade credit), TLT (long Treasuries) and AGG "
        "(aggregate). We normalize each fund's premium/discount as a 63-day "
        "rolling z-score, identify dislocation events at 1.5-, 2.0- and "
        "2.5-sigma thresholds, and trace forward returns and reversion "
        "dynamics in event time. We condition the analysis on volatility "
        "(VIX) and credit (ICE BofA option-adjusted spread) regimes, and "
        "evaluate a simple implementable strategy that buys deep discounts "
        "and fades rich premiums under realistic transaction costs, "
        "including a walk-forward split. Finally, the accompanying codebase "
        "operationalizes the signal as a live monitoring and alerting "
        "system."
    ),
    (
        "Our work sits in an established literature. Petajisto (2017) "
        "documents that ETF prices deviate economically from NAV far more "
        "often than commonly assumed, especially for funds holding illiquid "
        "assets, and shows the deviations are systematic enough to trade. "
        "Ben-David, Franzoni and Moussawi (2018) show the arbitrage channel "
        "itself can transmit liquidity shocks from the ETF to its "
        "constituents, increasing the volatility of underlying securities. "
        "Pan and Zeng (2019) highlight that bond ETF APs face inventory "
        "frictions that weaken arbitrage exactly when it is most needed, "
        "and Todorov (2021) and Haddad, Moreira and Muir (2021) analyze the "
        "March 2020 episode in detail. Relative to this literature, our "
        "contribution is a reproducible, end-to-end pipeline: event-study "
        "evidence with bootstrap inference, regime conditioning, a "
        "costed backtest, and an open-source alerting implementation. An "
        "important caveat applies throughout: where live NAV feeds are "
        "unavailable we simulate NAV with a calibrated AR(1) premium "
        "process, so empirical magnitudes illustrate the methodology rather "
        "than measure live market inefficiency."
    ),
]


def build(styles: dict) -> list[Flowable]:
    """Build the introduction flowables.

    Args:
        styles: Shared paragraph style dict from ``generate_paper.py``.

    Returns:
        List of ReportLab flowables.
    """
    flowables: list[Flowable] = [Paragraph("1. Introduction", styles["heading"])]
    flowables.extend(Paragraph(text, styles["body"]) for text in PARAGRAPHS)
    return flowables
