"""Data section of the research paper.

Describes the ETF universe, NAV construction (including the AR(1)
simulation caveat), the sample period, and a summary statistics table
computed dynamically from ``data/processed/spreads.csv``.

Outputs:
    :func:`build` returns ``list[Flowable]`` consumed by
    ``paper/generate_paper.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.platypus import Flowable, Paragraph, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

logger = config.get_logger(__name__)


def _summary_table(styles: dict) -> list[Flowable]:
    """Build the per-ETF summary statistics table.

    Args:
        styles: Shared paragraph style dict.

    Returns:
        List of flowables (caption + table), or a placeholder paragraph
        when the processed panel is unavailable.
    """
    path = config.DATA_PROCESSED_DIR / "spreads.csv"
    if not path.exists():
        logger.warning("Data section: %s missing; emitting placeholder", path)
        return [Paragraph("[Summary statistics table: run "
                          "analysis/spread.py to populate]", styles["body"])]
    panel = pd.read_csv(path, parse_dates=["date"])
    rows = [["ETF", "Obs.", "Mean prem. (%)", "Std (%)", "Min (%)",
             "Max (%)", "Start", "End"]]
    for ticker in config.ETFS:
        sub = panel[panel["ticker"] == ticker]
        premium = sub["premium_pct"]
        rows.append([
            ticker, f"{len(sub):,}", f"{premium.mean():.3f}",
            f"{premium.std():.3f}", f"{premium.min():.3f}",
            f"{premium.max():.3f}",
            sub["date"].min().strftime("%Y-%m-%d"),
            sub["date"].max().strftime("%Y-%m-%d"),
        ])
    table = Table(rows, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [
        Paragraph("Table 1. Premium/discount summary statistics by ETF.",
                  styles["caption"]),
        table,
        Spacer(1, 10),
    ]


def build(styles: dict) -> list[Flowable]:
    """Build the data section flowables.

    Args:
        styles: Shared paragraph style dict from ``generate_paper.py``.

    Returns:
        List of ReportLab flowables.
    """
    descriptions = "; ".join(
        f"{ticker} ({name})" for ticker, name in config.ETF_DESCRIPTIONS.items()
    )
    flowables: list[Flowable] = [
        Paragraph("2. Data", styles["heading"]),
        Paragraph(
            "The universe comprises five of the most actively traded U.S. "
            f"fixed income ETFs: {descriptions}. Daily open, high, low, "
            "close and volume series are obtained from Yahoo Finance for "
            f"the most recent {config.LOOKBACK_YEARS} years. Market "
            "regime variables are the CBOE VIX index (Yahoo Finance) and "
            "ICE BofA option-adjusted credit spreads from FRED: the US "
            "High Yield Master II OAS (series BAMLH0A0HYM2) and the US "
            "Corporate Master OAS (series BAMLC0A0CM).",
            styles["body"],
        ),
        Paragraph(
            "<b>NAV construction.</b> Official end-of-day NAVs are not "
            "distributed through free APIs, so NAV is simulated: the "
            "premium fraction follows a first-order autoregressive process "
            f"p<sub>t</sub> = {config.NAV_AR1_PHI} p<sub>t-1</sub> + "
            f"ε<sub>t</sub> with ε<sub>t</sub> ~ N(0, "
            f"{config.NAV_AR1_SIGMA}²), and NAV<sub>t</sub> = "
            "Price<sub>t</sub> / (1 + p<sub>t</sub>). The persistence "
            "(φ = 0.85) and innovation scale (σ = 0.3% per day) "
            "are calibrated to the empirical behavior of fixed income ETF "
            "premiums documented by Petajisto (2017). The simulation is "
            "seeded, making every result in this paper exactly "
            "reproducible. Because the dislocation signal is constructed "
            "from simulated NAV, the empirical sections should be read as "
            "a validation of the methodology and software pipeline; "
            "substituting an issuer or vendor NAV feed requires changing "
            "a single function in the data layer.",
            styles["body"],
        ),
    ]
    flowables.extend(_summary_table(styles))
    return flowables
