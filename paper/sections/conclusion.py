"""Conclusion and references sections of the research paper.

Summarizes the key findings, states the study's limitations (above all the
simulated NAV series), proposes extensions, and lists APA-formatted
references.

Outputs:
    :func:`build` returns ``list[Flowable]`` consumed by
    ``paper/generate_paper.py``.
"""

from __future__ import annotations

from reportlab.platypus import Flowable, Paragraph

REFERENCES = [
    "Ben-David, I., Franzoni, F., & Moussawi, R. (2018). Do ETFs increase "
    "volatility? <i>The Journal of Finance, 73</i>(6), 2471–2535. "
    "https://doi.org/10.1111/jofi.12727",

    "Boehmer, B., & Boehmer, E. (2003). Trading your neighbor's ETFs: "
    "Competition or fragmentation? <i>Journal of Banking &amp; Finance, "
    "27</i>(9), 1667–1703. https://doi.org/10.1016/S0378-4266(03)00095-6",

    "Engle, R., & Sarkar, D. (2006). Premiums-discounts and exchange traded "
    "funds. <i>The Journal of Derivatives, 13</i>(4), 27–45. "
    "https://doi.org/10.3905/jod.2006.635418",

    "Haddad, V., Moreira, A., & Muir, T. (2021). When selling becomes "
    "viral: Disruptions in debt markets in the COVID-19 crisis and the "
    "Fed's response. <i>The Review of Financial Studies, 34</i>(11), "
    "5309–5351. https://doi.org/10.1093/rfs/hhab026",

    "Lettau, M., & Madhavan, A. (2018). Exchange-traded funds 101 for "
    "economists. <i>Journal of Economic Perspectives, 32</i>(1), 135–154. "
    "https://doi.org/10.1257/jep.32.1.135",

    "Madhavan, A., & Sobczyk, A. (2016). Price dynamics and liquidity of "
    "exchange-traded funds. <i>Journal of Investment Management, 14</i>(2), "
    "1–17.",

    "Pan, K., & Zeng, Y. (2019). ETF arbitrage under liquidity mismatch "
    "(Working paper). Harvard University and University of Pennsylvania. "
    "https://doi.org/10.2139/ssrn.2895478",

    "Petajisto, A. (2017). Inefficiencies in the pricing of "
    "exchange-traded funds. <i>Financial Analysts Journal, 73</i>(1), "
    "24–54. https://doi.org/10.2469/faj.v73.n1.7",

    "Poterba, J. M., & Summers, L. H. (1988). Mean reversion in stock "
    "prices: Evidence and implications. <i>Journal of Financial Economics, "
    "22</i>(1), 27–59. https://doi.org/10.1016/0304-405X(88)90021-9",

    "Todorov, K. (2021). The anatomy of bond ETF arbitrage. <i>BIS "
    "Quarterly Review</i>, March 2021, 41–53.",
]

PARAGRAPHS = [
    (
        "<b>Key findings.</b> Normalizing fixed income ETF "
        "premium/discounts as rolling z-scores yields a tractable, "
        "comparable dislocation signal across credit, Treasury and "
        "aggregate funds. Dislocation episodes beyond two sigma are "
        "infrequent but cluster in stress windows, and the z-score "
        "reverts to zero within days—a direct consequence of the "
        "mean-reverting premium process, and the property an arbitrageur "
        "monetizes when the dislocation is real. The event study "
        "framework, bootstrap inference, regime conditioning and costed "
        "walk-forward backtest together form a complete template for "
        "evaluating NAV-basis strategies."
    ),
    (
        "<b>Limitations.</b> The central limitation is the NAV series: "
        "official NAVs are not freely distributed, so this study "
        "simulates the premium as a seeded AR(1) process calibrated to "
        "the empirical literature. The reported point estimates "
        "therefore validate the pipeline rather than measure live market "
        "inefficiency. Further, the backtest assumes fills at the close "
        "with linear costs, ignores borrow costs and shorting "
        "constraints on the premium leg, and equal-weights five funds "
        "whose dislocations are correlated in stress, understating "
        "portfolio concentration risk precisely when the signal fires "
        "most."
    ),
    (
        "<b>Extensions.</b> Three follow-ups are natural. First, replace "
        "the simulated NAV with issuer iNAV/NAV feeds or a "
        "vendor-supplied basis, at which point every table in Section 4 "
        "re-estimates live economics without code changes. Second, "
        "enrich the regime layer with funding and dealer balance sheet "
        "variables (e.g., primary dealer positions, repo spreads) that "
        "proxy the AP constraint channel of Pan and Zeng (2019). Third, "
        "extend the universe to bank loan, EM debt and municipal ETFs, "
        "where NAV staleness is more severe and the documented premia "
        "larger."
    ),
]


def build(styles: dict) -> list[Flowable]:
    """Build the conclusion and references flowables.

    Args:
        styles: Shared paragraph style dict from ``generate_paper.py``.

    Returns:
        List of ReportLab flowables.
    """
    flowables: list[Flowable] = [Paragraph("5. Conclusion", styles["heading"])]
    flowables.extend(Paragraph(text, styles["body"]) for text in PARAGRAPHS)
    flowables.append(Paragraph("References", styles["heading"]))
    flowables.extend(Paragraph(ref, styles["reference"]) for ref in REFERENCES)
    return flowables
