"""Master script that assembles the SSRN-ready research paper PDF.

Runs any missing analysis steps (data fetch, spread construction, event
study, regime analysis, backtest), then assembles the full paper from the
section modules in ``paper/sections/`` and renders it with ReportLab.

Layout: US Letter, Times New Roman 12 pt body text, bold 14 pt section
headers, page numbers and a running header with the paper title on every
page after the first.  Tables are rendered full width; body text is single
column.  A separate figures-only appendix PDF is also produced.

Inputs:
    ``data/processed/`` CSVs and ``outputs/figures/`` PNGs (regenerated on
    demand via the pipeline modules).

Outputs:
    * ``outputs/paper/etf_nav_arbitrage_study.pdf`` -- the full paper.
    * ``outputs/paper/figures_appendix.pdf`` -- figures-only appendix.

Usage:
    python paper/generate_paper.py             # run missing steps, build PDF
    python paper/generate_paper.py --rebuild   # force-rerun the pipeline
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from paper.sections import abstract, conclusion, data, introduction, methodology, results

logger = config.get_logger(__name__)

PAPER_TITLE = ("Mean Reversion in Fixed Income ETF Premiums and Discounts: "
               "Evidence, Regimes, and an Implementable Strategy")
AUTHOR_PLACEHOLDER = "[Author Name] — [Affiliation] — [email@institution.edu]"

PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 1.0 * inch


def build_styles() -> dict:
    """Create the shared paragraph styles used by every section module.

    Returns:
        Dict mapping style names (``title, author, heading, subheading,
        body, abstract, caption, reference``) to ``ParagraphStyle``
        objects.  Body text is Times-Roman 12 pt; headings are Times-Bold
        14 pt, per the SSRN-style typography requirement.
    """
    return {
        "title": ParagraphStyle(
            "title", fontName="Times-Bold", fontSize=18, leading=23,
            alignment=1, spaceAfter=18),
        "author": ParagraphStyle(
            "author", fontName="Times-Roman", fontSize=12, leading=15,
            alignment=1, spaceAfter=6),
        "heading": ParagraphStyle(
            "heading", fontName="Times-Bold", fontSize=14, leading=17,
            spaceBefore=16, spaceAfter=8),
        "subheading": ParagraphStyle(
            "subheading", fontName="Times-Bold", fontSize=12, leading=15,
            spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle(
            "body", fontName="Times-Roman", fontSize=12, leading=16,
            alignment=4, spaceAfter=10, firstLineIndent=18),
        "abstract": ParagraphStyle(
            "abstract", fontName="Times-Roman", fontSize=11, leading=14.5,
            alignment=4, leftIndent=36, rightIndent=36, spaceAfter=10),
        "caption": ParagraphStyle(
            "caption", fontName="Times-Italic", fontSize=10, leading=12.5,
            alignment=1, spaceBefore=4, spaceAfter=8),
        "reference": ParagraphStyle(
            "reference", fontName="Times-Roman", fontSize=11, leading=14,
            leftIndent=24, firstLineIndent=-24, spaceAfter=6),
    }


def _on_page(canvas, doc) -> None:  # noqa: ANN001 -- ReportLab callback signature
    """Draw the running header and page number on each page.

    Args:
        canvas: ReportLab canvas for the current page.
        doc: The document template (provides the page counter).

    Returns:
        None.
    """
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("Times-Italic", 9)
        header = PAPER_TITLE if len(PAPER_TITLE) <= 90 else PAPER_TITLE[:87] + "..."
        canvas.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 0.55 * inch, header)
        canvas.setStrokeColor(colors.gray)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, PAGE_HEIGHT - 0.65 * inch,
                    PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 0.65 * inch)
    canvas.setFont("Times-Roman", 10)
    canvas.drawCentredString(PAGE_WIDTH / 2, 0.5 * inch, str(doc.page))
    canvas.restoreState()


def _make_document(path: Path) -> BaseDocTemplate:
    """Create the page-numbered document template.

    Args:
        path: Output PDF path.

    Returns:
        Configured ``BaseDocTemplate``.
    """
    document = BaseDocTemplate(
        str(path), pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=PAPER_TITLE,
    )
    frame = Frame(MARGIN, MARGIN, PAGE_WIDTH - 2 * MARGIN,
                  PAGE_HEIGHT - 2 * MARGIN, id="main")
    document.addPageTemplates([PageTemplate(id="paper", frames=[frame],
                                            onPage=_on_page)])
    return document


def title_page(styles: dict) -> list:
    """Build the title page flowables.

    Args:
        styles: Shared paragraph style dict.

    Returns:
        List of flowables ending in a page break (abstract appears on the
        title page per SSRN convention, so the break follows it).
    """
    return [
        Spacer(1, 1.2 * inch),
        Paragraph(PAPER_TITLE, styles["title"]),
        Paragraph(AUTHOR_PLACEHOLDER, styles["author"]),
        Paragraph(date.today().strftime("%B %d, %Y"), styles["author"]),
        Spacer(1, 0.5 * inch),
        *abstract.build(styles),
        Paragraph(
            "<i>Keywords:</i> exchange-traded funds, net asset value, "
            "arbitrage, mean reversion, fixed income, market "
            "microstructure. <i>JEL:</i> G12, G14, G23.",
            styles["abstract"]),
        PageBreak(),
    ]


def run_pipeline(rebuild: bool = False) -> None:
    """Run any analysis steps whose outputs are missing.

    Each pipeline stage is skipped when its primary output already exists,
    unless ``rebuild`` forces a full rerun.  Stages run in dependency
    order: fetch -> spread -> reversion -> regime -> backtest.

    Args:
        rebuild: Force-rerun every stage regardless of existing outputs.

    Returns:
        None.
    """
    stages: list[tuple[Path, str]] = [
        (config.DATA_RAW_DIR / f"prices_{config.ETFS[0]}.csv", "data.fetch"),
        (config.DATA_PROCESSED_DIR / "spreads.csv", "analysis.spread"),
        (config.DATA_PROCESSED_DIR / "event_study_stats.csv", "analysis.reversion"),
        (config.DATA_PROCESSED_DIR / "regime_stats.csv", "analysis.regime"),
        (config.DATA_PROCESSED_DIR / "backtest_metrics.csv", "backtest.strategy"),
    ]
    import importlib

    for output, module_name in stages:
        if output.exists() and not rebuild:
            logger.info("Pipeline stage %s up to date (%s exists)",
                        module_name, output.name)
            continue
        logger.info("Running pipeline stage %s ...", module_name)
        module = importlib.import_module(module_name)
        module.main()


def build_paper() -> Path:
    """Assemble and render the full paper PDF.

    Returns:
        Path of the generated PDF.
    """
    styles = build_styles()
    flowables = [
        *title_page(styles),
        *introduction.build(styles),
        *data.build(styles),
        *methodology.build(styles),
        *results.build(styles),
        *conclusion.build(styles),
    ]
    out_path = config.PAPER_OUTPUT_DIR / "etf_nav_arbitrage_study.pdf"
    document = _make_document(out_path)
    document.build(flowables)
    logger.info("Paper written to %s", out_path)
    return out_path


def build_figures_appendix() -> Path:
    """Render the figures-only appendix PDF.

    Embeds every PNG found in ``outputs/figures/`` (sorted by name), one
    per block with a caption derived from the filename.

    Returns:
        Path of the generated appendix PDF.
    """
    from PIL import Image as PILImage

    styles = build_styles()
    flowables: list = [
        Paragraph("Figures Appendix", styles["title"]),
        Paragraph(PAPER_TITLE, styles["author"]),
        Spacer(1, 0.3 * inch),
    ]
    pngs = sorted(config.FIGURES_DIR.glob("*.png"))
    if not pngs:
        flowables.append(Paragraph(
            "[No figures found — run the analysis pipeline first]",
            styles["body"]))
    max_width = PAGE_WIDTH - 2 * MARGIN
    for i, png in enumerate(pngs, start=1):
        with PILImage.open(png) as im:
            aspect = im.height / im.width
        width = max_width
        height = width * aspect
        if height > 7.5 * inch:
            height = 7.5 * inch
            width = height / aspect
        caption = png.stem.replace("_", " ").capitalize()
        flowables.append(KeepTogether([
            Image(str(png), width=width, height=height),
            Paragraph(f"Figure A{i}. {caption}.", styles["caption"]),
            Spacer(1, 14),
        ]))
    out_path = config.PAPER_OUTPUT_DIR / "figures_appendix.pdf"
    document = _make_document(out_path)
    document.build(flowables)
    logger.info("Figures appendix written to %s", out_path)
    return out_path


def main() -> None:
    """CLI entry point: run pipeline as needed, then build both PDFs.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Generate the research paper")
    parser.add_argument("--rebuild", action="store_true",
                        help="force-rerun the full analysis pipeline first")
    args = parser.parse_args()

    config.ensure_directories()
    logger.info("=== Paper generation started ===")
    run_pipeline(rebuild=args.rebuild)
    build_paper()
    build_figures_appendix()
    logger.info("=== Paper generation complete ===")


if __name__ == "__main__":
    main()
