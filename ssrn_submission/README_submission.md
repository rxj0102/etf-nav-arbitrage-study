# SSRN Submission Checklist

Everything needed to put the paper on SSRN is in this folder:

| File | Purpose |
|---|---|
| `paper_final.pdf` | The full paper — upload this |
| `ssrn_abstract.txt` | 149-word abstract for the abstract field |
| `ssrn_metadata.txt` | Every form field pre-filled, copy/paste |
| `README_submission.md` | This checklist |

> **Before submitting:** replace `[Last Name]` with your surname in
> `ssrn_metadata.txt` and regenerate the PDF after editing
> `AUTHOR_PLACEHOLDER` in `paper/generate_paper.py`
> (`python paper/generate_paper.py`, then re-copy the PDF here).

## Upload steps

1. **Go to [ssrn.com](https://www.ssrn.com)** → sign in → **My Papers** →
   **Submit a Paper** (create a free author account first if needed).
2. **Paste metadata field by field** from `ssrn_metadata.txt`:
   title, author, affiliation, contact email, abstract (from
   `ssrn_abstract.txt` — already ≤150 words), JEL codes `G12, G14, G23`,
   and the six keywords.
3. **Upload `paper_final.pdf`** as the full-text document.
4. **Select the paper series / subject areas** listed in the metadata file:
   - *Financial Economics > Asset Pricing & Valuation*
   - *Financial Economics > Market Microstructure*
5. **Set visibility: Public** (Privately Available papers don't get
   indexed or counted in downloads).
6. **Submit.** SSRN's staff review typically takes **1–3 business days**;
   you'll receive an email when the paper is approved and assigned a
   permanent URL/abstract ID.

## After the paper is live

7. **Update the GitHub README** — add the SSRN URL under the *Citation*
   section of the repo's `README.md` (both as a link and inside the
   BibTeX entry's `url`/`note` field), commit and push.
8. **LinkedIn** — add the SSRN paper link to your profile's
   **Featured** section (Profile → Add section → Featured → Links).
9. **Resume** — add under *Research / Projects*:

   > ETF NAV Arbitrage and Premium/Discount Mean Reversion in Fixed
   > Income Markets (SSRN Working Paper, June 2026)

   Adjust `[Month Year]` to the month the paper goes live.

## Notes

- The PDF is regenerated from code: every table and figure pulls from
  `data/processed/` and `outputs/figures/`, so rerunning the pipeline and
  `python paper/generate_paper.py` keeps the paper in sync with results.
- The paper is transparent that NAV is simulated (seeded AR(1) premium
  process); the abstract frames the near-zero backtest edge as the
  expected placebo outcome. If SSRN reviewers or readers ask about data
  provenance, point them to Section 2 (Data) and the GitHub repository.
