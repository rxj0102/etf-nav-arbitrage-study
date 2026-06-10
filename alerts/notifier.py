"""Alert delivery channels: SMTP email and Slack incoming webhooks.

Provides :func:`send_alert`, which fans an alert out to every enabled
channel.  Channels are toggled independently via ``EMAIL_ENABLED`` /
``SLACK_ENABLED`` environment variables (see ``config.py`` and
``.env.example``); credentials are never hardcoded.

Email
-----
HTML-formatted message (Gmail-compatible SMTP with STARTTLS) containing the
ETF name, current z-score and premium/discount, historical context (how
often the threshold was crossed in the sample and the average forward
return at this threshold from the event study), a link to the dashboard,
and a small matplotlib chart of the recent z-score embedded inline as a
base64 ``<img>``.

Slack
-----
Block Kit formatted message with the ETF name, z-score, signal direction
and the expected reversion based on the historical average.  Posted to an
incoming webhook URL using only the standard library (``urllib``).

Failure handling is graceful: if email delivery fails the error is logged
and Slack is still attempted (and vice versa).

Inputs:
    ``data/processed/event_study_stats.csv`` and ``spreads.csv`` for
    historical context (optional -- alerts degrade gracefully without them).

Outputs:
    Side effects only (network sends).  Returns per-channel success flags.
"""

from __future__ import annotations

import base64
import io
import json
import smtplib
import sys
import urllib.request
from dataclasses import dataclass
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

logger = config.get_logger(__name__)


@dataclass
class Alert:
    """Container describing a single dislocation alert.

    Attributes:
        ticker: ETF ticker symbol.
        zscore: Current rolling z-score of the premium/discount.
        premium_pct: Current premium/discount in percent.
        threshold: Sigma threshold that was crossed.
        direction: ``"discount"`` (z below -threshold) or ``"premium"``.
        recent_zscores: Recent z-score series for the inline email chart
            (may be ``None``).
    """

    ticker: str
    zscore: float
    premium_pct: float
    threshold: float
    direction: str
    recent_zscores: pd.Series | None = None


def historical_context(ticker: str, threshold: float, direction: str) -> dict:
    """Pull historical context for an alert from the processed event study.

    Args:
        ticker: ETF ticker symbol.
        threshold: Sigma threshold crossed.
        direction: ``"discount"`` or ``"premium"``.

    Returns:
        Dict with ``n_crossings`` (historical event count for this ETF at
        this threshold/direction), ``avg_fwd_5d_pct`` and
        ``avg_fwd_10d_pct`` (pooled average forward returns from the event
        study) -- values are ``None`` when the stats file is unavailable.
    """
    context: dict = {"n_crossings": None, "avg_fwd_5d_pct": None,
                     "avg_fwd_10d_pct": None}
    stats_path = config.DATA_PROCESSED_DIR / "event_study_stats.csv"
    if not stats_path.exists():
        logger.warning("event_study_stats.csv missing; alert sent without "
                       "historical context")
        return context
    try:
        stats = pd.read_csv(stats_path)
        own = stats[(stats["ticker"] == ticker)
                    & (stats["threshold"] == threshold)
                    & (stats["direction"] == direction)]
        if not own.empty:
            context["n_crossings"] = int(own["n_events"].max())
        pooled = stats[(stats["ticker"] == "ALL")
                       & (stats["threshold"] == threshold)
                       & (stats["direction"] == direction)]
        for horizon, key in [(5, "avg_fwd_5d_pct"), (10, "avg_fwd_10d_pct")]:
            row = pooled[pooled["horizon"] == horizon]
            if not row.empty:
                context[key] = float(row["mean_fwd_ret_pct"].iloc[0])
    except Exception as exc:  # noqa: BLE001 -- context is best-effort
        logger.warning("Failed to load historical context: %s", exc)
    return context


def _inline_chart_png(alert: Alert) -> bytes | None:
    """Render a small z-score chart for embedding in the email.

    Args:
        alert: Alert payload (uses ``recent_zscores`` when present).

    Returns:
        PNG bytes, or ``None`` when no recent series is available.
    """
    if alert.recent_zscores is None or alert.recent_zscores.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.4))
    series = alert.recent_zscores
    ax.plot(series.index, series.to_numpy(), color="navy", lw=1.2)
    ax.axhline(alert.threshold, color="firebrick", ls="--", lw=0.8)
    ax.axhline(-alert.threshold, color="firebrick", ls="--", lw=0.8)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"{alert.ticker} premium z-score (last {len(series)} days)",
                 fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def build_email_html(alert: Alert, context: dict, chart_cid: str | None) -> str:
    """Build the HTML body of the alert email.

    Args:
        alert: Alert payload.
        context: Historical context from :func:`historical_context`.
        chart_cid: Content-ID of the inline chart image, or ``None``.

    Returns:
        HTML string.
    """
    signal = "BUY (discount, expect reversion up)" if alert.direction == "discount" \
        else "SELL/SHORT (premium, expect reversion down)"
    n_cross = context["n_crossings"]
    fwd5 = context["avg_fwd_5d_pct"]
    fwd10 = context["avg_fwd_10d_pct"]
    rows = [
        ("ETF", f"{alert.ticker} — {config.ETF_DESCRIPTIONS.get(alert.ticker, '')}"),
        ("Current z-score", f"{alert.zscore:+.2f}"),
        ("Premium/discount", f"{alert.premium_pct:+.3f}%"),
        ("Threshold crossed", f"{alert.threshold:.1f}σ ({alert.direction})"),
        ("Signal", signal),
        ("Historical crossings (this ETF)",
         f"{n_cross}" if n_cross is not None else "n/a"),
        ("Avg 5d fwd return at threshold (pooled)",
         f"{fwd5:+.2f}%" if fwd5 is not None else "n/a"),
        ("Avg 10d fwd return at threshold (pooled)",
         f"{fwd10:+.2f}%" if fwd10 is not None else "n/a"),
    ]
    table_rows = "".join(
        f"<tr><td style='padding:4px 12px 4px 0;color:#555'>{k}</td>"
        f"<td style='padding:4px 0;font-weight:600'>{v}</td></tr>"
        for k, v in rows
    )
    chart_html = (f"<p><img src='cid:{chart_cid}' alt='z-score chart' "
                  f"style='max-width:100%'></p>" if chart_cid else "")
    return f"""\
<html><body style="font-family:Arial,Helvetica,sans-serif;font-size:14px">
<h2 style="color:#1a237e">ETF NAV dislocation alert: {alert.ticker}</h2>
<table>{table_rows}</table>
{chart_html}
<p><a href="{config.DASHBOARD_URL}">Open the live dashboard</a></p>
<p style="color:#888;font-size:11px">Automated alert from the
etf-nav-arbitrage-study monitor. Not investment advice.</p>
</body></html>"""


def send_email(alert: Alert, context: dict) -> bool:
    """Send the alert by SMTP email.

    Args:
        alert: Alert payload.
        context: Historical context dict.

    Returns:
        ``True`` on success, ``False`` on any failure (logged, never
        raised, so other channels still run).
    """
    if not config.EMAIL_ENABLED:
        logger.info("Email channel disabled; skipping email for %s", alert.ticker)
        return False
    if not (config.SMTP_USER and config.SMTP_PASSWORD and config.ALERT_EMAIL_TO):
        logger.error("Email enabled but SMTP_USER/SMTP_PASSWORD/ALERT_EMAIL_TO "
                     "not configured; skipping email")
        return False
    try:
        msg = MIMEMultipart("related")
        msg["Subject"] = (f"[ETF Alert] {alert.ticker} premium/discount z-score "
                          f"crossed {alert.threshold:.1f}σ")
        msg["From"] = config.ALERT_EMAIL_FROM
        msg["To"] = ", ".join(config.ALERT_EMAIL_TO)

        chart = _inline_chart_png(alert)
        chart_cid = "zscore_chart" if chart else None
        html = build_email_html(alert, context, chart_cid)
        msg.attach(MIMEText(html, "html"))
        if chart:
            image = MIMEImage(chart, "png")
            image.add_header("Content-ID", f"<{chart_cid}>")
            image.add_header("Content-Disposition", "inline",
                             filename=f"{alert.ticker}_zscore.png")
            msg.attach(image)

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
            smtp.sendmail(config.ALERT_EMAIL_FROM, config.ALERT_EMAIL_TO,
                          msg.as_string())
        logger.info("Email alert sent for %s to %s", alert.ticker,
                    config.ALERT_EMAIL_TO)
        return True
    except Exception as exc:  # noqa: BLE001 -- must not block other channels
        logger.error("Email alert FAILED for %s: %s", alert.ticker, exc)
        return False


def build_slack_blocks(alert: Alert, context: dict) -> list[dict]:
    """Build the Slack Block Kit payload for an alert.

    Args:
        alert: Alert payload.
        context: Historical context dict.

    Returns:
        List of Block Kit block dicts.
    """
    emoji = ":chart_with_downwards_trend:" if alert.direction == "discount" \
        else ":chart_with_upwards_trend:"
    signal = "LONG (buy the discount)" if alert.direction == "discount" \
        else "SHORT (fade the premium)"
    fwd5 = context["avg_fwd_5d_pct"]
    expected = (f"{fwd5:+.2f}% avg 5-day forward return historically"
                if fwd5 is not None else "n/a (run the event study)")
    return [
        {
            "type": "header",
            "text": {"type": "plain_text",
                     "text": f"{emoji} ETF dislocation: {alert.ticker}",
                     "emoji": True},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Z-score:*\n{alert.zscore:+.2f}"},
                {"type": "mrkdwn",
                 "text": f"*Premium/discount:*\n{alert.premium_pct:+.3f}%"},
                {"type": "mrkdwn",
                 "text": f"*Threshold:*\n{alert.threshold:.1f}σ ({alert.direction})"},
                {"type": "mrkdwn", "text": f"*Signal:*\n{signal}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f"*Expected reversion:* {expected}\n"
                             f"<{config.DASHBOARD_URL}|Open dashboard>"},
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn",
                          "text": "etf-nav-arbitrage-study monitor — "
                                  "not investment advice"}],
        },
    ]


def send_slack(alert: Alert, context: dict) -> bool:
    """Post the alert to the configured Slack incoming webhook.

    Args:
        alert: Alert payload.
        context: Historical context dict.

    Returns:
        ``True`` on success, ``False`` on any failure (logged, never
        raised).
    """
    if not config.SLACK_ENABLED:
        logger.info("Slack channel disabled; skipping Slack for %s", alert.ticker)
        return False
    if not config.SLACK_WEBHOOK_URL:
        logger.error("Slack enabled but SLACK_WEBHOOK_URL not configured")
        return False
    try:
        payload = {
            "text": (f"[ETF Alert] {alert.ticker} z-score "
                     f"{alert.zscore:+.2f} crossed {alert.threshold:.1f}σ"),
            "blocks": build_slack_blocks(alert, context),
        }
        request = urllib.request.Request(
            config.SLACK_WEBHOOK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
        if body.strip() != "ok":
            raise RuntimeError(f"Slack webhook responded: {body!r}")
        logger.info("Slack alert sent for %s", alert.ticker)
        return True
    except Exception as exc:  # noqa: BLE001 -- must not block other channels
        logger.error("Slack alert FAILED for %s: %s", alert.ticker, exc)
        return False


def send_alert(alert: Alert) -> dict[str, bool]:
    """Send an alert through every enabled channel.

    Email failures never prevent the Slack attempt and vice versa.

    Args:
        alert: Alert payload.

    Returns:
        Dict ``{"email": bool, "slack": bool}`` of per-channel success.
    """
    context = historical_context(alert.ticker, alert.threshold, alert.direction)
    email_ok = send_email(alert, context)
    slack_ok = send_slack(alert, context)
    if not (email_ok or slack_ok):
        logger.warning("Alert for %s was not delivered on any channel "
                       "(channels disabled or failing)", alert.ticker)
    return {"email": email_ok, "slack": slack_ok}
