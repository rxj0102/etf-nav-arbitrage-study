"""Streamlit dashboard for the ETF NAV arbitrage study.

Five tabs: a live z-score monitor with gauges, historical premium/discount
analysis with stress and regime overlays, the interactive event study,
backtest results, and the alert log.  A sidebar offers an ETF selector,
threshold slider (1.5-3.0 sigma), lookback control, a regime overlay toggle
and a manual data refresh button (data otherwise auto-refreshes via a
cached loader on page load).

Inputs:
    ``data/processed/`` CSVs, ``data/raw/vix.csv`` and
    ``alerts/alert_log.csv`` produced by the pipeline and monitor.

Outputs:
    Interactive web app (no files written).

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

logger = config.get_logger(__name__)

st.set_page_config(page_title="ETF NAV Arbitrage Monitor", page_icon="📉",
                   layout="wide")


@st.cache_data(ttl=300)
def load_csv(name: str, parse_dates: list[str] | None = None,
             directory: str = "processed") -> pd.DataFrame | None:
    """Load a project CSV with caching.

    Args:
        name: File name within the chosen directory.
        parse_dates: Column names to parse as datetimes.
        directory: ``"processed"``, ``"raw"`` or ``"alerts"``.

    Returns:
        DataFrame, or ``None`` when the file does not exist.
    """
    base = {
        "processed": config.DATA_PROCESSED_DIR,
        "raw": config.DATA_RAW_DIR,
        "alerts": config.ALERTS_DIR,
    }[directory]
    path = base / name
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates or [])


def zscore_gauge(ticker: str, zscore: float, threshold: float) -> go.Figure:
    """Build a Plotly gauge for one ETF's current z-score.

    Args:
        ticker: ETF ticker symbol.
        zscore: Current z-score value.
        threshold: Alert threshold (colors the gauge bands).

    Returns:
        Plotly figure with an indicator gauge, color coded green inside
        the band, amber near the threshold and red beyond it.
    """
    color = ("#2e7d32" if abs(zscore) < threshold * 0.75
             else "#f9a825" if abs(zscore) < threshold else "#c62828")
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=zscore,
        number={"valueformat": "+.2f"},
        title={"text": ticker, "font": {"size": 16}},
        gauge={
            "axis": {"range": [-3.5, 3.5]},
            "bar": {"color": color},
            "steps": [
                {"range": [-3.5, -threshold], "color": "rgba(198,40,40,0.25)"},
                {"range": [-threshold, threshold], "color": "rgba(46,125,50,0.15)"},
                {"range": [threshold, 3.5], "color": "rgba(198,40,40,0.25)"},
            ],
            "threshold": {"line": {"color": "black", "width": 2},
                          "value": zscore},
        },
    ))
    figure.update_layout(height=210, margin=dict(l=12, r=12, t=36, b=8))
    return figure


def render_live_monitor(spreads: pd.DataFrame, alert_log: pd.DataFrame | None,
                        threshold: float) -> None:
    """Render Tab 1: per-ETF gauges, premium deltas and last alert times.

    Args:
        spreads: Long-format spread panel.
        alert_log: Alert log frame or ``None``.
        threshold: Selected sigma threshold.

    Returns:
        None.
    """
    st.subheader("Live monitor")
    st.caption("Latest available observation per ETF from the processed "
               "panel. Run `python data/fetch.py && python "
               "analysis/spread.py` (or the sidebar refresh) to update.")
    columns = st.columns(len(config.ETFS))
    for column, ticker in zip(columns, config.ETFS):
        sub = spreads[spreads["ticker"] == ticker].dropna(subset=["zscore"])
        if sub.empty:
            column.warning(f"No data for {ticker}")
            continue
        latest = sub.iloc[-1]
        prior = sub.iloc[-2] if len(sub) > 1 else latest
        with column:
            st.plotly_chart(zscore_gauge(ticker, float(latest["zscore"]),
                                         threshold),
                            width="stretch",
                            key=f"gauge_{ticker}")
            st.metric(
                "Premium/discount",
                f"{latest['premium_pct']:+.3f}%",
                delta=f"{latest['premium_pct'] - prior['premium_pct']:+.3f} pp "
                      "vs prior day",
            )
            if alert_log is not None and not alert_log.empty:
                alerts = alert_log[(alert_log["etf"] == ticker)
                                   & (alert_log["action"] == "alerted")]
                last = (pd.to_datetime(alerts["timestamp"]).max()
                        if not alerts.empty else None)
                st.caption(f"Last alert: {last:%Y-%m-%d %H:%M}" if last is not None
                           else "Last alert: never")
            else:
                st.caption("Last alert: never")


def render_historical(spreads: pd.DataFrame, vix: pd.DataFrame | None,
                      ticker: str, threshold: float, lookback_days: int,
                      show_regimes: bool) -> None:
    """Render Tab 2: premium time series with signals, stress and regimes.

    Args:
        spreads: Long-format spread panel.
        vix: VIX frame (``Date`` index column) or ``None``.
        ticker: Selected ETF.
        threshold: Selected sigma threshold.
        lookback_days: Trailing window length to display.
        show_regimes: Whether to shade VIX regime bands.

    Returns:
        None.
    """
    st.subheader(f"Historical analysis — {ticker}")
    sub = spreads[spreads["ticker"] == ticker].dropna(subset=["zscore"]).copy()
    if sub.empty:
        st.warning("No processed data. Run the analysis pipeline first.")
        return
    sub = sub.tail(lookback_days)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=sub["date"], y=sub["premium_pct"],
                                name="Premium/discount (%)",
                                line=dict(color="#1a237e", width=1.2)))
    signals = sub[sub["zscore"].abs() >= threshold]
    figure.add_trace(go.Scatter(
        x=signals["date"], y=signals["premium_pct"], mode="markers",
        name=f"|z| ≥ {threshold:.1f}σ",
        marker=dict(color="#c62828", size=6, symbol="diamond"),
    ))
    for name, (start, end) in config.STRESS_EPISODES.items():
        if pd.Timestamp(end) >= sub["date"].min():
            figure.add_vrect(x0=start, x1=end, fillcolor="crimson",
                             opacity=0.10, line_width=0,
                             annotation_text=name.split(" (")[0],
                             annotation_position="top left",
                             annotation_font_size=9)
    if show_regimes and vix is not None:
        vix_indexed = vix.set_index("Date")["VIX"].reindex(
            pd.DatetimeIndex(sub["date"])).ffill()
        high = vix_indexed > config.VIX_HIGH_MIN
        # Shade contiguous high-VIX stretches.
        in_block = False
        start_date = None
        dates = list(vix_indexed.index)
        for i, (day, flag) in enumerate(zip(dates, high)):
            if flag and not in_block:
                in_block, start_date = True, day
            last = i == len(dates) - 1
            if in_block and (not flag or last):
                figure.add_vrect(x0=start_date, x1=day, fillcolor="orange",
                                 opacity=0.10, line_width=0)
                in_block = False
    figure.update_layout(height=460, hovermode="x unified",
                         yaxis_title="Premium/discount (%)",
                         legend=dict(orientation="h", y=1.06))
    st.plotly_chart(figure, width="stretch")

    zfig = go.Figure()
    zfig.add_trace(go.Scatter(x=sub["date"], y=sub["zscore"], name="z-score",
                              line=dict(color="#00695c", width=1.2)))
    for level in (threshold, -threshold):
        zfig.add_hline(y=level, line_dash="dash", line_color="#c62828",
                       line_width=1)
    zfig.add_hline(y=0, line_color="gray", line_width=0.6)
    zfig.update_layout(height=300, yaxis_title="Rolling z-score",
                       hovermode="x unified")
    st.plotly_chart(zfig, width="stretch")


def render_event_study(paths: pd.DataFrame | None, stats: pd.DataFrame | None,
                       regime_events: pd.DataFrame | None,
                       threshold: float) -> None:
    """Render Tab 3: interactive reversion paths and regime comparison.

    Args:
        paths: Event path frame (``event_paths.csv``) or ``None``.
        stats: Event study statistics or ``None``.
        regime_events: Event-level regime frame or ``None``.
        threshold: Selected sigma threshold.

    Returns:
        None.
    """
    st.subheader("Event study")
    if paths is None or paths.empty:
        st.warning("Run `python analysis/reversion.py` to populate this tab.")
        return
    available = sorted(paths["threshold"].unique())
    nearest = min(available, key=lambda t: abs(t - threshold))
    if abs(nearest - threshold) > 1e-9:
        st.info(f"No event paths at {threshold:.1f}σ; showing nearest "
                f"computed threshold {nearest:.1f}σ.")
    figure = go.Figure()
    colors = {"discount": "#1565c0", "premium": "#c62828"}
    for direction in ["discount", "premium"]:
        sub = paths[(paths["threshold"] == nearest)
                    & (paths["direction"] == direction)]
        if sub.empty:
            continue
        figure.add_trace(go.Scatter(
            x=sub["day"], y=sub["ci_high"], line=dict(width=0),
            showlegend=False, hoverinfo="skip"))
        band_fill = ("rgba(21,101,192,0.15)" if direction == "discount"
                     else "rgba(198,40,40,0.15)")
        figure.add_trace(go.Scatter(
            x=sub["day"], y=sub["ci_low"], fill="tonexty",
            fillcolor=band_fill,
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(
            x=sub["day"], y=sub["mean_z"],
            name=f"{direction} (n={int(sub['n_events'].iloc[0])})",
            line=dict(color=colors[direction], width=2)))
    figure.add_hline(y=0, line_color="gray", line_width=0.6)
    figure.add_vline(x=0, line_dash="dash", line_color="gray", line_width=0.8)
    figure.update_layout(height=440, xaxis_title="Days from event",
                         yaxis_title="Average z-score",
                         title=f"Average reversion path, |z| > {nearest:.1f}σ "
                               "(95% bands)")
    st.plotly_chart(figure, width="stretch")

    if st.toggle("Regime-conditional comparison", value=False):
        if regime_events is None or regime_events.empty:
            st.warning("Run `python analysis/regime.py` first.")
        else:
            box = go.Figure()
            for regime in ["low", "medium", "high"]:
                sub = regime_events[regime_events["vix_regime"] == regime]
                box.add_trace(go.Box(
                    y=sub["fwd_ret_5d"] * 100, name=f"VIX {regime}",
                    boxmean=True))
            box.update_layout(height=380,
                              yaxis_title="5-day forward return (%)",
                              title="Forward returns by VIX regime "
                                    f"({config.ALERT_THRESHOLD:.1f}σ events)")
            st.plotly_chart(box, width="stretch")

    if stats is not None and not stats.empty:
        st.markdown("**Pooled forward return statistics**")
        pooled = stats[(stats["ticker"] == "ALL")
                       & (stats["threshold"] == nearest)]
        st.dataframe(pooled.drop(columns=["ticker"]).round(3),
                     width="stretch", hide_index=True)


def render_backtest(returns: pd.DataFrame | None,
                    metrics: pd.DataFrame | None) -> None:
    """Render Tab 4: backtest performance.

    Args:
        returns: Daily strategy/benchmark return frame or ``None``.
        metrics: Backtest metric table or ``None``.

    Returns:
        None.
    """
    st.subheader("Backtest")
    if returns is None or returns.empty:
        st.warning("Run `python backtest/strategy.py` to populate this tab.")
        return
    returns = returns.set_index("date")
    equity = (1 + returns[["strategy", "benchmark"]]).cumprod()

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=equity.index, y=equity["strategy"],
                                name="Strategy", line=dict(color="#1a237e")))
    figure.add_trace(go.Scatter(x=equity.index, y=equity["benchmark"],
                                name="Equal weight buy & hold",
                                line=dict(color="#ef6c00")))
    split = equity.index[0] + pd.DateOffset(years=config.TRAIN_YEARS)
    figure.add_vline(x=split, line_dash="dash", line_color="gray")
    figure.update_layout(height=420, yaxis_title="Growth of $1",
                         title="Cumulative returns (dashed line: "
                               "walk-forward split)")
    st.plotly_chart(figure, width="stretch")

    drawdown_fig = go.Figure()
    for column, color in [("strategy", "#1a237e"), ("benchmark", "#ef6c00")]:
        eq = (1 + returns[column]).cumprod()
        drawdown = (eq / eq.cummax() - 1) * 100
        drawdown_fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown,
                                          name=column, fill="tozeroy",
                                          line=dict(color=color, width=1)))
    drawdown_fig.update_layout(height=300, yaxis_title="Drawdown (%)",
                               title="Drawdowns")
    st.plotly_chart(drawdown_fig, width="stretch")

    if metrics is not None and not metrics.empty:
        st.markdown("**Summary statistics**")
        st.dataframe(metrics.round(2), width="stretch",
                     hide_index=True)

    monthly = (1 + returns["strategy"]).resample("ME").prod() - 1
    heat = pd.DataFrame({
        "year": monthly.index.year, "month": monthly.index.month,
        "ret": monthly.to_numpy() * 100,
    }).pivot(index="year", columns="month", values="ret")
    heat_fig = go.Figure(go.Heatmap(
        z=heat.to_numpy(), x=[f"{m:02d}" for m in heat.columns],
        y=heat.index.astype(str), colorscale="RdYlGn", zmid=0,
        texttemplate="%{z:.1f}", colorbar=dict(title="%")))
    heat_fig.update_layout(height=320, title="Strategy monthly returns (%)",
                           xaxis_title="Month", yaxis_title="Year")
    st.plotly_chart(heat_fig, width="stretch")


def render_alert_log(alert_log: pd.DataFrame | None) -> None:
    """Render Tab 5: filterable alert log.

    Args:
        alert_log: Alert log frame or ``None``.

    Returns:
        None.
    """
    st.subheader("Alert log")
    if alert_log is None or alert_log.empty:
        st.info("No alerts logged yet. Run `python alerts/monitor.py --once` "
                "to record a monitoring pass.")
        return
    log = alert_log.copy()
    log["timestamp"] = pd.to_datetime(log["timestamp"])

    filter_columns = st.columns(3)
    with filter_columns[0]:
        etfs = st.multiselect("ETF", options=sorted(log["etf"].unique()),
                              default=sorted(log["etf"].unique()))
    with filter_columns[1]:
        bounds = st.date_input(
            "Date range",
            value=(log["timestamp"].min().date(), log["timestamp"].max().date()),
        )
    with filter_columns[2]:
        min_threshold = st.number_input("Min threshold (σ)", value=0.0,
                                        step=0.5)

    mask = log["etf"].isin(etfs) & (log["threshold"] >= min_threshold)
    if isinstance(bounds, tuple) and len(bounds) == 2:
        mask &= ((log["timestamp"].dt.date >= bounds[0])
                 & (log["timestamp"].dt.date <= bounds[1]))
    st.dataframe(log[mask].sort_values("timestamp", ascending=False)
                 .round({"zscore": 2, "premium_pct": 3}),
                 width="stretch", hide_index=True)


def main() -> None:
    """Assemble the sidebar and tabs.

    Returns:
        None.
    """
    st.title("ETF NAV Arbitrage Monitor")
    st.caption("Premium/discount mean reversion in fixed income ETFs — "
               "research dashboard. NAV is simulated (see README).")

    with st.sidebar:
        st.header("Controls")
        ticker = st.selectbox("ETF", config.ETFS, index=0)
        threshold = st.slider("Dislocation threshold (σ)", 1.5, 3.0,
                              float(config.ALERT_THRESHOLD), 0.1)
        lookback = st.select_slider(
            "Lookback (trading days)",
            options=[63, 126, 252, 504, 756, 1260], value=504)
        show_regimes = st.toggle("Regime overlay (VIX bands)", value=True)
        if st.button("🔄 Refresh data", width="stretch"):
            load_csv.clear()
            st.rerun()

    spreads = load_csv("spreads.csv", parse_dates=["date"])
    if spreads is None:
        st.error("No processed data found. Run `python data/fetch.py` and "
                 "`python analysis/spread.py` first.")
        st.stop()

    vix = load_csv("vix.csv", parse_dates=["Date"], directory="raw")
    paths = load_csv("event_paths.csv")
    stats = load_csv("event_study_stats.csv")
    regime_events = load_csv("regime_events.csv", parse_dates=["date"])
    returns = load_csv("backtest_returns.csv", parse_dates=["date"])
    metrics = load_csv("backtest_metrics.csv")
    alert_log = load_csv("alert_log.csv", parse_dates=["timestamp"],
                         directory="alerts")

    tabs = st.tabs(["📡 Live Monitor", "📈 Historical Analysis",
                    "🔬 Event Study", "💼 Backtest", "🔔 Alert Log"])
    with tabs[0]:
        render_live_monitor(spreads, alert_log, threshold)
    with tabs[1]:
        render_historical(spreads, vix, ticker, threshold, lookback,
                          show_regimes)
    with tabs[2]:
        render_event_study(paths, stats, regime_events, threshold)
    with tabs[3]:
        render_backtest(returns, metrics)
    with tabs[4]:
        render_alert_log(alert_log)


main()
