import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import strategy  # noqa: E402


def _panel(n=300, seed=0, scale_late=1.0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.005, n)
    ret[n // 2:] *= scale_late
    close = 100 * np.cumprod(1 + ret)
    z = rng.normal(0, 1.5, n)
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n),
        "ticker": "X", "close": close, "zscore": z,
        "ret_1d": pd.Series(close).pct_change().to_numpy(),
    })


def test_sizing_has_no_lookahead():
    """Changing future data must not change earlier position sizes."""
    a = _panel(scale_late=1.0)
    b = _panel(scale_late=10.0)
    pa, _ = strategy.generate_positions(a)
    pb, _ = strategy.generate_positions(b)
    half = len(a) // 2
    assert np.allclose(pa[:half], pb[:half])


def test_positions_are_capped():
    from config import MAX_POSITION_MULTIPLE
    p, _ = strategy.generate_positions(_panel())
    assert p.abs().max() <= MAX_POSITION_MULTIPLE + 1e-12


def test_sortino_uses_downside_deviation():
    r = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005] * 50)
    m = strategy.compute_metrics(r, None, None, "t")
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean() * 252)
    cum = (1 + r).prod()
    ann = cum ** (252 / len(r)) - 1
    assert np.isclose(m["sortino"], ann / dd)


def test_exec_lag_delays_first_return():
    """With exec_lag=2 a decision at close t first earns day t+2's return."""
    panel = _panel()
    r1, _ = strategy.backtest_portfolio(panel, exec_lag=1)
    r2, _ = strategy.backtest_portfolio(panel, exec_lag=2)
    first1 = r1.index[(r1["strategy"] != 0)][0]
    first2 = r2.index[(r2["strategy"] != 0)][0]
    days = pd.bdate_range(first1, first2)
    assert len(days) == 2  # exactly one business day later


def test_exec_lag_one_matches_original_convention():
    panel = _panel()
    positions, _ = strategy.generate_positions(panel.sort_values("date").reset_index(drop=True), exec_lag=1)
    r1, _ = strategy.backtest_portfolio(panel, exec_lag=1)
    ret = panel["ret_1d"].fillna(0.0).to_numpy()
    lagged = positions.shift(1).fillna(0.0).to_numpy()
    turnover = pd.Series(lagged).diff().abs().fillna(abs(lagged[0])).to_numpy()
    expected = lagged * ret - turnover * 5.0 / 1e4
    assert np.allclose(r1["strategy"].to_numpy(), expected)


def test_trade_log_uses_fill_closes():
    panel = _panel()
    _, t1 = strategy.generate_positions(panel, exec_lag=1)
    _, t2 = strategy.generate_positions(panel, exec_lag=2)
    assert len(t1) == len(t2) > 0
    assert not np.allclose([t["gross_return_pct"] for t in t1], [t["gross_return_pct"] for t in t2])


def test_trade_pnl_uses_adjusted_close_when_present():
    panel = _panel()
    panel["adj_close"] = panel["close"] * np.linspace(1.0, 1.05, len(panel))
    _, t_adj = strategy.generate_positions(panel)
    _, t_raw = strategy.generate_positions(panel.drop(columns="adj_close"))
    assert not np.allclose([t["gross_return_pct"] for t in t_adj], [t["gross_return_pct"] for t in t_raw])
