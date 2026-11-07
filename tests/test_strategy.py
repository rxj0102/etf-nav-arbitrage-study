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
