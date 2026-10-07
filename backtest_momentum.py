"""Momentum backtest for a fixed symbol list.

Long when the 50-day return is positive and the close is above its
200-day average. Each name targets an equal share of equity.
"""

import numpy as np
from numba import njit

import pybroker
from pybroker import Strategy, YFinance, indicator, returns

pybroker.disable_progress_bar()
pybroker.disable_logging()

SYMBOLS = (
    "OXY",
    "VTI",
    "SCHD",
    "XLE",
    "F",
    "XLF",
    "VXUS",
    "BND",
    "KBH",
    "SPNT",
)


@njit(cache=True)
def _sma(values, period):
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan
    if period < 1:
        return out
    total = 0.0
    for i in range(n):
        total += values[i]
        if i >= period:
            total -= values[i - period]
        if i >= period - 1:
            out[i] = total / period
    return out


def _sma200(data):
    return _sma(data.close, 200)


def momentum_exec(ctx):
    """Hold an equal-weight long while the 50-day return and the
    200-day average agree.
    """
    roc50 = ctx.indicator("roc_50")
    sma200 = ctx.indicator("sma_200")
    if np.isnan(roc50[-1]) or np.isnan(sma200[-1]):
        return
    # Equal weight: a target of 1.0 on every name would each claim all equity.
    target = 1.0 / len(SYMBOLS)
    if roc50[-1] > 0 and ctx.close[-1] > sma200[-1]:
        ctx.set_target_shares(target, dir="long")
    elif ctx.long_pos():
        ctx.set_target_shares(0.0, dir="long")


if __name__ == "__main__":
    strategy = Strategy(
        YFinance(),
        start_date="2020-01-01",
        end_date="2025-12-31",
    )
    strategy.add_execution(
        momentum_exec,
        SYMBOLS,
        indicators=[
            returns("roc_50", "close", period=50),
            indicator("sma_200", _sma200),
        ],
    )
    result = strategy.backtest(warmup=200)
    print(result.metrics_df.to_string())
