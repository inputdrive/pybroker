"""Walkforward bootstrapped backtests on a fixed symbol list."""
import pybroker
from pybroker import Strategy, YFinance, indicator
from pybroker.vect import highv, lowv
import numpy as np
from numba import njit

pybroker.disable_progress_bar()
pybroker.disable_logging()

SYMBOLS = (
    "OXY","VTI","SCHD","XLE","F","XLF","VXUS","BND","KBH","SPNT"
)
START = "2020-01-01"
END = "2025-12-31"

@njit(cache=True)
def _sma(values, period):
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan
    total = 0.0
    for i in range(n):
        total += values[i]
        if i >= period:
            total -= values[i-period]
        if i >= period-1:
            out[i] = total/period
    return out

def _sma200(data):
    return _sma(data.close, 200)

def momentum_exec(ctx):
    from pybroker import returns
    roc50 = ctx.indicator("roc_50")
    sma200 = ctx.indicator("sma_200")
    if np.isnan(roc50[-1]) or np.isnan(sma200[-1]):
        return
    target = 1.0/len(SYMBOLS)
    if roc50[-1] > 0 and ctx.close[-1] > sma200[-1]:
        ctx.set_target_shares(target, dir="long")
    elif ctx.long_pos():
        ctx.set_target_shares(0.0, dir="long")

def build_momentum_strategy():
    strat = Strategy(YFinance(), start_date=START, end_date=END)
    strat.add_execution(momentum_exec, SYMBOLS, indicators=[
        indicator("roc_50", lambda data: returns(data.close, period=50).values),
        indicator("sma_200", _sma200),
    ])
    return strat

def hhv_llv_exec(ctx):
    hhv = ctx.indicator("hhv20")
    llv = ctx.indicator("llv20")
    if len(hhv) < 21:
        return
    close = ctx.close[-1]
    prev_hhv = hhv[-2]
    prev_llv = llv[-2]
    if not ctx.long_pos() and close > prev_hhv:
        ctx.buy_shares = 100
        ctx.stop_loss_pct = 5
        ctx.stop_profit_pct = 15
    elif ctx.long_pos() and close < prev_llv:
        ctx.sell_all_shares()

def build_hhv_strategy():
    hhv20 = indicator("hhv20", lambda b: highv(b.close, 20))
    llv20 = indicator("llv20", lambda b: lowv(b.close, 20))
    strat = Strategy(YFinance(), start_date=START, end_date=END)
    strat.add_execution(hhv_llv_exec, SYMBOLS, indicators=[hhv20, llv20])
    return strat

if __name__ == "__main__":
    print("=== Momentum 50/200 Walkforward ===")
    s = build_momentum_strategy()
    wf = s.walkforward(windows=5, train_size=0.6,
                       lookahead=1, calc_bootstrap=True, parallel_indicators=False)
    print(wf.metrics_df.to_string())
    # Save
    wf.metrics_df.to_csv("benchmarks/wf_momentum_metrics.csv")

    print("\n=== HHV/LLV Walkforward ===")
    s2 = build_hhv_strategy()
    wf2 = s2.walkforward(windows=5, train_size=0.6,
                         lookahead=1, calc_bootstrap=True, parallel_indicators=False)
    print(wf2.metrics_df.to_string())
    wf2.metrics_df.to_csv("benchmarks/wf_hhv_metrics.csv")
