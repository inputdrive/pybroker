"""
Walkforward bootstrapped backtests on combined universe.
"""
import pybroker
from pybroker import Strategy, StrategyConfig, YFinance, indicator
from pybroker.vect import highv, lowv
from pybroker import returns
import numpy as np
from numba import njit
from pybroker import enable_data_source_cache, enable_indicator_cache, enable_model_cache

enable_data_source_cache(namespace="agentic_universe")
enable_indicator_cache(namespace="agentic_universe")
enable_model_cache(namespace="agentic_universe")

pybroker.disable_progress_bar()
pybroker.disable_logging()

SYMBOLS = (
    "OXY","VTI","SCHD","XLE","F","XLF","VXUS","BND","KBH","SPNT",
    "BYDDY","SMR","LAES","MIR","KO","BAC","IYZ"
)
START = "2020-01-01"
END = "2025-12-31"
INITIAL_CASH = 1554.04

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
    roc50 = ctx.indicator("roc_50")
    sma200 = ctx.indicator("sma_200")
    if np.isnan(roc50[-1]) or np.isnan(sma200[-1]):
        return
    target = 1.0/len(SYMBOLS)
    if roc50[-1] > 0 and ctx.close[-1] > sma200[-1]:
        ctx.set_target_shares(target, dir="long")
    elif ctx.long_pos():
        ctx.set_target_shares(0.0, dir="long")

def build_momentum():
    strat = Strategy(YFinance(), start_date=START, end_date=END,
                     config=StrategyConfig(initial_cash=INITIAL_CASH))
    roc_ind = indicator("roc_50", lambda data: returns(data.close, period=50).values)
    sma_ind = indicator("sma_200", _sma200)
    strat.add_execution(momentum_exec, SYMBOLS, indicators=[roc_ind, sma_ind])
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

def build_hhv():
    hhv20 = indicator("hhv20", lambda b: highv(b.close, 20))
    llv20 = indicator("llv20", lambda b: lowv(b.close, 20))
    strat = Strategy(YFinance(), start_date=START, end_date=END,
                     config=StrategyConfig(initial_cash=INITIAL_CASH))
    strat.add_execution(hhv_llv_exec, SYMBOLS, indicators=[hhv20, llv20])
    return strat

# Rotational trading
def rank(ctxs):
    scores = {sym: ctx.indicator("roc_20")[-1] for sym, ctx in ctxs.items()}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_symbols = [s for s,_ in sorted_scores[:5]]
    pybroker.param("top_symbols", top_symbols)

def rotate(ctx):
    if ctx.long_pos():
        if ctx.symbol not in pybroker.param("top_symbols"):
            ctx.sell_all_shares()
        else:
            target_size = 1.0/2
            ctx.buy_shares = ctx.calc_target_shares(target_size)
            ctx.score = ctx.indicator("roc_20")[-1]
    elif ctx.symbol in pybroker.param("top_symbols"):
        target_size = 1.0/2
        ctx.buy_shares = ctx.calc_target_shares(target_size)
        ctx.score = ctx.indicator("roc_20")[-1]

def build_rotational():
    config = StrategyConfig(max_long_positions=2, initial_cash=INITIAL_CASH)
    roc20 = indicator("roc_20", lambda data: returns(data.close, period=20).values)
    strat = Strategy(YFinance(), start_date=START, end_date=END, config=config)
    strat.set_before_exec(rank)
    strat.add_execution(rotate, SYMBOLS, indicators=[roc20])
    return strat

if __name__ == "__main__":
    print("=== Momentum 50/200 Walkforward ===")
    wf = build_momentum().walkforward(windows=5, train_size=0.6, lookahead=1, calc_bootstrap=True)
    print(wf.metrics_df)
    wf.metrics_df.to_csv("/c/Users/ggutm/source/pybroker/benchmarks/wf_momentum_17.csv")

    print("\n=== HHV/LLV Walkforward ===")
    wf2 = build_hhv().walkforward(windows=5, train_size=0.6, lookahead=1, calc_bootstrap=True)
    print(wf2.metrics_df)
    wf2.metrics_df.to_csv("/c/Users/ggutm/source/pybroker/benchmarks/wf_hhv_17.csv")

    print("\n=== Rotational ROC20 Walkforward ===")
    wf3 = build_rotational().walkforward(windows=5, train_size=0.6, lookahead=1, calc_bootstrap=True)
    print(wf3.metrics_df)
    wf3.metrics_df.to_csv("/c/Users/ggutm/source/pybroker/benchmarks/wf_rotational_17.csv")
