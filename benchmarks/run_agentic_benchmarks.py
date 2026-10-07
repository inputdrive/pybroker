"""Run momentum, channel, and buy-and-hold backtests on a fixed symbol list."""
import numpy as np
import pybroker
from pybroker import Strategy, YFinance, indicator, returns
from pybroker.vect import highv, lowv

pybroker.disable_progress_bar()
pybroker.disable_logging()

_DELTA_NAMES = ("total_pnl", "total_return_pct", "sharpe", "win_rate")


def _begin(study):
    pybroker.enable_result_cache(study)


def _show_delta(study):
    latest = pybroker.list_runs()[-1]
    print(f"recorded {study} {latest['run_id']}")
    compared = pybroker.compare_runs(latest["run_id"])
    if not compared["deltas"]:
        print("no earlier run in this study")
        return
    for name in _DELTA_NAMES:
        row = compared["deltas"].get(name)
        if row is None:
            continue
        print(f"{name}: {row['before']} -> {row['after']} ({row['delta']})")

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

START = "2020-01-01"
END = "2025-12-31"

def run_momentum():
    """50-day ROC + 200-day SMA momentum."""
    # Simple SMA via Numba
    from numba import njit
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
                total -= values[i - period]
            if i >= period - 1:
                out[i] = total / period
        return out

    def _sma200(data):
        return _sma(data.close, 200)

    def exec_fn(ctx):
        roc50 = ctx.indicator("roc_50")
        sma200 = ctx.indicator("sma_200")
        if np.isnan(roc50[-1]) or np.isnan(sma200[-1]):
            return
        target = 1.0 / len(SYMBOLS)
        if roc50[-1] > 0 and ctx.close[-1] > sma200[-1]:
            ctx.set_target_shares(target, dir="long")
        elif ctx.long_pos():
            ctx.set_target_shares(0.0, dir="long")

    strat = Strategy(YFinance(), start_date=START, end_date=END)
    strat.add_execution(exec_fn, SYMBOLS,
                        indicators=[
                            returns("roc_50", "close", period=50),
                            indicator("sma_200", _sma200),
                        ])
    _begin("momentum")
    result = strat.backtest(warmup=200)
    print("=== Momentum 50/200 ===")
    print(result.metrics_df.to_string())
    _show_delta("momentum")
    return result

def run_hhv_llv():
    """HHV20/LLV20 crossover."""
    hhv20 = indicator("hhv20", lambda b: highv(b.close, 20))
    llv20 = indicator("llv20", lambda b: lowv(b.close, 20))
    def exec_fn(ctx):
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

    strat = Strategy(YFinance(), start_date=START, end_date=END)
    strat.add_execution(exec_fn, SYMBOLS, indicators=[hhv20, llv20])
    _begin("hhv-llv")
    result = strat.backtest(warmup=20)
    print("=== HHV20/LLV20 Crossover ===")
    print(result.metrics_df.to_string())
    _show_delta("hhv-llv")
    return result

def run_buy_hold():
    """Simple buy-and-hold."""
    def exec_fn(ctx):
        if not ctx.long_pos():
            ctx.set_target_shares(1.0/len(SYMBOLS), dir="long")

    strat = Strategy(YFinance(), start_date=START, end_date=END)
    strat.add_execution(exec_fn, SYMBOLS)
    _begin("buy-hold")
    result = strat.backtest(warmup=1)
    print("=== Buy & Hold ===")
    print(result.metrics_df.to_string())
    _show_delta("buy-hold")
    return result

if __name__ == "__main__":
    run_momentum()
    print()
    run_hhv_llv()
    print()
    run_buy_hold()
