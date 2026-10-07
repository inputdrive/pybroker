"""Tests for the run journal."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import pytest
from .fixtures import *
from pybroker.cache import (
    annotate_run,
    clear_result_cache,
    compare_runs,
    disable_result_cache,
    enable_result_cache,
    get_run,
    list_runs,
    record_run,
    set_parent_run,
)
from pybroker.indicator import indicator
from pybroker.optimize import hyperparam
from pybroker.strategy import Strategy
from pybroker.vect import highv


def _frame(n: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = np.linspace(10.0, 12.0, n)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "SPY",
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000.0),
        }
    )


def _buy(ctx):
    if not ctx.long_pos():
        ctx.buy_shares = 10


@pytest.fixture()
def journal(scope, tmp_path):
    enable_result_cache("study", cache_dir=str(tmp_path))
    yield
    disable_result_cache()


def test_record_note_and_compare(journal):
    first = record_run(
        kind="backtest",
        symbols=["SPY"],
        start_date="2020-01-02",
        end_date="2020-01-31",
        metrics={"total_pnl": 10.0, "trade_count": 1, "label": "a"},
        params={"shares": 10},
    )
    note = annotate_run(first, comment="size up", overrides={"shares": 20})
    stored = get_run(first)
    assert stored["metrics"]["total_pnl"] == 10.0
    assert stored["note"] == note
    assert note["overrides"] == {"shares": 20}

    second = record_run(
        kind="backtest",
        symbols=["SPY"],
        start_date="2020-01-02",
        end_date="2020-01-31",
        metrics={"total_pnl": 4.0, "trade_count": 2, "label": "b"},
        params={"shares": 20},
    )
    assert get_run(second)["parent_id"] == first
    compared = compare_runs(second)
    assert compared["baseline_id"] == first
    assert compared["deltas"]["total_pnl"] == {
        "before": 10.0,
        "after": 4.0,
        "delta": -6.0,
    }
    assert "label" not in compared["deltas"]


def test_set_parent_applies_once(journal):
    first = record_run(
        kind="backtest",
        symbols=["SPY"],
        start_date="2020-01-02",
        end_date="2020-01-03",
        metrics={"total_pnl": 1.0},
    )
    record_run(
        kind="backtest",
        symbols=["SPY"],
        start_date="2020-01-02",
        end_date="2020-01-03",
        metrics={"total_pnl": 2.0},
    )
    set_parent_run(first)
    third = record_run(
        kind="backtest",
        symbols=["SPY"],
        start_date="2020-01-02",
        end_date="2020-01-03",
        metrics={"total_pnl": 3.0},
    )
    assert get_run(third)["parent_id"] == first
    fourth = record_run(
        kind="backtest",
        symbols=["SPY"],
        start_date="2020-01-02",
        end_date="2020-01-03",
        metrics={"total_pnl": 4.0},
    )
    assert get_run(fourth)["parent_id"] == third


def test_annotate_return_is_a_copy(journal):
    run_id = record_run(
        kind="backtest",
        symbols=["SPY"],
        start_date="2020-01-02",
        end_date="2020-01-03",
        metrics={"total_pnl": 1.0},
    )
    note = annotate_run(run_id, overrides={"shares": 20})
    note["overrides"]["shares"] = 1
    assert get_run(run_id)["note"]["overrides"]["shares"] == 20


def test_get_run_is_a_copy(journal):
    run_id = record_run(
        kind="backtest",
        symbols=["SPY"],
        start_date="2020-01-02",
        end_date="2020-01-03",
        metrics={"total_pnl": 1.0},
    )
    got = get_run(run_id)
    got["metrics"]["total_pnl"] = 999.0
    assert get_run(run_id)["metrics"]["total_pnl"] == 1.0


def test_unknown_kind_and_missing_run(journal):
    with pytest.raises(ValueError, match="Unknown run kind"):
        record_run(
            kind="paper",
            symbols=["SPY"],
            start_date="2020-01-02",
            end_date="2020-01-03",
            metrics={},
        )
    with pytest.raises(ValueError, match="Run not found"):
        get_run("000001")


def test_clear_and_reopen(scope, tmp_path):
    enable_result_cache("study", cache_dir=str(tmp_path))
    record_run(
        kind="backtest",
        symbols=["SPY"],
        start_date="2020-01-02",
        end_date="2020-01-03",
        metrics={"total_pnl": 1.0},
    )
    clear_result_cache()
    assert list_runs() == []
    disable_result_cache()
    with pytest.raises(ValueError, match="enabled"):
        list_runs()
    enable_result_cache("study", cache_dir=str(tmp_path))
    assert list_runs() == []
    disable_result_cache()


def test_backtest_records_matching_metrics(scope, tmp_path):
    df = _frame()
    enable_result_cache("study", cache_dir=str(tmp_path))
    strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
    strategy.add_execution(_buy, ["SPY"])
    result = strategy.backtest()
    runs = list_runs()
    assert len(runs) == 1
    assert runs[0]["kind"] == "backtest"
    assert runs[0]["symbols"] == ["SPY"]
    assert runs[0]["metrics"]["trade_count"] == result.metrics.trade_count
    assert (
        runs[0]["metrics"]["total_pnl"]
        == result.metrics.to_json()["total_pnl"]
    )
    assert "trades" not in runs[0]

    disable_result_cache()
    strategy.backtest()
    enable_result_cache("study", cache_dir=str(tmp_path))
    assert len(list_runs()) == 1
    disable_result_cache()


def test_journal_failure_still_returns_result(scope, tmp_path):
    df = _frame()
    enable_result_cache("study", cache_dir=str(tmp_path))
    scope.result_parent_id = "999999"
    strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
    strategy.add_execution(_buy, ["SPY"])
    with pytest.warns(UserWarning, match="did not record"):
        result = strategy.backtest()
    assert (
        result.metrics.trade_count == result.metrics.to_json()["trade_count"]
    )
    assert list_runs() == []
    disable_result_cache()


def test_train_only_is_not_recorded(scope, tmp_path):
    df = _frame()
    enable_result_cache("study", cache_dir=str(tmp_path))
    strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
    strategy.add_execution(None, ["SPY"])
    strategy.backtest()
    assert list_runs() == []
    disable_result_cache()


def test_walkforward_one_window_keeps_its_kind(scope, tmp_path):
    df = _frame()
    enable_result_cache("study", cache_dir=str(tmp_path))
    strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
    strategy.add_execution(_buy, ["SPY"])
    strategy.walkforward(windows=1, train_size=0, lookahead=1)
    assert list_runs()[0]["kind"] == "walkforward"
    disable_result_cache()


def test_walkforward_kind(scope, tmp_path):
    df = _frame(12)
    enable_result_cache("study", cache_dir=str(tmp_path))
    strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
    strategy.add_execution(_buy, ["SPY"])
    strategy.walkforward(windows=2, train_size=0.5, lookahead=1)
    assert list_runs()[0]["kind"] == "walkforward"
    assert list_runs()[0]["windows"] == 2
    disable_result_cache()


def test_optimize_records_best_params(scope, tmp_path):
    df = _frame(12)
    lookback = hyperparam("lookback", default=2, low=2, high=3, step=1)
    hhv = indicator(
        "hhv_journal",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )

    def exec_fn(ctx):
        vals = ctx.indicator(hhv.name)
        if len(vals) > 0 and vals[-1] > 0 and not ctx.long_pos():
            ctx.buy_shares = 1

    enable_result_cache("opt", cache_dir=str(tmp_path))
    strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
    strategy.add_execution(exec_fn, ["SPY"], indicators=[hhv])
    opt = strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        train_size=0.5,
        parallel_indicators=False,
    )
    runs = list_runs()
    assert len(runs) == 1
    assert runs[0]["kind"] == "optimize"
    assert runs[0]["params"]["lookback"] == opt.best_params["lookback"]
    assert runs[0]["extra"]["n_trials"] == len(opt.study.trials)
    assert (
        runs[0]["metrics"]["total_pnl"]
        == opt.result.metrics.to_json()["total_pnl"]
    )
    disable_result_cache()


def test_optimize_windows_record_scores(scope, tmp_path):
    df = _frame(12)
    lookback = hyperparam("lookback_w", default=2, low=2, high=3, step=1)
    hhv = indicator(
        "hhv_journal_w",
        lambda data, period: highv(data.high, period),
        period=lookback,
    )

    def exec_fn(ctx):
        vals = ctx.indicator(hhv.name)
        if len(vals) > 0 and vals[-1] > 0 and not ctx.long_pos():
            ctx.buy_shares = 1

    enable_result_cache("opt-windows", cache_dir=str(tmp_path))
    strategy = Strategy(df, df["date"].iloc[0], df["date"].iloc[-1])
    strategy.add_execution(exec_fn, ["SPY"], indicators=[hhv])
    strategy.optimize(
        lambda r: r.metrics.total_pnl,
        sampler="grid",
        windows=2,
        train_size=0.5,
        parallel_indicators=False,
    )
    runs = list_runs()
    assert len(runs) == 1
    assert runs[0]["kind"] == "optimize"
    assert runs[0]["windows"] == 2
    assert len(runs[0]["extra"]["window_scores"]) == 2
    disable_result_cache()
