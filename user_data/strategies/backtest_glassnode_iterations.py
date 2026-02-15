#!/usr/bin/env python3
"""
Backtest runner for GlassnodeOnChainStrategy iterations.

Runs multiple parameter configurations against historical data and produces
a side-by-side comparison table of key performance metrics.

Usage:
    # Basic (uses your freqtrade config.json):
    python user_data/strategies/backtest_glassnode_iterations.py \
        --config user_data/config.json

    # With options:
    python user_data/strategies/backtest_glassnode_iterations.py \
        --config user_data/config.json \
        --timerange 20230101-20240101 \
        --pairs BTC/USDT \
        --stake-amount 1000 \
        --wallet 10000

Prerequisites:
    - Download data first: freqtrade download-data --pairs BTC/USDT -t 4h
    - A valid freqtrade config.json with exchange credentials
"""
import argparse
import logging
import sys
import time as _time
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure freqtrade is on the import path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from freqtrade.configuration import Configuration
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.history import get_timerange
from freqtrade.data.metrics import calculate_market_change
from freqtrade.enums import CandleType, RunMode
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.optimize_reports import generate_strategy_stats

logger = logging.getLogger(__name__)

# ============================================================================
# Strategy iteration definitions
# ============================================================================
# Each iteration is a dict with:
#   "name":   Human-readable label for comparison table
#   "params": Dict of strategy attribute overrides. Keys must match
#             the strategy's hyperoptable parameter names (attr.name)
#             or top-level strategy attributes (stoploss, minimal_roi, etc.)
#
# The BSS signal parameters (buy_bss_long_min, short_bss_short_min, etc.)
# accept values in [0.0, 1.0].  RSI parameters are integers.
# ============================================================================

ITERATIONS: list[dict[str, Any]] = [
    # ------------------------------------------------------------------
    # 1. Defaults — baseline with the strategy's built-in defaults
    # ------------------------------------------------------------------
    {
        "name": "Defaults (baseline)",
        "params": {},
    },
    # ------------------------------------------------------------------
    # 2. Conservative — higher signal thresholds, tighter risk
    # ------------------------------------------------------------------
    {
        "name": "Conservative",
        "params": {
            "buy_bss_long_min": 0.65,
            "short_bss_short_min": 0.65,
            "buy_rsi_max": 35,
            "short_rsi_min": 70,
            "exit_long_bss_short_min": 0.45,
            "exit_short_bss_long_min": 0.45,
            "stoploss": -0.05,
        },
    },
    # ------------------------------------------------------------------
    # 3. Aggressive — lower signal thresholds, wider stops
    # ------------------------------------------------------------------
    {
        "name": "Aggressive",
        "params": {
            "buy_bss_long_min": 0.35,
            "short_bss_short_min": 0.35,
            "buy_rsi_max": 50,
            "short_rsi_min": 55,
            "exit_long_bss_short_min": 0.60,
            "exit_short_bss_long_min": 0.60,
            "stoploss": -0.12,
        },
    },
    # ------------------------------------------------------------------
    # 4. Long-only — high BSS Long sensitivity, shorts disabled via
    #    impossible short threshold
    # ------------------------------------------------------------------
    {
        "name": "Long-only",
        "params": {
            "buy_bss_long_min": 0.45,
            "short_bss_short_min": 0.99,
            "buy_rsi_max": 45,
            "short_rsi_min": 85,
            "exit_long_bss_short_min": 0.55,
            "stoploss": -0.08,
        },
    },
    # ------------------------------------------------------------------
    # 5. Short-only — high BSS Short sensitivity, longs disabled
    # ------------------------------------------------------------------
    {
        "name": "Short-only",
        "params": {
            "buy_bss_long_min": 0.99,
            "short_bss_short_min": 0.45,
            "buy_rsi_max": 20,
            "short_rsi_min": 60,
            "exit_short_bss_long_min": 0.45,
            "stoploss": -0.08,
        },
    },
    # ------------------------------------------------------------------
    # 6. Tight exits — default entries but exits trigger earlier
    # ------------------------------------------------------------------
    {
        "name": "Tight exits",
        "params": {
            "exit_long_bss_short_min": 0.35,
            "exit_short_bss_long_min": 0.35,
        },
    },
    # ------------------------------------------------------------------
    # 7. Wide ROI — let winners run longer
    # ------------------------------------------------------------------
    {
        "name": "Wide ROI",
        "params": {
            "minimal_roi": {
                "0": 0.25,
                "1440": 0.10,
                "4320": 0.04,
                "8640": 0.01,
            },
            "trailing_stop_positive": 0.05,
            "trailing_stop_positive_offset": 0.08,
        },
    },
    # ------------------------------------------------------------------
    # 8. Tight stoploss — quick exit on drawdown
    # ------------------------------------------------------------------
    {
        "name": "Tight stoploss (-4%)",
        "params": {
            "stoploss": -0.04,
            "trailing_stop_positive": 0.02,
            "trailing_stop_positive_offset": 0.03,
        },
    },
]


# ============================================================================
# Synthetic BSS signal generation (for backtesting without Glassnode API)
# ============================================================================
def generate_synthetic_bss_signals(
    ohlcv_data: dict[str, pd.DataFrame],
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic BSS Long and BSS Short signals derived from price
    action so the strategy has data to work with during backtesting.

    The signals are modeled as smoothed momentum indicators in [0, 1]:
      - BSS Long:  rises when price momentum is positive and RSI-like
                   oscillator is recovering from oversold.
      - BSS Short: rises when price momentum is negative and RSI-like
                   oscillator is falling from overbought.

    Returns a dict of {"bss_long": DataFrame, "bss_short": DataFrame}
    where each DataFrame has columns [date, value].
    """
    rng = np.random.default_rng(seed)
    all_dates = []
    all_close = []

    # Combine all pair data to get a representative price series
    for pair, df in ohlcv_data.items():
        if "BTC" in pair.upper():
            all_dates = df["date"].tolist()
            all_close = df["close"].values
            break

    # Fallback: use the first pair if no BTC pair found
    if len(all_dates) == 0:
        first_pair = next(iter(ohlcv_data))
        df = ohlcv_data[first_pair]
        all_dates = df["date"].tolist()
        all_close = df["close"].values

    n = len(all_close)
    if n == 0:
        return {}

    # Compute momentum features
    close = pd.Series(all_close, dtype=float)
    returns_short = close.pct_change(periods=6).fillna(0)   # ~1 day at 4h
    returns_long = close.pct_change(periods=42).fillna(0)   # ~1 week at 4h

    # RSI-like oscillator (0-1 range)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50) / 100.0  # normalize to [0, 1]

    # BSS Long: high when momentum is positive and recovering from oversold
    long_raw = (
        0.4 * (returns_short.clip(-0.1, 0.1) / 0.1 * 0.5 + 0.5)
        + 0.3 * (returns_long.clip(-0.2, 0.2) / 0.2 * 0.5 + 0.5)
        + 0.3 * (1 - rsi)  # inverted: low RSI → high long signal
    )
    # Add noise and smooth
    long_raw += rng.normal(0, 0.05, n)
    bss_long = long_raw.rolling(12, min_periods=1).mean().clip(0, 1)

    # BSS Short: high when momentum is negative and overbought
    short_raw = (
        0.4 * ((-returns_short).clip(-0.1, 0.1) / 0.1 * 0.5 + 0.5)
        + 0.3 * ((-returns_long).clip(-0.2, 0.2) / 0.2 * 0.5 + 0.5)
        + 0.3 * rsi  # high RSI → high short signal
    )
    short_raw += rng.normal(0, 0.05, n)
    bss_short = short_raw.rolling(12, min_periods=1).mean().clip(0, 1)

    dates = pd.Series(all_dates)

    return {
        "bss_long": pd.DataFrame({"date": dates, "value": bss_long.values}),
        "bss_short": pd.DataFrame({"date": dates, "value": bss_short.values}),
    }


# ============================================================================
# Backtest runner
# ============================================================================
def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build a freqtrade config dict from CLI args and config file."""
    cli_args: dict[str, Any] = {
        "config": args.config,
        "strategy": "GlassnodeOnChainStrategy",
        "strategy_path": str(Path(args.config[0]).resolve().parent / "strategies")
        if args.strategy_path is None
        else args.strategy_path,
        "datadir": args.datadir,
        "timerange": args.timerange,
        "verbosity": 0,
    }
    # Remove None values so Configuration uses defaults
    cli_args = {k: v for k, v in cli_args.items() if v is not None}

    config = Configuration(cli_args, RunMode.BACKTEST).get_config()

    # Overrides from CLI
    config["strategy"] = "GlassnodeOnChainStrategy"
    config["dry_run"] = True
    config["export"] = "none"
    config["disableparamexport"] = True
    config["runmode"] = RunMode.BACKTEST

    if args.pairs:
        config["exchange"]["pair_whitelist"] = args.pairs
    if args.stake_amount:
        config["stake_amount"] = args.stake_amount
    if args.wallet:
        config["dry_run_wallet"] = args.wallet
    if args.max_open_trades:
        config["max_open_trades"] = args.max_open_trades
    if args.timeframe:
        config["timeframe"] = args.timeframe

    return config


def apply_iteration_params(
    strategy,
    params: dict[str, Any],
    defaults: dict[str, Any],
) -> None:
    """
    Apply parameter overrides to the strategy, resetting to defaults first.

    Handles both hyperoptable parameters (DecimalParameter/IntParameter)
    and top-level strategy attributes (stoploss, minimal_roi, etc.).
    """
    # First, reset everything to defaults
    for attr_name, attr in strategy.enumerate_parameters():
        if attr_name in defaults:
            attr.value = defaults[attr_name]
    for key in ("stoploss", "minimal_roi", "trailing_stop",
                "trailing_stop_positive", "trailing_stop_positive_offset",
                "trailing_only_offset_is_reached"):
        if key in defaults:
            setattr(strategy, key, defaults[key])

    # Then apply overrides from this iteration
    param_names = {name for name, _ in strategy.enumerate_parameters()}
    for key, value in params.items():
        if key in param_names:
            # Hyperoptable parameter — set via attr.value
            for attr_name, attr in strategy.enumerate_parameters():
                if attr_name == key:
                    attr.value = value
                    break
        else:
            # Top-level attribute (stoploss, minimal_roi, etc.)
            setattr(strategy, key, value)


def run_single_backtest(
    bt: Backtesting,
    processed_data: dict[str, pd.DataFrame],
    min_date: datetime,
    max_date: datetime,
    market_change: float,
) -> dict[str, Any]:
    """Run a single backtest and return the stats dict."""
    bt_results = bt.backtest(
        processed=deepcopy(processed_data),
        start_date=min_date,
        end_date=max_date,
    )
    bt_results.update({
        "backtest_start_time": int(datetime.now(timezone.utc).timestamp()),
        "backtest_end_time": int(datetime.now(timezone.utc).timestamp()),
    })

    strat_stats = generate_strategy_stats(
        list(processed_data.keys()),
        bt.strategy.get_strategy_name(),
        bt_results,
        min_date,
        max_date,
        market_change=market_change,
        is_hyperopt=True,
    )
    return strat_stats


def format_results_table(all_results: list[dict[str, Any]]) -> str:
    """Format backtest results into a readable comparison table."""
    if not all_results:
        return "No results to display."

    # Metrics to display: (display_name, key, format_spec)
    metrics = [
        ("Total trades", "total_trades", "d"),
        ("Long trades", "trade_count_long", "d"),
        ("Short trades", "trade_count_short", "d"),
        ("Trades/day", "trades_per_day", ".2f"),
        ("Win rate", "winrate", ".1%"),
        ("Profit total", "profit_total", ".2%"),
        ("Profit (abs)", "profit_total_abs", ".2f"),
        ("Profit factor", "profit_factor", ".2f"),
        ("Sharpe ratio", "sharpe", ".3f"),
        ("Sortino ratio", "sortino", ".3f"),
        ("Calmar ratio", "calmar", ".3f"),
        ("SQN", "sqn", ".3f"),
        ("CAGR", "cagr", ".2%"),
        ("Expectancy", "expectancy", ".4f"),
        ("Max drawdown", "max_drawdown_account", ".2%"),
        ("Max DD (abs)", "max_drawdown_abs", ".2f"),
        ("Final balance", "final_balance", ".2f"),
        ("Avg duration", "holding_avg_s", "duration"),
        ("Best pair", "best_pair_key", "s"),
        ("Worst pair", "worst_pair_key", "s"),
    ]

    # Extract iteration names and metric values
    names = [r["iteration_name"] for r in all_results]
    col_width = max(20, max(len(n) for n in names) + 2)

    lines = []
    # Header
    header = f"{'Metric':<22}" + "".join(f"{n:>{col_width}}" for n in names)
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))

    for display_name, key, fmt in metrics:
        row = f"{display_name:<22}"
        for r in all_results:
            stats = r["stats"]
            val = stats.get(key, "N/A")
            if val == "N/A" or val is None:
                cell = "N/A"
            elif fmt == "duration":
                # holding_avg is a timedelta, convert to readable
                holding = stats.get("holding_avg")
                if holding:
                    total_sec = int(holding.total_seconds())
                    hours, rem = divmod(total_sec, 3600)
                    mins, _ = divmod(rem, 60)
                    cell = f"{hours}h {mins}m"
                else:
                    cell = "N/A"
            elif fmt == "s":
                cell = str(val)
            else:
                try:
                    cell = f"{val:{fmt}}"
                except (ValueError, TypeError):
                    cell = str(val)
            row += f"{cell:>{col_width}}"
        lines.append(row)

    lines.append("=" * len(header))

    # Parameter summary below the table
    lines.append("")
    lines.append("Parameter overrides per iteration:")
    lines.append("-" * 60)
    for r in all_results:
        params = r.get("params", {})
        if params:
            lines.append(f"  {r['iteration_name']}:")
            for k, v in params.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"  {r['iteration_name']}: (defaults)")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GlassnodeOnChainStrategy backtests with multiple parameter iterations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python user_data/strategies/backtest_glassnode_iterations.py \\
      --config user_data/config.json

  python user_data/strategies/backtest_glassnode_iterations.py \\
      --config user_data/config.json \\
      --timerange 20230101-20240101 \\
      --pairs BTC/USDT ETH/USDT \\
      --wallet 10000

  python user_data/strategies/backtest_glassnode_iterations.py \\
      --config user_data/config.json \\
      --no-synthetic-signals
        """,
    )
    parser.add_argument(
        "-c", "--config", nargs="+", required=True,
        help="Path(s) to freqtrade config file(s)",
    )
    parser.add_argument(
        "--timerange", type=str, default=None,
        help="Timerange for backtesting (e.g. 20230101-20240101)",
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Trading pairs (e.g. BTC/USDT ETH/USDT)",
    )
    parser.add_argument(
        "--timeframe", type=str, default=None,
        help="Override timeframe (default: strategy's 4h)",
    )
    parser.add_argument(
        "--stake-amount", type=float, default=None,
        help="Stake amount per trade",
    )
    parser.add_argument(
        "--wallet", type=float, default=None,
        help="Starting wallet balance",
    )
    parser.add_argument(
        "--max-open-trades", type=int, default=None,
        help="Maximum number of simultaneous open trades",
    )
    parser.add_argument(
        "--strategy-path", type=str, default=None,
        help="Path to strategies directory",
    )
    parser.add_argument(
        "--datadir", type=str, default=None,
        help="Path to data directory",
    )
    parser.add_argument(
        "--no-synthetic-signals", action="store_true",
        help="Skip synthetic BSS signal injection (use only if you have "
             "real Glassnode data cached or GLASSNODE_API_KEY is set)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic signal generation (default: 42)",
    )
    parser.add_argument(
        "--export-csv", type=str, default=None,
        help="Export results comparison to CSV file",
    )

    args = parser.parse_args()

    # ---- Setup logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("\n" + "=" * 70)
    print(" GlassnodeOnChainStrategy — Multi-Iteration Backtest Runner")
    print("=" * 70)

    # ---- Build config ----
    print("\n[1/5] Loading configuration...")
    config = build_config(args)
    print(f"  Exchange:    {config['exchange']['name']}")
    print(f"  Pairs:       {config['exchange']['pair_whitelist']}")
    print(f"  Timeframe:   {config.get('timeframe', '4h')}")
    print(f"  Timerange:   {config.get('timerange', 'all available')}")
    print(f"  Wallet:      {config.get('dry_run_wallet', 'default')}")

    # ---- Initialize backtester ----
    print("\n[2/5] Initializing backtesting engine...")
    bt = Backtesting(config)
    bt._set_strategy(bt.strategylist[0])
    strategy = bt.strategy
    print(f"  Strategy:    {strategy.get_strategy_name()}")

    # ---- Capture default parameter values ----
    defaults: dict[str, Any] = {}
    for attr_name, attr in strategy.enumerate_parameters():
        defaults[attr_name] = attr.value
    for key in ("stoploss", "minimal_roi", "trailing_stop",
                "trailing_stop_positive", "trailing_stop_positive_offset",
                "trailing_only_offset_is_reached"):
        defaults[key] = deepcopy(getattr(strategy, key))

    # ---- Load data ----
    print("\n[3/5] Loading historical data...")
    data, timerange = bt.load_bt_data()
    print(f"  Pairs loaded: {list(data.keys())}")
    for pair, df in data.items():
        print(f"    {pair}: {len(df)} candles "
              f"({df['date'].iloc[0]} → {df['date'].iloc[-1]})")

    # ---- Inject synthetic BSS signals if needed ----
    if not args.no_synthetic_signals:
        print("\n  Generating synthetic BSS signals for backtesting...")
        synthetic = generate_synthetic_bss_signals(data, seed=args.seed)
        if synthetic:
            strategy._glassnode_cache = synthetic
            strategy._glassnode_last_fetch = datetime.now(timezone.utc)
            print(f"  Injected synthetic signals: {list(synthetic.keys())}")
            for name, df in synthetic.items():
                print(f"    {name}: {len(df)} data points, "
                      f"range [{df['value'].min():.3f}, {df['value'].max():.3f}]")
        else:
            print("  WARNING: Could not generate synthetic signals (no price data).")
    else:
        print("\n  Skipping synthetic signals (--no-synthetic-signals)")

    # ---- Run iterations ----
    print(f"\n[4/5] Running {len(ITERATIONS)} backtest iterations...")
    all_results: list[dict[str, Any]] = []

    for i, iteration in enumerate(ITERATIONS, 1):
        name = iteration["name"]
        params = iteration["params"]

        print(f"\n  [{i}/{len(ITERATIONS)}] {name}...")
        t_start = _time.monotonic()

        # Reset strategy to defaults, then apply this iteration's params
        apply_iteration_params(strategy, params, defaults)

        # Re-run indicator calculation (needed because signals are merged
        # in populate_indicators and params may affect entry/exit logic)
        processed = strategy.advise_all_indicators(deepcopy(data))

        # Trim to timerange
        trimmed = trim_dataframes(processed, timerange, bt.required_startup)
        min_date, max_date = get_timerange(trimmed)
        market_change = calculate_market_change(trimmed, "close", min_date=min_date)

        # Run backtest
        stats = run_single_backtest(bt, processed, min_date, max_date, market_change)

        elapsed = _time.monotonic() - t_start

        # Extract best/worst pair names for the table
        best_pair = stats.get("best_pair", {})
        worst_pair = stats.get("worst_pair", {})
        stats["best_pair_key"] = best_pair.get("key", "N/A") if best_pair else "N/A"
        stats["worst_pair_key"] = worst_pair.get("key", "N/A") if worst_pair else "N/A"

        all_results.append({
            "iteration_name": name,
            "params": params,
            "stats": stats,
        })

        total_trades = stats.get("total_trades", 0)
        profit = stats.get("profit_total", 0)
        sharpe = stats.get("sharpe", 0)
        dd = stats.get("max_drawdown_account", 0)
        print(f"    Trades: {total_trades:>4}  |  "
              f"Profit: {profit:>8.2%}  |  "
              f"Sharpe: {sharpe:>7.3f}  |  "
              f"Max DD: {dd:>7.2%}  |  "
              f"Time: {elapsed:.1f}s")

    # ---- Display results ----
    print(f"\n[5/5] Results comparison\n")
    table = format_results_table(all_results)
    print(table)

    # ---- Export CSV if requested ----
    if args.export_csv:
        rows = []
        for r in all_results:
            row = {"iteration": r["iteration_name"]}
            row.update({k: v for k, v in r["stats"].items()
                        if isinstance(v, (int, float, str)) and k != "holding_avg"})
            # Add params as columns
            for pk, pv in r["params"].items():
                row[f"param_{pk}"] = pv
            rows.append(row)
        csv_df = pd.DataFrame(rows)
        csv_df.to_csv(args.export_csv, index=False)
        print(f"\nResults exported to: {args.export_csv}")

    # ---- Cleanup ----
    Backtesting.cleanup()
    print("\nDone.")


if __name__ == "__main__":
    main()
