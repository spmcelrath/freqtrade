# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    timeframe_to_minutes,
)

import talib.abstract as ta
from technical import qtpylib


logger = logging.getLogger(__name__)


class GlassnodeOnChainStrategy(IStrategy):
    """
    A Freqtrade strategy that combines Glassnode on-chain metrics with
    technical analysis for BTC trading signals.

    On-chain metrics used (fetched daily from the Glassnode API):
        - MVRV Z-Score:  Market cycle valuation (overvalued/undervalued)
        - SOPR:          Spent Output Profit Ratio (profit-taking behavior)
        - NUPL:          Net Unrealized Profit/Loss (market-wide sentiment)
        - Exchange Net Position Change: Supply/demand pressure on exchanges
        - Active Addresses: Network health and adoption

    Technical indicators used for entry/exit timing:
        - RSI, EMA 20/50, Bollinger Bands

    Configuration:
        Set your Glassnode API key via one of:
        1. Environment variable: GLASSNODE_API_KEY
        2. Strategy config in your freqtrade config.json:
           "glassnode_api_key": "your_key_here"

    Pair requirements:
        This strategy is designed for BTC pairs (e.g. BTC/USDT).
        On-chain metrics are Bitcoin-specific.
    """

    INTERFACE_VERSION = 3

    can_short: bool = False

    # ROI: hold positions longer since on-chain signals are slow-moving
    minimal_roi = {
        "0": 0.15,     # 15% target at entry
        "720": 0.08,   # 8% after 12 hours
        "1440": 0.04,  # 4% after 24 hours
        "4320": 0.01,  # 1% after 3 days
    }

    stoploss = -0.08

    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    # 4h timeframe suits on-chain signals (daily granularity)
    timeframe = "4h"

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 50

    # ---------------------------------------------------------------
    # Hyperoptable parameters
    # ---------------------------------------------------------------
    buy_rsi = IntParameter(low=15, high=45, default=35, space="buy", optimize=True)
    sell_rsi = IntParameter(low=55, high=85, default=70, space="sell", optimize=True)

    # On-chain thresholds (hyperoptable)
    buy_mvrv_z_max = DecimalParameter(
        low=-0.5, high=3.0, default=1.5, decimals=1, space="buy", optimize=True,
    )
    sell_mvrv_z_min = DecimalParameter(
        low=2.0, high=7.0, default=3.5, decimals=1, space="sell", optimize=True,
    )
    buy_nupl_max = DecimalParameter(
        low=-0.2, high=0.5, default=0.3, decimals=2, space="buy", optimize=True,
    )
    sell_nupl_min = DecimalParameter(
        low=0.4, high=0.8, default=0.6, decimals=2, space="sell", optimize=True,
    )
    buy_sopr_max = DecimalParameter(
        low=0.90, high=1.02, default=0.98, decimals=2, space="buy", optimize=True,
    )

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "ema20": {"color": "blue"},
            "ema50": {"color": "orange"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
            "On-Chain Score": {
                "onchain_score": {"color": "green"},
            },
            "MVRV Z-Score": {
                "mvrv_z": {"color": "purple"},
            },
            "SOPR": {
                "sopr": {"color": "teal"},
            },
        },
    }

    # ---------------------------------------------------------------
    # Glassnode data management
    # ---------------------------------------------------------------
    GLASSNODE_BASE_URL = "https://api.glassnode.com/v1/metrics"
    # Metrics: (category, metric_name, column_name)
    GLASSNODE_METRICS = [
        ("market", "mvrv_z_score", "mvrv_z"),
        ("indicators", "sopr", "sopr"),
        ("indicators", "net_unrealized_profit_loss", "nupl"),
        ("distribution", "exchange_net_position_change", "exchange_netflow"),
        ("addresses", "active_count", "active_addresses"),
    ]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._glassnode_api_key: str = config.get(
            "glassnode_api_key", os.environ.get("GLASSNODE_API_KEY", "")
        )
        # Cache: maps metric column name -> DataFrame
        self._glassnode_cache: dict[str, pd.DataFrame] = {}
        self._glassnode_last_fetch: Optional[datetime] = None
        # Refresh at most once per hour (the data is daily anyway)
        self._glassnode_refresh_interval = timedelta(hours=1)

        if not self._glassnode_api_key:
            logger.warning(
                "GlassnodeOnChainStrategy: No Glassnode API key configured. "
                "Set GLASSNODE_API_KEY env var or 'glassnode_api_key' in config. "
                "Strategy will use technical indicators only."
            )

    # ------------------------------------------------------------------
    # Glassnode API helpers
    # ------------------------------------------------------------------
    def _fetch_glassnode_metric(
        self, category: str, metric: str, asset: str = "BTC",
        since: str = "2020-01-01", interval: str = "24h",
    ) -> Optional[pd.DataFrame]:
        """Fetch a single metric from the Glassnode API with error handling."""
        url = f"{self.GLASSNODE_BASE_URL}/{category}/{metric}"
        params = {
            "a": asset,
            "s": since,
            "i": interval,
            "api_key": self._glassnode_api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                logger.warning("Glassnode returned empty data for %s/%s", category, metric)
                return None
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["t"], unit="s", utc=True)
            df.rename(columns={"v": "value"}, inplace=True)
            df = df[["date", "value"]].sort_values("date").reset_index(drop=True)
            return df
        except requests.RequestException as e:
            logger.error("Glassnode API error for %s/%s: %s", category, metric, e)
            return None
        except (KeyError, ValueError) as e:
            logger.error("Glassnode parse error for %s/%s: %s", category, metric, e)
            return None

    def _refresh_glassnode_data(self) -> None:
        """Fetch all configured Glassnode metrics and populate the cache."""
        if not self._glassnode_api_key:
            return

        now = datetime.now(timezone.utc)
        if (
            self._glassnode_last_fetch is not None
            and (now - self._glassnode_last_fetch) < self._glassnode_refresh_interval
        ):
            return  # still fresh

        logger.info("GlassnodeOnChainStrategy: refreshing on-chain data from Glassnode")
        for category, metric, col_name in self.GLASSNODE_METRICS:
            df = self._fetch_glassnode_metric(category, metric)
            if df is not None:
                self._glassnode_cache[col_name] = df
            # Be respectful of rate limits
            time.sleep(1)

        self._glassnode_last_fetch = now
        logger.info(
            "Glassnode data refreshed: %d/%d metrics loaded",
            len(self._glassnode_cache),
            len(self.GLASSNODE_METRICS),
        )

    def _get_onchain_value_for_date(self, col_name: str, dt: datetime) -> Optional[float]:
        """Look up the most recent on-chain value at or before the given datetime."""
        if col_name not in self._glassnode_cache:
            return None
        df = self._glassnode_cache[col_name]
        mask = df["date"] <= dt
        if mask.any():
            return float(df.loc[mask, "value"].iloc[-1])
        return None

    def _merge_onchain_column(
        self, dataframe: DataFrame, col_name: str,
    ) -> DataFrame:
        """Merge a cached on-chain metric into the OHLCV dataframe using as-of join."""
        if col_name not in self._glassnode_cache:
            dataframe[col_name] = np.nan
            return dataframe

        onchain_df = self._glassnode_cache[col_name].copy()
        onchain_df.rename(columns={"value": col_name}, inplace=True)

        # Ensure both sides are timezone-aware UTC for the merge
        if dataframe["date"].dt.tz is None:
            candle_dates = dataframe["date"].dt.tz_localize("UTC")
        else:
            candle_dates = dataframe["date"].dt.tz_convert("UTC")

        if onchain_df["date"].dt.tz is None:
            onchain_df["date"] = onchain_df["date"].dt.tz_localize("UTC")
        else:
            onchain_df["date"] = onchain_df["date"].dt.tz_convert("UTC")

        # Use pandas merge_asof to align daily on-chain data to candle timestamps
        temp = pd.DataFrame({"date": candle_dates, "_idx": dataframe.index})
        temp = temp.sort_values("date")
        onchain_df = onchain_df.sort_values("date")

        merged = pd.merge_asof(
            temp, onchain_df, on="date", direction="backward",
        )
        merged = merged.set_index("_idx").sort_index()
        dataframe[col_name] = merged[col_name].values

        return dataframe

    # ------------------------------------------------------------------
    # Strategy lifecycle hooks
    # ------------------------------------------------------------------
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """Refresh Glassnode data at the start of each bot loop."""
        if self.dp and self.dp.runmode.value in ("live", "dry_run"):
            self._refresh_glassnode_data()

    def bot_start(self, **kwargs) -> None:
        """Fetch initial on-chain data when the bot starts."""
        self._refresh_glassnode_data()

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- Technical indicators ---
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2,
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]

        # --- Glassnode on-chain metrics ---
        for _category, _metric, col_name in self.GLASSNODE_METRICS:
            dataframe = self._merge_onchain_column(dataframe, col_name)

        # --- Composite on-chain score ---
        # Each sub-score ranges from -1 (bearish) to +1 (bullish)
        dataframe["onchain_score"] = self._compute_onchain_score(dataframe)

        return dataframe

    def _compute_onchain_score(self, dataframe: DataFrame) -> pd.Series:
        """
        Build a composite on-chain score from individual metrics.
        Returns a Series in [-1, +1] where positive = bullish, negative = bearish.
        Missing metrics are ignored (score derived from available data only).
        """
        scores = []
        weights = []

        # MVRV Z-Score: < 0 very bullish, 0-2 neutral, > 3 bearish, > 6 very bearish
        if "mvrv_z" in dataframe.columns:
            mvrv = dataframe["mvrv_z"]
            mvrv_score = pd.Series(np.where(
                mvrv < 0, 1.0,
                np.where(mvrv < 1.5, 0.5,
                np.where(mvrv < 3.0, 0.0,
                np.where(mvrv < 5.0, -0.5, -1.0)))
            ), index=dataframe.index)
            mvrv_score = mvrv_score.where(mvrv.notna(), np.nan)
            scores.append(mvrv_score)
            weights.append(2.0)  # highest weight

        # SOPR: < 1 selling at loss (bullish for reversal), > 1.05 profit-taking (bearish)
        if "sopr" in dataframe.columns:
            sopr = dataframe["sopr"]
            sopr_score = pd.Series(np.where(
                sopr < 0.95, 1.0,
                np.where(sopr < 1.0, 0.5,
                np.where(sopr < 1.02, 0.0,
                np.where(sopr < 1.05, -0.5, -1.0)))
            ), index=dataframe.index)
            sopr_score = sopr_score.where(sopr.notna(), np.nan)
            scores.append(sopr_score)
            weights.append(1.5)

        # NUPL: < 0 capitulation (bullish), 0-0.25 hope, 0.25-0.5 optimism,
        #        0.5-0.75 belief, > 0.75 euphoria (bearish)
        if "nupl" in dataframe.columns:
            nupl = dataframe["nupl"]
            nupl_score = pd.Series(np.where(
                nupl < 0, 1.0,
                np.where(nupl < 0.25, 0.5,
                np.where(nupl < 0.5, 0.0,
                np.where(nupl < 0.75, -0.5, -1.0)))
            ), index=dataframe.index)
            nupl_score = nupl_score.where(nupl.notna(), np.nan)
            scores.append(nupl_score)
            weights.append(1.5)

        # Exchange net flows: negative = outflows (bullish), positive = inflows (bearish)
        if "exchange_netflow" in dataframe.columns:
            flow = dataframe["exchange_netflow"]
            # Normalize relative to a 30-period rolling window
            roll_std = flow.rolling(30, min_periods=5).std()
            roll_mean = flow.rolling(30, min_periods=5).mean()
            z_flow = (flow - roll_mean) / roll_std.replace(0, np.nan)
            flow_score = (-z_flow).clip(-1, 1)
            flow_score = flow_score.where(flow.notna(), np.nan)
            scores.append(flow_score)
            weights.append(1.0)

        # Active addresses: rising = healthy (bullish), falling = weakening
        if "active_addresses" in dataframe.columns:
            aa = dataframe["active_addresses"]
            aa_pct = aa.pct_change(periods=7)  # 7-day change
            addr_score = pd.Series(np.where(
                aa_pct > 0.05, 1.0,
                np.where(aa_pct > 0, 0.3,
                np.where(aa_pct > -0.05, -0.3, -1.0))
            ), index=dataframe.index)
            addr_score = addr_score.where(aa.notna(), np.nan)
            scores.append(addr_score)
            weights.append(0.5)

        if not scores:
            return pd.Series(0.0, index=dataframe.index)

        # Weighted average, ignoring NaN contributions
        score_df = pd.DataFrame(scores).T
        weight_arr = np.array(weights)
        valid_mask = score_df.notna()
        weighted_sum = (score_df.fillna(0) * weight_arr).sum(axis=1)
        total_weight = (valid_mask.astype(float) * weight_arr).sum(axis=1)
        composite = (weighted_sum / total_weight.replace(0, np.nan)).fillna(0)
        return composite

    # ------------------------------------------------------------------
    # Entry signals
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # On-chain: composite score is bullish
                (dataframe["onchain_score"] > 0.2)
                # On-chain thresholds (hyperoptable)
                & (
                    dataframe["mvrv_z"].isna()
                    | (dataframe["mvrv_z"] < self.buy_mvrv_z_max.value)
                )
                & (
                    dataframe["nupl"].isna()
                    | (dataframe["nupl"] < self.buy_nupl_max.value)
                )
                & (
                    dataframe["sopr"].isna()
                    | (dataframe["sopr"] < self.buy_sopr_max.value)
                )
                # Technical confirmation
                & (dataframe["rsi"] < self.buy_rsi.value)
                & (dataframe["ema20"] > dataframe["ema50"])
                & (dataframe["close"] < dataframe["bb_mid"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    # ------------------------------------------------------------------
    # Exit signals
    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # On-chain: composite score turns bearish
                (dataframe["onchain_score"] < -0.2)
                # On-chain thresholds (hyperoptable)
                & (
                    dataframe["mvrv_z"].isna()
                    | (dataframe["mvrv_z"] > self.sell_mvrv_z_min.value)
                )
                & (
                    dataframe["nupl"].isna()
                    | (dataframe["nupl"] > self.sell_nupl_min.value)
                )
                # Technical confirmation
                & (dataframe["rsi"] > self.sell_rsi.value)
                & (dataframe["ema20"] < dataframe["ema50"])
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe

    # ------------------------------------------------------------------
    # Custom stoploss: tighten stop when on-chain score deteriorates
    # ------------------------------------------------------------------
    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime,
        current_rate: float, current_profit: float, after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """
        Dynamically tighten the stoploss when on-chain conditions worsen,
        even if the trade is still in profit.
        """
        if not self.dp:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]
        onchain_score = last_candle.get("onchain_score", 0)

        # If on-chain score turns negative while in profit, tighten stop
        if onchain_score < -0.3 and current_profit > 0.02:
            return -0.03  # tight 3% stop
        if onchain_score < 0 and current_profit > 0.01:
            return -0.05  # moderate 5% stop

        return None  # use default stoploss
