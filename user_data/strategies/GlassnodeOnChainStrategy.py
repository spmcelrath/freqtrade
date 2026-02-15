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
    A Freqtrade strategy driven by Glassnode's proprietary Bitcoin Sharpe
    Signal (BSS) pair, with technical analysis for entry/exit timing.

    Glassnode signals used:
        - BSS Long  (signals/btc_sharpe_signal):  ML-based signal in [0, 1].
          High values indicate favorable risk-adjusted conditions for longs.
        - BSS Short (signals/btc_bss_short):  ML-based signal in [0, 1].
          Values > 0.5 indicate high confidence in an imminent downturn (short).

    Technical indicators used for confirmation:
        - RSI (14), EMA 20/50, Bollinger Bands (20, 2)

    Configuration:
        Set your Glassnode API key via one of:
        1. Environment variable: GLASSNODE_API_KEY
        2. Strategy config in your freqtrade config.json:
           "glassnode_api_key": "your_key_here"

    Pair requirements:
        Designed for BTC pairs (e.g. BTC/USDT). The Glassnode BSS signals
        are Bitcoin-specific.
    """

    INTERFACE_VERSION = 3

    # Enable shorting — BSS Short drives short entries
    can_short: bool = True

    # ROI: on-chain signals are slower-moving, allow positions time to develop
    minimal_roi = {
        "0": 0.15,
        "720": 0.08,
        "1440": 0.04,
        "4320": 0.01,
    }

    stoploss = -0.08

    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    # 4h candles — BSS signals update hourly, 4h smooths noise
    timeframe = "4h"

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 50

    # ---------------------------------------------------------------
    # Hyperoptable parameters
    # ---------------------------------------------------------------
    # BSS Long thresholds
    buy_bss_long_min = DecimalParameter(
        low=0.3, high=0.8, default=0.5, decimals=2, space="buy", optimize=True,
    )
    buy_rsi_max = IntParameter(low=20, high=50, default=40, space="buy", optimize=True)

    # BSS Short thresholds
    short_bss_short_min = DecimalParameter(
        low=0.3, high=0.8, default=0.5, decimals=2, space="sell", optimize=True,
    )
    short_rsi_min = IntParameter(low=55, high=85, default=65, space="sell", optimize=True)

    # Exit thresholds — signal weakening triggers exit
    exit_long_bss_short_min = DecimalParameter(
        low=0.3, high=0.7, default=0.5, decimals=2, space="sell", optimize=True,
    )
    exit_short_bss_long_min = DecimalParameter(
        low=0.3, high=0.7, default=0.5, decimals=2, space="buy", optimize=True,
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
            "BSS Long": {
                "bss_long": {"color": "green"},
            },
            "BSS Short": {
                "bss_short": {"color": "magenta"},
            },
        },
    }

    # ---------------------------------------------------------------
    # Glassnode data management
    # ---------------------------------------------------------------
    GLASSNODE_BASE_URL = "https://api.glassnode.com/v1/metrics"

    # (category, metric_name, dataframe_column_name)
    GLASSNODE_SIGNALS = [
        ("signals", "btc_sharpe_signal", "bss_long"),
        ("signals", "btc_bss_short", "bss_short"),
    ]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._glassnode_api_key: str = config.get(
            "glassnode_api_key", os.environ.get("GLASSNODE_API_KEY", "")
        )
        # Cache: column_name -> DataFrame with columns [date, value]
        self._glassnode_cache: dict[str, pd.DataFrame] = {}
        self._glassnode_last_fetch: Optional[datetime] = None
        # Signals update hourly; refresh every 30 min to stay current
        self._glassnode_refresh_interval = timedelta(minutes=30)

        if not self._glassnode_api_key:
            logger.warning(
                "GlassnodeOnChainStrategy: No Glassnode API key configured. "
                "Set GLASSNODE_API_KEY env var or 'glassnode_api_key' in config. "
                "Strategy will fall back to technical indicators only."
            )

    # ------------------------------------------------------------------
    # Glassnode API helpers
    # ------------------------------------------------------------------
    def _fetch_glassnode_metric(
        self,
        category: str,
        metric: str,
        asset: str = "BTC",
        since: str = "2020-01-01",
        interval: str = "24h",
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
        """Fetch all configured Glassnode signals and update the cache."""
        if not self._glassnode_api_key:
            return

        now = datetime.now(timezone.utc)
        if (
            self._glassnode_last_fetch is not None
            and (now - self._glassnode_last_fetch) < self._glassnode_refresh_interval
        ):
            return  # cache is still fresh

        logger.info("GlassnodeOnChainStrategy: refreshing BSS signals from Glassnode")
        for category, metric, col_name in self.GLASSNODE_SIGNALS:
            df = self._fetch_glassnode_metric(category, metric)
            if df is not None:
                self._glassnode_cache[col_name] = df
            time.sleep(1)  # respect rate limits

        self._glassnode_last_fetch = now
        cached = len(self._glassnode_cache)
        total = len(self.GLASSNODE_SIGNALS)
        logger.info("Glassnode BSS signals refreshed: %d/%d loaded", cached, total)

    def _merge_signal_column(self, dataframe: DataFrame, col_name: str) -> DataFrame:
        """Merge a cached Glassnode signal into the OHLCV dataframe via as-of join."""
        if col_name not in self._glassnode_cache:
            dataframe[col_name] = np.nan
            return dataframe

        signal_df = self._glassnode_cache[col_name].copy()
        signal_df.rename(columns={"value": col_name}, inplace=True)

        # Ensure timezone-aware UTC on both sides
        if dataframe["date"].dt.tz is None:
            candle_dates = dataframe["date"].dt.tz_localize("UTC")
        else:
            candle_dates = dataframe["date"].dt.tz_convert("UTC")

        if signal_df["date"].dt.tz is None:
            signal_df["date"] = signal_df["date"].dt.tz_localize("UTC")
        else:
            signal_df["date"] = signal_df["date"].dt.tz_convert("UTC")

        # merge_asof: align signal data (hourly/daily) to each candle timestamp
        temp = pd.DataFrame({"date": candle_dates, "_idx": dataframe.index})
        temp = temp.sort_values("date")
        signal_df = signal_df.sort_values("date")

        merged = pd.merge_asof(temp, signal_df, on="date", direction="backward")
        merged = merged.set_index("_idx").sort_index()
        dataframe[col_name] = merged[col_name].values

        return dataframe

    # ------------------------------------------------------------------
    # Strategy lifecycle hooks
    # ------------------------------------------------------------------
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """Refresh Glassnode signals at the start of each bot loop."""
        if self.dp and self.dp.runmode.value in ("live", "dry_run"):
            self._refresh_glassnode_data()

    def bot_start(self, **kwargs) -> None:
        """Fetch initial signal data when the bot starts."""
        self._refresh_glassnode_data()

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- Technical indicators for confirmation ---
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2,
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]

        # --- Glassnode BSS signals ---
        for _category, _metric, col_name in self.GLASSNODE_SIGNALS:
            dataframe = self._merge_signal_column(dataframe, col_name)

        return dataframe

    # ------------------------------------------------------------------
    # Entry signals
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ---- LONG entry: BSS Long signal is high ----
        dataframe.loc[
            (
                # Primary: BSS Long signal above threshold
                (dataframe["bss_long"] >= self.buy_bss_long_min.value)
                # Confirmation: BSS Short is NOT active (no conflicting short signal)
                & (
                    dataframe["bss_short"].isna()
                    | (dataframe["bss_short"] < self.short_bss_short_min.value)
                )
                # TA confirmation: RSI not overbought, trend aligned
                & (dataframe["rsi"] < self.buy_rsi_max.value)
                & (dataframe["ema20"] > dataframe["ema50"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        # ---- SHORT entry: BSS Short signal is active ----
        dataframe.loc[
            (
                # Primary: BSS Short signal above threshold (>0.5 = high confidence)
                (dataframe["bss_short"] >= self.short_bss_short_min.value)
                # Confirmation: BSS Long is NOT strong (no conflicting long signal)
                & (
                    dataframe["bss_long"].isna()
                    | (dataframe["bss_long"] < self.buy_bss_long_min.value)
                )
                # TA confirmation: RSI not oversold, trend aligned
                & (dataframe["rsi"] > self.short_rsi_min.value)
                & (dataframe["ema20"] < dataframe["ema50"])
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    # ------------------------------------------------------------------
    # Exit signals
    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ---- Exit LONG: BSS Short activates (downturn expected) ----
        dataframe.loc[
            (
                (dataframe["bss_short"] >= self.exit_long_bss_short_min.value)
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        # ---- Exit SHORT: BSS Long activates (recovery expected) ----
        dataframe.loc[
            (
                (dataframe["bss_long"] >= self.exit_short_bss_long_min.value)
                & (dataframe["volume"] > 0)
            ),
            "exit_short",
        ] = 1

        return dataframe

    # ------------------------------------------------------------------
    # Custom stoploss: tighten based on opposing signal strength
    # ------------------------------------------------------------------
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """
        Dynamically tighten stoploss when the opposing BSS signal strengthens,
        even if the position is still in profit.
        """
        if not self.dp:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        bss_long = last.get("bss_long", np.nan)
        bss_short = last.get("bss_short", np.nan)

        if trade.is_short:
            # In a short: tighten stop if BSS Long is rising
            if not np.isnan(bss_long) and bss_long > 0.6 and current_profit > 0.01:
                return -0.03
            if not np.isnan(bss_long) and bss_long > 0.4 and current_profit > 0.005:
                return -0.05
        else:
            # In a long: tighten stop if BSS Short is rising
            if not np.isnan(bss_short) and bss_short > 0.6 and current_profit > 0.01:
                return -0.03
            if not np.isnan(bss_short) and bss_short > 0.4 and current_profit > 0.005:
                return -0.05

        return None  # use default stoploss
