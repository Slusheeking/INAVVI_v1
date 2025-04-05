"""
Feature Calculator for the AI Day Trading Bot.

Calculates technical indicators and other features needed for ML models,
leveraging RAPIDS cuDF for GPU acceleration.
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple

# Import cuDF if available, otherwise log error
try:
    import cudf
    import cupy as cp # cupy is often used with cudf
    CUDF_AVAILABLE = True
except ImportError:
    cudf = None
    cp = None
    CUDF_AVAILABLE = False

logger = logging.getLogger(__name__)

if not CUDF_AVAILABLE:
    logger.error("cuDF library not found. FeatureCalculator cannot use GPU acceleration.")
    # Optionally, raise an error or implement a fallback pandas version here
    # For now, we'll let it fail later if cudf is called

class FeatureCalculator:
    """Calculates features based on historical price/volume data using cuDF."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FeatureCalculator with configurable periods.

        Args:
            config: Configuration dictionary (optional).
        """
        if not CUDF_AVAILABLE:
            raise ImportError("cuDF is required for GPU-accelerated FeatureCalculator but not installed.")

        self.config = config or {}
        # Define periods for indicators - allow overrides from config
        self.sma_short_period = int(self.config.get("SMA_SHORT_PERIOD", 10))
        self.sma_long_period = int(self.config.get("SMA_LONG_PERIOD", 30))
        self.rsi_period = int(self.config.get("RSI_PERIOD", 14))
        self.macd_fast_period = int(self.config.get("MACD_FAST", 12))
        self.macd_slow_period = int(self.config.get("MACD_SLOW", 26))
        self.macd_signal_period = int(self.config.get("MACD_SIGNAL", 9))
        self.bbands_period = int(self.config.get("BBANDS_PERIOD", 20))
        self.bbands_std_dev = float(self.config.get("BBANDS_STDDEV", 2.0))
        self.atr_period = int(self.config.get("ATR_PERIOD", 14))

        # Validate periods
        if any(p <= 0 for p in [self.sma_short_period, self.sma_long_period, self.rsi_period,
                                 self.macd_fast_period, self.macd_slow_period, self.macd_signal_period,
                                 self.bbands_period, self.atr_period]):
            raise ValueError("Indicator periods must be positive integers.")
        if self.macd_fast_period >= self.macd_slow_period:
             raise ValueError("MACD fast period must be less than slow period.")

        logger.info("GPU FeatureCalculator initialized with indicator periods.")

    def _calculate_sma(self, series: 'cudf.Series', period: int) -> 'cudf.Series':
        """Calculates Simple Moving Average using cuDF."""
        if len(series) < period: return cudf.Series([np.nan] * len(series), index=series.index)
        # min_periods=1 ensures output length matches input length, NaNs where window not full
        return series.rolling(window=period, min_periods=1).mean()

    def _calculate_ema(self, series: 'cudf.Series', period: int) -> 'cudf.Series':
        """Calculates Exponential Moving Average using cuDF."""
        if len(series) < period: return cudf.Series([np.nan] * len(series), index=series.index)
        # Use span for period-like behavior in EWM
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    def _calculate_rsi(self, series: 'cudf.Series', period: int) -> 'cudf.Series':
        """Calculates Relative Strength Index using cuDF."""
        if len(series) <= period: return cudf.Series([np.nan] * len(series), index=series.index)
        delta = series.diff()
        gain = delta.applymap(lambda x: x if x > 0 else 0.0)
        loss = delta.applymap(lambda x: -x if x < 0 else 0.0)

        # Use Exponential Moving Average for RSI calculation
        avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Handle potential division by zero if avg_loss is 0 (replace inf with 100)
        # Handle initial NaNs (replace with 50 - neutral RSI)
        # cuDF fillna needs a scalar or another cuDF Series/DataFrame
        rsi = rsi.replace(cp.inf, 100.0).fillna(50.0)
        return rsi


    def _calculate_macd(self, series: 'cudf.Series', fast: int, slow: int, signal: int) -> Tuple['cudf.Series', 'cudf.Series', 'cudf.Series']:
        """Calculates MACD, Signal Line, and Histogram using cuDF."""
        if len(series) < slow:
            nan_series = cudf.Series([np.nan] * len(series), index=series.index)
            return nan_series, nan_series, nan_series

        ema_fast = self._calculate_ema(series, fast)
        ema_slow = self._calculate_ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        macd_hist = macd_line - signal_line
        return macd_line, signal_line, macd_hist

    def _calculate_bbands(self, series: 'cudf.Series', period: int, std_dev: float) -> Tuple['cudf.Series', 'cudf.Series', 'cudf.Series']:
        """Calculates Bollinger Bands using cuDF."""
        if len(series) < period:
            nan_series = cudf.Series([np.nan] * len(series), index=series.index)
            return nan_series, nan_series, nan_series

        middle_band = self._calculate_sma(series, period)
        rolling_std = series.rolling(window=period, min_periods=1).std()
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        return upper_band, middle_band, lower_band

    def _calculate_atr(self, high: 'cudf.Series', low: 'cudf.Series', close: 'cudf.Series', period: int) -> 'cudf.Series':
        """Calculates Average True Range using cuDF."""
        if len(high) <= period: # Need at least period+1 for shift
             return cudf.Series([np.nan] * len(close), index=close.index)

        high_low = high - low
        high_close_prev = (high - close.shift(1)).abs()
        low_close_prev = (low - close.shift(1)).abs()

        # Combine the three components into a DataFrame to find the max per row
        tr_df = cudf.DataFrame({
            'hl': high_low,
            'hcp': high_close_prev,
            'lcp': low_close_prev
        })
        tr = tr_df.max(axis=1) # Get the True Range

        # Use Exponential Moving Average (Wilder's smoothing) for ATR
        # Note: cuDF's ewm alpha calculation might differ slightly from pandas' default for ATR.
        # Using com (center of mass) is often preferred for Wilder's smoothing.
        # com = period - 1 is equivalent to alpha = 1 / period for large periods.
        atr = tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()
        return atr

    async def calculate_features(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Calculates all required features for a given symbol using cuDF.

        Args:
            symbol: The stock symbol.
            df: DataFrame with historical OHLCV data ('o', 'h', 'l', 'c', 'v').
                Index should be datetime.

        Returns:
            A dictionary containing the calculated features for the *latest* timestamp,
            or None if calculation fails.
        """
        if not CUDF_AVAILABLE:
            logger.error("cuDF not available, cannot calculate features on GPU.")
            return None

        required_cols = {'o', 'h', 'l', 'c', 'v'}
        if df is None or df.empty:
            logger.warning(f"No historical data provided for {symbol} feature calculation.")
            return None
        if not required_cols.issubset(df.columns):
            logger.error(f"Missing required columns ({required_cols - set(df.columns)}) in historical data for {symbol}.")
            return None
        # cuDF works best with default integer index, keep datetime index for potential future use if needed
        # if not isinstance(df.index, pd.DatetimeIndex):
        #      logger.error(f"Historical data for {symbol} must have a DatetimeIndex.")
        #      return None

        logger.debug(f"Calculating features for {symbol} using {len(df)} data points on GPU.")
        features = {}
        try:
            # Convert pandas DataFrame to cuDF DataFrame
            gdf = cudf.from_pandas(df)

            # Ensure correct data types (float for calculations)
            for col in ['o', 'h', 'l', 'c', 'v']:
                 if col in gdf.columns:
                      gdf[col] = gdf[col].astype('float32') # Use float32 for GPU

            close = gdf['c']
            high = gdf['h']
            low = gdf['l']
            volume = gdf['v']

            # --- Calculate Indicators using cuDF ---
            sma_short = self._calculate_sma(close, self.sma_short_period)
            sma_long = self._calculate_sma(close, self.sma_long_period)
            rsi = self._calculate_rsi(close, self.rsi_period)
            macd_line, signal_line, macd_hist = self._calculate_macd(close, self.macd_fast_period, self.macd_slow_period, self.macd_signal_period)
            upper_bb, middle_bb, lower_bb = self._calculate_bbands(close, self.bbands_period, self.bbands_std_dev)
            atr = self._calculate_atr(high, low, close, self.atr_period)

            # --- Extract Latest Values ---
            # Accessing the last element in cuDF can be done via index or .iloc
            # Using .iloc[-1] might be less efficient than direct indexing if index is known
            # Alternatively, convert the last value to CPU memory: .values_host[-1]
            last_idx = len(gdf) - 1
            if last_idx < 0:
                 logger.warning(f"DataFrame for {symbol} became empty after conversion/cleaning.")
                 return None

            features[f'sma_{self.sma_short_period}'] = sma_short.values_host[-1] if len(sma_short) > 0 else np.nan
            features[f'sma_{self.sma_long_period}'] = sma_long.values_host[-1] if len(sma_long) > 0 else np.nan
            features[f'rsi_{self.rsi_period}'] = rsi.values_host[-1] if len(rsi) > 0 else np.nan

            features['macd'] = macd_line.values_host[-1] if len(macd_line) > 0 else np.nan
            features['macd_signal'] = signal_line.values_host[-1] if len(signal_line) > 0 else np.nan
            features['macd_hist'] = macd_hist.values_host[-1] if len(macd_hist) > 0 else np.nan

            features['bb_upper'] = upper_bb.values_host[-1] if len(upper_bb) > 0 else np.nan
            features['bb_middle'] = middle_bb.values_host[-1] if len(middle_bb) > 0 else np.nan
            features['bb_lower'] = lower_bb.values_host[-1] if len(lower_bb) > 0 else np.nan

            # Calculate BB Width safely
            bb_middle_last = features['bb_middle']
            bb_upper_last = features['bb_upper']
            bb_lower_last = features['bb_lower']
            if pd.notna(bb_middle_last) and bb_middle_last > 1e-6 and pd.notna(bb_upper_last) and pd.notna(bb_lower_last):
                 features['bb_width'] = (bb_upper_last - bb_lower_last) / bb_middle_last
            else: features['bb_width'] = np.nan

            features[f'atr_{self.atr_period}'] = atr.values_host[-1] if len(atr) > 0 else np.nan

            # Add price/volume features from the latest row
            features['price'] = close.values_host[-1] # Use latest close as current price reference
            features['volume'] = volume.values_host[-1]

            # Add price change from previous close safely
            if len(close) > 1:
                 # Calculate pct_change on GPU, then get last value
                 price_change_pct = close.pct_change()
                 features['price_change_pct'] = price_change_pct.values_host[-1] if len(price_change_pct) > 0 else np.nan
            else: features['price_change_pct'] = np.nan


            # Clean NaN/Inf values - replace with 0 or handle appropriately
            final_features = {}
            for k, v in features.items():
                 # Check for standard NaN, numpy NaN, and cupy NaN/Inf if cupy is involved
                 is_nan = pd.isna(v)
                 is_inf = isinstance(v, (float, np.float32, np.float64)) and np.isinf(v)
                 if cp and isinstance(v, cp.ndarray): # Check if value is a cupy scalar/array
                      is_nan = is_nan or cp.isnan(v).any()
                      is_inf = is_inf or cp.isinf(v).any()

                 if is_nan or is_inf:
                      # logger.debug(f"Feature '{k}' for {symbol} is NaN/Inf, replacing with 0.0")
                      final_features[k] = 0.0 # Replace NaN/Inf with 0.0 (float)
                 else:
                      final_features[k] = float(v) # Ensure result is standard Python float

            logger.debug(f"Calculated features for {symbol} on GPU: {list(final_features.keys())}")
            return final_features

        except Exception as e:
            logger.error(f"Error calculating features for {symbol} on GPU: {e}", exc_info=True)
            return None
