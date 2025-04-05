"""
Risk Manager for the AI Day Trading Bot.

Handles position sizing, daily risk limits, and other risk checks.
"""
import logging
import asyncio
from typing import Optional, Tuple, Any, TYPE_CHECKING, Dict # Import Dict
from datetime import datetime, timezone

# Assuming Config and Redis client are passed or accessible
from ai_day_trader.utils.config import Config # Use new utils path
# Use the new redis client functions
from ai_day_trader.clients.redis_client import get_async_redis_client, set_redis_key, get_redis_key

# Import FeatureCalculator for type hinting
if TYPE_CHECKING:
    from ai_day_trader.feature_calculator import FeatureCalculator

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages risk parameters and calculates position sizes."""

    def __init__(
        self,
        config: Config,
        feature_calculator: 'FeatureCalculator', # Add feature_calculator dependency
        redis_client: Optional[Any] = None
    ):
        """
        Initialize the RiskManager.

        Args:
            config: Application configuration object.
            redis_client: Optional async Redis client instance. If None, will try to get one.
        """
        self.config = config
        self.feature_calculator = feature_calculator # Store feature_calculator
        self.redis_client = redis_client
        self.portfolio_size = config.get_float("PORTFOLIO_SIZE", 100000.0)
        # Load MAX_DAILY_RISK_AMOUNT, handling the optional None case
        _max_daily_risk_amount_str = config.get("MAX_DAILY_RISK_AMOUNT") # Get raw string value first
        self.max_daily_risk_amount = config.get_float("MAX_DAILY_RISK_AMOUNT", 0.0) # Load as float with 0.0 default
        if _max_daily_risk_amount_str is None: # If the env var wasn't set at all
            self.max_daily_risk_amount = None # Set back to None
        self.max_daily_risk_pct = config.get_float("MAX_DAILY_RISK_PCT", 0.01) # Percentage fallback
        self.max_trade_risk_pct = config.get_float("MAX_TRADE_RISK_PCT", 0.005)
        # Configuration for stop loss type and parameters
        self.stop_loss_type = config.get_str("STOP_LOSS_TYPE", "percentage").lower() # 'percentage' or 'atr'
        self.stop_loss_pct = config.get_float("STOP_LOSS_PCT", 0.02) # Used if type is 'percentage'
        self.atr_stop_multiplier = config.get_float("ATR_STOP_MULTIPLIER", 2.0) # Used if type is 'atr'
        # Ensure feature_calculator has the attribute before accessing it
        atr_period = getattr(self.feature_calculator, 'atr_period', 14) # Default if not found
        self.atr_feature_name = f"atr_{atr_period}" # Get expected ATR feature name

        self.daily_risk_key = f"risk:daily_limit_used:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

        if self.portfolio_size <= 0:
            raise ValueError("PORTFOLIO_SIZE must be positive.")
        if self.max_daily_risk_amount is not None and self.max_daily_risk_amount <= 0:
             logger.warning("MAX_DAILY_RISK_AMOUNT is set but not positive. Falling back to percentage.")
             self.max_daily_risk_amount = None # Fallback to percentage
        if self.max_daily_risk_amount is None and not (0 < self.max_daily_risk_pct < 1):
             raise ValueError("MAX_DAILY_RISK_PCT must be between 0 and 1 if fixed amount is not set.")
        if not (0 < self.max_trade_risk_pct < 1):
             raise ValueError("MAX_TRADE_RISK_PCT must be between 0 and 1.")
        if self.stop_loss_type not in ["percentage", "atr"]:
             raise ValueError("STOP_LOSS_TYPE must be 'percentage' or 'atr'.")
        if self.stop_loss_type == "percentage" and not (0 < self.stop_loss_pct < 1):
             raise ValueError("STOP_LOSS_PCT must be between 0 and 1 for percentage stop loss.")
        if self.stop_loss_type == "atr" and self.atr_stop_multiplier <= 0:
             raise ValueError("ATR_STOP_MULTIPLIER must be positive for ATR stop loss.")


        logger.info(f"RiskManager initialized (Stop Loss Type: {self.stop_loss_type}).")
        # Log the effective daily risk limit
        effective_limit = self.get_max_daily_risk()
        logger.info(f"Effective Max Daily Risk Limit: ${effective_limit:,.2f}")


    async def _get_redis_client(self) -> Optional[Any]:
        """Gets the Redis client, attempting to initialize if needed."""
        if self.redis_client:
            return self.redis_client
        try:
            self.redis_client = await get_async_redis_client()
            return self.redis_client
        except Exception as e:
            logger.error(f"Failed to get Redis client in RiskManager: {e}")
            return None

    def get_max_daily_risk(self) -> float:
        """Returns the maximum daily risk amount."""
        if self.max_daily_risk_amount is not None and self.max_daily_risk_amount > 0:
            return self.max_daily_risk_amount
        else:
            return self.portfolio_size * self.max_daily_risk_pct

    async def get_daily_limit_used(self) -> float:
        """Gets the amount of risk already used today from Redis."""
        client = await self._get_redis_client()
        if not client: return 0.0 # Assume 0 if Redis unavailable

        try:
            used_bytes = await client.get(self.daily_risk_key)
            return float(used_bytes.decode('utf-8')) if used_bytes else 0.0
        except Exception as e:
            logger.error(f"Error getting daily limit used from Redis: {e}")
            return 0.0 # Assume 0 on error

    async def update_daily_limit_used(self, amount: float):
        """Updates the daily risk limit used in Redis."""
        client = await self._get_redis_client()
        if not client: return

        try:
            # Use INCRBYFLOAT for atomic update
            new_total = await client.incrbyfloat(self.daily_risk_key, amount)
            # Set expiry at the end of the day (e.g., 24 hours from creation)
            await client.expire(self.daily_risk_key, 86400)
            logger.info(f"Updated daily risk used by ${amount:,.2f}. New total: ${new_total:,.2f}")
        except Exception as e:
            logger.error(f"Error updating daily limit used in Redis: {e}")

    async def get_remaining_daily_limit(self) -> float:
        """Calculates the remaining daily risk limit."""
        max_daily_risk = self.get_max_daily_risk()
        used_today = await self.get_daily_limit_used()
        remaining = max(0, max_daily_risk - used_today)
        logger.debug(f"Remaining daily risk limit: ${remaining:,.2f} (Max: ${max_daily_risk:,.2f}, Used: ${used_today:,.2f})")
        return remaining

    async def calculate_position_size(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_portfolio_value: float, # Use current value if available
        remaining_daily_limit: float,
        latest_features: Optional[Dict[str, Any]] = None # Add features argument
    ) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        """
        Calculates the position size based on risk parameters.

        Args:
            symbol: The stock symbol.
            side: 'buy' or 'sell'.
            entry_price: The estimated entry price.
            current_portfolio_value: Current total portfolio value.
            remaining_daily_limit: Remaining $ amount allowed for daily risk.
            latest_features: Optional dictionary containing latest calculated features (e.g., ATR).

        Returns:
            Tuple[Optional[int], Optional[float], Optional[float]]:
                (Quantity in shares, Estimated position value, Calculated stop loss price) or (None, None, None)
        """
        if entry_price <= 0:
            logger.error(f"Invalid entry price ({entry_price}) for {symbol}. Cannot calculate size.")
            return None, None, None

        # 1. Determine Max Risk per Trade ($)
        # Use the smaller of remaining daily limit and max trade risk % of portfolio
        max_risk_from_trade_pct = current_portfolio_value * self.max_trade_risk_pct
        max_risk_per_trade = min(remaining_daily_limit, max_risk_from_trade_pct)

        if max_risk_per_trade <= 0:
            logger.warning(f"No risk budget remaining for {symbol}. Max Trade Risk: ${max_risk_from_trade_pct:.2f}, Remaining Daily: ${remaining_daily_limit:.2f}")
            return None, None, None

        # 2. Determine Stop Loss Price based on configured type
        stop_loss_price = None
        risk_per_share = None

        if self.stop_loss_type == "atr":
            if latest_features and self.atr_feature_name in latest_features:
                atr_value = latest_features[self.atr_feature_name]
                if atr_value is not None and atr_value > 0:
                    stop_loss_price = entry_price - (atr_value * self.atr_stop_multiplier) if side == 'buy' else entry_price + (atr_value * self.atr_stop_multiplier)
                    risk_per_share = abs(entry_price - stop_loss_price)
                    logger.debug(f"Using ATR stop for {symbol}. ATR: {atr_value:.4f}, Multiplier: {self.atr_stop_multiplier}, Stop: {stop_loss_price:.2f}")
                else:
                    logger.warning(f"Invalid ATR value ({atr_value}) for {symbol}. Falling back to percentage stop for sizing.")
            else:
                logger.warning(f"ATR feature '{self.atr_feature_name}' not found for {symbol}. Falling back to percentage stop for sizing.")

        # Fallback to percentage if ATR failed or type is 'percentage'
        if stop_loss_price is None:
            stop_loss_price = entry_price * (1 - self.stop_loss_pct) if side == 'buy' else entry_price * (1 + self.stop_loss_pct)
            risk_per_share = abs(entry_price - stop_loss_price)
            logger.debug(f"Using percentage stop ({self.stop_loss_pct*100}%) for {symbol}. Stop: {stop_loss_price:.2f}")


        if risk_per_share is None or risk_per_share <= 1e-6: # Avoid division by zero
            logger.warning(f"Risk per share for {symbol} is zero or negligible. Cannot calculate size.")
            return None, None, None

        # 3. Calculate Quantity
        quantity = int(max_risk_per_trade / risk_per_share) # Round down to whole shares

        if quantity <= 0:
            logger.debug(f"Calculated quantity for {symbol} is zero based on risk parameters.")
            return None, None, None

        # 4. Calculate Estimated Position Value
        estimated_value = quantity * entry_price

        logger.info(f"Calculated size for {symbol} ({side}): Qty={quantity}, Value=${estimated_value:,.2f}, MaxTradeRisk=${max_risk_per_trade:,.2f}, Risk/Share=${risk_per_share:.2f}, StopPrice={stop_loss_price:.2f}")
        return quantity, estimated_value, stop_loss_price

    async def check_entry_risk(self, symbol: str, side: str, quantity: int, entry_price: float) -> bool:
        """
        Performs final risk checks before placing an entry order.
        (Currently integrated into calculate_position_size, but could be separate)

        Args:
            symbol: Stock symbol.
            side: 'buy' or 'sell'.
            quantity: Proposed quantity.
            entry_price: Estimated entry price.

        Returns:
            bool: True if the trade passes risk checks, False otherwise.
        """
        # This logic is effectively handled by calculate_position_size returning None if risk limits exceeded.
        # Could add more checks here if needed (e.g., max exposure per symbol).
        logger.debug(f"Performing entry risk check for {side} {quantity} {symbol} @ {entry_price}")
        # Re-calculate potential loss to double-check against remaining limit
        stop_loss_price = entry_price * (1 - self.stop_loss_pct) if side == 'buy' else entry_price * (1 + self.stop_loss_pct)
        potential_loss = abs(entry_price - stop_loss_price) * quantity
        remaining_limit = await self.get_remaining_daily_limit()

        if potential_loss > remaining_limit:
             logger.warning(f"Entry order for {symbol} blocked: Potential loss ${potential_loss:.2f} exceeds remaining daily limit ${remaining_limit:.2f}")
             return False

        # Add other checks like max position size, diversification rules, etc.

        return True
