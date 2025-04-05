"""Base classes and shared types for the execution system."""
import time
import csv
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any, TypeVar
from datetime import datetime # Added missing import

# Import base class and potentially shared types/exceptions
# from trading_engine.base import TradingEngine # Removed obsolete import
# Define types locally to avoid import issues
from typing import TypedDict, Literal

# Import Config object, exceptions and logger from new utils path
from ai_day_trader.utils.config import Config # Import the main Config class
from ai_day_trader.utils.exceptions import (
    TradingError, APIError as TradingAPIError, ExecutionError,
    OrderError, CircuitBreakerError
)
from ai_day_trader.utils.logging_config import get_logger

# Define what a 'Position' dictionary looks like
class PositionInfo(TypedDict, total=False): # Use total=False for flexibility
    symbol: str
    quantity: float
    entry_price: float
    current_price: Optional[float]
    market_value: Optional[float]
    cost_basis: Optional[float]
    unrealized_pl: Optional[float]
    unrealized_plpc: Optional[float]
    entry_time: Optional[datetime] # Time when the position was entered
    lastday_price: Optional[float]
    change_today: Optional[float]
    last_update_time: Optional[datetime] # Time of last price update

# Define what 'TradeExecutionDetails' looks like
class TradeExecutionDetails(TypedDict, total=False): # Use total=False
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    quantity: float
    side: Literal['buy', 'sell']
    order_type: str
    time_in_force: Optional[str]
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: Literal['submitted', 'accepted', 'new', 'filled', 'partially_filled', 'canceled', 'rejected', 'error', 'expired', 'pending_cancel']
    fill_price: Optional[float]
    fill_quantity: Optional[float]
    timestamp: float # Unix timestamp
    cancelled_at: Optional[float] # Unix timestamp
    error_message: Optional[str]


# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Define log directory and file path
LOG_DIR = Path("./logs")
PAPER_TRADE_LOG_FILE = LOG_DIR / "paper_trades.csv"
PAPER_TRADE_LOG_FIELDNAMES = [
    "timestamp", "order_id", "client_order_id", "symbol", "side", "quantity",
    "order_type", "limit_price", "stop_price", "status", "fill_price",
    "fill_quantity", "commission", "error_message" # Added commission
]

def _setup_paper_trade_log():
    """Creates log directory and writes header if file doesn't exist."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if not PAPER_TRADE_LOG_FILE.exists():
            with open(PAPER_TRADE_LOG_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=PAPER_TRADE_LOG_FIELDNAMES)
                writer.writeheader()
    except Exception as e:
        get_logger(__name__).error(f"Failed to setup paper trade log: {e}")

_setup_paper_trade_log()  # Ensure log file is ready on module load

class ExecutionSystem(ABC):
    """Abstract base class for order execution systems."""

    def __init__(self, config: Config):
        """
        Initialize the execution system.

        Args:
            config: The main application Config object.
        """
        if not isinstance(config, Config):
             # Ensure a proper Config object is passed
             raise TypeError("ExecutionSystem requires a valid Config object.")
        self.config: Config = config # Store the main Config object
        self.logger = get_logger(self.__class__.__name__)
        self.broker_name = self.__class__.__name__.lower().replace("execution", "")
        # Subclasses should load their specific config (like API keys) using self.config

    @abstractmethod
    async def execute_order(self, order: Dict) -> TradeExecutionDetails:
        """Execute a trading order asynchronously."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order asynchronously."""
        pass

    @abstractmethod
    async def get_positions(self) -> Dict[str, PositionInfo]:
        """Get current portfolio positions asynchronously."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[TradeExecutionDetails]:
        """Get the status of a specific order asynchronously."""
        pass

    async def initialize(self):
        """Perform any necessary async initialization."""
        self.logger.info(f"Initializing {self.__class__.__name__} execution system...")
        pass

    async def close(self):
        """Perform any necessary async cleanup."""
        self.logger.info(f"Closing {self.__class__.__name__} execution system...")
        pass
