# AI Day Trading Strategy (GH200 Optimized)

## 1. Overview

This document outlines an autonomous AI-driven day trading strategy focused on maximizing dollar profit while managing risk, designed to leverage high-performance computing environments like the NVIDIA GH200. The strategy operates strictly intraday with no overnight positions.

**Core Principles:**

*   **Day Trading Only:** All positions are opened and closed within the same trading day via automated EOD closure.
*   **Profit Focus:** Prioritizes capturing dollar profit per trade through timely exits.
*   **Peak Exit Strategy:** Utilizes peak/trough detection algorithms (`trading_engine.peak_detection`) potentially combined with ML predictions to attempt exits near price extremes.
*   **Risk Management:**
    *   Strict daily position value limit (configurable, e.g., $5000 total across all open positions), tracked via Redis.
    *   Risk-based position sizing (configurable % of daily limit per trade).
    *   Configurable percentage-based stop-loss orders.
    *   Mandatory closure of all positions before market close.
*   **AI-Driven:** Leverages Machine Learning (primarily XGBoost via `ml_engine`) for entry signals and potentially exit refinement.
*   **Performance:** Designed with asynchronous operations and potential GPU acceleration (leveraging PyTorch, CuPy, TensorRT, XGBoost on GH200) in mind.

## 2. System Architecture & Environment

### 2.1 Hardware & Environment

*   **Compute:** NVIDIA GH200 Grace Hopper Superchip (ARM64 CPU + Hopper GPU)
*   **GPU:** NVIDIA H100 (Integrated within GH200) with 480GB+ High-Bandwidth Memory (HBM).
*   **CUDA Version:** 12.8 (as reported by `nvidia-smi`)
*   **Key Libraries:** PyTorch, CuPy, TensorRT, XGBoost (GPU-enabled), Pandas, NumPy, Redis, Alpaca Trading API, Polygon.io API Client.

### 2.2 Code Structure (`ai_day_trader/`)

The core logic resides within the `ai_day_trader` directory, distinct from the original `trading_engine` and `ml_engine` structures, while reusing essential components.

*   **Main Entry Point:** `ai_day_trader/main.py` initializes all components and manages the main application lifecycle and graceful shutdown.
*   **Configuration:** `ai_day_trader/config.py` loads base configuration (`utils/config.py`) and defines/validates strategy-specific parameters (risk levels, thresholds, etc.).
*   **Strategy Orchestration:** `ai_day_trader/strategy.py` (`AIStrategyRunner` class) contains the main asynchronous loop that runs periodically during market hours. It coordinates fetching signals, checking risk, monitoring positions, and triggering entries/exits.
*   **Signal Generation:** `ai_day_trader/signal_generator.py` (`SignalGenerator` class) is responsible for producing entry signals. It integrates:
    *   Rule-based strategies (e.g., ORB, VWAP Reversion from `trading_engine.day_trading`).
    *   ML model predictions (using `ml_engine.predictor`).
*   **Risk Management:** `ai_day_trader/risk_manager.py` (`RiskManager` class) handles:
    *   Calculating position sizes based on configured risk parameters (e.g., % risk of daily limit, stop-loss).
    *   Tracking and checking the daily position value limit against Redis.
*   **Execution & Broker Interface:** Reuses components from `trading_engine/`:
    *   `trading_engine.execution.live` (`LiveExecution`): Interfaces with Alpaca API for live trading.
    *   `trading_engine.trade_manager` (`TradeManager`): Manages order lifecycle and rollback for transactional integrity.
    *   `trading_engine.execution.circuit_breaker`: Provides resilience against repeated execution failures.
*   **Data Handling & Clients:** Reuses:
    *   `api_clients/`: Polygon REST/WebSocket clients, Alpaca client (via `LiveExecution`).
    *   `utils/redis_base.py`: For accessing Redis (latest ticks, daily limit tracking).
*   **Analysis Tools:** Reuses:
    *   `trading_engine.peak_detection` (`PeakDetector`): Used by `AIStrategyRunner` for exit logic.
*   **Machine Learning:** Reuses `ml_engine/` components:
    *   `ml_engine.predictor`: For loading trained models and making predictions.
    *   `ml_engine.trainers/`: For training entry and exit models (likely XGBoost).
    *   `ml_engine.xgboost_model`: Wrapper for GPU-accelerated XGBoost.

## 3. Algorithm Details (Implemented in `ai_day_trader/`)

### 3.1. Market Hours & Scheduling

*   `AIStrategyRunner` uses an Alpaca client instance (obtained via `LiveExecution`) to check `is_market_open`.
*   The main strategy loop in `AIStrategyRunner._run_loop` only executes fully during regular market hours.
*   EOD check (`strategy_logic.handle_eod_closure`) runs at the beginning of each loop cycle.

### 3.2. Signal Generation (`signal_generator.py`)

*   `SignalGenerator.generate_signals` is called by the `AIStrategyRunner`.
*   It currently integrates rule-based strategies from `trading_engine.day_trading` (ORB, VWAP).
*   **ML Integration:** Contains placeholders to call an ML predictor (e.g., `ml_engine.predictor`) to get entry signals based on trained models.
*   Returns a list of potential entry signals (`{'symbol': 'X', 'side': 'buy', 'source': '...', 'confidence': 0.85}`).

### 3.3. Position Sizing & Risk Management (`risk_manager.py`)

*   **Daily Limit Check:** `RiskManager.get_remaining_daily_limit` queries Redis for the `daily_used_capital:<YYYY-MM-DD>` key to determine available capital for the day against the `DAILY_POSITION_LIMIT` config value.
*   **Position Sizing:** `RiskManager.calculate_position_size` calculates order quantity based on:
    *   Configured risk percentage per trade (`RISK_PCT_PER_TRADE`) applied to the *total daily limit*.
    *   Configured stop-loss percentage (`STOP_LOSS_PCT`) to determine risk per share.
    *   Checks if the calculated position value exceeds the *remaining* daily limit.
    *   Checks if adding the position would exceed the *total* daily limit.
*   **Limit Update:** `RiskManager.update_daily_limit_used` is called by `AIStrategyRunner` after *attempting* an order submission to atomically increment the used capital in Redis for the current day.

### 3.4. Entry Execution (`strategy.py` -> `risk_manager.py` -> `trade_manager.py`)

*   `AIStrategyRunner._process_entries` iterates through signals from `SignalGenerator`.
*   For each valid signal on a symbol without an existing position, it calls `RiskManager.calculate_position_size`.
*   If sizing is successful and limits allow (`can_enter` is True), it calls `AIStrategyRunner._execute_order_managed`.
*   `_execute_order_managed` submits the order via `TradeManager` (for rollback) or directly to `ExecutionSystem`.
*   If order submission is attempted, `RiskManager.update_daily_limit_used` is called.

### 3.5. Exit Logic (`strategy.py` -> `strategy_logic.py`)

*   `AIStrategyRunner._monitor_exits` iterates through currently held positions fetched via `ExecutionSystem.get_positions`.
*   For each position, it calls `strategy_logic.check_exit_conditions`.
*   `check_exit_conditions` evaluates multiple exit criteria in order:
    *   **Peak/Trough Detection:** Calls `PeakDetector.find_peaks` (or `find_troughs` for shorts - *TODO*) on recent historical data. If the price drops below a peak by `PEAK_EXIT_DROP_THRESHOLD`, an exit signal is generated.
    *   **ML Exit Model:** (Placeholder) Checks if an ML exit predictor exists and if its prediction meets the `ML_EXIT_THRESHOLD`.
    *   **Stop Loss:** Calculates the stop price based on `avg_entry_price` and `STOP_LOSS_PCT`. Triggers an exit if the current price breaches the stop level.
*   If any exit condition is met, `_execute_order_managed` is called with a market IOC order to close the position.

### 3.6. End-of-Day (EOD) Closure (`strategy_logic.py`)

*   `strategy_logic.handle_eod_closure` is called at the start of the `AIStrategyRunner._run_loop`.
*   It checks the time until market close using the Alpaca clock.
*   If within the `EOD_CLOSE_MINUTES_BEFORE` window:
    *   Fetches all open positions.
    *   Submits market IOC orders via `_execute_order_managed` to close each position.
    *   Returns `True` to signal the main loop to halt further strategy execution for the day.

## 4. Machine Learning Models (`ml_engine/`)

*   The system leverages the existing `ml_engine` components.
*   **Entry Signal Model:**
    *   Trained using `ml_engine/trainers/signal_detection.py` (or similar).
    *   Likely an XGBoost Classifier or Regressor.
    *   Features potentially include technical indicators, price/volume patterns, volatility.
    *   Loaded and used by an instance of `ml_engine.predictor.Predictor` (integration placeholder in `ai_day_trader/signal_generator.py`).
*   **Exit Strategy Model:**
    *   Trained using `ml_engine/trainers/exit_strategy.py`.
    *   Likely an XGBoost Regressor predicting exit suitability/score.
    *   Features similar to entry model, plus potentially trade-specific features (P/L, duration).
    *   Loaded and used by an instance of `ml_engine.predictor.Predictor` (integration placeholder in `ai_day_trader/strategy.py`).

## 5. Setup & Execution

*   **Configuration:** Set API keys, symbols, risk parameters, etc., in environment variables (prefixed with `TRADING_`) or a `.env` file. See `utils/config.py` and `ai_day_trader/config.py` for details.
*   **Dependencies:** Install using `pip install -r requirements.txt`. Ensure GPU drivers, CUDA, PyTorch, CuPy, TensorRT, and XGBoost are correctly installed for the GH200 environment.
*   **Redis:** Ensure a Redis server is running and accessible per the configuration.
*   **Model Training:** Train the necessary ML models using the scripts/processes within `ml_engine/`. Ensure trained models (`.json` or `.xgb`), scalers, and feature lists are saved to the configured `MODEL_DIR`.
*   **Execution:** Run the bot using `python ai_day_trader/main.py`.

## 6. Code Structure Notes

*   The core strategy logic is now encapsulated within the `ai_day_trader` directory.
*   Reusable components (API clients, utils, execution, ML engine) are imported from their respective locations.
*   Files aim to be concise and focused.
*   Placeholders for ML model integration and advanced signal/risk logic remain but the structure is in place.
