# AI Day Trader

An advanced algorithmic trading system using machine learning for real-time market analysis and automated trading.

## Project Overview

This AI Day Trading system integrates multiple components:

- Real-time market data processing
- ML-powered prediction models (XGBoost with GPU acceleration)
- Automated trading execution
- Risk management
- Performance analytics

## Features

- **High-Performance Trading**: Optimized for low-latency execution
- **Machine Learning**: XGBoost models for market prediction
- **GPU Acceleration**: CUDA-enabled processing for faster model inference
- **Risk Management**: Sophisticated position sizing and drawdown prevention
- **Multiple Data Sources**: Integration with Polygon.io, Unusual Whales, and other providers
- **Paper & Live Trading**: Supports both simulation and real trading environments

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, for accelerated processing)
- Redis
- API access to supported data providers

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/slusheeking/AI-Day-Trader.git
   cd AI-Day-Trader
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   cp .env.sample .env
   # Edit .env with your API keys and configuration
   ```

### Running the System

Start the trading system:

```
./start_ai_trader.sh
```

For development and testing, individual components can be run separately:

```
python -m ai_day_trader.main
```

## Project Structure

- `ai_day_trader/`: Core application code
  - `clients/`: API client implementations
  - `ml/`: Machine learning models and predictors
  - `trading/`: Trading logic and execution
  - `utils/`: Utility functions and helpers
- `tests/`: Test suite
- `logs/`: Trading logs and performance data

## Development

### Running Tests

```
pytest
```

### GPU Setup

For GPU acceleration:

```
./build_xgboost_cuda.sh
```

## License

Proprietary - All rights reserved

## Contact

For questions or support, contact [zach@getmeslushed.com](mailto:zach@getmeslushed.com)