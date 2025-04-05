#!/bin/bash

# Simple startup script for the AI Day Trading Bot

# Activate virtual environment if needed (adjust path if necessary)
# source venv/bin/activate

# Set environment variables if not using a .env file (optional)
# export TRADING_POLYGON_API_KEY="your_polygon_key"
# export TRADING_APCA_API_KEY_ID="your_alpaca_id"
# export TRADING_APCA_API_SECRET_KEY="your_alpaca_secret"
# export TRADING_REDIS_PASSWORD="your_redis_password"
# ... add other necessary env vars

# Run the main application
echo "Starting AI Day Trading Bot..."
python ai_day_trader/main.py

echo "AI Day Trading Bot finished."
