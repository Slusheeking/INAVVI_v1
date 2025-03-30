#!/usr/bin/env python3
"""
Test suite for the stock_selection module

This test suite verifies the functionality of the StockSelectionCore class:
- Initialization and Redis connection
- Lifecycle methods (start/stop)
- Core selection methods
"""

import unittest
import asyncio
from unittest.mock import MagicMock, patch
import logging

from stock_selection.core import StockSelectionCore
from utils.exceptions import RedisConnectionError

class TestStockSelectionCoreInitialization(unittest.TestCase):
    """Test the initialization of StockSelectionCore"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_redis = MagicMock()
        self.mock_redis.ping.return_value = True

    def test_init_default(self):
        """Test initialization with default Redis client"""
        core = StockSelectionCore()
        self.assertIsNotNone(core.redis_client)

    def test_init_with_redis(self):
        """Test initialization with provided Redis client"""
        core = StockSelectionCore(redis_client=self.mock_redis)
        self.assertEqual(core.redis_client, self.mock_redis)

    def test_redis_connection_failure(self):
        """Test Redis connection validation failure"""
        self.mock_redis.ping.return_value = False
        with self.assertRaises(RedisConnectionError):
            StockSelectionCore(redis_client=self.mock_redis)

    def test_redis_connection_error(self):
        """Test Redis connection error handling"""
        self.mock_redis.ping.side_effect = Exception("Connection failed")
        with self.assertRaises(RedisConnectionError):
            StockSelectionCore(redis_client=self.mock_redis)


class TestStockSelectionCoreLifecycle(unittest.IsolatedAsyncioTestCase):
    """Test the lifecycle methods of StockSelectionCore"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_redis = MagicMock()
        self.mock_redis.ping.return_value = True
        self.core = StockSelectionCore(redis_client=self.mock_redis)

    async def test_start(self):
        """Test the start method"""
        with patch.object(logging.getLogger("stock_selection_core"), 'info') as mock_log:
            await self.core.start()
            mock_log.assert_called_once_with("Starting Stock Selection Core")

    async def test_stop(self):
        """Test the stop method"""
        with patch.object(logging.getLogger("stock_selection_core"), 'info') as mock_log:
            await self.core.stop()
            mock_log.assert_called_once_with("Stock Selection Core stopped")


class TestStockSelectionCoreMethods(unittest.IsolatedAsyncioTestCase):
    """Test the core selection methods of StockSelectionCore"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_redis = MagicMock()
        self.mock_redis.ping.return_value = True
        self.core = StockSelectionCore(redis_client=self.mock_redis)

    async def test_build_universe(self):
        """Test build_universe method (placeholder)"""
        result = await self.core.build_universe()
        self.assertEqual(result, [])

    async def test_score_stocks(self):
        """Test score_stocks method (placeholder)"""
        result = await self.core.score_stocks(["AAPL", "MSFT"])
        self.assertEqual(result, {})


if __name__ == '__main__':
    unittest.main()
