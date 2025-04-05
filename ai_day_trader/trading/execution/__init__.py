"""
Execution module for trading engine.

This package provides execution systems for trading operations, including:
- Base classes and interfaces
- Circuit breaker functionality
- Live trading implementation (Alpaca)
- Paper trading implementation

The module is structured to allow easy extension with new execution systems
while maintaining a consistent interface.
"""

# Re-export main classes using relative imports
from .base import ExecutionSystem
from .circuit_breaker import (
    ExecutionCircuitBreaker,
    ExecutionCircuitBreakerRegistry,
    execution_circuit_breaker_registry,
    CircuitBreakerState
)
from .live import LiveExecution
from .paper import PaperExecution

# For backward compatibility with code that imports from trading_engine.execution
# We re-export everything that was previously in the single file
__all__ = [
    'ExecutionSystem',
    'ExecutionCircuitBreaker',
    'ExecutionCircuitBreakerRegistry',
    'execution_circuit_breaker_registry',
    'CircuitBreakerState',
    'LiveExecution',
    'PaperExecution',
]
