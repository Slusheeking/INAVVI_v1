"""Learning Engine Module

Provides automated model training, evaluation, deployment and monitoring capabilities.

Key Components:
- LearningEngineMain: Main orchestrator class
- ModelRegistry: Model versioning and storage
- ModelDeployer: Model deployment management
- ModelPerformanceTracker: Performance monitoring
- LearningEngine: Base training functionality
"""

__version__ = "1.0.0"

from .base import LearningEngine
from .main import LearningEngineMain
from .registry import ModelRegistry
from .deployer import ModelDeployer
from .tracker import ModelPerformanceTracker
from .drift_detector import DriftDetector

__all__ = [
    "LearningEngine",
    "LearningEngineMain",
    "ModelRegistry",
    "ModelDeployer",
    "ModelPerformanceTracker",
    "DriftDetector"
]
