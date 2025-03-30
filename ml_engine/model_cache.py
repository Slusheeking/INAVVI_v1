#!/usr/bin/env python3
"""
Model Cache Module

This module provides a sophisticated caching system for ML models:
1. Time-based expiration (TTL)
2. LRU (Least Recently Used) eviction policy
3. Size-based limits
4. Thread-safe operations
5. Metrics tracking
6. Persistent storage option

The cache improves performance by avoiding repeated model loading
and provides graceful degradation through fallback mechanisms.
"""

import os
import time
import json
import pickle
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict

from utils.logging_config import get_logger
from utils.metrics_registry import CACHE_HIT_COUNT, CACHE_MISS_COUNT, CACHE_SIZE

# Configure logging
logger = get_logger("ml_engine.model_cache")


class ModelCache:
    """
    Advanced caching system for ML models with TTL and LRU eviction
    """
    
    def __init__(self, max_size: int = 10, ttl_seconds: int = 3600, 
                persistent: bool = False, cache_dir: str = None):
        """
        Initialize model cache
        
        Args:
            max_size: Maximum number of models in cache
            ttl_seconds: Time-to-live in seconds
            persistent: Whether to persist cache to disk
            cache_dir: Directory for persistent cache
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.persistent = persistent
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "model_cache")
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Create cache directory if persistent
        if self.persistent and not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info(f"Created cache directory: {self.cache_dir}")
            except Exception as e:
                logger.error(f"Error creating cache directory: {e}")
                self.persistent = False
        
        # Load persistent cache if enabled
        if self.persistent:
            self._load_cache_metadata()
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache and not expired"""
        with self.lock:
            if key not in self.cache:
                return False
            
            # Check if entry is expired
            if self.cache[key]["expiry"] < datetime.now():
                # Remove expired entry
                self._remove_item(key)
                return False
            
            return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item from cache
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached item or default
        """
        with self.lock:
            if key not in self:
                # Record cache miss
                CACHE_MISS_COUNT.labels(client="model_cache", cache_type="model").inc()
                return default
            
            # Record cache hit
            CACHE_HIT_COUNT.labels(client="model_cache", cache_type="model").inc()
            
            # Move to end to mark as most recently used
            self.cache.move_to_end(key)
            
            # Load model from disk if persistent
            if self.persistent and "model" not in self.cache[key]:
                try:
                    model_path = os.path.join(self.cache_dir, f"{key}.pkl")
                    if os.path.exists(model_path):
                        with open(model_path, "rb") as f:
                            self.cache[key]["model"] = pickle.load(f)
                    else:
                        logger.warning(f"Model file not found: {model_path}")
                        return default
                except Exception as e:
                    logger.error(f"Error loading model from disk: {e}")
                    return default
            
            return self.cache[key]["model"]
    
    def get_metadata(self, key: str, field: str = None) -> Any:
        """
        Get metadata for cached item
        
        Args:
            key: Cache key
            field: Specific metadata field or None for all
            
        Returns:
            Metadata or specific field
        """
        with self.lock:
            if key not in self:
                return None
            
            if field:
                return self.cache[key].get(field)
            else:
                # Return copy of metadata without model
                metadata = self.cache[key].copy()
                metadata.pop("model", None)
                return metadata
    
    def set(self, key: str, model: Any, ttl: Optional[int] = None, 
           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add item to cache
        
        Args:
            key: Cache key
            model: Model to cache
            ttl: Time-to-live in seconds (overrides default)
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Check if cache is full
            if key not in self.cache and len(self.cache) >= self.max_size:
                # Remove oldest item (first item in OrderedDict)
                self._remove_oldest()
            
            # Calculate expiry time
            ttl = ttl or self.ttl_seconds
            expiry = datetime.now() + timedelta(seconds=ttl)
            
            # Create cache entry
            entry = {
                "model": model,
                "expiry": expiry,
                "created": datetime.now(),
                "last_accessed": datetime.now(),
                "access_count": 0
            }
            
            # Add metadata if provided
            if metadata:
                entry.update(metadata)
            
            # Add to cache
            self.cache[key] = entry
            
            # Move to end to mark as most recently used
            self.cache.move_to_end(key)
            
            # Save to disk if persistent
            if self.persistent:
                try:
                    # Save model
                    model_path = os.path.join(self.cache_dir, f"{key}.pkl")
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
                    
                    # Save metadata
                    self._save_cache_metadata()
                    
                    logger.info(f"Saved model to disk: {model_path}")
                except Exception as e:
                    logger.error(f"Error saving model to disk: {e}")
                    return False
            
            # Update cache size metric
            CACHE_SIZE.labels(client="model_cache", cache_type="model").set(len(self.cache))
            
            return True
    
    def remove(self, key: str) -> bool:
        """
        Remove item from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if removed, False if not found
        """
        with self.lock:
            if key not in self.cache:
                return False
            
            self._remove_item(key)
            return True
    
    def clear(self) -> None:
        """Clear all items from cache"""
        with self.lock:
            # Remove all model files if persistent
            if self.persistent:
                for key in list(self.cache.keys()):
                    model_path = os.path.join(self.cache_dir, f"{key}.pkl")
                    if os.path.exists(model_path):
                        try:
                            os.remove(model_path)
                        except Exception as e:
                            logger.error(f"Error removing model file: {e}")
            
            # Clear cache
            self.cache.clear()
            
            # Update cache size metric
            CACHE_SIZE.labels(client="model_cache", cache_type="model").set(0)
    
    def cleanup(self) -> int:
        """
        Remove expired items from cache
        
        Returns:
            Number of items removed
        """
        with self.lock:
            now = datetime.now()
            expired_keys = [k for k, v in self.cache.items() if v["expiry"] < now]
            
            for key in expired_keys:
                self._remove_item(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary of cache statistics
        """
        with self.lock:
            stats = {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "persistent": self.persistent,
                "items": {}
            }
            
            # Add item stats
            for key, entry in self.cache.items():
                stats["items"][key] = {
                    "created": entry["created"].isoformat(),
                    "expiry": entry["expiry"].isoformat(),
                    "ttl_remaining": (entry["expiry"] - datetime.now()).total_seconds(),
                    "access_count": entry.get("access_count", 0)
                }
            
            return stats
    
    def _remove_item(self, key: str) -> None:
        """
        Remove item from cache and disk
        
        Args:
            key: Cache key
        """
        # Remove from disk if persistent
        if self.persistent:
            model_path = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                except Exception as e:
                    logger.error(f"Error removing model file: {e}")
        
        # Remove from cache
        self.cache.pop(key, None)
        
        # Update cache size metric
        CACHE_SIZE.labels(client="model_cache", cache_type="model").set(len(self.cache))
    
    def _remove_oldest(self) -> None:
        """Remove oldest item from cache"""
        if not self.cache:
            return
        
        # Get oldest key (first item in OrderedDict)
        oldest_key = next(iter(self.cache))
        
        # Remove item
        self._remove_item(oldest_key)
    
    def _save_cache_metadata(self) -> bool:
        """
        Save cache metadata to disk
        
        Returns:
            True if successful, False otherwise
        """
        if not self.persistent:
            return False
        
        try:
            # Create metadata without model objects
            metadata = {}
            for key, entry in self.cache.items():
                metadata[key] = entry.copy()
                metadata[key].pop("model", None)
                
                # Convert datetime objects to strings
                metadata[key]["expiry"] = metadata[key]["expiry"].isoformat()
                metadata[key]["created"] = metadata[key]["created"].isoformat()
                metadata[key]["last_accessed"] = metadata[key]["last_accessed"].isoformat()
            
            # Save metadata
            metadata_path = os.path.join(self.cache_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            
            return True
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
            return False
    
    def _load_cache_metadata(self) -> bool:
        """
        Load cache metadata from disk
        
        Returns:
            True if successful, False otherwise
        """
        if not self.persistent:
            return False
        
        try:
            metadata_path = os.path.join(self.cache_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                return False
            
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Convert string timestamps to datetime objects
            for key, entry in metadata.items():
                entry["expiry"] = datetime.fromisoformat(entry["expiry"])
                entry["created"] = datetime.fromisoformat(entry["created"])
                entry["last_accessed"] = datetime.fromisoformat(entry["last_accessed"])
                
                # Add to cache without loading model
                self.cache[key] = entry
            
            # Update cache size metric
            CACHE_SIZE.labels(client="model_cache", cache_type="model").set(len(self.cache))
            
            return True
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
            return False


# Global model cache instance
MODEL_CACHE = ModelCache(
    max_size=int(os.environ.get("MODEL_CACHE_SIZE", "10")),
    ttl_seconds=int(os.environ.get("MODEL_CACHE_TTL", "3600")),
    persistent=os.environ.get("MODEL_CACHE_PERSISTENT", "false").lower() == "true",
    cache_dir=os.environ.get("MODEL_CACHE_DIR", None)
)


def get_model(key: str, loader_fn=None, ttl: Optional[int] = None, 
             metadata: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get model from cache or load using loader function
    
    Args:
        key: Cache key
        loader_fn: Function to load model if not in cache
        ttl: Time-to-live in seconds
        metadata: Additional metadata
        
    Returns:
        Cached model or loaded model
    """
    # Check if model is in cache
    model = MODEL_CACHE.get(key)
    
    # If not in cache and loader function provided, load and cache
    if model is None and loader_fn is not None:
        try:
            # Load model
            start_time = time.time()
            model = loader_fn()
            load_time = time.time() - start_time
            
            # Add load time to metadata
            if metadata is None:
                metadata = {}
            metadata["load_time"] = load_time
            
            # Cache model
            if model is not None:
                MODEL_CACHE.set(key, model, ttl, metadata)
                logger.info(f"Loaded and cached model: {key} (load time: {load_time:.2f}s)")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    return model


def invalidate_model(key: str) -> bool:
    """
    Invalidate model in cache
    
    Args:
        key: Cache key
        
    Returns:
        True if invalidated, False otherwise
    """
    return MODEL_CACHE.remove(key)


def get_model_metadata(key: str, field: str = None) -> Any:
    """
    Get metadata for cached model
    
    Args:
        key: Cache key
        field: Specific metadata field or None for all
        
    Returns:
        Metadata or specific field
    """
    return MODEL_CACHE.get_metadata(key, field)


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics
    
    Returns:
        Dictionary of cache statistics
    """
    return MODEL_CACHE.get_stats()


def cleanup_cache() -> int:
    """
    Remove expired items from cache
    
    Returns:
        Number of items removed
    """
    return MODEL_CACHE.cleanup()