#!/usr/bin/env python3
"""
Batch Processing Module

This module provides optimized batch processing for ML models:
1. Memory-efficient batch processing
2. Parallel processing capabilities
3. GPU acceleration
4. Progress tracking and monitoring
5. Error handling and recovery
6. Performance metrics collection

These utilities help optimize throughput for large-scale predictions.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Iterator
from datetime import datetime
import threading
import queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from utils.logging_config import get_logger
from utils.metrics_registry import PREDICTION_THROUGHPUT, PREDICTION_LATENCY, BATCH_PROCESSING_TIME

# Configure logging
logger = get_logger("ml_engine.batch_processor")

# Try to import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    logger.info(f"PyTorch is available (CUDA: {CUDA_AVAILABLE})")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    logger.warning("PyTorch is not available. Install with 'pip install torch' for GPU acceleration")

# Try to import Ray for distributed processing
try:
    import ray
    RAY_AVAILABLE = True
    logger.info("Ray is available for distributed processing")
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray is not available. Install with 'pip install ray' for distributed processing")


class BatchProcessor:
    """
    Optimized batch processor for ML models
    """
    
    def __init__(self, model: Any, batch_size: int = 1000, 
                use_gpu: bool = True, num_workers: int = 4,
                use_ray: bool = False, memory_limit_mb: int = 4096):
        """
        Initialize batch processor
        
        Args:
            model: ML model for predictions
            batch_size: Batch size for processing
            use_gpu: Whether to use GPU acceleration
            num_workers: Number of worker processes/threads
            use_ray: Whether to use Ray for distributed processing
            memory_limit_mb: Memory limit in MB
        """
        self.model = model
        self.batch_size = batch_size
        self.use_gpu = use_gpu and CUDA_AVAILABLE and TORCH_AVAILABLE
        self.num_workers = num_workers
        self.use_ray = use_ray and RAY_AVAILABLE
        self.memory_limit_mb = memory_limit_mb
        
        # Initialize Ray if using distributed processing
        if self.use_ray and RAY_AVAILABLE:
            if not ray.is_initialized():
                ray.init(num_cpus=self.num_workers, ignore_reinit_error=True)
                logger.info(f"Initialized Ray with {self.num_workers} CPUs")
        
        # Set device for PyTorch
        if self.use_gpu and TORCH_AVAILABLE:
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for processing")
    
    def optimize_batch_size(self, sample_data: Union[pd.DataFrame, np.ndarray]) -> int:
        """
        Optimize batch size based on memory constraints
        
        Args:
            sample_data: Sample data to estimate memory usage
            
        Returns:
            Optimized batch size
        """
        try:
            # Estimate memory per sample (in bytes)
            if isinstance(sample_data, pd.DataFrame):
                # For DataFrames, use memory_usage
                bytes_per_sample = sample_data.memory_usage(deep=True).sum() / len(sample_data)
            else:
                # For numpy arrays, calculate based on dtype
                bytes_per_sample = sample_data.nbytes / len(sample_data)
            
            # Add overhead for processing (approximately 2x)
            bytes_per_sample_with_overhead = bytes_per_sample * 2
            
            # Convert memory limit to bytes
            memory_limit_bytes = self.memory_limit_mb * 1024 * 1024
            
            # Calculate maximum batch size based on memory
            max_batch_size = int(memory_limit_bytes // bytes_per_sample_with_overhead)
            
            # Adjust for GPU memory if using GPU
            if self.use_gpu:
                # GPU memory is more constrained, use a more conservative estimate
                max_batch_size = max_batch_size // 2
            
            # Ensure batch size is at least 1
            max_batch_size = max(1, max_batch_size)
            
            # Limit batch size to a reasonable maximum
            max_batch_size = min(max_batch_size, 10000)
            
            logger.info(f"Optimized batch size: {max_batch_size} (estimated memory per sample: {bytes_per_sample_with_overhead:.2f} bytes)")
            
            return max_batch_size
            
        except Exception as e:
            logger.error(f"Error optimizing batch size: {e}")
            # Return default batch size
            return self.batch_size
    
    def create_batches(self, data: Union[pd.DataFrame, np.ndarray], 
                      batch_size: Optional[int] = None) -> Iterator:
        """
        Create batches from input data
        
        Args:
            data: Input data
            batch_size: Batch size (if None, use self.batch_size)
            
        Returns:
            Iterator of batches
        """
        batch_size = batch_size or self.batch_size
        
        # Get total number of samples
        num_samples = len(data)
        
        # Calculate number of batches
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        logger.info(f"Creating {num_batches} batches of size {batch_size} from {num_samples} samples")
        
        # Create batches
        for i in range(0, num_samples, batch_size):
            yield data[i:i+batch_size]
    
    def process_batch(self, batch: Union[pd.DataFrame, np.ndarray], 
                     predict_fn: Optional[Callable] = None) -> np.ndarray:
        """
        Process a single batch
        
        Args:
            batch: Batch of data
            predict_fn: Custom prediction function (if None, use model.predict)
            
        Returns:
            Batch predictions
        """
        try:
            start_time = time.time()
            
            # Use custom prediction function if provided
            if predict_fn is not None:
                predictions = predict_fn(batch)
            else:
                # Use model's predict method
                predictions = self.model.predict(batch)
            
            # Record metrics
            batch_size = len(batch)
            processing_time = time.time() - start_time
            
            # Record prediction latency
            PREDICTION_LATENCY.labels(model_name=getattr(self.model, "model_version", "unknown")).observe(processing_time)
            
            # Record batch throughput
            PREDICTION_THROUGHPUT.labels(
                model_name=getattr(self.model, "model_version", "unknown"),
                batch_size=str(batch_size)
            ).inc()
            
            # Record batch processing time
            BATCH_PROCESSING_TIME.labels(
                model_name=getattr(self.model, "model_version", "unknown"),
                batch_size=str(batch_size)
            ).observe(processing_time)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return empty array with same shape as expected output
            if hasattr(self.model, "output_shape"):
                output_shape = (len(batch),) + self.model.output_shape[1:]
                return np.zeros(output_shape)
            else:
                # Default to 1D output
                return np.zeros(len(batch))
    
    def process_parallel(self, data: Union[pd.DataFrame, np.ndarray],
                        predict_fn: Optional[Callable] = None,
                        batch_size: Optional[int] = None,
                        use_processes: bool = False,
                        show_progress: bool = True) -> np.ndarray:
        """
        Process data in parallel using multiple threads or processes
        
        Args:
            data: Input data
            predict_fn: Custom prediction function
            batch_size: Batch size
            use_processes: Whether to use processes instead of threads
            show_progress: Whether to show progress
            
        Returns:
            Predictions for all data
        """
        batch_size = batch_size or self.batch_size
        
        # Optimize batch size if needed
        if batch_size is None:
            batch_size = self.optimize_batch_size(data[:min(1000, len(data))])
        
        # Create batches
        batches = list(self.create_batches(data, batch_size))
        num_batches = len(batches)
        
        # Create result array
        results = [None] * num_batches
        
        # Create progress tracking
        if show_progress:
            progress_queue = queue.Queue()
            stop_event = threading.Event()
            
            # Start progress thread
            progress_thread = threading.Thread(
                target=self._show_progress,
                args=(progress_queue, num_batches, stop_event)
            )
            progress_thread.daemon = True
            progress_thread.start()
        
        # Process batches in parallel
        start_time = time.time()
        
        try:
            # Choose executor based on use_processes flag
            executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
            
            with executor_class(max_workers=self.num_workers) as executor:
                # Submit all batch processing tasks
                future_to_idx = {
                    executor.submit(self.process_batch, batch, predict_fn): i
                    for i, batch in enumerate(batches)
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                        
                        # Update progress
                        if show_progress:
                            progress_queue.put(1)
                            
                    except Exception as e:
                        logger.error(f"Error processing batch {idx}: {e}")
                        # Create empty result with appropriate shape
                        if idx > 0 and results[0] is not None:
                            # Use shape from first result
                            batch_size = len(batches[idx])
                            results[idx] = np.zeros((batch_size,) + results[0].shape[1:])
                        else:
                            # Default shape
                            results[idx] = np.zeros(len(batches[idx]))
            
            # Stop progress thread
            if show_progress:
                stop_event.set()
                progress_thread.join()
            
            # Combine results
            if all(r is not None for r in results):
                combined_results = np.vstack(results)
            else:
                # Handle case where some batches failed
                valid_results = [r for r in results if r is not None]
                if valid_results:
                    # Use shape from valid results
                    output_shape = (len(data),) + valid_results[0].shape[1:]
                    combined_results = np.zeros(output_shape)
                    
                    # Fill in valid results
                    start_idx = 0
                    for i, result in enumerate(results):
                        if result is not None:
                            batch_size = len(batches[i])
                            combined_results[start_idx:start_idx+batch_size] = result
                        start_idx += len(batches[i])
                else:
                    # All batches failed
                    combined_results = np.zeros(len(data))
            
            # Record total processing time
            total_time = time.time() - start_time
            logger.info(f"Processed {len(data)} samples in {total_time:.2f}s ({len(data)/total_time:.2f} samples/s)")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            # Stop progress thread
            if show_progress:
                stop_event.set()
                if progress_thread.is_alive():
                    progress_thread.join()
            
            # Return empty result
            return np.zeros(len(data))
    
    def process_ray(self, data: Union[pd.DataFrame, np.ndarray],
                   predict_fn: Optional[Callable] = None,
                   batch_size: Optional[int] = None,
                   show_progress: bool = True) -> np.ndarray:
        """
        Process data using Ray for distributed computing
        
        Args:
            data: Input data
            predict_fn: Custom prediction function
            batch_size: Batch size
            show_progress: Whether to show progress
            
        Returns:
            Predictions for all data
        """
        if not RAY_AVAILABLE:
            logger.warning("Ray is not available. Falling back to parallel processing.")
            return self.process_parallel(data, predict_fn, batch_size, show_progress=show_progress)
        
        batch_size = batch_size or self.batch_size
        
        # Optimize batch size if needed
        if batch_size is None:
            batch_size = self.optimize_batch_size(data[:min(1000, len(data))])
        
        # Create batches
        batches = list(self.create_batches(data, batch_size))
        num_batches = len(batches)
        
        # Define remote function for batch processing
        @ray.remote
        def process_batch_ray(batch, model_ref):
            try:
                # Get model from reference
                model = ray.get(model_ref)
                
                # Process batch
                if predict_fn is not None:
                    return predict_fn(batch)
                else:
                    return model.predict(batch)
            except Exception as e:
                logger.error(f"Error in Ray batch processing: {e}")
                return np.zeros(len(batch))
        
        # Create progress tracking
        if show_progress:
            progress_queue = queue.Queue()
            stop_event = threading.Event()
            
            # Start progress thread
            progress_thread = threading.Thread(
                target=self._show_progress,
                args=(progress_queue, num_batches, stop_event)
            )
            progress_thread.daemon = True
            progress_thread.start()
        
        # Process batches using Ray
        start_time = time.time()
        
        try:
            # Put model in Ray object store
            model_ref = ray.put(self.model)
            
            # Submit all batch processing tasks
            result_refs = [
                process_batch_ray.remote(batch, model_ref)
                for batch in batches
            ]
            
            # Process results as they complete
            results = []
            for result_ref in ray.get(result_refs):
                results.append(result_ref)
                
                # Update progress
                if show_progress:
                    progress_queue.put(1)
            
            # Stop progress thread
            if show_progress:
                stop_event.set()
                progress_thread.join()
            
            # Combine results
            combined_results = np.vstack(results)
            
            # Record total processing time
            total_time = time.time() - start_time
            logger.info(f"Processed {len(data)} samples using Ray in {total_time:.2f}s ({len(data)/total_time:.2f} samples/s)")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in Ray processing: {e}")
            # Stop progress thread
            if show_progress:
                stop_event.set()
                if progress_thread.is_alive():
                    progress_thread.join()
            
            # Fall back to parallel processing
            logger.info("Falling back to parallel processing")
            return self.process_parallel(data, predict_fn, batch_size, show_progress=show_progress)
    
    def process(self, data: Union[pd.DataFrame, np.ndarray],
               predict_fn: Optional[Callable] = None,
               batch_size: Optional[int] = None,
               parallel: bool = True,
               use_ray: Optional[bool] = None,
               show_progress: bool = True) -> np.ndarray:
        """
        Process data in batches
        
        Args:
            data: Input data
            predict_fn: Custom prediction function
            batch_size: Batch size
            parallel: Whether to use parallel processing
            use_ray: Whether to use Ray for distributed processing
            show_progress: Whether to show progress
            
        Returns:
            Predictions for all data
        """
        # Use instance setting if not specified
        use_ray = use_ray if use_ray is not None else self.use_ray
        
        # Choose processing method
        if use_ray and RAY_AVAILABLE:
            return self.process_ray(data, predict_fn, batch_size, show_progress)
        elif parallel:
            return self.process_parallel(data, predict_fn, batch_size, show_progress=show_progress)
        else:
            return self.process_sequential(data, predict_fn, batch_size, show_progress)
    
    def process_sequential(self, data: Union[pd.DataFrame, np.ndarray],
                          predict_fn: Optional[Callable] = None,
                          batch_size: Optional[int] = None,
                          show_progress: bool = True) -> np.ndarray:
        """
        Process data sequentially in batches
        
        Args:
            data: Input data
            predict_fn: Custom prediction function
            batch_size: Batch size
            show_progress: Whether to show progress
            
        Returns:
            Predictions for all data
        """
        batch_size = batch_size or self.batch_size
        
        # Optimize batch size if needed
        if batch_size is None:
            batch_size = self.optimize_batch_size(data[:min(1000, len(data))])
        
        # Create batches
        batches = list(self.create_batches(data, batch_size))
        num_batches = len(batches)
        
        # Create progress tracking
        if show_progress:
            progress_queue = queue.Queue()
            stop_event = threading.Event()
            
            # Start progress thread
            progress_thread = threading.Thread(
                target=self._show_progress,
                args=(progress_queue, num_batches, stop_event)
            )
            progress_thread.daemon = True
            progress_thread.start()
        
        # Process batches sequentially
        start_time = time.time()
        results = []
        
        try:
            for batch in batches:
                # Process batch
                batch_result = self.process_batch(batch, predict_fn)
                results.append(batch_result)
                
                # Update progress
                if show_progress:
                    progress_queue.put(1)
            
            # Stop progress thread
            if show_progress:
                stop_event.set()
                progress_thread.join()
            
            # Combine results
            combined_results = np.vstack(results)
            
            # Record total processing time
            total_time = time.time() - start_time
            logger.info(f"Processed {len(data)} samples sequentially in {total_time:.2f}s ({len(data)/total_time:.2f} samples/s)")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in sequential processing: {e}")
            # Stop progress thread
            if show_progress:
                stop_event.set()
                if progress_thread.is_alive():
                    progress_thread.join()
            
            # Return empty result
            return np.zeros(len(data))
    
    def _show_progress(self, progress_queue: queue.Queue, total: int, stop_event: threading.Event) -> None:
        """
        Show progress for batch processing
        
        Args:
            progress_queue: Queue for progress updates
            total: Total number of batches
            stop_event: Event to signal stopping
        """
        processed = 0
        start_time = time.time()
        
        while not stop_event.is_set():
            try:
                # Get progress updates (non-blocking)
                try:
                    count = progress_queue.get(timeout=0.1)
                    processed += count
                    progress_queue.task_done()
                except queue.Empty:
                    continue
                
                # Calculate progress
                percent = (processed / total) * 100
                elapsed = time.time() - start_time
                
                # Calculate ETA
                if processed > 0:
                    eta = (elapsed / processed) * (total - processed)
                    eta_str = f"ETA: {eta:.1f}s"
                else:
                    eta_str = "ETA: N/A"
                
                # Log progress
                logger.info(f"Progress: {processed}/{total} batches ({percent:.1f}%) - {eta_str}")
                
            except Exception as e:
                logger.error(f"Error in progress tracking: {e}")
        
        # Final progress update
        logger.info(f"Completed {processed}/{total} batches in {time.time() - start_time:.2f}s")


def batch_predict(model: Any, data: Union[pd.DataFrame, np.ndarray],
                 batch_size: int = 1000, use_gpu: bool = True,
                 parallel: bool = True, num_workers: int = 4,
                 use_ray: bool = False, show_progress: bool = True) -> np.ndarray:
    """
    Convenience function for batch prediction
    
    Args:
        model: ML model for predictions
        data: Input data
        batch_size: Batch size for processing
        use_gpu: Whether to use GPU acceleration
        parallel: Whether to use parallel processing
        num_workers: Number of worker processes/threads
        use_ray: Whether to use Ray for distributed processing
        show_progress: Whether to show progress
        
    Returns:
        Predictions for all data
    """
    # Create batch processor
    processor = BatchProcessor(
        model=model,
        batch_size=batch_size,
        use_gpu=use_gpu,
        num_workers=num_workers,
        use_ray=use_ray
    )
    
    # Process data
    return processor.process(
        data=data,
        parallel=parallel,
        use_ray=use_ray,
        show_progress=show_progress
    )

# Add process_features method to BatchProcessor class
BatchProcessor.process_features = lambda self, features, processor_fn, batch_size=None, parallel=True, use_ray=None, show_progress=True: (
    # Use instance setting if not specified
    (lambda use_ray_val=use_ray if use_ray is not None else self.use_ray, 
            batch_size_val=batch_size or self.batch_size,
            original_model=self.model: (
        # Create custom prediction function that applies processor_fn
        (lambda custom_predict_fn=lambda batch: processor_fn(batch): (
            # Try-finally block to restore original model
            (lambda: (
                # Set model to None to avoid using it in processing
                setattr(self, 'model', None),
                
                # Process features using the custom prediction function
                (lambda result=(
                    self.process_ray(features, custom_predict_fn, batch_size_val, show_progress)
                    if use_ray_val and RAY_AVAILABLE
                    else (
                        self.process_parallel(features, custom_predict_fn, batch_size_val, show_progress=show_progress)
                        if parallel
                        else self.process_sequential(features, custom_predict_fn, batch_size_val, show_progress)
                    )
                ): (
                    # Restore original model
                    setattr(self, 'model', original_model),
                    result
                )[1])()
            ))()
        ))()
    ))()
)

# Add docstring to process_features method
BatchProcessor.process_features.__doc__ = """
Process features in batches using a custom processor function

Args:
    features: Input features
    processor_fn: Function to process features
    batch_size: Batch size
    parallel: Whether to use parallel processing
    use_ray: Whether to use Ray for distributed processing
    show_progress: Whether to show progress
    
Returns:
    Processed features
"""