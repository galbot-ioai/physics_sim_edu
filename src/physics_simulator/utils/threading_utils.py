"""Threading utilities for physics simulator.

This module provides thread-safe utilities and context managers for 
managing concurrent access to simulation resources.
"""

import threading
import time
from contextlib import contextmanager
from typing import Optional, Any, Callable
import functools

from auro_utils import Logger
from .constants import DEFAULT_THREAD_TIMEOUT


class SimulatorLock:
    """Thread-safe lock manager for simulator operations.
    
    This class provides a more structured approach to managing locks
    compared to the global lock pattern used before.
    """
    
    def __init__(self, name: str = "SimulatorLock"):
        self._lock = threading.RLock()
        self._name = name
        self._owner = None
        
    @contextmanager
    def acquire(self, timeout: Optional[float] = None):
        """Context manager for acquiring the lock.
        
        Args:
            timeout: Maximum time to wait for lock acquisition
            
        Yields:
            bool: True if lock was acquired successfully
            
        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        acquired = False
        try:
            acquired = self._lock.acquire(timeout=timeout or DEFAULT_THREAD_TIMEOUT)
            if not acquired:
                raise TimeoutError(f"Failed to acquire {self._name} within {timeout}s")
            self._owner = threading.current_thread().name
            yield acquired
        finally:
            if acquired:
                self._owner = None
                self._lock.release()
    
    def is_locked(self) -> bool:
        """Check if the lock is currently held."""
        return self._lock._count > 0 if hasattr(self._lock, '_count') else False
    
    def get_owner(self) -> Optional[str]:
        """Get the name of the thread currently holding the lock."""
        return self._owner


def thread_safe(lock: Optional[SimulatorLock] = None, timeout: Optional[float] = None):
    """Decorator to make methods thread-safe.
    
    Args:
        lock: Lock instance to use. If None, a new lock is created.
        timeout: Maximum time to wait for lock acquisition
        
    Returns:
        Decorated function that is thread-safe
    """
    def decorator(func: Callable) -> Callable:
        # Create a lock if none provided (one per function)
        nonlocal lock
        if lock is None:
            lock = SimulatorLock(f"{func.__module__}.{func.__name__}")
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock.acquire(timeout=timeout):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class ThreadSafeCounter:
    """Thread-safe counter for tracking operations."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += 1
            return self._value
    
    def decrement(self) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self._value -= 1
            return self._value
    
    def get(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._value
    
    def reset(self) -> int:
        """Reset counter to zero and return previous value."""
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value


@contextmanager
def time_limit(seconds: float, operation_name: str = "operation"):
    """Context manager that enforces a time limit on operations.
    
    Args:
        seconds: Maximum time allowed for the operation
        operation_name: Name of the operation for error messages
        
    Raises:
        TimeoutError: If operation exceeds time limit
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if elapsed > seconds:
            # Create a temporary logger for warning
            logger = Logger()
            logger.log_warning(
                f"{operation_name} took {elapsed:.3f}s, which exceeds the {seconds}s limit"
            )


class ResourceManager:
    """Thread-safe resource manager for simulation objects."""
    
    def __init__(self):
        self._resources = {}
        self._lock = threading.RLock()
        self._usage_counter = ThreadSafeCounter()
    
    def register(self, key: str, resource: Any) -> None:
        """Register a resource with a unique key."""
        with self._lock:
            if key in self._resources:
                # Create a temporary logger for warning
                logger = Logger()
                logger.log_warning(f"Resource {key} already exists, overwriting")
            self._resources[key] = resource
            self._usage_counter.increment()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a resource by key."""
        with self._lock:
            return self._resources.get(key)
    
    def remove(self, key: str) -> bool:
        """Remove a resource by key."""
        with self._lock:
            if key in self._resources:
                del self._resources[key]
                self._usage_counter.decrement()
                return True
            return False
    
    def clear(self) -> int:
        """Clear all resources and return count of removed items."""
        with self._lock:
            count = len(self._resources)
            self._resources.clear()
            self._usage_counter.reset()
            return count
    
    def list_keys(self) -> list[str]:
        """Get list of all registered resource keys."""
        with self._lock:
            return list(self._resources.keys())
    
    def get_usage_count(self) -> int:
        """Get current number of registered resources."""
        return self._usage_counter.get() 