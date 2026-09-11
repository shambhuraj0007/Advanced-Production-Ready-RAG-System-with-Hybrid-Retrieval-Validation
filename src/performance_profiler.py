"""
Performance profiler for RAG system.
Helps identify bottlenecks in query processing.
"""

import time
from typing import Dict, List
from functools import wraps


class PerformanceProfiler:
    """Profiles execution time of different operations."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = {}
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return TimingContext(self, operation_name)
    
    def record_time(self, operation_name: str, duration: float):
        """Record timing for an operation."""
        if self.enabled:
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(duration)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        stats = {}
        for op_name, times in self.timings.items():
            if times:
                stats[op_name] = {
                    'count': len(times),
                    'total': sum(times),
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'last': times[-1]
                }
        return stats
    
    def print_stats(self):
        """Print timing statistics."""
        if not self.enabled:
            return
        
        stats = self.get_stats()
        if not stats:
            print("No timing data collected.")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS")
        print("="*60)
        print(f"{'Operation':<30} {'Count':<8} {'Avg (s)':<12} {'Total (s)':<12} {'Last (s)':<12}")
        print("-"*60)
        
        # Sort by average time (descending)
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['avg'], reverse=True)
        
        for op_name, stat in sorted_stats:
            print(f"{op_name:<30} {stat['count']:<8} {stat['avg']:<12.4f} {stat['total']:<12.4f} {stat['last']:<12.4f}")
        
        total_time = sum(s['total'] for s in stats.values())
        print("-"*60)
        print(f"{'TOTAL':<30} {'':<8} {'':<12} {total_time:<12.4f}")
        print("="*60 + "\n")
    
    def reset(self):
        """Reset all timing data."""
        self.timings.clear()


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        if self.profiler.enabled:
            self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler.enabled and self.start_time:
            duration = time.time() - self.start_time
            self.profiler.record_time(self.operation_name, duration)


def profile_function(profiler: PerformanceProfiler, operation_name: str = None):
    """Decorator to profile a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            with profiler.time_operation(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

