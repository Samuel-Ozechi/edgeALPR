"""Performance metrics collection for edgeALPR pipeline."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Metrics for model inference performance."""
    
    model_name: str
    total_inferences: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_inference(self, latency_ms: float) -> None:
        """Record an inference latency.
        
        Args:
            latency_ms: Inference time in milliseconds
        """
        self.total_inferences += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.recent_latencies.append(latency_ms)
    
    def get_average_latency(self) -> float:
        """Get average inference latency in milliseconds.
        
        Returns:
            Average latency, or 0.0 if no inferences recorded
        """
        if self.total_inferences == 0:
            return 0.0
        return self.total_latency_ms / self.total_inferences
    
    def get_recent_average_latency(self) -> float:
        """Get average of recent inferences (last 100).
        
        Returns:
            Recent average latency
        """
        if not self.recent_latencies:
            return 0.0
        return sum(self.recent_latencies) / len(self.recent_latencies)
    
    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary.
        
        Returns:
            Metrics dict with all fields
        """
        return {
            "model_name": self.model_name,
            "total_inferences": self.total_inferences,
            "average_latency_ms": self.get_average_latency(),
            "recent_average_latency_ms": self.get_recent_average_latency(),
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "recent_sample_count": len(self.recent_latencies)
        }


@dataclass
class CacheMetrics:
    """Metrics for cache performance."""
    
    hits: int = 0
    misses: int = 0
    
    def add_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1
    
    def add_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate (0-1).
        
        Returns:
            Cache hit rate, or 0.0 if no accesses
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_accesses": self.hits + self.misses,
            "hit_rate": self.get_hit_rate()
        }


class MetricsCollector:
    """Thread-safe metrics collector for pipeline components."""
    
    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._lock = Lock()
        self._inference_metrics: dict[str, InferenceMetrics] = {}
        self._cache_metrics: dict[str, CacheMetrics] = {}
        self._start_time = datetime.now(timezone.utc)
        logger.info("MetricsCollector initialized")
    
    def record_inference(self, model_name: str, latency_ms: float) -> None:
        """Record model inference latency.
        
        Args:
            model_name: Name of the model (e.g., "vehicle_detector", "ocr_engine")
            latency_ms: Inference time in milliseconds
        """
        with self._lock:
            if model_name not in self._inference_metrics:
                self._inference_metrics[model_name] = InferenceMetrics(model_name)
            self._inference_metrics[model_name].add_inference(latency_ms)
    
    def record_cache_hit(self, cache_name: str) -> None:
        """Record a cache hit.
        
        Args:
            cache_name: Name of the cache
        """
        with self._lock:
            if cache_name not in self._cache_metrics:
                self._cache_metrics[cache_name] = CacheMetrics()
            self._cache_metrics[cache_name].add_hit()
    
    def record_cache_miss(self, cache_name: str) -> None:
        """Record a cache miss.
        
        Args:
            cache_name: Name of the cache
        """
        with self._lock:
            if cache_name not in self._cache_metrics:
                self._cache_metrics[cache_name] = CacheMetrics()
            self._cache_metrics[cache_name].add_miss()
    
    def get_inference_metrics(self, model_name: str | None = None) -> dict[str, Any]:
        """Get inference metrics.
        
        Args:
            model_name: Specific model name, or None for all models
            
        Returns:
            Metrics dict or dict of all metrics
        """
        with self._lock:
            if model_name:
                if model_name in self._inference_metrics:
                    return self._inference_metrics[model_name].to_dict()
                return {}
            
            return {
                name: metrics.to_dict()
                for name, metrics in self._inference_metrics.items()
            }
    
    def get_cache_metrics(self, cache_name: str | None = None) -> dict[str, Any]:
        """Get cache metrics.
        
        Args:
            cache_name: Specific cache name, or None for all caches
            
        Returns:
            Metrics dict or dict of all metrics
        """
        with self._lock:
            if cache_name:
                if cache_name in self._cache_metrics:
                    return self._cache_metrics[cache_name].to_dict()
                return {}
            
            return {
                name: metrics.to_dict()
                for name, metrics in self._cache_metrics.items()
            }
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of all metrics.
        
        Returns:
            Summary dict with inference and cache metrics
        """
        uptime_seconds = (
            datetime.now(timezone.utc) - self._start_time
        ).total_seconds()
        
        with self._lock:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": uptime_seconds,
                "inference_models": len(self._inference_metrics),
                "cache_name": len(self._cache_metrics),
                "inference_metrics": self.get_inference_metrics(),
                "cache_metrics": self.get_cache_metrics()
            }
    
    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._inference_metrics.clear()
            self._cache_metrics.clear()
            self._start_time = datetime.now(timezone.utc)
            logger.info("Metrics reset")


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector (thread-safe singleton).
    
    Returns:
        Global MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
