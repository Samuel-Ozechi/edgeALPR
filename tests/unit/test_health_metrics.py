"""Tests for health check and metrics modules."""

import pytest
import tempfile
import sqlite3
from pathlib import Path

from src.health.health_check import HealthCheck
from src.health.metrics import (
    InferenceMetrics,
    CacheMetrics,
    MetricsCollector,
    get_metrics_collector,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with expected schema."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    # Create schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE vehicles (
            id INTEGER PRIMARY KEY,
            plate_text TEXT UNIQUE
        )
    """)
    cursor.execute("""
        CREATE TABLE access_events (
            id INTEGER PRIMARY KEY,
            vehicle_id INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink()


class TestHealthCheck:
    """Tests for HealthCheck system diagnostics."""
    
    def test_init(self):
        """Test HealthCheck initialization."""
        checker = HealthCheck("/tmp/test.db")
        assert checker.db_path == Path("/tmp/test.db")
    
    def test_check_database_connectivity_file_not_found(self):
        """Test database check when file doesn't exist."""
        checker = HealthCheck("/nonexistent/path/db.sqlite")
        result = checker.check_database_connectivity()
        
        assert result["healthy"] is False
        assert "not found" in result["details"]
    
    def test_check_database_connectivity_valid(self, temp_db):
        """Test database check with valid schema."""
        checker = HealthCheck(temp_db)
        result = checker.check_database_connectivity()
        
        assert result["healthy"] is True
        assert "vehicles" in result["details"]
        assert "access_events" in result["details"]
        assert result["tables_found"] == 2
    
    def test_check_model_files_all_missing(self):
        """Test model file check when all are missing."""
        checker = HealthCheck("/tmp/test.db")
        result = checker.check_model_files({
            "vehicle_detector": "/nonexistent/vehicle.pt",
            "plate_detector": "/nonexistent/plate.pt"
        })
        
        assert result["healthy"] is False
        assert len(result["missing_models"]) == 2
        assert len(result["available_models"]) == 0
    
    def test_check_model_files_all_available(self, temp_db):
        """Test model file check when files exist."""
        # Create temp files
        with tempfile.NamedTemporaryFile() as f1, tempfile.NamedTemporaryFile() as f2:
            result = checker.check_model_files({
                "vehicle_detector": f1.name,
                "plate_detector": f2.name
            })
        
            assert result["healthy"] is True
            assert len(result["available_models"]) == 2
            assert len(result["missing_models"]) == 0
    
    def test_check_hailo_availability(self):
        """Test Hailo availability check."""
        checker = HealthCheck("/tmp/test.db")
        result = checker.check_hailo_availability()
        
        assert "available" in result
        assert "details" in result
        # Result depends on installation
    
    def test_check_onnx_availability(self):
        """Test ONNX availability check."""
        checker = HealthCheck("/tmp/test.db")
        result = checker.check_onnx_availability()
        
        assert "available" in result
        assert "details" in result
    
    def test_get_full_health_status(self, temp_db):
        """Test full health status report."""
        checker = HealthCheck(temp_db)
        result = checker.get_full_health_status({
            "vehicle_detector": "/nonexistent/vehicle.pt"
        })
        
        assert "timestamp" in result
        assert "overall_status" in result
        assert "database" in result
        assert "models" in result
        assert "accelerators" in result
        assert result["overall_status"] == "degraded"  # Missing model


class TestInferenceMetrics:
    """Tests for InferenceMetrics."""
    
    def test_init(self):
        """Test initialization."""
        metrics = InferenceMetrics("test_model")
        assert metrics.model_name == "test_model"
        assert metrics.total_inferences == 0
        assert metrics.total_latency_ms == 0.0
    
    def test_add_inference(self):
        """Test recording inference latency."""
        metrics = InferenceMetrics("model")
        metrics.add_inference(10.5)
        metrics.add_inference(12.3)
        
        assert metrics.total_inferences == 2
        assert metrics.total_latency_ms == pytest.approx(22.8)
    
    def test_get_average_latency(self):
        """Test average latency calculation."""
        metrics = InferenceMetrics("model")
        metrics.add_inference(10.0)
        metrics.add_inference(20.0)
        
        assert metrics.get_average_latency() == pytest.approx(15.0)
    
    def test_get_average_latency_no_inferences(self):
        """Test average with no inferences."""
        metrics = InferenceMetrics("model")
        assert metrics.get_average_latency() == 0.0
    
    def test_min_max_latency(self):
        """Test min/max latency tracking."""
        metrics = InferenceMetrics("model")
        metrics.add_inference(5.0)
        metrics.add_inference(15.0)
        metrics.add_inference(10.0)
        
        assert metrics.min_latency_ms == 5.0
        assert metrics.max_latency_ms == 15.0
    
    def test_to_dict(self):
        """Test metrics export to dict."""
        metrics = InferenceMetrics("model")
        metrics.add_inference(10.0)
        
        result = metrics.to_dict()
        assert result["model_name"] == "model"
        assert result["total_inferences"] == 1
        assert "average_latency_ms" in result


class TestCacheMetrics:
    """Tests for CacheMetrics."""
    
    def test_init(self):
        """Test initialization."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
    
    def test_add_hit(self):
        """Test recording cache hit."""
        metrics = CacheMetrics()
        metrics.add_hit()
        assert metrics.hits == 1
    
    def test_add_miss(self):
        """Test recording cache miss."""
        metrics = CacheMetrics()
        metrics.add_miss()
        assert metrics.misses == 1
    
    def test_get_hit_rate(self):
        """Test hit rate calculation."""
        metrics = CacheMetrics()
        metrics.add_hit()
        metrics.add_hit()
        metrics.add_miss()
        
        assert metrics.get_hit_rate() == pytest.approx(2.0 / 3.0)
    
    def test_get_hit_rate_no_accesses(self):
        """Test hit rate with no accesses."""
        metrics = CacheMetrics()
        assert metrics.get_hit_rate() == 0.0
    
    def test_to_dict(self):
        """Test metrics export to dict."""
        metrics = CacheMetrics()
        metrics.add_hit()
        metrics.add_miss()
        
        result = metrics.to_dict()
        assert result["hits"] == 1
        assert result["misses"] == 1
        assert "hit_rate" in result


class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_init(self):
        """Test initialization."""
        collector = MetricsCollector()
        assert collector.get_inference_metrics() == {}
        assert collector.get_cache_metrics() == {}
    
    def test_record_inference(self):
        """Test recording inference."""
        collector = MetricsCollector()
        collector.record_inference("model1", 10.5)
        collector.record_inference("model1", 12.3)
        
        metrics = collector.get_inference_metrics("model1")
        assert metrics["total_inferences"] == 2
    
    def test_record_cache_hit(self):
        """Test recording cache hit."""
        collector = MetricsCollector()
        collector.record_cache_hit("cache1")
        
        metrics = collector.get_cache_metrics("cache1")
        assert metrics["hits"] == 1
        assert metrics["misses"] == 0
    
    def test_record_cache_miss(self):
        """Test recording cache miss."""
        collector = MetricsCollector()
        collector.record_cache_miss("cache1")
        
        metrics = collector.get_cache_metrics("cache1")
        assert metrics["hits"] == 0
        assert metrics["misses"] == 1
    
    def test_get_summary(self):
        """Test metrics summary."""
        collector = MetricsCollector()
        collector.record_inference("model1", 10.0)
        collector.record_cache_hit("cache1")
        
        summary = collector.get_summary()
        assert "timestamp" in summary
        assert "uptime_seconds" in summary
        assert "inference_models" in summary
        assert "cache_name" in summary
    
    def test_thread_safety(self):
        """Test thread-safe concurrent access."""
        import threading
        collector = MetricsCollector()
        
        def worker():
            for i in range(100):
                collector.record_inference("model", float(i))
                collector.record_cache_hit("cache")
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        metrics = collector.get_inference_metrics("model")
        # Should have recorded 500 inferences (5 threads × 100 each)
        assert metrics["total_inferences"] == 500
    
    def test_reset(self):
        """Test metrics reset."""
        collector = MetricsCollector()
        collector.record_inference("model", 10.0)
        collector.record_cache_hit("cache")
        
        collector.reset()
        
        assert collector.get_inference_metrics() == {}
        assert collector.get_cache_metrics() == {}


class TestGlobalMetricsCollector:
    """Tests for global metrics collector singleton."""
    
    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should be same instance
        assert collector1 is collector2
