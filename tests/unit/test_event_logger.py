"""Unit tests for logging module."""

import pytest
import json
from pathlib import Path
import tempfile
from src.logging.event_logger import JsonlEventLogger


class TestJsonlEventLogger:
    """Test suite for JsonlEventLogger."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file path."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        yield path
        Path(path).unlink(missing_ok=True)
    
    @pytest.fixture
    def logger(self, temp_log_file):
        """Create a logger for testing."""
        return JsonlEventLogger(temp_log_file)
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample event."""
        return {
            "timestamp": "2024-06-02T12:00:00Z",
            "device_id": "test_device",
            "decision": "allow",
            "plate_text": "ABC123"
        }
    
    def test_init_creates_directory(self, temp_log_file):
        """Test that __init__ creates parent directory if needed."""
        nested_path = str(Path(temp_log_file).parent / "subdir" / "test.jsonl")
        logger = JsonlEventLogger(nested_path)
        assert Path(nested_path).parent.exists()
        Path(nested_path).parent.rmdir()
    
    def test_log_single_event(self, logger, temp_log_file, sample_event):
        """Test logging a single event."""
        logger.log(sample_event)
        
        # Verify event was written
        with open(temp_log_file, 'r') as f:
            line = f.readline()
        
        logged_event = json.loads(line)
        assert logged_event["device_id"] == "test_device"
        assert logged_event["decision"] == "allow"
    
    def test_log_multiple_events(self, logger, temp_log_file, sample_event):
        """Test logging multiple events creates JSONL file."""
        event1 = sample_event.copy()
        event2 = sample_event.copy()
        event2["plate_text"] = "XYZ999"
        
        logger.log(event1)
        logger.log(event2)
        
        # Verify both events were written
        with open(temp_log_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        events = [json.loads(line) for line in lines]
        assert events[0]["plate_text"] == "ABC123"
        assert events[1]["plate_text"] == "XYZ999"
    
    def test_log_event_with_special_characters(self, logger, temp_log_file):
        """Test logging events with special characters."""
        event = {
            "timestamp": "2024-06-02T12:00:00Z",
            "description": "Test with émojis 🚗 and special chars: ñ, ü, é",
            "plate_text": "ÖBE-2024"
        }
        
        logger.log(event)
        
        with open(temp_log_file, 'r') as f:
            line = f.readline()
        
        logged_event = json.loads(line)
        assert "émojis" in logged_event["description"]
        assert "ÖBE-2024" in logged_event["plate_text"]
    
    def test_thread_safety_mock(self, logger, temp_log_file):
        """Test that logger uses lock for thread safety."""
        # Verify that the logger has a lock
        assert hasattr(logger, '_lock')
        assert logger._lock is not None
    
    def test_log_with_numeric_values(self, logger, temp_log_file):
        """Test logging events with numeric values."""
        event = {
            "timestamp": "2024-06-02T12:00:00Z",
            "confidence": 0.95,
            "latency_ms": 123.45,
            "count": 42
        }
        
        logger.log(event)
        
        with open(temp_log_file, 'r') as f:
            line = f.readline()
        
        logged_event = json.loads(line)
        assert logged_event["confidence"] == 0.95
        assert logged_event["latency_ms"] == 123.45
        assert logged_event["count"] == 42
