"""Test configuration and fixtures for edgeALPR tests."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import sqlite3


@pytest.fixture
def sample_image():
    """Create a sample image (BGR format) for testing."""
    # Create a random 480x640 BGR image
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def empty_image():
    """Create an empty/invalid image for testing error handling."""
    return np.array([], dtype=np.uint8)


@pytest.fixture
def sample_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create vehicles table
    cur.execute("""
        CREATE TABLE vehicles (
            id INTEGER PRIMARY KEY,
            plate_text TEXT UNIQUE,
            status TEXT,
            vehicle_class TEXT
        )
    """)
    
    # Create access_events table
    cur.execute("""
        CREATE TABLE access_events (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            device_id TEXT,
            site_id TEXT,
            plate_text TEXT,
            ocr_confidence REAL,
            detected_vehicle_class TEXT,
            vehicle_confidence REAL,
            plate_confidence REAL,
            decision TEXT,
            reason TEXT,
            total_latency_ms REAL,
            vehicle_latency_ms REAL,
            plate_latency_ms REAL,
            ocr_latency_ms REAL,
            actuator_triggered INTEGER,
            actuator_reason TEXT
        )
    """)
    
    # Insert sample vehicle
    cur.execute("""
        INSERT INTO vehicles (plate_text, status, vehicle_class)
        VALUES (?, ?, ?)
    """, ("ABC123", "authorized", "car"))
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_event():
    """Create a sample access event for testing."""
    return {
        "timestamp": "2024-06-02T12:00:00.000Z",
        "device_id": "test_device_01",
        "site_id": "test_site_01",
        "plate_text": "ABC123",
        "ocr_confidence": 0.95,
        "detected_vehicle_class": "car",
        "vehicle_confidence": 0.92,
        "plate_confidence": 0.88,
        "decision": "allow",
        "reason": "authorized_match",
        "total_latency_ms": 125.5,
        "vehicle_latency_ms": 45.2,
        "plate_latency_ms": 32.1,
        "ocr_latency_ms": 48.2,
        "actuator_triggered": 1,
        "actuator_reason": "access_allowed",
    }
