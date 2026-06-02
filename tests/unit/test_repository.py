"""Unit tests for VehicleRepository."""

import pytest
from src.db.repository import VehicleRepository


class TestVehicleRepository:
    """Test suite for VehicleRepository."""
    
    @pytest.fixture
    def repository(self, sample_db):
        """Create a repository for testing."""
        return VehicleRepository(sample_db, fuzzy_threshold=85.0)
    
    def test_get_vehicle_exact_match(self, repository):
        """Test exact plate match."""
        vehicle = repository.get_vehicle_by_plate("ABC123")
        assert vehicle is not None
        assert vehicle["plate_text"] == "ABC123"
        assert vehicle["status"] == "authorized"
        assert vehicle["match_type"] == "exact"
        assert vehicle["match_score"] == 100.0
    
    def test_get_vehicle_not_found(self, repository):
        """Test that non-existent vehicle returns None."""
        vehicle = repository.get_vehicle_by_plate("NOTHERE")
        assert vehicle is None
    
    def test_get_vehicle_empty_plate_text(self, repository):
        """Test that empty plate_text returns None."""
        vehicle = repository.get_vehicle_by_plate("")
        assert vehicle is None
    
    def test_get_vehicle_none_input(self, repository):
        """Test that None input returns None."""
        vehicle = repository.get_vehicle_by_plate(None)
        assert vehicle is None
    
    def test_insert_event_valid(self, repository, sample_event):
        """Test inserting a valid event."""
        # Should not raise
        repository.insert_event(sample_event)
        
        # Verify event was inserted by reading from database
        import sqlite3
        conn = sqlite3.connect(repository.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM access_events")
        count = cur.fetchone()[0]
        conn.close()
        
        assert count == 1
    
    def test_insert_event_invalid_empty_dict(self, repository):
        """Test that inserting empty dict raises ValueError."""
        with pytest.raises(ValueError):
            repository.insert_event({})
    
    def test_insert_event_invalid_none(self, repository):
        """Test that inserting None raises ValueError."""
        with pytest.raises(ValueError):
            repository.insert_event(None)
    
    def test_insert_event_multiple(self, repository, sample_event):
        """Test inserting multiple events."""
        repository.insert_event(sample_event)
        
        # Create another event with different data
        event2 = sample_event.copy()
        event2["plate_text"] = "XYZ999"
        repository.insert_event(event2)
        
        # Verify both were inserted
        import sqlite3
        conn = sqlite3.connect(repository.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM access_events")
        count = cur.fetchone()[0]
        conn.close()
        
        assert count == 2
    
    def test_cache_functionality(self, repository):
        """Test that caching works for repeated queries."""
        # First query
        vehicle1 = repository.get_vehicle_by_plate("ABC123")
        assert vehicle1 is not None
        
        # Second query (should be cached)
        vehicle2 = repository.get_vehicle_by_plate("ABC123")
        assert vehicle2 is not None
        assert vehicle1 == vehicle2
    
    def test_get_stats(self, repository, sample_event):
        """Test get_stats returns expected structure."""
        repository.insert_event(sample_event)
        # Stats method would be implemented in a subclass or extended version
        # This is a placeholder test for future implementation
        pass
