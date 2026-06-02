"""Abstract base classes for data persistence in edgeALPR."""

from abc import ABC, abstractmethod
from typing import Any
import logging

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    \"\"\"Abstract base class for data repositories.
    
    Subclasses can implement different storage backends (SQLite, PostgreSQL, MongoDB, etc.)
    while maintaining a consistent interface with the pipeline.
    
    Example:
        class MongoRepository(BaseRepository):
            def __init__(self, connection_string):
                self.client = MongoClient(connection_string)
                self.db = self.client['edgealpr']
                
            def get_vehicle_by_plate(self, plate_text):
                return self.db.vehicles.find_one({'plate': plate_text})
    \"\"\"

    @abstractmethod
    def get_vehicle_by_plate(self, plate_text: str) -> dict | None:
        \"\"\"Retrieve vehicle information by license plate.
        
        Args:
            plate_text: License plate text to search for
            
        Returns:
            Vehicle record dictionary with keys: id, plate_text, status, vehicle_class, etc.
            or None if not found
        \"\"\"
        pass

    @abstractmethod
    def insert_event(self, event: dict) -> None:
        \"\"\"Record an access event.
        
        Args:
            event: Event dictionary with access information
            
        Raises:
            ValueError: If event is invalid
            RuntimeError: If storage fails
        \"\"\"
        pass

    @abstractmethod
    def get_events(
        self, 
        device_id: str | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        \"\"\"Retrieve historical events.
        
        Args:
            device_id: Filter by device ID (optional)
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List of event dictionaries
        \"\"\"
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        \"\"\"Get repository statistics.
        
        Returns:
            Dictionary with: total_vehicles, total_events, authorized_count, etc.
        \"\"\"
        pass
