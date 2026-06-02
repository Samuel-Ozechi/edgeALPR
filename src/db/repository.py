# src/db/repository.py

"""Database repository module for a vision-based access control system.
This module defines a VehicleRepository class that provides methods to interact with the SQLite database for retrieving vehicle information and inserting access events.
The get_vehicle_by_plate method retrieves vehicle details based on the license plate text.
The insert_event method records access events with detailed information about the detection and OCR results, as well as latency metrics."""

# Import required libraries
import sqlite3
import logging
import time
from rapidfuzz import fuzz
from src.configs.settings import settings

logger = logging.getLogger(__name__)

# Define the VehicleRepository class
class VehicleRepository:
    """VehicleRepository class that provides methods to interact with the SQLite database for retrieving vehicle information and inserting access events.
    The get_vehicle_by_plate method retrieves vehicle details based on the license plate text.
    The insert_event method records access events with detailed information about the detection and OCR results, as well as latency metrics."""

    def __init__(self, db_path: str, fuzzy_threshold: float = 85.0):
        self.db_path = db_path
        self.fuzzy_threshold = fuzzy_threshold
        self._apply_migrations()
        self._plate_cache = {}  # Simple cache to avoid repeated fuzzy matching
        self._cache_ttl_seconds = 300  # 5 minute TTL

    def _apply_migrations(self) -> None:
        """Adds any missing columns to existing databases without touching existing data."""
        migrations = [
            "ALTER TABLE access_events ADD COLUMN actuator_triggered INTEGER",
            "ALTER TABLE access_events ADD COLUMN actuator_reason TEXT",
        ]
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for statement in migrations:
            try:
                cur.execute(statement)
            except sqlite3.OperationalError:
                pass  # Column already exists
        conn.commit()
        conn.close()

    def get_vehicle_by_plate(self, plate_text: str) -> dict | None:
        """Get vehicle record by license plate text using exact match first, then fuzzy matching.
        
        Args:
            plate_text: License plate text to search for
            
        Returns:
            Dictionary with vehicle record and match metadata, or None if no match found
        """
        if not plate_text or not isinstance(plate_text, str):
            logger.warning(f"Invalid plate_text input: {plate_text}")
            return None
        
        plate_text = plate_text.strip()
        
        # Check cache first
        cache_key = plate_text.upper()
        if cache_key in self._plate_cache:
            cached_result, timestamp = self._plate_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                logger.debug(f"Cache hit for plate: {plate_text}")
                return cached_result
            else:
                del self._plate_cache[cache_key]  # Expired entry

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Try exact match first 
            cur.execute(
                "SELECT * FROM vehicles WHERE plate_text = ?",
                (plate_text,)
            )
            row = cur.fetchone()

            if row:
                result = dict(row)
                result["match_type"] = "exact"
                result["match_score"] = 100.0
                conn.close()
                self._plate_cache[cache_key] = (result, time.time())
                return result

            # Fuzzy match fallback - but fetch only plates and IDs first for efficiency
            cur.execute("SELECT id, plate_text FROM vehicles LIMIT 1000")  # Limit to 1000 records
            all_plates = cur.fetchall()
            conn.close()

            best_match = None
            best_score = 0.0
            best_id = None

            for plate_row in all_plates:
                db_plate = plate_row["plate_text"]
                score = fuzz.ratio(plate_text.upper(), db_plate.upper())

                if score > best_score and score >= self.fuzzy_threshold:
                    best_score = score
                    best_id = plate_row["id"]

            # If fuzzy match found, fetch full record
            if best_id is not None:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM vehicles WHERE id = ?", (best_id,))
                best_match = cur.fetchone()
                conn.close()
                
                if best_match:
                    result = dict(best_match)
                    result["match_type"] = "fuzzy"
                    result["match_score"] = best_score
                    logger.info(f"Fuzzy match for plate '{plate_text}' → '{result.get('plate_text')}' (score: {best_score})")
                    self._plate_cache[cache_key] = (result, time.time())
                    return result

            logger.debug(f"No vehicle found for plate: {plate_text}")
            return None
            
        except sqlite3.DatabaseError as e:
            logger.error(f"Database error querying vehicle by plate: {e}", exc_info=True)
            return None

    def insert_event(self, event: dict) -> None:
        """Insert an access event into the database.
        
        Args:
            event: Dictionary containing event data with keys:
                - timestamp, device_id, site_id, plate_text, ocr_confidence,
                - detected_vehicle_class, vehicle_confidence, plate_confidence,
                - decision, reason, total_latency_ms, vehicle_latency_ms,
                - plate_latency_ms, ocr_latency_ms, actuator_triggered, actuator_reason
                
        Raises:
            ValueError: If event is missing required fields
            sqlite3.DatabaseError: If database insertion fails
        """
        if not event or not isinstance(event, dict):
            raise ValueError("Event must be a non-empty dictionary")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO access_events (
                    timestamp,
                    device_id,
                    site_id,
                    plate_text,
                    ocr_confidence,
                    detected_vehicle_class,
                    vehicle_confidence,
                    plate_confidence,
                    decision,
                    reason,
                    total_latency_ms,
                    vehicle_latency_ms,
                    plate_latency_ms,
                    ocr_latency_ms,
                    actuator_triggered,
                    actuator_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("timestamp"),
                    event.get("device_id"),
                    event.get("site_id"),
                    event.get("plate_text"),
                    event.get("ocr_confidence"),
                    event.get("detected_vehicle_class"),
                    event.get("vehicle_confidence"),
                    event.get("plate_confidence"),
                    event.get("decision"),
                    event.get("reason"),
                    event.get("total_latency_ms"),
                    event.get("vehicle_latency_ms"),
                    event.get("plate_latency_ms"),
                    event.get("ocr_latency_ms"),
                    event.get("actuator_triggered"),
                    event.get("actuator_reason"),
                )
            )

            conn.commit()
            conn.close()
            
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to insert event: {e}", exc_info=True)
            raise