# src/db/repository.py

"""Database repository module for a vision-based access control system.
This module defines a VehicleRepository class that provides methods to interact with the SQLite database for retrieving vehicle information and inserting access events.
The get_vehicle_by_plate method retrieves vehicle details based on the license plate text.
The insert_event method records access events with detailed information about the detection and OCR results, as well as latency metrics."""

# Import required libraries
import sqlite3
from rapidfuzz import fuzz
from src.configs.settings import settings

# Define the VehicleRepository class
class VehicleRepository:
    """VehicleRepository class that provides methods to interact with the SQLite database for retrieving vehicle information and inserting access events.
    The get_vehicle_by_plate method retrieves vehicle details based on the license plate text.
    The insert_event method records access events with detailed information about the detection and OCR results, as well as latency metrics."""

    def __init__(self, db_path: str, fuzzy_threshold: float = 85.0):
        self.db_path = db_path
        self.fuzzy_threshold = fuzzy_threshold
        self._apply_migrations()

    def _apply_migrations(self):
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
        if not plate_text:
            return None

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
            conn.close()
            result = dict(row)
            result["match_type"] = "exact"
            result["match_score"] = 100.0
            return result

        # Fuzzy match fallback - fetch all plates and find best match
        cur.execute("SELECT * FROM vehicles")
        all_vehicles = cur.fetchall()
        conn.close()

        best_match = None
        best_score = 0.0

        for vehicle in all_vehicles:
            db_plate = vehicle["plate_text"]
            score = fuzz.ratio(plate_text.upper(), db_plate.upper())

            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = vehicle

        if best_match:
            result = dict(best_match)
            result["match_type"] = "fuzzy"
            result["match_score"] = best_score
            return result

        return None

    def insert_event(self, event: dict) -> None:
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