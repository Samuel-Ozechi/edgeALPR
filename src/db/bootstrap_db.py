# src/db/bootstrap_db.py

"""Database initialization script for a vision-based access control system.
This script sets up the SQLite database with the required schema for storing vehicle information and access events."""

# Import required libraries
import sqlite3
from pathlib import Path
from src.configs.settings import settings

# Define the database path from settings
DB_PATH = settings.database.db_path


# Define the database schema
# The vehicles table stores authorized and blocked vehicle information
# The access_events table logs each access attempt with info about the detection results and latency metrics
SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles ( 
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_text TEXT UNIQUE NOT NULL,
    vehicle_class TEXT,
    owner_name TEXT,
    status TEXT NOT NULL DEFAULT 'authorized'
);

CREATE TABLE IF NOT EXISTS access_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
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
);
"""

# Seed data for demonstration purposes
SEED = [
    ("RSH129SA", "car", "Demo User 1", "authorized"),
    ("KTU538HN", "car", "Demo User 2", "authorized"),
    ("KRD624JT", "truck", "Blocked Vehicle", "blocked"),
    ("KTU656KH", "car", "Demo User 3", "authorized"),
    ("UWN162AU", "car", "Demo User 4", "authorized"),
    ("AGL871HV", "car", "Blocked Vehicle", "blocked"),
    ("LSD176JL", "car", "Demo User 5", "authorized"),
    ("EKY587FY", "car", "Demo User 6", "authorized"),
    ("LND300EC", "car", "Demo User 7", "authorized"),
    ("ABC108DB", "car", "Blocked Vehicle", "blocked"),
    ("FST643HJ", "car", "Demo User 8", "authorized"),
    ("EKY722BY", "car", "Demo User 9", "authorized"),
]

# Main function to initialize the database and insert seed data
def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)

    for row in SEED:
        cur.execute(
            """
            INSERT OR IGNORE INTO vehicles
            (plate_text, vehicle_class, owner_name, status)
            VALUES (?, ?, ?, ?)
            """,
            row
        )

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()