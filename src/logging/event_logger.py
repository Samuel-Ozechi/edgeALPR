# src/logging/event_logger.py

"""Event logging module for a vision-based access control system. 
This module defines a JsonlEventLogger class that provides functionality to log events in JSON Lines format. 
Each event is a dictionary that gets serialized to JSON and appended to a log file. 
The logger ensures that the log directory exists before writing events."""   

# Import required libraries
import json
from pathlib import Path
from src.configs.settings import settings

# Event logger that writes events to a JSON Lines file
class JsonlEventLogger:
    def __init__(self, log_path: str):

        """Initializes the event logger and ensures the log directory exists."""
        
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")