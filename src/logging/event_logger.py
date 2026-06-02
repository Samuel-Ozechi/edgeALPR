# src/logging/event_logger.py

"""Event logging module for a vision-based access control system. 
This module defines a JsonlEventLogger class that provides thread-safe logging of events in JSON Lines format. 
Each event is a dictionary that gets serialized to JSON and appended to a log file. 
The logger ensures that the log directory exists before writing events and uses file locking for thread-safety."""   

# Import required libraries
import json
import threading
from pathlib import Path
from src.configs.settings import settings

# Event logger that writes events to a JSON Lines file with thread-safe access
class JsonlEventLogger:
    def __init__(self, log_path: str):
        """Initializes the event logger and ensures the log directory exists.
        
        Args:
            log_path: Path to the JSONL log file
        """
        
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: dict) -> None:
        """Log an event to the JSONL file in a thread-safe manner.
        
        Args:
            event: Dictionary containing event data to log
            
        Raises:
            IOError: If the log file cannot be written
        """
        with self._lock:
            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except IOError as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to write event log: {e}", exc_info=True)
                raise