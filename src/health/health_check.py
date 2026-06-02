"""Health check diagnostics for edgeALPR system."""

import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HealthCheck:
    """System health diagnostics and status monitoring."""
    
    def __init__(self, db_path: str | Path) -> None:
        """Initialize health checker.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.logger = logger
    
    def check_database_connectivity(self) -> dict[str, Any]:
        """Check if database is accessible and has expected schema.
        
        Returns:
            Status dict with keys: healthy, details, tables_found
        """
        try:
            if not self.db_path.exists():
                return {
                    "healthy": False,
                    "details": f"Database file not found: {self.db_path}",
                    "tables_found": 0
                }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for expected tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            expected_tables = {"vehicles", "access_events"}
            has_expected = expected_tables.issubset(set(tables))
            
            return {
                "healthy": has_expected,
                "details": f"Found tables: {tables}",
                "tables_found": len(tables)
            }
        except sqlite3.DatabaseError as e:
            return {
                "healthy": False,
                "details": f"Database error: {e}",
                "tables_found": 0
            }
        except Exception as e:
            return {
                "healthy": False,
                "details": f"Unexpected error: {e}",
                "tables_found": 0
            }
    
    def check_model_files(self, model_paths: dict[str, str | Path]) -> dict[str, Any]:
        """Check if required model files exist and are accessible.
        
        Args:
            model_paths: Dict of model name -> file path
            
        Returns:
            Status dict with keys: healthy, missing_models, available_models
        """
        missing = []
        available = []
        
        for name, path in model_paths.items():
            path_obj = Path(path)
            if path_obj.exists():
                available.append(name)
            else:
                missing.append(f"{name} ({path})")
        
        return {
            "healthy": len(missing) == 0,
            "missing_models": missing,
            "available_models": available,
            "total_models": len(model_paths)
        }
    
    def check_hailo_availability(self) -> dict[str, Any]:
        """Check if Hailo accelerator is available.
        
        Returns:
            Status dict with keys: available, details
        """
        try:
            from hailo_platform import VDevice  # noqa: F401
            return {
                "available": True,
                "details": "Hailo SDK is installed"
            }
        except ImportError:
            return {
                "available": False,
                "details": "Hailo SDK not installed - CPU mode will be used"
            }
        except Exception as e:
            return {
                "available": False,
                "details": f"Hailo check failed: {e}"
            }
    
    def check_onnx_availability(self) -> dict[str, Any]:
        """Check if ONNX Runtime is available.
        
        Returns:
            Status dict with keys: available, details
        """
        try:
            import onnxruntime  # noqa: F401
            return {
                "available": True,
                "details": "ONNX Runtime is installed"
            }
        except ImportError:
            return {
                "available": False,
                "details": "ONNX Runtime not installed"
            }
        except Exception as e:
            return {
                "available": False,
                "details": f"ONNX check failed: {e}"
            }
    
    def get_full_health_status(self, model_paths: dict[str, str | Path]) -> dict[str, Any]:
        """Get complete system health status.
        
        Args:
            model_paths: Dict of model name -> file path
            
        Returns:
            Comprehensive health status dict
        """
        db_status = self.check_database_connectivity()
        model_status = self.check_model_files(model_paths)
        hailo_status = self.check_hailo_availability()
        onnx_status = self.check_onnx_availability()
        
        # System is healthy if database is OK and has models available
        overall_healthy = (
            db_status["healthy"] and 
            model_status["healthy"]
        )
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "healthy" if overall_healthy else "degraded",
            "database": db_status,
            "models": model_status,
            "accelerators": {
                "hailo": hailo_status,
                "onnx": onnx_status
            }
        }
