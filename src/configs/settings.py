# src/configs/settings.py

"""Centralized configuration management using Pydantic for a vision-based access control system. 
This module defines structured settings for system identifiers, video capture, model paths, detection thresholds, business rules, and logging. 
It ensures all necessary directories are created at runtime and supports environment variable overrides."""

# Import required libraries
from pathlib import Path
from typing import Set, Literal, Union
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the project root for relative path management
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


# System identifiers and device info
class SystemConfig(BaseSettings):
    site_id: str = "gate_01"
    device_id: str = "pi5_unit_01"


# Model paths and parameters
class ModelsConfig(BaseSettings):
    models_dir: Path = PROJECT_ROOT / "models"
    vehicle_detector: Path = PROJECT_ROOT / "models" / "vehicle_detector.pt" 
    vehicle_detector_hef: Path = PROJECT_ROOT / "models" / "vehicle_detector.hef" 
    vehicle_target_classes: Set[str] = {"car", "truck", "bus", "motorcycle"}
    plate_detector: Path = PROJECT_ROOT / "models" / "plate_detector.pt"
    plate_detector_hef: Path = PROJECT_ROOT / "models" / "plate_detector.hef" 
    license_recognizer: Path = PROJECT_ROOT / "models" / "license_recognizer" / "lprnet.pth"
    pretrained_lprnet: Path = PROJECT_ROOT / "models" / "license_recognizer" / "lprnet_pretrained.pth"
    ocr_lang: str = "en"
    ocr_use_angle_cls: bool = True
    

    

# Detection confidence thresholds
class ThresholdsConfig(BaseSettings):
    vehicle_conf: float = Field(0.6, ge=0, le=1)
    plate_conf: float = Field(0.6, ge=0, le=1)
    ocr_conf: float = Field(0.7, ge=0, le=1)


# Business rules for access control decisions
class RulesConfig(BaseSettings):
    require_class_match: bool = False
    default_action_on_low_confidence: Literal["allow", "deny", "review"] = "review"

# Logging configuration for event recording and failed crop management
class LoggingConfig(BaseSettings):
    log_path: Path = PROJECT_ROOT / "data" / "logs" / "events.jsonl"
    save_failed_crops: bool = True
    failed_crop_dir: Path =   PROJECT_ROOT / "data"  / "logs" / "failed_crops"


# Database configuration for access control records
class DatabaseConfig(BaseSettings):
    db_path: Path = PROJECT_ROOT / "data" / "access_control.db"

# Actuator safety configuration for barrier gating behavior
class ActuatorConfig(BaseModel):
    mode: str = "mock"
    relay_pin: int = 17
    relay_active_high: bool = True
    open_pulse_seconds: float = 1.0
    open_hold_seconds: int = 3
    same_plate_cooldown_seconds: int = 10
    global_cooldown_seconds: int = 2
    fail_safe_action: str = "review"

# Temporal voting configuration for multi-frame plate decision making
class TemporalConfig(BaseModel):
    enabled: bool = True
    window_size: int = 5
    min_votes: int = 2
    max_window_time_ms: int = 70
    decision_cooldown_ms: int = 200
    use_confidence_weighting: bool = True


# Video configuration for handling video input and processing
class VideoConfig(BaseModel):
    source: int | str = 0 # 0 for CSI/USB, or "path/to/video.mp4"
    frame_width: int = 1280
    frame_height: int = 720
    display: bool = True
    process_every_n_frames: int = 5
    save_review_frames: bool = True
    review_frame_dir: str = "data/logs/review_frames"


class RuntimeConfig(BaseModel):
    detector_backend: str = "hailo"  # Options: "hailo", "torch"

# Main settings class that aggregates all configurations and ensures directory creation at runtime
class Settings(BaseSettings):
    # Setup Pydantic to read from environment variables or .env file
    model_config = SettingsConfigDict(env_nested_delimiter='__', env_file='.env')

    # Path Management
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VEHICLE_DIR: Path= DATA_DIR / "vehicle_identification"
    VEHICLE_INPUT_DIR: Path = VEHICLE_DIR / "inputs"
    VEHICLE_OUTPUT_DIR: Path = VEHICLE_DIR / "outputs"

    PLATE_DIR: Path = DATA_DIR / "plate_detection"
    PLATE_INPUT_DIR: Path = PLATE_DIR / "inputs"
    PLATE_OUTPUT_DIR: Path = PLATE_DIR / "outputs"

    LICENSE_DIR: Path = DATA_DIR / "license_recognition"
    LICENSE_INPUT_DIR: Path = LICENSE_DIR / "inputs"
    LICENSE_OUTPUT_DIR: Path = LICENSE_DIR / "outputs"

    # Nested Configurations
    system: SystemConfig = SystemConfig()
    video: VideoConfig = VideoConfig()
    models: ModelsConfig = ModelsConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    rules: RulesConfig = RulesConfig()
    logging: LoggingConfig = LoggingConfig()
    database: DatabaseConfig = DatabaseConfig()
    temporal: TemporalConfig = TemporalConfig()
    actuator: ActuatorConfig = ActuatorConfig()
    runtime: RuntimeConfig = RuntimeConfig()


    # Method to create necessary directories at runtime
    def create_directories(self):
        """Ensures all production directories exist at runtime."""
        dirs = [
            self.DATA_DIR, 
            self.VEHICLE_INPUT_DIR, 
            self.VEHICLE_OUTPUT_DIR, 
            self.PLATE_INPUT_DIR,
            self.PLATE_OUTPUT_DIR,
            self.LICENSE_INPUT_DIR,
            self.LICENSE_OUTPUT_DIR,
            self.logging.failed_crop_dir
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

# Initialize settings
settings = Settings()
settings.create_directories()

if __name__ == "__main__":
    # Sample configs
    print(f"\n--- Vision Based Access Intelligence ---")
    print(f"Site ID: {settings.system.site_id}")
    print(f"Vehicle Model: {settings.models.vehicle_detector}")
    print(f"Target Classes: {settings.models.vehicle_target_classes}")
    print(f"Thresholds: {settings.thresholds.model_dump()}")