# src/pipeline/run_image_pipeline.py

"""Image processing pipeline for a vision-based access control system.
This script defines a function to process a single image through the entire pipeline, including vehicle detection, license plate detection, OCR recognition, access decision making, and event logging. 
It also includes a main block to run the pipeline on a set of test images and print the results."""

# Import required libraries
import time
from datetime import datetime
from pathlib import Path
import logging


# Import local modules
from src.configs.settings import settings
from src.vision.image_ops import load_image, crop_bbox, refine_plate_crop
from src.vision.vehicle_detector import VehicleDetector
from src.vision.plate_detector import PlateDetector
from src.vision.plate_ocr import PlateOCR
from src.db.repository import VehicleRepository
from src.decision.rules_engine import AccessRulesEngine
from src.actuator.mock_actuator import MockActuator
from src.logging.event_logger import JsonlEventLogger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define a function to process a single image through the entire pipeline
def run_single_image(image_path: Path):
    total_start = time.perf_counter()

    logger.info(f"initializing detectors......")
    vehicle_detector = VehicleDetector(
        model_path=settings.models.vehicle_detector,
        target_classes=settings.models.vehicle_target_classes,
        conf_threshold=settings.thresholds.vehicle_conf
    )

    plate_detector = PlateDetector(
        model_path=settings.models.plate_detector,
        conf_threshold=settings.thresholds.plate_conf
    )

    plate_ocr = PlateOCR(
        lang=settings.models.ocr_lang,
        use_angle_cls=settings.models.ocr_use_angle_cls
    )

    repository = VehicleRepository(settings.database.db_path)

    rules_engine = AccessRulesEngine(
        ocr_conf_threshold=settings.thresholds.ocr_conf,
        require_class_match=False
    )

    actuator = MockActuator()

    event_logger = JsonlEventLogger(
        log_path=str(Path(settings.logging.log_path))
    )

    logger.info(f"loading input frame.....")
    image = load_image(image_path)

    logger.info(f"running vehicle detection.....")
    vehicle_detections, vehicle_latency = vehicle_detector.detect(image)
    best_vehicle = vehicle_detector.select_best_vehicle(vehicle_detections)

    if best_vehicle is None:
        decision = "review"
        reason = "no_vehicle_detected"
        actuator.execute(decision)

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "device_id": settings.system.device_id,
            "site_id": settings.system.site_id,
            "plate_text": None,
            "ocr_confidence": 0.0,
            "detected_vehicle_class": None,
            "vehicle_confidence": 0.0,
            "plate_confidence": 0.0,
            "decision": decision,
            "reason": reason,
            "total_latency_ms": round((time.perf_counter() - total_start) * 1000, 2),
            "vehicle_latency_ms": round(vehicle_latency, 2),
            "plate_latency_ms": 0.0,
            "ocr_latency_ms": 0.0,
        }

        repository.insert_event(event)
        event_logger.log(event)
        return event

    logger.info(f"running plate detection.....")
    vehicle_crop = crop_bbox(image, best_vehicle["bbox"])

    logger.info(f"running plate OCR.....")
    plate_detections, plate_latency = plate_detector.detect(vehicle_crop)
    best_plate = plate_detector.select_best_plate(plate_detections)

    if best_plate is None:
        decision = "review"
        reason = "no_plate_detected"
        actuator.execute(decision)

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "device_id": settings.system.device_id,
            "site_id": settings.system.site_id,
            "plate_text": None,
            "ocr_confidence": 0.0,
            "detected_vehicle_class": best_vehicle["class_name"],
            "vehicle_confidence": best_vehicle["confidence"],
            "plate_confidence": 0.0,
            "decision": decision,
            "reason": reason,
            "total_latency_ms": round((time.perf_counter() - total_start) * 1000, 2),
            "vehicle_latency_ms": round(vehicle_latency, 2),
            "plate_latency_ms": round(plate_latency, 2),
            "ocr_latency_ms": 0.0,
        }

        repository.insert_event(event)
        event_logger.log(event)
        return event

    plate_crop = crop_bbox(vehicle_crop, best_plate["bbox"])

    logger.info(f"running OCR on license plate.....")
    ocr_result, ocr_latency = plate_ocr.recognize(plate_crop)

    logger.info(f"fetching vehicle record from database.....")
    vehicle_record = repository.get_vehicle_by_plate(ocr_result["plate_text"])

    logger.info(f"making access decision.....")
    decision, reason = rules_engine.decide(
        plate_text=ocr_result["plate_text"],
        ocr_confidence=ocr_result["confidence"],
        detected_vehicle_class=best_vehicle["class_name"],
        vehicle_record=vehicle_record
    )

    actuator.execute(decision)

    total_latency = (time.perf_counter() - total_start) * 1000

    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "device_id": settings.system.device_id,
        "site_id": settings.system.site_id,
        "plate_text": ocr_result["plate_text"],
        "ocr_confidence": round(ocr_result["confidence"], 4),
        "detected_vehicle_class": best_vehicle["class_name"],
        "vehicle_confidence": round(best_vehicle["confidence"], 4),
        "plate_confidence": round(best_plate["confidence"], 4),
        "decision": decision,
        "reason": reason,
        "total_latency_ms": round(total_latency, 2),
        "vehicle_latency_ms": round(vehicle_latency, 2),
        "plate_latency_ms": round(plate_latency, 2),
        "ocr_latency_ms": round(ocr_latency, 2),
    }

    logger.info(f"logging event and inserting into database.....")
    repository.insert_event(event)
    event_logger.log(event)

    return event


if __name__ == "__main__":
    # Data Configs
    input_dir = Path("data/pipeline_test/vehicles")

    # Find all image files in the input directory
    image_paths = (
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpeg"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    # Process each image and print the results
    for image_path in image_paths[:5]:  # Process only first 5 images for testing
        print(f"\nProcessing: {image_path.name}")
        result = run_single_image(image_path)
        print(result)