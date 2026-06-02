"""Complete example: Full access control workflow.

This example demonstrates the complete edgeALPR pipeline including:
- Vehicle detection
- Plate detection and OCR
- Database lookup
- Access decision logic
- Action execution

Usage:
    python examples/with_access_control.py path/to/image.jpg
"""

import sys
from pathlib import Path

import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configs.settings import settings
from src.vision.detector_factory import build_vehicle_detector, build_plate_detector
from src.vision.plate_ocr import PlateOCR
from src.decision.rules_engine import AccessRulesEngine
from src.actuator.mock_actuator import MockActuator
from src.db.repository import VehicleRepository


def main(image_path: str) -> None:
    """Run complete access control workflow.
    
    Args:
        image_path: Path to input image file
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    print(f"Loaded image: {image_path} ({image.shape})")
    print("-" * 60)
    
    # Initialize components
    print("Initializing components...")
    vehicle_detector = build_vehicle_detector()
    plate_detector = build_plate_detector()
    ocr_engine = PlateOCR()
    rules_engine = AccessRulesEngine(
        ocr_confidence_threshold=settings.thresholds.ocr_conf
    )
    actuator = MockActuator()
    repository = VehicleRepository(settings.db.database_path)
    
    print("  ✓ Vehicle detector")
    print("  ✓ Plate detector")
    print("  ✓ OCR engine")
    print("  ✓ Rules engine")
    print("  ✓ Actuator")
    print("  ✓ Database")
    print()
    
    # Step 1: Detect vehicle
    print("Step 1: Vehicle Detection")
    print("-" * 40)
    vehicles, vehicle_latency = vehicle_detector.detect(image)
    print(f"  Found {len(vehicles)} vehicle(s) in {vehicle_latency:.2f}ms")
    
    if not vehicles:
        print("  → No vehicles detected, skipping access control")
        return
    
    vehicle = vehicles[0]  # Process first vehicle
    print(f"  → Using best match (confidence: {vehicle['confidence']:.2%})")
    print()
    
    # Step 2: Detect and recognize plate
    print("Step 2: Plate Detection & OCR")
    print("-" * 40)
    
    # Detect plate in full image
    detections, detect_latency = plate_detector.detect(image)
    if not detections:
        print("  [Error] No plates detected")
        return
    
    plate_detection = detections[0]
    bbox = plate_detection.get("bbox", (0, 0, 10, 10))
    x1, y1, x2, y2 = bbox
    plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
    
    print(f"  Plate confidence: {plate_detection['confidence']:.2%}")
    
    # Recognize text
    try:
        result, ocr_latency = ocr_engine.recognize(plate_crop)
        plate_text = result.get("plate_text", "")
        ocr_confidence = result.get("confidence", 0.0)
        
        print(f"  Recognized plate: {plate_text}")
        print(f"  OCR confidence: {ocr_confidence:.2%} ({ocr_latency:.2f}ms)")
    except Exception as e:
        print(f"  [Error] OCR failed: {e}")
        return
    
    print()
    
    # Step 3: Database lookup
    print("Step 3: Vehicle Authorization Lookup")
    print("-" * 40)
    
    vehicle_record = repository.get_vehicle_by_plate(plate_text)
    if vehicle_record:
        print(f"  ✓ Vehicle found in database")
        print(f"    ID: {vehicle_record['id']}")
        print(f"    Class: {vehicle_record.get('class', 'unknown')}")
        print(f"    Status: {vehicle_record.get('status', 'unknown')}")
    else:
        print(f"  ✗ Vehicle NOT found in database")
    
    print()
    
    # Step 4: Make access decision
    print("Step 4: Access Control Decision")
    print("-" * 40)
    
    decision, reason = rules_engine.decide(
        plate_text=plate_text,
        ocr_confidence=ocr_confidence,
        vehicle_class=vehicle.get("class_name", "unknown"),
        vehicle_found=vehicle_record is not None
    )
    
    print(f"  Decision: {decision.upper()}")
    print(f"  Reason: {reason}")
    print()
    
    # Step 5: Execute action
    print("Step 5: Action Execution")
    print("-" * 40)
    
    try:
        actuator.execute(decision)  # type: ignore
        state = actuator.get_state()
        print(f"  ✓ Action executed")
        print(f"  Last action: {state['last_action']}")
        print(f"  Total actions: {state['action_count']}")
    except Exception as e:
        print(f"  [Error] Action failed: {e}")
    
    print()
    
    # Step 6: Log event
    print("Step 6: Event Logging")
    print("-" * 40)
    
    event = {
        "timestamp": settings.get_timestamp(),
        "vehicle_confidence": vehicle["confidence"],
        "plate_text": plate_text,
        "ocr_confidence": ocr_confidence,
        "decision": decision,
        "vehicle_found": vehicle_record is not None
    }
    
    try:
        repository.insert_event(event)
        print(f"  ✓ Event logged to database")
    except Exception as e:
        print(f"  [Error] Logging failed: {e}")
    
    print()
    print("=" * 60)
    print("ACCESS CONTROL WORKFLOW COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python with_access_control.py <image_path>")
        sys.exit(1)
    
    main(sys.argv[1])
