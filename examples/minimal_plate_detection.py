"""Minimal example: Detect and recognize license plates from an image.

This example shows the simplest way to use edgeALPR for plate detection and OCR.

Usage:
    python examples/minimal_plate_detection.py path/to/image.jpg
"""

import sys
from pathlib import Path

import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configs.settings import settings
from src.vision.detector_factory import build_plate_detector
from src.vision.plate_ocr import PlateOCR


def main(image_path: str) -> None:
    """Detect and recognize plates in an image.
    
    Args:
        image_path: Path to input image file
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    print(f"Loaded image: {image_path} ({image.shape})")
    
    # Initialize detector and OCR
    print("Initializing models...")
    plate_detector = build_plate_detector()
    ocr_engine = PlateOCR()
    
    # Detect plates
    print("Detecting plates...")
    detections, detect_latency = plate_detector.detect(image)
    print(f"  Found {len(detections)} plate(s) in {detect_latency:.2f}ms")
    
    if not detections:
        print("No plates detected in image")
        return
    
    # Process each plate
    for i, detection in enumerate(detections):
        confidence = detection.get("confidence", 0.0)
        bbox = detection.get("bbox", (0, 0, 10, 10))
        
        print(f"\nPlate {i+1}:")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Bounding box: {bbox}")
        
        # Extract plate region
        x1, y1, x2, y2 = bbox
        plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
        
        if plate_crop.size == 0:
            print("  [Skipped: Invalid crop region]")
            continue
        
        # Recognize text
        try:
            result, ocr_latency = ocr_engine.recognize(plate_crop)
            plate_text = result.get("plate_text", "")
            ocr_conf = result.get("confidence", 0.0)
            
            print(f"  Recognized: {plate_text}")
            print(f"  OCR confidence: {ocr_conf:.2%}")
            print(f"  OCR latency: {ocr_latency:.2f}ms")
        except Exception as e:
            print(f"  [Error: {e}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python minimal_plate_detection.py <image_path>")
        sys.exit(1)
    
    main(sys.argv[1])
