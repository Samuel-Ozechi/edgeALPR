# src/vision/plate_detector.py

"""Plate detection module for a vision-based access control system. 
This module defines a PlateDetector class that utilizes a YOLO-based model to detect license plates in images.
It supports configurable model paths and confidence thresholds. 
The detect method returns a list of detected plates with their confidence scores and bounding boxes, along with the inference latency in milliseconds. 
The select_best_plate static method allows selecting the most confident detection from a list of plate detections."""

# Import required libraries
import time
import logging
from pathlib import Path
from typing import Any
import numpy as np
from ultralytics import YOLO
from src.vision.base_detector import BaseDetector

logger = logging.getLogger(__name__)

DetectionDict = dict[str, float | tuple[int, int, int, int]]

# Define the PlateDetector class
class PlateDetector(BaseDetector):
    """PlateDetector class that uses a YOLO-based model to detect license plates in images.
    The constructor initializes the model with the specified path and confidence threshold.
    The detect method processes an input image and returns a list of detected plates along with inference latency.
    The select_best_plate static method allows selecting the most confident detection from a list of plate detections.

    Args:
        model_path (str | Path): The file path to the YOLO model weights.
        conf_threshold (float): The confidence threshold for filtering detections (between 0 and 1).
        
    Raises:
        FileNotFoundError: If model_path does not exist
        ValueError: If conf_threshold is not between 0 and 1
    """
    
    def __init__(self, model_path: str | Path, conf_threshold: float) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Plate detector model not found: {model_path}")
        
        super().__init__(conf_threshold)
        
        try:
            self.model = YOLO(str(model_path))
            self.model_path = model_path
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            raise
        
        logger.info(f"PlateDetector initialized with model: {model_path}")

    def detect(self, image: np.ndarray) -> tuple[list[DetectionDict], float]:
        """Detect license plates in an image.
        
        Args:
            image: Input image as numpy array (BGR format expected)
            
        Returns:
            Tuple of (detections list, latency_ms float)
            Each detection is a dict with keys: confidence, bbox
            
        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if image.size == 0:
            raise ValueError("Image is empty")
        
        start = time.perf_counter()
        try:
            results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            logger.error(f"Plate detection failed: {e}", exc_info=True)
            raise
            
        latency_ms = (time.perf_counter() - start) * 1000

        detections: list[DetectionDict] = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append({
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        logger.debug(f"Detected {len(detections)} plates in {latency_ms:.2f}ms")
        return detections, latency_ms

    @staticmethod
    def select_best_plate(detections: list[DetectionDict]) -> DetectionDict | None:
        """Select the plate with highest confidence.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Best detection dict or None if list is empty
        """
        if not detections:
            return None
        return detections[0]
    
    def get_model_info(self) -> dict[str, Any]:
        """Get metadata about the plate detector model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": "YOLOv8 Plate Detector",
            "backend": "PyTorch (ultralytics)",
            "model_path": str(self.model_path),
            "confidence_threshold": self.conf_threshold,
            "input_shape": (640, 640, 3),  # YOLO default
            "output_shape": "variable",
        }
    

if __name__ == "__main__":
    from src.configs.settings import settings
    import cv2
    # Example usage
    plate_detector = PlateDetector(
        model_path=settings.models.plate_detector,
        conf_threshold=settings.thresholds.plate_conf
    )

    test_image_path = "data/pipeline_test/plates/vehicle_0.jpg"
    image = cv2.imread(test_image_path)
    detections, latency = plate_detector.detect(image)
    best_plate = plate_detector.select_best_plate(detections)

    # save best plate crop
    if best_plate is not None:
        x1, y1, x2, y2 = best_plate["bbox"]
        plate_crop = image[y1:y2, x1:x2]
        cv2.imwrite("data/pipeline_test/license/best_plate_crop.jpg", plate_crop)

    print(f"Detections: {detections}")
    print(f"Best Plate: {best_plate}")
    print(f"Latency (ms): {latency:.2f}")