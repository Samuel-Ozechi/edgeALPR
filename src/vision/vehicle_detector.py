# src/vision/vehicle_detector.py

"""Vehicle detection module for a vision-based access control system.
This module defines a VehicleDetector class that utilizes a YOLO-based model to detect vehicles in images. 
It supports configurable model paths, target classes, and confidence thresholds. 
The detect method returns a list of detected vehicles with their class names, confidence scores, and bounding boxes, along with the inference latency in milliseconds. 
The select_best_vehicle static method allows selecting the most confident detection from a list of vehicle detections. """

# Import required libraries
import time
import logging
from pathlib import Path
from typing import Literal, Any
import numpy as np
from ultralytics import YOLO
from src.vision.base_detector import BaseClassifier

logger = logging.getLogger(__name__)

DetectionDict = dict[str, float | int | str | tuple[int, int, int, int]]

# Define the VehicleDetector class
class VehicleDetector(BaseClassifier):
    """VehicleDetector class that uses a YOLO-based model to detect vehicles in images.
    The constructor initializes the model with the specified path, target classes, and confidence threshold.
    The detect method processes an input image and returns a list of detected vehicles along with inference latency.
    The select_best_vehicle static method allows selecting the most confident detection from a list of vehicle detections.
    
    Args:
        model_path (str | Path): The file path to the YOLO model weights.
        target_classes (set[str]): A set of class names to filter detections (e.g., {"car", "truck", "bus", "motorcycle"}).
        conf_threshold (float): The confidence threshold for filtering detections (between 0 and 1).
        
    Raises:
        FileNotFoundError: If model_path does not exist
        ValueError: If conf_threshold is not between 0 and 1
    """

    def __init__(self, model_path: str | Path, target_classes: set[str], conf_threshold: float) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Vehicle detector model not found: {model_path}")
        
        if not target_classes:
            raise ValueError("target_classes cannot be empty")
        
        super().__init__(conf_threshold, target_classes)
        
        try:
            self.model = YOLO(str(model_path))
            self.model_path = model_path
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            raise
        
        logger.info(f"VehicleDetector initialized with model: {model_path}")

    def detect(self, image: np.ndarray) -> tuple[list[DetectionDict], float]:
        """Detect vehicles in an image.
        
        Args:
            image: Input image as numpy array (BGR format expected)
            
        Returns:
            Tuple of (detections list, latency_ms float)
            Each detection is a dict with keys: class_id, class_name, confidence, bbox
            
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
            logger.error(f"Vehicle detection failed: {e}", exc_info=True)
            raise
            
        latency_ms = (time.perf_counter() - start) * 1000

        detections: list[DetectionDict] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id].lower()

                if cls_name not in self.target_classes:
                    continue

                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        logger.debug(f"Detected {len(detections)} vehicles in {latency_ms:.2f}ms")
        return detections, latency_ms

    @staticmethod
    def select_best_vehicle(detections: list[DetectionDict]) -> DetectionDict | None:
        """Select the vehicle with highest confidence.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Best detection dict or None if list is empty
        """
        if not detections:
            return None
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)[0]
    
    def get_model_info(self) -> dict[str, Any]:
        """Get metadata about the vehicle detector model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": "YOLOv8 Vehicle Detector",
            "backend": "PyTorch (ultralytics)",
            "model_path": str(self.model_path),
            "target_classes": sorted(self.target_classes),
            "confidence_threshold": self.conf_threshold,
            "input_shape": (640, 640, 3),  # YOLO default
            "output_shape": "variable",
        }
    

if __name__ == "__main__":
    import cv2
    from src.configs.settings import settings
    import os

    # Example usage
    vehicle_detector = VehicleDetector(
        model_path=settings.models.vehicle_detector,
        target_classes=settings.models.vehicle_target_classes,
        conf_threshold=settings.thresholds.vehicle_conf
    )

    test_image_path = "data/pipeline_test/vehicles/2_jpg.rf.3fb6ec616e366fe1688fdba2d6ba919f.jpg"

    image = cv2.imread(test_image_path)
    detections, latency = vehicle_detector.detect(image)

    # Save cropped images to a defined output directory
    output_dir = "data/pipeline_test/plates"
    os.makedirs(output_dir, exist_ok=True)
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        cropped = image[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(output_dir, f"vehicle_{idx}.jpg"), cropped)

    print(f"Detected vehicles: {detections}")
    print(f"Inference latency: {latency:.2f} ms")