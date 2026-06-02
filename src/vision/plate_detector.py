# src/vision/plate_detector.py

"""Plate detection module for a vision-based access control system. 
This module defines a PlateDetector class that utilizes a YOLO-based model to detect license plates in images.
It supports configurable model paths and confidence thresholds. 
The detect method returns a list of detected plates with their confidence scores and bounding boxes, along with the inference latency in milliseconds. 
The select_best_plate static method allows selecting the most confident detection from a list of plate detections."""

# Import required libraries
import time
import numpy as np
from ultralytics import YOLO

#  Define the PlateDetector class
class PlateDetector:
    """PlateDetector class that uses a YOLO-based model to detect license plates in images.
    The constructor initializes the model with the specified path and confidence threshold.
    The detect method processes an input image and returns a list of detected plates along with inference latency.
    The select_best_plate static method allows selecting the most confident detection from a list of plate detections.

    Args:
        model_path (str): The file path to the YOLO model weights.
        conf_threshold (float): The confidence threshold for filtering detections (between 0 and 1)."""
    
    def __init__(self, model_path: str, conf_threshold: float):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> tuple[list[dict], float]:
        start = time.perf_counter()
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        latency_ms = (time.perf_counter() - start) * 1000

        detections = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append({
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        return detections, latency_ms

    @staticmethod
    def select_best_plate(detections: list[dict]) -> dict | None:
        if not detections:
            return None
        return detections[0]
    

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