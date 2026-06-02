# src/vision/vehicle_detector.py

"""Vehicle detection module for a vision-based access control system.
This module defines a VehicleDetector class that utilizes a YOLO-based model to detect vehicles in images. 
It supports configurable model paths, target classes, and confidence thresholds. 
The detect method returns a list of detected vehicles with their class names, confidence scores, and bounding boxes, along with the inference latency in milliseconds. 
The select_best_vehicle static method allows selecting the most confident detection from a list of vehicle detections. """

# Import required libraries
import time
import numpy as np
from ultralytics import YOLO


# Define the VehicleDetector class
class VehicleDetector:
    """VehicleDetector class that uses a YOLO-based model to detect vehicles in images.
    The constructor initializes the model with the specified path, target classes, and confidence threshold.
    The detect method processes an input image and returns a list of detected vehicles along with inference latency.
    The select_best_vehicle static method allows selecting the most confident detection from a list of vehicle detections.
    
    Args:
    model_path (str): The file path to the YOLO model weights.
    target_classes (list[str]): A list of class names to filter detections (e.g., ["car", "truck", "bus", "motorcycle"]).
    conf_threshold (float): The confidence threshold for filtering detections (between 0 and 1)."""

    def __init__(self, model_path: str, target_classes: list[str], conf_threshold: float):
        self.model = YOLO(model_path)
        self.target_classes = set(target_classes)
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> tuple[list[dict], float]:
        start = time.perf_counter()
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        latency_ms = (time.perf_counter() - start) * 1000

        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]

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

        return detections, latency_ms

    @staticmethod
    def select_best_vehicle(detections: list[dict]) -> dict | None:
        if not detections:
            return None
        return sorted(detections, key=lambda x: x["confidence"], reverse=True)[0]
    

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