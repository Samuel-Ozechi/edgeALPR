# src/vision/plate_ocr.py

"""License plate OCR module for a vision-based access control system.   
This module defines a PlateOCR class that utilizes the PaddleOCR library to recognize text from license plate images."""

# Import required libraries
import re
import time
import numpy as np
import cv2
from paddleocr import PaddleOCR
from src.vision.image_ops import refine_plate_crop


# Define the PlateOCR class
class PlateOCR:
    """PlateOCR class that uses the PaddleOCR library to recognize text from license plate images.
    The constructor initializes the OCR engine with the specified language and angle classification settings.
    The normalize_plate_text static method processes raw OCR text to produce a cleaned license plate string.
    The recognize method processes an input image and returns the recognized text along with confidence scores and inference latency.
    Args:
        lang (str, optional): The language setting for OCR. Defaults to "en".
        use_angle_cls (bool, optional): Whether to use angle classification. Defaults to True."""   
    
    def __init__(self, lang: str = "en", use_angle_cls: bool = True):
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            show_log=False
        )

    @staticmethod
    def normalize_plate_text(text: str | None) -> str:
        if not text:
            return ""
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text
    

    def recognize(self, image: np.ndarray) -> tuple[dict, float]:
        start = time.perf_counter()

        image = refine_plate_crop(image)
        
        result = self.ocr.ocr(image, det=False, rec=True, cls=False)

        latency_ms = (time.perf_counter() - start) * 1000

        text = ""
        confidence = 0.0

        try:
            if result and result[0]:
                text, confidence = result[0][0]
                confidence = float(confidence)
        except Exception:
            text = ""
            confidence = 0.0

        normalized = self.normalize_plate_text(text)

        return {
            "raw_text": text,
            "plate_text": normalized,
            "confidence": confidence
        }, latency_ms
    

if __name__ == "__main__":
    from src.configs.settings import settings
    import cv2

    # Example usage
    plate_ocr = PlateOCR(
        lang=settings.models.ocr_lang,
        use_angle_cls=settings.models.ocr_use_angle_cls
    )

    test_image_path = "data/pipeline_test/license/best_plate_crop.jpg"
    image = cv2.imread(test_image_path)

    ocr_result, latency = plate_ocr.recognize(image)
    print(f"OCR Result: {ocr_result}, Latency: {latency:.2f} ms")   