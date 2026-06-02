# src/vision/plate_ocr.py

"""License plate OCR module for a vision-based access control system.   
This module defines a PlateOCR class that utilizes the PaddleOCR library to recognize text from license plate images."""

# Import required libraries
import re
import time
import logging
import numpy as np
import cv2
from paddleocr import PaddleOCR
from src.vision.image_ops import refine_plate_crop

logger = logging.getLogger(__name__)

OCRResultDict = dict[str, str | float]

# Define the PlateOCR class
class PlateOCR:
    """PlateOCR class that uses the PaddleOCR library to recognize text from license plate images.
    The constructor initializes the OCR engine with the specified language and angle classification settings.
    The normalize_plate_text static method processes raw OCR text to produce a cleaned license plate string.
    The recognize method processes an input image and returns the recognized text along with confidence scores and inference latency.
    
    Args:
        lang (str, optional): The language setting for OCR. Defaults to "en".
        use_angle_cls (bool, optional): Whether to use angle classification. Defaults to True.
        
    Raises:
        ValueError: If lang is not a valid language code
    """   
    
    def __init__(self, lang: str = "en", use_angle_cls: bool = True) -> None:
        valid_langs = {"en", "ch", "fr", "de", "es", "pt", "ru", "ar", "ja", "ko"}
        if lang not in valid_langs:
            logger.warning(f"Language '{lang}' may not be supported, proceeding anyway")
        
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                show_log=False
            )
            logger.info(f"PlateOCR initialized (lang={lang}, angle_cls={use_angle_cls})")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}", exc_info=True)
            raise

    @staticmethod
    def normalize_plate_text(text: str | None) -> str:
        """Normalize OCR output to clean license plate format.
        
        Args:
            text: Raw OCR text (may be None or empty)
            
        Returns:
            Normalized plate text (uppercase, alphanumeric only)
        """
        if not text:
            return ""
        text = str(text).upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text
    

    def recognize(self, image: np.ndarray) -> tuple[OCRResultDict, float]:
        """Recognize license plate text from an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (result_dict, latency_ms)
            result_dict contains: raw_text, plate_text, confidence
            
        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if image.size == 0:
            raise ValueError("Image is empty")
        
        start = time.perf_counter()

        try:
            image = refine_plate_crop(image)
        except Exception as e:
            logger.warning(f"Image refinement failed: {e}")
            # Continue with original image
        
        try:
            result = self.ocr.ocr(image, det=False, rec=True, cls=False)
        except Exception as e:
            logger.warning(f"OCR inference failed: {e}", exc_info=True)
            result = None

        latency_ms = (time.perf_counter() - start) * 1000

        text = ""
        confidence = 0.0

        try:
            if result and result[0]:
                text, confidence = result[0][0]
                confidence = float(confidence)
        except (IndexError, TypeError, ValueError) as e:
            logger.warning(f"OCR result parsing failed: {e}")
            text = ""
            confidence = 0.0

        normalized = self.normalize_plate_text(text)

        result_dict: OCRResultDict = {
            "raw_text": text,
            "plate_text": normalized,
            "confidence": confidence
        }
        
        logger.debug(f"OCR result: {normalized} (confidence: {confidence:.3f}, latency: {latency_ms:.2f}ms)")
        return result_dict, latency_ms
    

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
    print(f"OCR Result: {ocr_result}")
    print(f"Latency (ms): {latency:.2f}")
