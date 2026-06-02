import cv2
import numpy as np
from typing import Tuple

def clamp_bbox(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2

def crop_image(image: np.ndarray, bbox) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(*bbox, w, h)
    return image[y1:y2, x1:x2].copy()

def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)