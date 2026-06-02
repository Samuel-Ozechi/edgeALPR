# src/vision/image_ops.py

"""Utility functions for image processing in a vision-based access control system.
This module provides functions to load images, crop bounding boxes, and refine license plate crops for better OCR performance. 
It ensures that bounding boxes are clamped within image dimensions and supports optional histogram equalization"""

# Import required libraries
from pathlib import Path
import cv2
import numpy as np

# Load an image from a given path
def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from the specified path. 
    Raises FileNotFoundError if the image cannot be loaded.
    
    Args:
        image_path (str | Path): The file path to the image."""
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


# Ensure bounding box coordinates are within image dimensions
def clamp_bbox(x1, y1, x2, y2, width, height) -> tuple[int, int, int, int]:
    """Clamp bounding box coordinates within image dimensions.
    
    Args:
        x1, y1, x2, y2 (int): Bounding box coordinates.
        width, height (int): Image dimensions.
    
    Returns:
        tuple[int, int, int, int]: Clamped bounding box coordinates."""
    
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2


# Crop a region from the image based on the provided bounding box
def crop_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a region from the image based on the provided bounding box.
    The bounding box is clamped to ensure it stays within the image dimensions.
    
    Args:
        image (np.ndarray): The input image.
        bbox (tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).
    
    Returns:
        np.ndarray: The cropped image region."""
    
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(*bbox, w, h)
    return image[y1:y2, x1:x2].copy()


# Refine a license plate crop for better OCR performance
def refine_plate_crop(
    plate_crop: np.ndarray,
    target_size: tuple[int, int] = (320, 96),
    apply_equalization: bool = True
) -> np.ndarray | None:
    
    """Refine a license plate crop for better OCR performance.
    This function applies optional histogram equalization and resizes the crop to a target size.    
    Args:
        plate_crop (np.ndarray): The cropped license plate image.
        target_size (tuple[int, int], optional): The desired output size (width, height). Defaults to (320, 96).
        apply_equalization (bool, optional): Whether to apply histogram equalization. Defaults to True. """
    
    if plate_crop is None or plate_crop.size == 0:
        return None

    refined = plate_crop.copy()

    if apply_equalization:
        gray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        refined = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    refined = cv2.resize(refined, target_size)
    return refined