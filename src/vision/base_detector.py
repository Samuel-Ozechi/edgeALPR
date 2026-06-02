"""Abstract base classes for detectors in edgeALPR."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

DetectionDict = dict[str, Any]


class BaseDetector(ABC):
    """Abstract base class for all object detectors.
    
    Subclasses should implement the detect() method to support different model backends
    (PyTorch, ONNX, Hailo, TensorRT, etc.) while maintaining a consistent interface.
    
    Example:
        class CustomONNXDetector(BaseDetector):
            def __init__(self, model_path):
                self.session = ort.InferenceSession(model_path)
                
            def detect(self, image):
                # Custom implementation
                pass
    """

    def __init__(self, conf_threshold: float) -> None:
        """Initialize detector with confidence threshold.
        
        Args:
            conf_threshold: Confidence threshold (0-1)
            
        Raises:
            ValueError: If conf_threshold is not in valid range
        """
        if not (0 <= conf_threshold <= 1):
            raise ValueError(f"conf_threshold must be between 0 and 1, got {conf_threshold}")
        self.conf_threshold = conf_threshold
        logger.info(f"{self.__class__.__name__} initialized (threshold={conf_threshold})")

    @abstractmethod
    def detect(self, image: np.ndarray) -> tuple[list[DetectionDict], float]:
        """Detect objects in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (detections list, latency_ms float)
            Each detection should have standard keys: confidence, bbox
            (and optionally: class_id, class_name for classified detectors)
            
        Raises:
            ValueError: If image is invalid
            RuntimeError: If inference fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get metadata about the model.
        
        Returns:
            Dictionary with keys: name, version, backend, input_shape, output_shape, etc.
        """
        pass


class BaseClassifier(BaseDetector):
    """Abstract base class for classifying detectors (vehicles, plates, etc.).
    
    Adds class-specific detection metadata beyond base detector.
    """

    def __init__(self, conf_threshold: float, target_classes: set[str] | None = None) -> None:
        """Initialize classifier.
        
        Args:
            conf_threshold: Confidence threshold (0-1)
            target_classes: Set of class names to filter for
        """
        super().__init__(conf_threshold)
        self.target_classes = target_classes or set()

    @abstractmethod
    def detect(self, image: np.ndarray) -> tuple[list[DetectionDict], float]:
        """Detect and classify objects in an image.
        
        Returns:
            Each detection should have: class_id, class_name, confidence, bbox
        """
        pass


class BaseOCR(ABC):
    """Abstract base class for OCR engines."""

    def __init__(self) -> None:
        """Initialize OCR engine."""
        logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def recognize(self, image: np.ndarray) -> tuple[dict[str, str | float], float]:
        """Recognize text from an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (result_dict, latency_ms)
            result_dict should have: raw_text, plate_text (normalized), confidence
        """
        pass

    @staticmethod
    @abstractmethod
    def normalize_text(text: str | None) -> str:
        """Normalize raw text output to desired format."""
        pass
