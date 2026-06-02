"""ONNX detector backend for efficient inference on CPU and NPU devices.

This module provides a detector implementation using ONNX Runtime for
cross-platform inference, supporting CPU, GPU, and specialized accelerators.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as rt
except ImportError:
    rt = None

from src.vision.base_detector import DetectionDict, BaseClassifier

logger = logging.getLogger(__name__)


class ONNXDetector(BaseClassifier):
    """ONNX-based detector for CPU/GPU inference.
    
    Uses ONNX Runtime for efficient cross-platform model inference.
    Supports vehicle and plate detection through inherited BaseClassifier.
    
    Example:
        detector = ONNXDetector(
            model_path="models/vehicle_detector.onnx",
            target_classes=["car", "truck"],
            conf_threshold=0.5
        )
        detections, inference_time = detector.detect(image)
    """
    
    def __init__(
        self,
        model_path: str,
        target_classes: list[str],
        conf_threshold: float = 0.45,
        providers: Optional[list[str]] = None,
    ) -> None:
        """Initialize ONNX detector.
        
        Args:
            model_path: Path to ONNX model file
            target_classes: List of class names to detect
            conf_threshold: Confidence threshold (0-1)
            providers: ONNX Runtime execution providers
                      (e.g., ["CUDAExecutionProvider", "CPUExecutionProvider"])
            
        Raises:
            ImportError: If onnxruntime not installed
            FileNotFoundError: If model file doesn't exist
            ValueError: If conf_threshold not in [0, 1]
            ValueError: If target_classes is empty
        """
        # Validate parameters first before checking imports
        if conf_threshold < 0 or conf_threshold > 1:
            raise ValueError(f"conf_threshold must be in [0, 1], got {conf_threshold}")
        
        if not target_classes:
            raise ValueError("target_classes cannot be empty")
        
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        # Convert list to set for BaseClassifier compatibility
        target_classes_set = set(c.lower() for c in target_classes)
        
        # Call parent __init__ with correct parameter order: conf_threshold, target_classes
        super().__init__(conf_threshold, target_classes_set)
        
        # Now check for onnxruntime
        if rt is None:
            raise ImportError(
                "onnxruntime is required for ONNX detector. "
                "Install with: pip install onnxruntime"
            )
        
        # Default to CPU if not specified
        if providers is None:
            providers = ["CPUExecutionProvider"]
        
        try:
            self.session = rt.InferenceSession(
                str(model_path),
                providers=providers
            )
            logger.info(
                f"ONNX detector loaded: {model_path} "
                f"(providers: {self.session.get_providers()})"
            )
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
        
        self.model_path = str(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        logger.debug(
            f"ONNX model inputs: {self.input_name}, outputs: {self.output_names}"
        )
    
    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[list[DetectionDict], float]:
        """Run inference on image using ONNX Runtime.
        
        Note: This is a simplified implementation. Real ONNX detector would:
        - Handle various input/output tensor formats
        - Implement proper post-processing (NMS, etc.)
        - Scale inputs to model requirements
        
        For production, use properly formatted ONNX YOLOv8 models with
        dynamic input shapes: [1, 3, H, W] (NCHW format)
        
        Args:
            image: Input image as numpy array (BGR, HxWx3)
            
        Returns:
            Tuple of (detections, inference_time_ms)
            
        Raises:
            ValueError: If image is not ndarray or is empty
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image must be numpy.ndarray, got {type(image)}")
        
        if image.size == 0:
            raise ValueError("Image is empty")
        
        import time
        start_time = time.time()
        
        try:
            # Placeholder: Real implementation would:
            # 1. Resize image to model input size
            # 2. Normalize pixels (e.g., [0,255] -> [0,1])
            # 3. Convert BGR to RGB if needed
            # 4. Transpose to NCHW format: (1, 3, H, W)
            # 5. Run inference
            # 6. Post-process outputs (NMS, confidence filtering)
            # 7. Scale boxes back to original image size
            
            logger.debug(
                f"ONNX inference on image shape {image.shape} "
                f"using {self.session.get_providers()}"
            )
            
            # For now, return empty detections to avoid actual inference
            # In production, implement full inference pipeline
            detections: list[DetectionDict] = []
            
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}", exc_info=True)
            detections = []
        
        inference_time = (time.time() - start_time) * 1000  # ms
        logger.debug(f"ONNX inference time: {inference_time:.2f}ms")
        
        return detections, inference_time
    
    def get_model_info(self) -> dict:
        """Get ONNX model metadata.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": "ONNX",
            "model_path": self.model_path,
            "target_classes": self.target_classes,
            "conf_threshold": self.conf_threshold,
            "input_name": self.input_name,
            "output_names": self.output_names,
            "providers": self.session.get_providers(),
        }
