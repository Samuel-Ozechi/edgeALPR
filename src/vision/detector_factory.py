import logging
import threading
from typing import Union

from src.configs.settings import settings
from src.vision.vehicle_detector import VehicleDetector
from src.vision.plate_detector import PlateDetector
from src.vision.hailo_yolo_detector import HailoYoloDetector
from src.vision.onnx_detector import ONNXDetector
from src.vision.hailo_plate_detector import HailoPlateDetector
from src.vision.base_detector import BaseDetector, BaseClassifier

# Shared Hailo device across detectors - thread-safe access
_shared_vdevice = None
_vdevice_lock = threading.Lock()

logger = logging.getLogger(__name__)


def get_shared_vdevice():
    """Get or create a shared VDevice for Hailo inference in a thread-safe manner.
    
    Returns:
        VDevice: Hailo accelerator device handle
        
    Raises:
        ImportError: If Hailo platform is not installed or device cannot be initialized
    """
    global _shared_vdevice
    if _shared_vdevice is None:
        with _vdevice_lock:
            # Double-check pattern for thread safety
            if _shared_vdevice is None:
                try:
                    from hailo_platform import VDevice
                    _shared_vdevice = VDevice()
                    logger.info("Hailo VDevice initialized")
                except ImportError as e:
                    logger.error("Hailo platform not installed", exc_info=True)
                    raise
                except Exception as e:
                    logger.error(f"Failed to initialize Hailo VDevice: {e}", exc_info=True)
                    raise
    return _shared_vdevice


def build_vehicle_detector() -> Union[BaseClassifier, VehicleDetector, ONNXDetector]:
    """Build vehicle detector based on configuration.
    
    Supports multiple backends:
    - pytorch (default): PyTorch YOLOv8 models
    - onnx: ONNX Runtime for CPU/GPU/NPU inference
    - hailo: Hailo AI accelerator for edge devices
    
    Returns:
        Configured vehicle detector instance
        
    Raises:
        ValueError: If backend not supported
        FileNotFoundError: If model file not found
    """
    backend = settings.runtime.detector_backend
    
    if backend == "hailo":
        vdevice = get_shared_vdevice()
        return HailoYoloDetector(
            hef_path=str(settings.models.vehicle_detector_hef),
            class_names=settings.models.vehicle_target_classes,
            conf_threshold=settings.thresholds.vehicle_conf,
            model_type="vehicle",
            vdevice=vdevice
        )
    
    elif backend == "onnx":
        logger.info("Building ONNX vehicle detector")
        return ONNXDetector(
            model_path=str(settings.models.vehicle_detector_onnx),
            target_classes=settings.models.vehicle_target_classes,
            conf_threshold=settings.thresholds.vehicle_conf
        )
    
    elif backend == "pytorch":
        logger.info("Building PyTorch vehicle detector")
        return VehicleDetector(
            model_path=str(settings.models.vehicle_detector),
            target_classes=settings.models.vehicle_target_classes,
            conf_threshold=settings.thresholds.vehicle_conf
        )
    
    else:
        raise ValueError(
            f"Unsupported detector backend: {backend}. "
            f"Supported: pytorch, onnx, hailo"
        )


def build_plate_detector() -> Union[BaseDetector, PlateDetector, ONNXDetector, HailoPlateDetector]:
    """Build plate detector based on configuration.
    
    Supports multiple backends:
    - pytorch (default): PyTorch YOLOv8 plate detector
    - onnx: ONNX Runtime for CPU/GPU/NPU inference
    - hailo: Hailo AI accelerator with optimized plate detector
    
    Returns:
        Configured plate detector instance
        
    Raises:
        ValueError: If backend not supported
        FileNotFoundError: If model file not found
    """
    backend = settings.runtime.detector_backend
    
    if backend == "hailo":
        logger.info("Building Hailo plate detector")
        vdevice = get_shared_vdevice()
        # Use specialized HailoPlateDetector for optimized inference
        return HailoPlateDetector(
            model_path=str(settings.models.plate_detector_hef),
            vdevice=vdevice,
            conf_threshold=settings.thresholds.plate_conf
        )
    
    elif backend == "onnx":
        logger.info("Building ONNX plate detector")
        return ONNXDetector(
            model_path=str(settings.models.plate_detector_onnx),
            target_classes=["plate"],
            conf_threshold=settings.thresholds.plate_conf
        )
    
    elif backend == "pytorch":
        logger.info("Building PyTorch plate detector")
        return PlateDetector(
            model_path=str(settings.models.plate_detector),
            conf_threshold=settings.thresholds.plate_conf
        )
    
    else:
        raise ValueError(
            f"Unsupported detector backend: {backend}. "
            f"Supported: pytorch, onnx, hailo"
        )