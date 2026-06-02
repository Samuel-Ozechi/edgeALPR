from src.configs.settings import settings
from src.vision.vehicle_detector import VehicleDetector
from src.vision.plate_detector import PlateDetector
from src.vision.hailo_yolo_detector import HailoYoloDetector

# Shared Hailo device across detectors
_shared_vdevice = None


def get_shared_vdevice():
    """Get or create a shared VDevice for Hailo inference."""
    global _shared_vdevice
    if _shared_vdevice is None:
        from hailo_platform import VDevice
        _shared_vdevice = VDevice()
    return _shared_vdevice


def build_vehicle_detector():
    if settings.runtime.detector_backend == "hailo":
        vdevice = get_shared_vdevice()
        return HailoYoloDetector(
            hef_path=str(settings.models.vehicle_detector_hef),
            class_names=settings.models.vehicle_target_classes,
            conf_threshold=settings.thresholds.vehicle_conf,
            model_type="vehicle",
            vdevice=vdevice
        )

    return VehicleDetector(
        model_path=settings.models.vehicle_detector,
        target_classes=settings.models.vehicle_target_classes,
        conf_threshold=settings.thresholds.vehicle_conf
    )


def build_plate_detector():
    if settings.runtime.detector_backend == "hailo":
        vdevice = get_shared_vdevice()
        return HailoYoloDetector(
            hef_path=str(settings.models.plate_detector_hef),
            class_names=["plate"],
            conf_threshold=settings.thresholds.plate_conf,
            model_type="plate",
            vdevice=vdevice
        )

    return PlateDetector(
        model_path=settings.models.plate_detector,
        conf_threshold=settings.thresholds.plate_conf
    )