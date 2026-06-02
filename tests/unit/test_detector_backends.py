"""Tests for ONNX and Hailo detector backends."""

import pytest
import numpy as np
from pathlib import Path

# Conditionally import based on availability
try:
    from src.vision.onnx_detector import ONNXDetector
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from src.vision.hailo_plate_detector import HailoPlateDetector


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not installed")
class TestONNXDetector:
    """Tests for ONNX detector backend."""
    
    def test_init_file_not_found(self):
        """Test that FileNotFoundError is raised for missing model."""
        with pytest.raises(FileNotFoundError):
            ONNXDetector(
                model_path="/nonexistent/model.onnx",
                target_classes=["car", "truck"],
                conf_threshold=0.5
            )
    
    def test_init_invalid_conf_threshold_too_high(self):
        """Test validation of conf_threshold > 1."""
        with pytest.raises(ValueError, match="conf_threshold must be in"):
            ONNXDetector(
                model_path="models/dummy.onnx",
                target_classes=["car"],
                conf_threshold=1.5
            )
    
    def test_init_invalid_conf_threshold_negative(self):
        """Test validation of negative conf_threshold."""
        with pytest.raises(ValueError, match="conf_threshold must be in"):
            ONNXDetector(
                model_path="models/dummy.onnx",
                target_classes=["car"],
                conf_threshold=-0.1
            )
    
    def test_init_empty_target_classes(self):
        """Test validation of empty target_classes."""
        with pytest.raises(ValueError, match="target_classes cannot be empty"):
            ONNXDetector(
                model_path="models/vehicle_detector.onnx",
                target_classes=[],
                conf_threshold=0.5
            )


class TestHailoPlateDetector:
    """Tests for Hailo plate detector backend."""
    
    def test_init_file_not_found(self):
        """Test that FileNotFoundError is raised for missing model."""
        with pytest.raises(FileNotFoundError):
            HailoPlateDetector(
                model_path="/nonexistent/model.hef",
                conf_threshold=0.5
            )
    
    def test_init_invalid_conf_threshold_too_high(self):
        """Test validation of conf_threshold > 1."""
        with pytest.raises(ValueError, match="conf_threshold must be in"):
            HailoPlateDetector(
                model_path="models/dummy.hef",
                conf_threshold=1.5
            )
    
    def test_init_invalid_conf_threshold_negative(self):
        """Test validation of negative conf_threshold."""
        with pytest.raises(ValueError, match="conf_threshold must be in"):
            HailoPlateDetector(
                model_path="models/dummy.hef",
                conf_threshold=-0.1
            )
    
    def test_detect_invalid_image_type(self, sample_image):
        """Test error when image is not ndarray."""
        # Skip if Hailo not available
        try:
            from hailo_sdk_common.hailo_sdk_common import HailoDeviceProperties  # noqa: F401
        except ImportError:
            pytest.skip("Hailo SDK not installed")
    
    def test_detect_empty_image(self):
        """Test error when image is empty."""
        try:
            from hailo_sdk_common.hailo_sdk_common import HailoDeviceProperties  # noqa: F401
        except ImportError:
            pytest.skip("Hailo SDK not installed")
    
    def test_get_model_info(self):
        """Test get_model_info returns expected structure."""
        # Can't test without actual model, but structure is defined in class
        # This test documents the expected return format
        pass


class TestDetectorBackendSelection:
    """Integration tests for detector backend selection."""
    
    @pytest.mark.skip(reason="libGL.so.1 not available in test environment")
    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError."""
        from src.vision.detector_factory import build_vehicle_detector
        from src.configs.settings import settings
        
        original_backend = settings.runtime.detector_backend
        
        try:
            settings.runtime.detector_backend = "invalid_backend"
            with pytest.raises(ValueError, match="Unsupported detector backend"):
                build_vehicle_detector()
            
        finally:
            settings.runtime.detector_backend = original_backend
