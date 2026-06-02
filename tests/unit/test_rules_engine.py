"""Unit tests for the AccessRulesEngine."""

import pytest
from src.decision.rules_engine import AccessRulesEngine


class TestAccessRulesEngine:
    """Test suite for AccessRulesEngine decision logic."""
    
    @pytest.fixture
    def engine(self):
        """Create a rules engine for testing."""
        return AccessRulesEngine(ocr_conf_threshold=0.7, require_class_match=False)
    
    @pytest.fixture
    def authorized_vehicle(self):
        """Create an authorized vehicle record."""
        return {
            "id": 1,
            "plate_text": "ABC123",
            "status": "authorized",
            "vehicle_class": "car"
        }
    
    @pytest.fixture
    def unauthorized_vehicle(self):
        """Create an unauthorized vehicle record."""
        return {
            "id": 2,
            "plate_text": "XYZ999",
            "status": "revoked",
            "vehicle_class": "truck"
        }
    
    def test_init_valid_threshold(self):
        """Test initialization with valid threshold."""
        engine = AccessRulesEngine(ocr_conf_threshold=0.5)
        assert engine.ocr_conf_threshold == 0.5
        assert engine.require_class_match is False
    
    def test_init_invalid_threshold_too_high(self):
        """Test initialization rejects threshold > 1."""
        with pytest.raises(ValueError):
            AccessRulesEngine(ocr_conf_threshold=1.5)
    
    def test_init_invalid_threshold_negative(self):
        """Test initialization rejects negative threshold."""
        with pytest.raises(ValueError):
            AccessRulesEngine(ocr_conf_threshold=-0.1)
    
    def test_decide_no_plate_text(self, engine):
        """Test decision when plate_text is None."""
        decision, reason = engine.decide(None, 0.9, "car", {"status": "authorized"})
        assert decision == "review"
        assert reason == "no_plate_text"
    
    def test_decide_empty_plate_text(self, engine):
        """Test decision when plate_text is empty string."""
        decision, reason = engine.decide("", 0.9, "car", {"status": "authorized"})
        assert decision == "review"
        assert reason == "no_plate_text"
    
    def test_decide_low_ocr_confidence(self, engine):
        """Test decision when OCR confidence is below threshold."""
        decision, reason = engine.decide("ABC123", 0.5, "car", {"status": "authorized"})
        assert decision == "review"
        assert reason == "low_ocr_confidence"
    
    def test_decide_plate_not_found(self, engine):
        """Test decision when vehicle record is not found."""
        decision, reason = engine.decide("UNKNOWN", 0.95, "car", None)
        assert decision == "deny"
        assert reason == "plate_not_found"
    
    def test_decide_vehicle_not_authorized(self, engine, unauthorized_vehicle):
        """Test decision when vehicle is not authorized."""
        decision, reason = engine.decide("XYZ999", 0.95, "truck", unauthorized_vehicle)
        assert decision == "deny"
        assert reason == "vehicle_not_authorized"
    
    def test_decide_authorized_allow(self, engine, authorized_vehicle):
        """Test decision allows authorized vehicle."""
        decision, reason = engine.decide("ABC123", 0.95, "car", authorized_vehicle)
        assert decision == "allow"
        assert reason == "authorized_match"
    
    def test_decide_class_match_enabled_mismatch(self, authorized_vehicle):
        """Test decision with class matching enabled detects mismatch."""
        engine = AccessRulesEngine(ocr_conf_threshold=0.7, require_class_match=True)
        decision, reason = engine.decide("ABC123", 0.95, "truck", authorized_vehicle)
        assert decision == "review"
        assert reason == "vehicle_class_mismatch"
    
    def test_decide_class_match_enabled_match(self, authorized_vehicle):
        """Test decision with class matching enabled allows correct match."""
        engine = AccessRulesEngine(ocr_conf_threshold=0.7, require_class_match=True)
        decision, reason = engine.decide("ABC123", 0.95, "car", authorized_vehicle)
        assert decision == "allow"
        assert reason == "authorized_match"
    
    def test_decide_invalid_ocr_confidence_too_high(self, engine):
        """Test that invalid OCR confidence is rejected."""
        with pytest.raises(ValueError):
            engine.decide("ABC123", 1.5, "car", {"status": "authorized"})
    
    def test_decide_invalid_ocr_confidence_negative(self, engine):
        """Test that negative OCR confidence is rejected."""
        with pytest.raises(ValueError):
            engine.decide("ABC123", -0.1, "car", {"status": "authorized"})
    
    def test_get_stats(self, engine):
        """Test that get_stats returns expected fields."""
        stats = engine.get_stats()
        assert "engine_type" in stats
        assert "ocr_confidence_threshold" in stats
        assert stats["ocr_confidence_threshold"] == 0.7
        assert stats["require_class_match"] is False
