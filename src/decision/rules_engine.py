# src/decision/rules_engine.py

"""Rules engine module for a vision-based access control system.
This module defines an AccessRulesEngine class that implements the decision logic for allowing, denying, or flagging access attempts.
The decision responses are based on OCR confidence, vehicle authorization status, and optional class matching.
The decide method evaluates the input parameters and returns a decision along with a reason for that decision."""

# Import required libraries
import logging
from typing import Literal
from src.configs.settings import settings
from src.decision.base_rules_engine import BaseRulesEngine

logger = logging.getLogger(__name__)

Decision = Literal["allow", "deny", "review"]
Reason = str

# Define the AccessRulesEngine class
class AccessRulesEngine(BaseRulesEngine):
    """
    AccessRulesEngine implements the decision logic for a vision-based access control system.
    It evaluates OCR confidence, vehicle authorization status, and optional class matching to determine access decisions.

    Args:
        ocr_conf_threshold (float): The minimum OCR confidence required for a valid detection (0-1).
        require_class_match (bool): If True, requires detected vehicle class to match registered class for authorization.
        
    Raises:
        ValueError: If ocr_conf_threshold is not between 0 and 1
    """
    def __init__(self, ocr_conf_threshold: float, require_class_match: bool = False) -> None:
        if not (0 <= ocr_conf_threshold <= 1):
            raise ValueError(f"ocr_conf_threshold must be between 0 and 1, got {ocr_conf_threshold}")
        
        self.ocr_conf_threshold = ocr_conf_threshold
        self.require_class_match = require_class_match
        logger.info(f"AccessRulesEngine initialized (threshold={ocr_conf_threshold}, class_match={require_class_match})")

    def decide(
        self,
        plate_text: str | None,
        ocr_confidence: float,
        detected_vehicle_class: str | None,
        vehicle_record: dict | None
    ) -> tuple[Decision, Reason]:
        """Determine access decision based on OCR result and vehicle record.
        
        Args:
            plate_text: Recognized license plate text (may be None or empty)
            ocr_confidence: OCR confidence score (0-1)
            detected_vehicle_class: Detected vehicle class name (e.g., 'car', 'truck')
            vehicle_record: Vehicle record from database (dict) or None if not found
            
        Returns:
            Tuple of (decision, reason) where decision is 'allow'|'deny'|'review'
            
        Raises:
            ValueError: If ocr_confidence is not between 0 and 1
        """
        if not (0 <= ocr_confidence <= 1):
            raise ValueError(f"ocr_confidence must be between 0 and 1, got {ocr_confidence}")
        
        # Decision tree with early returns
        if not plate_text or not plate_text.strip():
            reason = "no_plate_text"
            logger.debug(f"Decision: review - {reason}")
            return "review", reason

        if ocr_confidence < self.ocr_conf_threshold:
            reason = "low_ocr_confidence"
            logger.debug(f"Decision: review - {reason} ({ocr_confidence} < {self.ocr_conf_threshold})")
            return "review", reason

        if vehicle_record is None:
            reason = "plate_not_found"
            logger.info(f"Decision: deny - {reason} (plate: {plate_text})")
            return "deny", reason

        if vehicle_record.get("status") != "authorized":
            reason = "vehicle_not_authorized"
            logger.info(f"Decision: deny - {reason} (plate: {plate_text})")
            return "deny", reason

        if self.require_class_match:
            registered_class = (vehicle_record.get("vehicle_class") or "").lower()
            detected_class = (detected_vehicle_class or "").lower()

            if registered_class and detected_class and registered_class != detected_class:
                reason = "vehicle_class_mismatch"
                logger.warning(f"Decision: review - {reason} (registered: {registered_class}, detected: {detected_class})")
                return "review", reason

        reason = "authorized_match"
        logger.info(f"Decision: allow - {reason} (plate: {plate_text})")
        return "allow", reason

    def get_stats(self) -> dict:
        """Get decision engine statistics.
        
        Returns:
            Dictionary with decision statistics
        """
        return {
            "engine_type": "AccessRulesEngine",
            "ocr_confidence_threshold": self.ocr_conf_threshold,
            "require_class_match": self.require_class_match,
            "configuration": "production"
        }