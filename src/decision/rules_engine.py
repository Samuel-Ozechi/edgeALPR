# src/decision/rules_engine.py

"""Rules engine module for a vision-based access control system.
This module defines an AccessRulesEngine class that implements the decision logic for allowing, denying, or flagging access attempts.
The decision responses are based on OCR confidence, vehicle authorization status, and optional class matching.
The decide method evaluates the input parameters and returns a decision along with a reason for that decision."""

# Import required libraries
from src.configs.settings import settings

# Define the AccessRulesEngine class
class AccessRulesEngine:
    """
    AccessRulesEngine implements the decision logic for a vision-based access control system.
    It evaluates OCR confidence, vehicle authorization status, and optional class matching to determine access decisions.

    Args:
        ocr_conf_threshold (float): The minimum OCR confidence required for a valid detection.
        require_class_match (bool): If True, requires detected vehicle class to match registered class for authorization.
    """
    def __init__(self, ocr_conf_threshold: float, require_class_match: bool = False):
        self.ocr_conf_threshold = ocr_conf_threshold
        self.require_class_match = require_class_match

    def decide(
        self,
        plate_text: str,
        ocr_confidence: float,
        detected_vehicle_class: str | None,
        vehicle_record: dict | None
    ) -> tuple[str, str]:
        if not plate_text:
            return "review", "no_plate_text"

        if ocr_confidence < self.ocr_conf_threshold:
            return "review", "low_ocr_confidence"

        if vehicle_record is None:
            return "deny", "plate_not_found"

        if vehicle_record["status"] != "authorized":
            return "deny", "vehicle_not_authorized"

        if self.require_class_match:
            registered_class = (vehicle_record.get("vehicle_class") or "").lower()
            detected_class = (detected_vehicle_class or "").lower()

            if registered_class and detected_class and registered_class != detected_class:
                return "review", "vehicle_class_mismatch"

        return "allow", "authorized_match"