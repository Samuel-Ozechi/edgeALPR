"""Abstract base classes for decision engines in edgeALPR."""

from abc import ABC, abstractmethod
from typing import Literal
import logging

logger = logging.getLogger(__name__)

Decision = Literal["allow", "deny", "review"]


class BaseRulesEngine(ABC):
    """Abstract base class for access control decision engines.
    
    Subclasses can implement custom decision logic (ML models, custom rules, etc.)
    while maintaining a consistent interface with the pipeline.
    
    Example:
        class MLDecisionEngine(BaseRulesEngine):
            def __init__(self, model_path):
                self.model = load_model(model_path)
                
            def decide(self, plate_text, ...):
                features = extract_features(plate_text, ...)
                return self.model.predict(features)
    """

    @abstractmethod
    def decide(
        self,
        plate_text: str | None,
        ocr_confidence: float,
        detected_vehicle_class: str | None,
        vehicle_record: dict | None
    ) -> tuple[Decision, str]:
        """Determine access decision based on available information.
        
        Args:
            plate_text: Recognized license plate text
            ocr_confidence: OCR confidence score (0-1)
            detected_vehicle_class: Detected vehicle class (e.g., 'car', 'truck')
            vehicle_record: Vehicle record from database (if found)
            
        Returns:
            Tuple of (decision, reason) where decision is 'allow'|'deny'|'review'
            reason is a string explaining the decision
            
        Raises:
            ValueError: If inputs are invalid
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get decision engine statistics.
        
        Returns:
            Dictionary with keys: decisions_made, allow_count, deny_count, review_count, etc.
        """
        pass
