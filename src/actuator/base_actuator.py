"""Abstract base classes for actuators and hardware control in edgeALPR."""

from abc import ABC, abstractmethod
from typing import Literal
import logging

logger = logging.getLogger(__name__)

Action = Literal["allow", "deny", "review"]


class BaseActuator(ABC):
    """Abstract base class for hardware actuators (relays, barriers, lights, etc.).
    
    Subclasses can implement specific hardware control while maintaining a consistent interface.
    
    Example:
        class ServoActuator(BaseActuator):
            def __init__(self, pin):
                self.servo = Servo(pin)
                
            def execute(self, action):
                if action == 'allow':
                    self.servo.rotate(90)
    """

    @abstractmethod
    def execute(self, action: Action) -> None:
        """Execute an action (open/close barrier, toggle light, etc.).
        
        Args:
            action: 'allow' (open), 'deny' (keep closed), or 'review' (warning signal)
            
        Raises:
            RuntimeError: If hardware operation fails
        """
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Get current state of the actuator.
        
        Returns:
            Dictionary with keys: state, last_action, last_action_time, error_count, etc.
        """
        pass

    def test(self) -> bool:
        """Test that the actuator is functional.
        
        Returns:
            True if actuator responds correctly, False otherwise
        """
        try:
            state = self.get_state()
            logger.info(f"{self.__class__.__name__} test passed: {state}")
            return True
        except Exception as e:
            logger.error(f"{self.__class__.__name__} test failed: {e}")
            return False
