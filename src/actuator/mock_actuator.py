"""Mock actuator for testing without real hardware."""

import logging
from datetime import datetime, timezone
from typing import Literal
from src.actuator.base_actuator import BaseActuator

logger = logging.getLogger(__name__)

Action = Literal["allow", "deny", "review"]


class MockActuator(BaseActuator):
    """Mock actuator that logs actions without controlling hardware.
    
    Useful for testing and development without physical relay or barrier hardware.
    """
    
    def __init__(self) -> None:
        """Initialize the mock actuator."""
        super().__init__()
        self.last_action: Action | None = None
        self.last_action_time: str | None = None
        self.action_count = 0
        logger.info("MockActuator initialized")

    def open_barrier(self) -> None:
        """Log barrier open action."""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"MOCK: OPEN BARRIER at {timestamp}")
        self.last_action = "allow"
        self.last_action_time = timestamp
        self.action_count += 1

    def deny_access(self) -> None:
        """Log deny access action."""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"MOCK: DENY ACCESS at {timestamp}")
        self.last_action = "deny"
        self.last_action_time = timestamp
        self.action_count += 1

    def manual_review(self) -> None:
        """Log manual review required action."""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.warning(f"MOCK: MANUAL REVIEW REQUIRED at {timestamp}")
        self.last_action = "review"
        self.last_action_time = timestamp
        self.action_count += 1

    def close_barrier(self) -> None:
        """Log barrier close action."""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"MOCK: CLOSE BARRIER at {timestamp}")

    def execute(self, action: Action) -> None:
        """Execute the specified action.
        
        Args:
            action: 'allow' (open), 'deny' (close), or 'review' (warning)
        """
        if action == "allow":
            self.open_barrier()
        elif action == "deny":
            self.deny_access()
        else:
            self.manual_review()
    
    def get_state(self) -> dict:
        """Get current state of the mock actuator.
        
        Returns:
            Dictionary with actuator state
        """
        return {
            "type": "MockActuator",
            "last_action": self.last_action,
            "last_action_time": self.last_action_time,
            "total_actions": self.action_count,
            "operational": True,
        }
