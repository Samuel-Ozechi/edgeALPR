"""Unit tests for MockActuator."""

import pytest
from src.actuator.mock_actuator import MockActuator


class TestMockActuator:
    """Test suite for MockActuator."""
    
    @pytest.fixture
    def actuator(self):
        """Create a mock actuator for testing."""
        return MockActuator()
    
    def test_init(self, actuator):
        """Test actuator initialization."""
        assert actuator.last_action is None
        assert actuator.last_action_time is None
        assert actuator.action_count == 0
    
    def test_execute_allow(self, actuator):
        """Test executing allow action."""
        actuator.execute("allow")
        assert actuator.last_action == "allow"
        assert actuator.last_action_time is not None
        assert actuator.action_count == 1
    
    def test_execute_deny(self, actuator):
        """Test executing deny action."""
        actuator.execute("deny")
        assert actuator.last_action == "deny"
        assert actuator.last_action_time is not None
        assert actuator.action_count == 1
    
    def test_execute_review(self, actuator):
        """Test executing review action."""
        actuator.execute("review")
        assert actuator.last_action == "review"
        assert actuator.last_action_time is not None
        assert actuator.action_count == 1
    
    def test_execute_multiple_actions(self, actuator):
        """Test executing multiple actions increments counter."""
        actuator.execute("allow")
        actuator.execute("deny")
        actuator.execute("review")
        assert actuator.action_count == 3
    
    def test_get_state(self, actuator):
        """Test getting actuator state."""
        actuator.execute("allow")
        state = actuator.get_state()
        
        assert state["type"] == "MockActuator"
        assert state["last_action"] == "allow"
        assert state["last_action_time"] is not None
        assert state["total_actions"] == 1
        assert state["operational"] is True
    
    def test_get_state_before_action(self, actuator):
        """Test getting state before any action."""
        state = actuator.get_state()
        
        assert state["last_action"] is None
        assert state["last_action_time"] is None
        assert state["total_actions"] == 0
    
    def test_open_barrier(self, actuator):
        """Test open_barrier directly."""
        actuator.open_barrier()
        assert actuator.last_action == "allow"
    
    def test_deny_access(self, actuator):
        """Test deny_access directly."""
        actuator.deny_access()
        assert actuator.last_action == "deny"
    
    def test_manual_review(self, actuator):
        """Test manual_review directly."""
        actuator.manual_review()
        assert actuator.last_action == "review"
