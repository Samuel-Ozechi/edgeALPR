# src/actuator/factory.py

"""Actuator factory — builds the correct base actuator based on settings.actuator.mode
and wraps it in SafeActuatorController."""

from src.configs.settings import settings
from src.actuator.mock_actuator import MockActuator
from src.actuator.safe_actuator import SafeActuatorController


def build_actuator() -> SafeActuatorController:
    if settings.actuator.mode == "gpio":
        from src.actuator.gpio_relay_actuator import GPIORelayActuator
        base_actuator = GPIORelayActuator(
            relay_pin=settings.actuator.relay_pin,
            active_high=settings.actuator.relay_active_high,
            open_pulse_seconds=settings.actuator.open_pulse_seconds,
        )
    else:
        base_actuator = MockActuator()

    return SafeActuatorController(
        base_actuator=base_actuator,
        open_hold_seconds=settings.actuator.open_hold_seconds,
        same_plate_cooldown_seconds=settings.actuator.same_plate_cooldown_seconds,
        global_cooldown_seconds=settings.actuator.global_cooldown_seconds,
    )
