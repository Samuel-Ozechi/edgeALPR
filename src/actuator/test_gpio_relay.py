# src/actuator/test_gpio_relay.py

"""Standalone relay test script. Run on Raspberry Pi to verify wiring before
integrating with the full pipeline.

Usage:
    python -m src.actuator.test_gpio_relay

You should hear the relay click or see the relay LED activate briefly.
"""

import time
from src.actuator.gpio_relay_actuator import GPIORelayActuator


def main():
    actuator = GPIORelayActuator(
        relay_pin=17,
        active_high=True,
        open_pulse_seconds=1.0,
    )

    try:
        print("Testing relay in 3 seconds...")
        time.sleep(3)

        print("Triggering relay...")
        actuator.open_barrier()

        print("Test complete.")

    finally:
        actuator.cleanup()


if __name__ == "__main__":
    main()
