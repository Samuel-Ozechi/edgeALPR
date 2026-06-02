# src/actuator/gpio_relay_actuator.py

"""Raspberry Pi GPIO relay actuator for barrier control.

Sends a short GPIO pulse to a relay module that triggers the access barrier.

Wiring:
    Raspberry Pi GPIO17  -> Relay IN
    Raspberry Pi 5V      -> Relay VCC
    Raspberry Pi GND     -> Relay GND
    Relay COM            -> Barrier controller open input common
    Relay NO             -> Barrier controller open trigger input

Requires: pip install gpiozero
"""

import time
from datetime import datetime


class GPIORelayActuator:
    """
    Raspberry Pi GPIO relay actuator.

    Sends a short pulse on a configured GPIO pin to trigger a relay module,
    which in turn activates the physical access barrier.
    """

    def __init__(
        self,
        relay_pin: int = 17,
        active_high: bool = True,
        open_pulse_seconds: float = 1.0,
    ):
        from gpiozero import OutputDevice

        self.relay_pin = relay_pin
        self.open_pulse_seconds = open_pulse_seconds

        self.relay = OutputDevice(
            pin=relay_pin,
            active_high=active_high,
            initial_value=False,
        )

    def open_barrier(self):
        print(f"[{datetime.now().isoformat()}] GPIO RELAY: OPEN PULSE ON GPIO{self.relay_pin}")
        self.relay.on()
        time.sleep(self.open_pulse_seconds)
        self.relay.off()
        print(f"[{datetime.now().isoformat()}] GPIO RELAY: OPEN PULSE COMPLETE")

    def close_barrier(self):
        """Many barrier controllers auto-close after a configured time.
        If the controller has a separate CLOSE input, wire and implement it here."""
        print(f"[{datetime.now().isoformat()}] GPIO RELAY: CLOSE requested — no-op unless a close relay is configured")

    def deny_access(self):
        print(f"[{datetime.now().isoformat()}] GPIO RELAY: DENY ACCESS — no relay trigger")

    def manual_review(self):
        print(f"[{datetime.now().isoformat()}] GPIO RELAY: MANUAL REVIEW — no relay trigger")

    def cleanup(self):
        self.relay.off()
        self.relay.close()
