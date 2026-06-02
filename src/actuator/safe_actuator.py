# src/actuator/safe_actuator.py

"""Safety wrapper around barrier actuation.

Prevents repeated opening for the same plate, rapid repeated triggers,
and unsafe duplicate commands while the barrier is already open."""

import time
from datetime import datetime


class SafeActuatorController:
    """
    Safety wrapper around barrier actuation.

    Prevents:
    - repeated opening for the same plate
    - rapid repeated actuator triggers
    - unsafe duplicate commands while barrier is open
    """

    def __init__(
        self,
        base_actuator,
        open_hold_seconds: int = 3,
        same_plate_cooldown_seconds: int = 10,
        global_cooldown_seconds: int = 2,
    ):
        self.base_actuator = base_actuator
        self.open_hold_seconds = open_hold_seconds
        self.same_plate_cooldown_seconds = same_plate_cooldown_seconds
        self.global_cooldown_seconds = global_cooldown_seconds

        self.last_trigger_time = 0.0
        self.last_plate_seen: dict[str, float] = {}
        self.barrier_is_open = False
        self.barrier_opened_at = None

    def _now(self) -> float:
        return time.perf_counter()

    def _plate_in_cooldown(self, plate_text: str) -> bool:
        if not plate_text:
            return False
        last_seen = self.last_plate_seen.get(plate_text)
        if last_seen is None:
            return False
        return (self._now() - last_seen) < self.same_plate_cooldown_seconds

    def _global_in_cooldown(self) -> bool:
        return (self._now() - self.last_trigger_time) < self.global_cooldown_seconds

    def _mark_plate_seen(self, plate_text: str):
        if plate_text:
            self.last_plate_seen[plate_text] = self._now()

    def _auto_close_if_needed(self):
        if not self.barrier_is_open or self.barrier_opened_at is None:
            return
        if (self._now() - self.barrier_opened_at) >= self.open_hold_seconds:
            self.close_barrier()

    def open_barrier(self, plate_text: str | None = None) -> tuple[bool, str]:
        self._auto_close_if_needed()

        if self.barrier_is_open:
            return False, "barrier_already_open"
        if self._global_in_cooldown():
            return False, "global_actuator_cooldown"
        if plate_text and self._plate_in_cooldown(plate_text):
            return False, "same_plate_cooldown"

        self.base_actuator.open_barrier()

        self.barrier_is_open = True
        self.barrier_opened_at = self._now()
        self.last_trigger_time = self._now()
        self._mark_plate_seen(plate_text)

        return True, "barrier_opened"

    def close_barrier(self) -> tuple[bool, str]:
        if not self.barrier_is_open:
            return False, "barrier_already_closed"

        if hasattr(self.base_actuator, "close_barrier"):
            self.base_actuator.close_barrier()
        else:
            print(f"[{datetime.now().isoformat()}] ACTUATOR: CLOSE BARRIER")

        self.barrier_is_open = False
        self.barrier_opened_at = None

        return True, "barrier_closed"

    def deny_access(self) -> tuple[bool, str]:
        self._auto_close_if_needed()
        self.base_actuator.deny_access()
        return True, "access_denied"

    def manual_review(self) -> tuple[bool, str]:
        self._auto_close_if_needed()
        self.base_actuator.manual_review()
        return True, "manual_review_required"

    def execute(self, decision: str, plate_text: str | None = None) -> tuple[bool, str]:
        self._auto_close_if_needed()

        if decision == "allow":
            return self.open_barrier(plate_text=plate_text)
        if decision == "deny":
            return self.deny_access()
        return self.manual_review()
