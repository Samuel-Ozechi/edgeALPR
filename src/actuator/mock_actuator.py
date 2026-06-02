from datetime import datetime


class MockActuator:
    def open_barrier(self):
        print(f"[{datetime.now().isoformat()}] ACTUATOR: OPEN BARRIER")

    def deny_access(self):
        print(f"[{datetime.now().isoformat()}] ACTUATOR: DENY ACCESS")

    def manual_review(self):
        print(f"[{datetime.now().isoformat()}] ACTUATOR: MANUAL REVIEW REQUIRED")

    def close_barrier(self):
        print(f"[{datetime.now().isoformat()}] ACTUATOR: CLOSE BARRIER")

    def execute(self, decision: str):
        if decision == "allow":
            self.open_barrier()
        elif decision == "deny":
            self.deny_access()
        else:
            self.manual_review()