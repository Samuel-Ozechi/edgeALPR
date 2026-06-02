"""Example: GPIO relay control for barrier/access gate.

This example shows how to integrate edgeALPR with physical hardware
like servo motors, stepper motors, or relay-controlled barriers.

Usage:
    python examples/with_gpio_relay.py
    
Note:
    Requires gpiozero library: pip install gpiozero
    Requires appropriate GPIO wiring and power supply
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from gpiozero import Servo, MotionSensor, DistanceSensor
    from gpiozero.pins.pigpio import PiGPIOFactory
    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False
    print("Warning: gpiozero not installed")
    print("Install with: pip install gpiozero")


class GPIORelayActuator:
    """GPIO relay controller for physical barrier/gate access."""
    
    def __init__(self, servo_pin: int = 17, motion_pin: int = 27):
        """Initialize GPIO relay actuator.
        
        Args:
            servo_pin: GPIO pin for servo motor (barrier control)
            motion_pin: GPIO pin for motion sensor (vehicle detection)
        """
        if not GPIOZERO_AVAILABLE:
            raise ImportError("gpiozero library required for GPIO control")
        
        self.servo_pin = servo_pin
        self.motion_pin = motion_pin
        
        # Initialize GPIO devices
        self.servo = Servo(servo_pin, pin_factory=PiGPIOFactory())
        self.motion = MotionSensor(motion_pin, pin_factory=PiGPIOFactory())
        
        print(f"GPIO Relay Actuator initialized")
        print(f"  Servo pin: {servo_pin}")
        print(f"  Motion sensor pin: {motion_pin}")
    
    def open_barrier(self, duration: float = 5.0) -> None:
        """Open barrier/gate for vehicle passage.
        
        Args:
            duration: Time to keep barrier open (seconds)
        """
        print(f"  [BARRIER] Opening for {duration}s...")
        
        # Servo position: -1 (min) to 1 (max)
        # Adjust based on physical setup
        self.servo.value = 1  # Open position
        
        time.sleep(duration)
        
        self.servo.value = -1  # Closed position
        print(f"  [BARRIER] Closed")
    
    def deny_access(self) -> None:
        """Lock barrier (deny access)."""
        print(f"  [BARRIER] LOCKED (access denied)")
        self.servo.value = -1
        
        # Optional: Flash warning light
        # self.warning_light.on()
        # time.sleep(2)
        # self.warning_light.off()
    
    def manual_review(self) -> None:
        """Open barrier for manual review by operator.
        
        Vehicle stops, operator checks if entry should be allowed.
        """
        print(f"  [BARRIER] Manual review required")
        print(f"  [LIGHT] Flashing yellow warning light...")
        
        # Flash warning light
        # while True:
        #     self.warning_light.on()
        #     time.sleep(0.5)
        #     self.warning_light.off()
        #     time.sleep(0.5)
        #     # Wait for operator input...
    
    def wait_for_vehicle(self, timeout: float = 30.0) -> bool:
        """Wait for vehicle to approach (motion detected).
        
        Args:
            timeout: Maximum time to wait for motion (seconds)
            
        Returns:
            True if motion detected, False if timeout
        """
        print(f"  [MOTION] Waiting for vehicle... ({timeout}s timeout)")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.motion.is_active:
                print(f"  [MOTION] Vehicle detected!")
                return True
            time.sleep(0.1)
        
        print(f"  [MOTION] Timeout - no vehicle detected")
        return False
    
    def cleanup(self) -> None:
        """Clean up GPIO resources."""
        print("Cleaning up GPIO...")
        self.servo.close()
        self.motion.close()


def main() -> None:
    """Main GPIO relay control example."""
    print("edgeALPR GPIO Relay Control Example")
    print("=" * 60)
    print()
    
    if not GPIOZERO_AVAILABLE:
        print("ERROR: gpiozero library not installed")
        print("Install with: pip install gpiozero")
        sys.exit(1)
    
    # Initialize actuator
    try:
        actuator = GPIORelayActuator(servo_pin=17, motion_pin=27)
    except Exception as e:
        print(f"ERROR: Failed to initialize GPIO: {e}")
        print("Make sure:")
        print("  - Running on Raspberry Pi with GPIO pins available")
        print("  - GPIO pins are properly wired")
        print("  - pigpiod daemon is running (sudo pigpiod)")
        sys.exit(1)
    
    try:
        # Example workflow
        print("Workflow Example:")
        print("-" * 40)
        
        # Wait for vehicle
        vehicle_detected = actuator.wait_for_vehicle(timeout=10.0)
        
        if vehicle_detected:
            # In real scenario, run access control workflow here
            # For now, simulate access decisions
            
            print("\n1. Authorized vehicle - ALLOW")
            actuator.open_barrier(duration=5.0)
            
            print("\n2. Unauthorized vehicle - DENY")
            actuator.deny_access()
            
            print("\n3. Unknown vehicle - MANUAL REVIEW")
            # actuator.manual_review()
        
        print("\nWorkflow completed")
    
    finally:
        actuator.cleanup()


if __name__ == "__main__":
    main()
