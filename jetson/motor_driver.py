#!/usr/bin/env python3
"""L298N motor driver for the Jetson Nano — maps policy actions [left, right] to the 2 H-bridge channels.

Wiring (Jetson Nano BOARD pin -> L298N):
  29 -> IN1 , 31 -> IN2   (left pair  = channel A = OUT1/OUT2)
  35 -> IN3 , 37 -> IN4   (right pair = channel B = OUT3/OUT4)
  32 -> ENA , 33 -> ENB   (hardware PWM speed; remove the ENA/ENB jumpers)
  39 -> GND               (COMMON GROUND to the L298N — required)

action convention: left,right in [-1, 1].  +1 = full forward, -1 = full reverse, 0 = stop.
(If a side spins the wrong way, swap that motor pair's two output wires on the L298N — the hardware
equivalent of the sim's mirrored-axle sign.)

Run directly for a SAFE test sequence (put the wheels OFF the ground first):
  PATH=$HOME/.local/bin:$PATH OPENBLAS_CORETYPE=ARMV8 python3 jetson/motor_driver.py
  add '--signals' to only toggle the pins with NO motors moving (check with an LED/multimeter first).
"""
import sys
import time
import Jetson.GPIO as GPIO

IN1, IN2, IN3, IN4 = 29, 31, 35, 37
ENA, ENB = 33, 32     # ENA on pin 33, ENB on pin 32 (hardware PWM pins)
PWM_HZ = 1000
MIN_DUTY = 0.0          # some gearmotors need a minimum duty to overcome stiction; raise if they buzz but don't turn
# 6V "yellow TT" motors on a ~12.5V LiPo (L298N drops ~2V -> ~10.5V) would be over-volted at full duty.
# Cap the duty so the AVERAGE voltage stays ~6V: 55% * 10.5V ≈ 5.8V. Raise/lower to taste.
MAX_DUTY = 55.0


class MotorDriver:
    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        for p in (IN1, IN2, IN3, IN4):
            GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(ENA, GPIO.OUT)
        GPIO.setup(ENB, GPIO.OUT)
        self.pwm_a = GPIO.PWM(ENA, PWM_HZ)
        self.pwm_b = GPIO.PWM(ENB, PWM_HZ)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

    @staticmethod
    def _duty(x):
        x = max(-1.0, min(1.0, float(x)))
        d = abs(x) * MAX_DUTY                     # scale [0,1] -> [0, MAX_DUTY] to protect 6V motors
        return 0.0 if d == 0 else max(MIN_DUTY, d)

    def drive(self, left, right):
        # left channel (A)
        GPIO.output(IN1, GPIO.HIGH if left > 0 else GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH if left < 0 else GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(self._duty(left))
        # right channel (B)
        GPIO.output(IN3, GPIO.HIGH if right > 0 else GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH if right < 0 else GPIO.LOW)
        self.pwm_b.ChangeDutyCycle(self._duty(right))

    def stop(self):
        for p in (IN1, IN2, IN3, IN4):
            GPIO.output(p, GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)

    def close(self):
        self.stop()
        self.pwm_a.stop()
        self.pwm_b.stop()
        GPIO.cleanup()


def _signals_only():
    """Toggle the pins with NO motion expected — verify wiring with an LED/multimeter, motors optional."""
    GPIO.setwarnings(False); GPIO.setmode(GPIO.BOARD)
    for p in (IN1, IN2, IN3, IN4, ENA, ENB):
        GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)
    print("Toggling each control pin HIGH for 1s (measure with a multimeter/LED):")
    for name, p in [("IN1", IN1), ("IN2", IN2), ("IN3", IN3), ("IN4", IN4), ("ENA", ENA), ("ENB", ENB)]:
        GPIO.output(p, GPIO.HIGH); print(f"  {name} (pin {p}) = HIGH"); time.sleep(1); GPIO.output(p, GPIO.LOW)
    GPIO.cleanup(); print("signal test done.")


def _motor_test():
    print(">>> WHEELS OFF THE GROUND. Starting in 3s...  (Ctrl-C to abort)")
    time.sleep(3)
    m = MotorDriver()
    try:
        seq = [("LEFT forward",  0.6, 0.0), ("LEFT reverse", -0.6, 0.0),
               ("RIGHT forward", 0.0, 0.6), ("RIGHT reverse", 0.0, -0.6),
               ("BOTH forward",  0.6, 0.6), ("SPIN in place", 0.6, -0.6)]
        for label, l, r in seq:
            print(f"  {label}  (left={l}, right={r})")
            m.drive(l, r); time.sleep(1.5); m.stop(); time.sleep(0.5)
        print("motor test done.")
    finally:
        m.close()


if __name__ == "__main__":
    if "--signals" in sys.argv:
        _signals_only()
    else:
        try:
            _motor_test()
        except KeyboardInterrupt:
            GPIO.cleanup(); print("\naborted, GPIO cleaned up.")
