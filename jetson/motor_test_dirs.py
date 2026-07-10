#!/usr/bin/env python3
"""Direction-only motor test for the L298N with ENA/ENB JUMPERS ON (full speed).

Safe even if the pin-32/33 wires are still plugged in — it never drives those pins, so there is no
conflict with the enable jumpers. Uses only the 4 direction pins.

Wiring: Jetson BOARD 29->IN1, 31->IN2 (left), 35->IN3, 37->IN4 (right), a GND pin -> L298N GND.
ENA/ENB jumpers ON, motors on OUT1/OUT2 (left) and OUT3/OUT4 (right), 9V on VS.

Run on the Jetson Nano (wheels OFF the ground):
  python3 ~/jetson/motor_test_dirs.py
"""
import time
import Jetson.GPIO as GPIO

IN1, IN2, IN3, IN4 = 29, 31, 35, 37


def main():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    for p in (IN1, IN2, IN3, IN4):
        GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)

    def drive(left, right):
        GPIO.output(IN1, GPIO.HIGH if left > 0 else GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH if left < 0 else GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH if right > 0 else GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH if right < 0 else GPIO.LOW)

    def stop():
        for p in (IN1, IN2, IN3, IN4):
            GPIO.output(p, GPIO.LOW)

    seq = [("LEFT fwd", 1, 0), ("RIGHT fwd", 0, 1), ("BOTH fwd", 1, 1), ("SPIN", 1, -1)]
    try:
        for label, l, r in seq:
            print(label); drive(l, r); time.sleep(2.0); stop(); time.sleep(0.6)
        print("done")
    finally:
        stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
