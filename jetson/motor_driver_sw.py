#!/usr/bin/env python3
"""SOFTWARE-PWM L298N driver (SINGLE-THREAD) — ENA/ENB JUMPERS ON, no hardware-PWM pins used.

The Jetson Nano can't run two hardware PWMs at once, so we PWM the 4 direction pins in software.
A SINGLE timing thread drives both channels in one loop (no thread contention -> no random dropouts).
Speed range is wide so the difference is visible. Use the LiPo on VS.

Wiring: 29->IN1,31->IN2 (left/A), 35->IN3,37->IN4 (right/B), GND common, BOTH jumpers ON, LiPo on VS.
"""
import threading
import time
import Jetson.GPIO as GPIO

IN1, IN2, IN3, IN4 = 29, 31, 35, 37
FREQ = 100          # software PWM frequency (Hz)
MAX_DUTY = 0.82     # full command -> ~8.6V on the LiPo (just under the 9V max)
MIN_MOVE = 0.42     # slowest moving command -> ~4.4V (overcome stiction)
DEADZONE = 0.05


class MotorDriver:
    def __init__(self, freq=FREQ):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        for p in (IN1, IN2, IN3, IN4):
            GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)
        self.period = 1.0 / freq
        self._cmd = [0.0, 0.0]           # [left(A), right(B)]
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def drive(self, left, right):
        with self._lock:
            self._cmd = [max(-1.0, min(1.0, float(left))), max(-1.0, min(1.0, float(right)))]

    def stop(self):
        self.drive(0.0, 0.0)

    @staticmethod
    def _duty_dir(c, fwd_pin, rev_pin):
        """return (duty, active_pin, other_pin) — active_pin=None means channel off."""
        mag = abs(c)
        if mag < DEADZONE:
            return 0.0, None, None
        duty = MIN_MOVE + mag * (MAX_DUTY - MIN_MOVE)
        return (duty, fwd_pin, rev_pin) if c > 0 else (duty, rev_pin, fwd_pin)

    def _loop(self):
        while self._running:
            with self._lock:
                cA, cB = self._cmd
            dA, actA, othA = self._duty_dir(cA, IN1, IN2)
            dB, actB, othB = self._duty_dir(cB, IN3, IN4)
            # off channels: both pins low
            if actA is None:
                GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.LOW)
            if actB is None:
                GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.LOW)
            # start ON phase for active channels (other pin low, active pin high)
            t0 = time.perf_counter()
            if actA is not None:
                GPIO.output(othA, GPIO.LOW); GPIO.output(actA, GPIO.HIGH)
            if actB is not None:
                GPIO.output(othB, GPIO.LOW); GPIO.output(actB, GPIO.HIGH)
            # schedule the two turn-off events in one sorted timeline (single thread, no contention)
            events = []
            if actA is not None:
                events.append((self.period * dA, actA))
            if actB is not None:
                events.append((self.period * dB, actB))
            for when, pin in sorted(events):
                dt = when - (time.perf_counter() - t0)
                if dt > 0:
                    time.sleep(dt)
                GPIO.output(pin, GPIO.LOW)
            rem = self.period - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

    def close(self):
        self._running = False
        self._thread.join(timeout=0.5)
        for p in (IN1, IN2, IN3, IN4):
            GPIO.output(p, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    m = MotorDriver()
    try:
        for label, spd in [("SLOW  0.30", 0.30), ("MEDIUM 0.60", 0.60), ("FAST  1.00", 1.00)]:
            print(label, flush=True); m.drive(spd, spd); time.sleep(3.0); m.stop(); time.sleep(0.8)
        print("done")
    finally:
        m.close()
