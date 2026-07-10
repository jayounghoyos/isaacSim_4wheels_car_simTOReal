#!/usr/bin/env python3
"""Sim-to-real MIRROR server (runs on the Jetson Nano).

Listens on UDP for [left, right] motor commands (2 x float32, big-endian) streamed from the PC's
simulation, and drives the L298N via motor_driver. The real wheels mirror the sim robot live.

SAFETY WATCHDOG: if no command arrives for WATCHDOG_S seconds (PC closed, WiFi dropped), the motors
are stopped automatically — the robot never runs away on a lost link.

Run on the Jetson Nano (wheels off the ground for the first test):
  PATH=$HOME/.local/bin:$PATH python3 ~/jetson/mirror_server.py
"""
import os
import sys
import socket
import struct
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# SOFTWARE-PWM driver: BOTH jumpers ON, pins 32/33 unused. Avoids the Jetson Nano's dual-hardware-PWM
# limitation (can't run pin 32 AND 33 at once). Software-PWMs the IN pins -> both channels work.
from motor_driver_sw import MotorDriver

PORT = 5005
WATCHDOG_S = 0.4          # stop motors if no command within this window
PKT = struct.Struct("!ff")   # left, right


def main():
    m = MotorDriver()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(WATCHDOG_S)
    print(f"[mirror] listening on UDP :{PORT}  (watchdog {WATCHDOG_S}s -> auto-stop)")
    n = 0
    last_report = time.time()
    stopped = True
    try:
        while True:
            try:
                data, addr = sock.recvfrom(32)
                if len(data) >= PKT.size:
                    left, right = PKT.unpack(data[:PKT.size])
                    m.drive(left, right)
                    stopped = False
                    n += 1
                    now = time.time()
                    if now - last_report >= 1.0:
                        print(f"[mirror] {n} cmd/s from {addr[0]}  last=({left:+.2f},{right:+.2f})")
                        n = 0; last_report = now
            except socket.timeout:
                if not stopped:
                    m.stop(); stopped = True
                    print("[mirror] no commands -> STOPPED (watchdog)")
    except KeyboardInterrupt:
        pass
    finally:
        m.close()
        print("[mirror] closed, GPIO cleaned up.")


if __name__ == "__main__":
    main()
