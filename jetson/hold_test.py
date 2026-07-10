#!/usr/bin/env python3
"""Hold a motor command for N seconds — for wire-wiggling / re-seating debug.

Usage (on the Jetson Nano):
  python3 ~/jetson/hold_test.py            # both forward, 15s
  python3 ~/jetson/hold_test.py 1 0 15     # LEFT only, 15s
  python3 ~/jetson/hold_test.py 0 1 15     # RIGHT only, 15s
  python3 ~/jetson/hold_test.py 1 1 20     # BOTH forward, 20s
"""
import sys
import time
import motor_driver as md

left = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
right = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

m = md.MotorDriver()
print(f"driving left={left} right={right} for {secs:.0f}s — wiggle/re-seat the VS + GROUND wires now")
try:
    m.drive(left, right)
    time.sleep(secs)
finally:
    m.stop()
    m.close()
    print("done")
