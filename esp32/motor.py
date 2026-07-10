# ESP32 (MicroPython) L298N motor driver — reliable hardware PWM.
# Wiring:  ENA=25,IN1=26,IN2=27 (left/A) ; ENB=33,IN3=32,IN4=14 (right/B) ; ESP32 GND -> L298N GND.
# L298N jumpers OFF (ESP32 drives ENA/ENB). Motors on OUT1/2 (left) & OUT3/4 (right). LiPo on VS.
from machine import Pin, PWM

ENA, IN1, IN2 = 25, 26, 27
ENB, IN3, IN4 = 33, 32, 14
FREQ = 1000
MAX_DUTY = 0.85   # full cmd -> ~8.4V on LiPo (under 9V max)
MIN_MOVE = 0.45   # slowest moving cmd -> ~4.7V
DEADZONE = 0.05

_ena = PWM(Pin(ENA), freq=FREQ, duty=0)
_enb = PWM(Pin(ENB), freq=FREQ, duty=0)
_in1 = Pin(IN1, Pin.OUT); _in2 = Pin(IN2, Pin.OUT)
_in3 = Pin(IN3, Pin.OUT); _in4 = Pin(IN4, Pin.OUT)

def _duty(x):
    if x < -1: x = -1
    if x > 1: x = 1
    m = x if x >= 0 else -x
    if m < DEADZONE:
        return 0
    return int((MIN_MOVE + m * (MAX_DUTY - MIN_MOVE)) * 1023)

def drive(left, right):
    _in1.value(1 if left > 0 else 0); _in2.value(1 if left < 0 else 0); _ena.duty(_duty(left))
    _in3.value(1 if right > 0 else 0); _in4.value(1 if right < 0 else 0); _enb.duty(_duty(right))

def stop():
    _in1.value(0); _in2.value(0); _in3.value(0); _in4.value(0)
    _ena.duty(0); _enb.duty(0)
