# ESP32 (MicroPython) sim-to-real MIRROR receiver.
# WiFi -> UDP :5005 -> motor.drive(left,right).  Watchdog stops motors if the link drops.
# WiFi creds come from wifi_config.py (SSID, PASSWORD) — kept off the git repo.
import network, socket, struct, time
import motor

PORT = 5005
WATCHDOG_S = 0.4
try:
    import wifi_config
    SSID, PASSWORD = wifi_config.SSID, wifi_config.PASSWORD
except Exception:
    SSID, PASSWORD = "", ""

def wifi_connect():
    w = network.WLAN(network.STA_IF); w.active(True)
    if not w.isconnected():
        w.connect(SSID, PASSWORD)
        for _ in range(40):
            if w.isconnected(): break
            time.sleep(0.5)
    return w

w = wifi_connect()
if w.isconnected():
    print("WIFI_OK ip=%s" % w.ifconfig()[0])
else:
    print("WIFI_FAIL ssid=%r" % SSID); motor.stop()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", PORT)); s.settimeout(WATCHDOG_S)
print("LISTENING udp %d" % PORT)
while True:
    try:
        data, addr = s.recvfrom(32)
        if len(data) >= 8:
            l, r = struct.unpack("!ff", data[:8])   # MicroPython struct has no Struct class
            motor.drive(l, r)
    except OSError:            # timeout -> watchdog stop
        motor.stop()
