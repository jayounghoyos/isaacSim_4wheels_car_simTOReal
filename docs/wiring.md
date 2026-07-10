# Robot Power & Wiring — Jetson Nano + L298N + 3S LiPo

**Components**
- **3S LiPo** Turnigy 11.1 V nominal (12.6 V full / ~9.9 V empty), 3300 mAh, 30C — main power.
- **XL4015E** buck converter (4–36 V in → 1.2–35 V out, 5 A / 160 W, with display) — makes 5 V for Jetson.
- **L298N** dual H-bridge (2 channels: left pair + right pair of motors).
- **Jetson Nano 4GB** — needs a clean **5 V** supply.
- **LiPo low-voltage alarm** ("salvalipo") — plugs into the balance lead, beeps on low cell.
- Recommended adds: **inline fuse (5–7.5 A)** on LiPo +, a **main switch**, decent-gauge wire.

```
                              ┌─────────── inline FUSE (5–7.5A) ──── main SWITCH ───┐
   3S LiPo (12.6V) ── + ──────┤                                                      │
                     │        │                                    ┌────────────────┴─► XL4015E  IN+
                     │        └────────────────────────────────────┤                     XL4015E  OUT+ ──► Jetson barrel-jack CENTER (+5.0–5.1V)
                     │                                              │
                     │                                              └─► L298N  VS (motor +12V screw terminal)
                     │
   3S LiPo ── − ─────┴──────────── COMMON GROUND rail ───────┬──► XL4015E IN− and OUT−
                                                             ├──► L298N  GND (screw terminal)
                                                             └──► Jetson GND  (barrel-jack SLEEVE + a GND GPIO pin)

   LiPo BALANCE connector (4-pin JST-XH) ──► LiPo alarm  (set to beep at ~3.4 V/cell)

   Jetson GPIO (3.3V logic) ──► L298N control:
       IN1, IN2  = left-pair direction        ENA = left-pair PWM speed  (remove ENA jumper)
       IN3, IN4  = right-pair direction        ENB = right-pair PWM speed (remove ENB jumper)
```

## ⚠️ Critical safety order (do NOT skip)
1. **Set the buck output to 5.0–5.1 V BEFORE connecting the Jetson.** Power the XL4015E from the LiPo,
   watch its display, turn the pot until it reads **5.0–5.1 V** with NO load. A fresh buck can output
   20 V+ — connecting that to the Jetson kills it instantly. Only after it reads ~5 V do you connect
   the Jetson.
2. **Never let any LiPo cell drop below ~3.3 V** (≈9.9 V total). Set the alarm to **3.4 V/cell**. Below
   ~3.0 V/cell = permanent damage / fire risk. Store the pack at ~3.8 V/cell (≈11.4 V), not full.
3. **CHECK YOUR MOTOR VOLTAGE RATING before feeding them 12.6 V** (see Motor Power).
4. **One common ground** for everything (LiPo −, buck in/out −, L298N GND, Jetson GND). Skipping this is
   the #1 beginner failure — the PWM/direction signals won't be referenced and motors act erratic.
5. Add an **inline fuse** on LiPo + — a LiPo can dump ~100 A into a short.

## Jetson power (barrel jack, not micro-USB)
- Micro-USB is only 5 V/2 A → throttles/shuts down under CNN load. Use the **barrel jack (5.5×2.1 mm,
  center +), 5 V/4 A**.
- **Set the Nano jumper `J48`** to enable barrel-jack power (without it the barrel jack does nothing).
- Buck OUT+ → barrel center; Buck OUT− → barrel sleeve (and to the common ground rail).
- Jetson Nano peaks ~4 A @ 5 V (20 W). The XL4015E (5 A/160 W) handles it easily; give it a heatsink,
  it may run warm at 20 W. From the 12.6 V LiPo that's only ~1.7 A input.

## Motor power (the voltage caveat)
- The LiPo + (after fuse/switch) → **L298N VS** screw terminal. Motors then see ~VS − 2 V ≈ **10.5 V**
  (the L298N drops ~1.5–2.5 V).
- **⚠️ If your motors are 3–6 V "TT/yellow gearmotors," ~10.5 V will overheat/burn them.** You were
  previously running them off a 9 V battery (~7 V at the motor), so 10.5 V is meaningfully more.
  - **If motors are ~12 V rated:** feed VS from the LiPo directly — fine.
  - **If motors are 6 V rated:** either (a) limit PWM duty to ~55–60% in software (our policy rarely
    commands full throttle, so this is workable), or (b) get a *second* small buck / UBEC set to ~6 V
    to feed VS. Don't run 6 V motors at 10.5 V full-throttle continuously.
- Keep the L298N's **on-board 5 V jumper ON** (its 78M05 handles 12.6 V input fine; it powers the
  driver's own logic). Do **not** wire that 5 V pin to the Jetson — the Jetson gets 5 V from the buck.

## Signal wiring (Jetson GPIO → L298N)
- Jetson GPIO is **3.3 V logic**; the L298N input HIGH threshold is ~2.3 V, so **3.3 V works directly —
  no level shifter needed**.
- 6 signals: `IN1,IN2` (left dir), `IN3,IN4` (right dir), `ENA,ENB` (PWM). **Remove the ENA/ENB jumpers**
  to control speed with PWM (leaving them on = always full speed).
- `ENA/ENB` want PWM: use the Nano's **2 hardware-PWM pins (board 32 & 33)** — enable them once with
  `sudo /opt/nvidia/jetson-io/jetson-io.py` → configure pins 32/33 as PWM. (Software PWM via
  `Jetson.GPIO` also works but is jittery.)
- **Policy → motors mapping** (our ONNX outputs `[left, right]` in [−1, 1]):
  - `left  > 0` → IN1=HIGH, IN2=LOW ; `left  < 0` → IN1=LOW, IN2=HIGH ; ENA duty = `|left|`
  - `right > 0` → IN3=HIGH, IN4=LOW ; `right < 0` → IN3=LOW, IN4=HIGH ; ENB duty = `|right|`
  - (If a side spins the wrong way, swap that motor's two output wires — matches the sim's
    "mirrored axle" sign handling, but done in hardware.)

## First-power checklist
1. Buck set to ~5.0–5.1 V (no load) ✔ before Jetson connected.
2. Common ground everywhere ✔.
3. Fuse + switch on LiPo + ✔.
4. LiPo alarm on balance lead, set 3.4 V/cell ✔.
5. Motor voltage rating confirmed (or PWM-limited) ✔.
6. ENA/ENB jumpers removed if using PWM ✔.
7. Wheels off the ground for the first motor test.
```
```
