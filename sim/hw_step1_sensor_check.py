r"""
Hardware step 1: sensor check (without propellers!)

What it does
  1) Live view: keeps showing the drone's roll, pitch and gyro (roll/pitch rate) on screen
  2) Direction test: tilt the drone by hand as instructed; it checks whether each sign matches the simulator
  3) Saves the result to hw_signs.json (step 2 and later use these signs)

Simulator convention (Crazyflie body frame: x = forward, y = left, z = up)
  - right side down  -> roll  +,  roll rate  + while tilting
  - nose down        -> pitch +,  pitch rate + while tilting

Run (.venv-sim, in the sim folder):
    python hw_step1_sensor_check.py
    python hw_step1_sensor_check.py --uri radio://0/80/2M/E7E7E7E7E7
"""
import argparse
import json
import os
import time

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--uri', default='radio://0/80/2M/E7E7E7E7E7')
ap.add_argument('--live-sec', type=float, default=15.0, help='live view duration (s)')
ap.add_argument('--record-sec', type=float, default=3.0, help='recording time for one direction test (s)')
args = ap.parse_args()

VARS = ['stabilizer.roll', 'stabilizer.pitch', 'gyro.x', 'gyro.y']


def make_log():
    lg = LogConfig(name='att', period_in_ms=10)   # 100 Hz
    for v in VARS:
        lg.add_variable(v, 'float')
    return lg


def record(scf, seconds, live=False):
    """Record for `seconds`. Returns the times (t) and an array of [roll, pitch, gx, gy]."""
    t, X = [], []
    t0 = time.time(); last_print = 0.0
    with SyncLogger(scf, make_log()) as logger:
        for entry in logger:
            _, data, _ = entry
            now = time.time() - t0
            t.append(now); X.append([data[v] for v in VARS])
            if live and now - last_print > 0.1:
                r, pch, gx, gy = X[-1]
                print(f'\r  roll {r:7.1f}°   pitch {pch:7.1f}°   roll rate {gx:8.1f}°/s   pitch rate {gy:8.1f}°/s   ', end='', flush=True)
                last_print = now
            if now >= seconds:
                break
    if live:
        print()
    return np.array(t), np.array(X)


def direction_test(scf, title, instruction):
    print(f'\n[{title}]')
    print(f'  Ready: hold the drone level.')
    print(f'  Press Enter to record for {args.record_sec:.0f} s. During that time, {instruction} (slowly, to about 30°, then hold)')
    input('  Press Enter when ready > ')
    t, X = record(scf, args.record_sec)
    n0 = max(3, int(0.3 / max(np.median(np.diff(t)), 1e-3)))
    start, end = X[:n0].mean(axis=0), X[-n0:].mean(axis=0)
    dt = np.diff(t, prepend=t[0])
    gyro_int = (X[:, 2:4] * dt[:, None]).sum(axis=0)   # rotation from integrating the gyro (deg)
    d_roll, d_pitch = end[0] - start[0], end[1] - start[1]
    print(f'  Angle change: roll {d_roll:+.1f}°, pitch {d_pitch:+.1f}°   /   gyro integral: roll {gyro_int[0]:+.1f}°, pitch {gyro_int[1]:+.1f}°')
    return d_roll, d_pitch, gyro_int


cflib.crtp.init_drivers()
print(f'Connecting: {args.uri}')
try:
    scf_ctx = SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache=os.path.join(HERE, '..', 'cache')))
    scf = scf_ctx.__enter__()
except Exception as e:
    print(f'\nConnection failed: {e}')
    print('Drones found nearby:', [u[0] for u in cflib.crtp.scan_interfaces()])
    print('Run again with one of these addresses as --uri.')
    raise SystemExit(1)

try:
    print('Connected. Make sure the propellers are off. (This script does not spin the motors.)\n')
    print(f'1) Live view for {args.live_sec:.0f} s: tilt the drone in different directions.')
    t, X = record(scf, args.live_sec, live=True)
    rate = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0
    print(f'  Receive rate: about {rate:.0f} Hz (target 100 Hz)')

    print('\n2) Direction test (twice)')
    dr1, dp1, g1 = direction_test(scf, 'Test A: roll', 'tilt the right side of the drone down')
    dr2, dp2, g2 = direction_test(scf, 'Test B: pitch', 'tilt the nose of the drone down')
finally:
    scf_ctx.__exit__(None, None, None)

ok = True
if abs(dr1) < 10 or abs(dp2) < 10:
    print('\n[warning] The tilt was less than 10°. Tilt more (about 30°) and run again.')
    ok = False
if abs(dp1) > abs(dr1) or abs(dr2) > abs(dp2):
    print('\n[warning] In the roll test pitch changed more (or the reverse). The axes may be swapped or the tilt direction was wrong.')
    ok = False

signs = {
    'roll': 1 if dr1 > 0 else -1,          # multiply the real value by this sign to get the simulator convention
    'pitch': 1 if dp2 > 0 else -1,
    'roll_rate': 1 if g1[0] > 0 else -1,
    'pitch_rate': 1 if g2[1] > 0 else -1,
}
print('\n=== Result: real drone vs simulator ===')
for k, desc in [('roll', 'right side down -> roll'), ('pitch', 'nose down -> pitch'),
                ('roll_rate', 'while rolling right side down -> roll rate'), ('pitch_rate', 'while pitching nose down -> pitch rate')]:
    print(f'  {desc:<44}: sim +,  real {"+" if signs[k] > 0 else "-"}   -> {"same" if signs[k] > 0 else "opposite (flip the sign)"}')

if ok:
    out = os.path.join(HERE, 'hw_signs.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'uri': args.uri, 'signs': signs, 'log_rate_hz': round(rate, 1),
                   'checked': time.strftime('%Y-%m-%d %H:%M')}, f, indent=2)
    print(f'\nSaved: {out}')
else:
    print('\nNot saved because of the warning above. Please run again.')
