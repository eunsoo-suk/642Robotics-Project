r"""
Hardware step 2: live loop check (without propellers! commands are only shown on screen, never sent to the drone)

What it does
  1) Receives the drone sensors (roll, pitch, gyro) at 100 Hz
  2) Converts them to the simulator convention with the signs in hw_signs.json (step 1 result: only the pitch angle is flipped)
  3) The trained DQN picks a command (target roll/pitch rate) at 50 Hz  -> shown on screen only, not sent
  4) At the end, summarizes the loop rate, the delays, and whether the command pointed back toward level, and saves a CSV

How to check: slowly tilt the drone by hand while it runs.
  - right side down -> the roll command should be negative (-) (back toward the left)
  - nose down       -> the pitch command should be negative (-) (back toward the rear)
  (all angles and commands shown use the simulator sign convention)

Run (.venv-sim, in the sim folder):
    python hw_step2_dry_run.py
    python hw_step2_dry_run.py --model results\dqn_25a_sw0.05_prop_s2_20261107_174607\best_model.zip --seconds 60
"""
import argparse
import csv
import itertools
import json
import os
import threading
import time

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from stable_baselines3 import DQN

from attitude_env import load_env_config

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.path.join(HERE, 'results', 'dqn_25a_sw0.05_prop_s1_20261107_174517', 'best_model.zip'))
ap.add_argument('--seconds', type=float, default=40.0)
ap.add_argument('--ctrl-hz', type=float, default=50.0)
args = ap.parse_args()

with open(os.path.join(HERE, 'hw_signs.json'), encoding='utf-8') as f:
    hw = json.load(f)
S = hw['signs']
print(f"Signs (hw_signs.json, {hw['checked']}): roll {S['roll']:+d}, pitch {S['pitch']:+d}, roll rate {S['roll_rate']:+d}, pitch rate {S['pitch_rate']:+d}")

# ---- same observation/action definition as training (attitude_env.py defaults) ----
cfg = load_env_config(args.model)
LEVELS = cfg.get('rate_levels', [-90.0, 0.0, 90.0])
ACTIONS = np.array(list(itertools.product(LEVELS, LEVELS)), dtype=float)
TILT_NORM, OMEGA_NORM, RATE_MAX = cfg.get('tilt_norm_deg', 60.0), 300.0, max(abs(x) for x in LEVELS)
model = DQN.load(args.model)
print(f'Model: {args.model}\n{len(ACTIONS)} actions ({LEVELS} deg/s per axis)')


def obs_vector(roll, pitch, gx, gy, prev_cmd):
    o = np.concatenate([np.array([roll, pitch]) / TILT_NORM, np.array([gx, gy]) / OMEGA_NORM, prev_cmd / RATE_MAX])
    return np.clip(o, -5.0, 5.0).astype(np.float32)


# ---- sensor stream (callback, keeps only the latest value) ----
latest = {'t': None, 'v': None, 'n': 0}
lock = threading.Lock()


def on_log(ts, data, conf):
    with lock:
        latest['t'] = time.perf_counter()
        latest['v'] = (S['roll'] * data['stabilizer.roll'], S['pitch'] * data['stabilizer.pitch'],
                       S['roll_rate'] * data['gyro.x'], S['pitch_rate'] * data['gyro.y'])
        latest['n'] += 1


cflib.crtp.init_drivers()
rows = []
with SyncCrazyflie(hw['uri'], cf=Crazyflie(rw_cache=os.path.join(HERE, '..', 'cache'))) as scf:
    lg = LogConfig(name='att', period_in_ms=10)
    for v in ['stabilizer.roll', 'stabilizer.pitch', 'gyro.x', 'gyro.y']:
        lg.add_variable(v, 'float')
    scf.cf.log.add_config(lg)
    lg.data_received_cb.add_callback(on_log)
    lg.start()
    print('\nConnected. Propellers off. Commands are NOT sent to the drone.')
    print(f'Slowly tilt the drone for {args.seconds:.0f} s (right side down, nose down, etc.).\n')
    while latest['v'] is None:
        time.sleep(0.01)

    dt = 1.0 / args.ctrl_hz
    prev_cmd = np.zeros(2)
    t0 = time.perf_counter(); next_tick = t0; last_print = 0.0
    while True:
        now = time.perf_counter()
        if now - t0 >= args.seconds:
            break
        if now < next_tick:
            time.sleep(min(0.001, next_tick - now))
            continue
        next_tick += dt
        with lock:
            roll, pitch, gx, gy = latest['v']; age = now - latest['t']
        t_inf = time.perf_counter()
        a = int(model.predict(obs_vector(roll, pitch, gx, gy, prev_cmd), deterministic=True)[0])
        inf_ms = (time.perf_counter() - t_inf) * 1000
        cmd = ACTIONS[a]
        rows.append(dict(t=round(now - t0, 4), roll=roll, pitch=pitch, roll_rate=gx, pitch_rate=gy,
                         cmd_roll=cmd[0], cmd_pitch=cmd[1], sensor_age_ms=age * 1000, infer_ms=inf_ms))
        prev_cmd = cmd
        if now - last_print > 0.15:
            print(f'\r  roll {roll:6.1f}°  pitch {pitch:6.1f}°  |  DQN command: roll {cmd[0]:+5.0f}°/s  pitch {cmd[1]:+5.0f}°/s  |  '
                  f'sensor {age * 1000:4.1f} ms  compute {inf_ms:4.1f} ms   ', end='', flush=True)
            last_print = now
    lg.stop()

print('\n')
out = os.path.join(HERE, 'results', time.strftime('hw_step2_%Y%m%d_%H%M%S.csv'))
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

T = np.array([r['t'] for r in rows]); dts = np.diff(T) * 1000
age = np.array([r['sensor_age_ms'] for r in rows]); inf = np.array([r['infer_ms'] for r in rows])
R = np.array([[r['roll'], r['pitch'], r['cmd_roll'], r['cmd_pitch']] for r in rows])
print('=== Step 2 summary ===')
print(f'  Loop rate      : {(len(T) - 1) / (T[-1] - T[0]):.1f} Hz (target {args.ctrl_hz:g} Hz),  mean interval {dts.mean():.1f} ms / max {dts.max():.1f} ms')
print(f'  Sensor age     : mean {age.mean():.1f} ms, 95% {np.percentile(age, 95):.1f} ms  (how old the latest sensor value on the PC is)')
print(f'  DQN compute    : mean {inf.mean():.2f} ms, max {inf.max():.2f} ms')
for i, name in [(0, 'roll'), (1, 'pitch')]:
    for side, mask in [('+5° or more', R[:, i] > 5), ('-5° or less', R[:, i] < -5)]:
        n = int(mask.sum())
        if n == 0:
            print(f'  {name} {side}: no samples (tilt in that direction too)')
            continue
        want = -1 if side.startswith('+') else 1
        right = np.mean(np.sign(R[mask, 2 + i]) == want) * 100
        zero = np.mean(R[mask, 2 + i] == 0) * 100
        print(f'  {name} {side}, {n} steps: correcting command {right:.0f}%,  zero command {zero:.0f}%,  same direction (wrong) {100 - right - zero:.0f}%')
print(f'\nDetailed log: {out}')
