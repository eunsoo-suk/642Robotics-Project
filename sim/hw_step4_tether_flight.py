r"""
Hardware step 4: tethered flight - take off with the PID, then switch attitude control to the DQN (propellers on)

Keys (with this PowerShell window selected)
    Space      : emergency stop (motors off at once, program ends)
    W / S      : thrust up / down (--thrust-step per press);  H : jump to take-off thrust (--h-thrust)
    D          : switch to the DQN (the trained DQN controls attitude, rate mode);  T : take-off mode (rate mode, zero rate command)
    P          : switch to the PID (firmware default control, angle mode)
    Q          : land slowly and exit (thrust ramps down to 0)
    Ctrl+C     : same as emergency stop

Auto stop (the program cuts the motors before a person can react; --no-auto-stop turns off items 1 and 3 below)
    - roll or pitch goes over --max-tilt degrees (default 40)
    - no sensor data for more than --max-age seconds (default 0.1 s; radio lost)
    - flight time goes over --max-time seconds (default 60 s)

Run (.venv-sim, in the sim folder):
    python hw_step4_tether_flight.py
"""
import argparse
import csv
import itertools
import json
import msvcrt
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
ap.add_argument('--thrust-step', type=int, default=1000)
ap.add_argument('--thrust-max', type=int, default=60000, help='thrust limit (0-65535)')
ap.add_argument('--max-tilt', type=float, default=40.0)
ap.add_argument('--max-age', type=float, default=0.1)
ap.add_argument('--max-time', type=float, default=60.0)
ap.add_argument('--h-thrust', type=int, default=40000, help='thrust set at once by the H key')
ap.add_argument('--no-auto-stop', action='store_true', help='turn off the tilt and time auto stops (sensor-loss stop and stop keys stay on)')
args = ap.parse_args()

with open(os.path.join(HERE, 'hw_signs.json'), encoding='utf-8') as f:
    hw = json.load(f)
S, CS = hw['signs'], hw['cmd_signs']
cfg = load_env_config(args.model)
LEVELS = cfg.get('rate_levels', [-90.0, 0.0, 90.0])
ACTIONS = np.array(list(itertools.product(LEVELS, LEVELS)), dtype=float)
TILT_NORM, OMEGA_NORM, RATE_MAX = cfg.get('tilt_norm_deg', 60.0), 300.0, max(abs(x) for x in LEVELS)
model = DQN.load(args.model)
model.predict(np.zeros(6, dtype=np.float32), deterministic=True)   # warm-up call (avoids a delay on the first step)
MODE_PARAMS = ['flightmode.stabModeRoll', 'flightmode.stabModePitch']

print('Keys: Space = emergency stop, W/S = thrust, H = take-off thrust, T = take-off mode, D = DQN, P = PID, Q = land')

latest = {'t': None, 'v': None}
lock = threading.Lock()


def on_log(ts, data, conf):
    with lock:
        latest['t'] = time.perf_counter()
        latest['v'] = (S['roll'] * data['stabilizer.roll'], S['pitch'] * data['stabilizer.pitch'],
                       S['roll_rate'] * data['gyro.x'], S['pitch_rate'] * data['gyro.y'])


diag = {'m1': 0, 'm2': 0, 'm3': 0, 'm4': 0, 'fw_cmd_roll': 0.0, 'fw_cmd_pitch': 0.0, 'vbat': 0.0, 'yaw_rate': 0.0}


def on_diag(ts, data, conf):
    with lock:
        diag.update(m1=data['motor.m1'], m2=data['motor.m2'], m3=data['motor.m3'], m4=data['motor.m4'],
                    fw_cmd_roll=data['controller.cmd_roll'], fw_cmd_pitch=data['controller.cmd_pitch'], vbat=data['pm.vbat'], yaw_rate=data['gyro.z'])


def set_mode(cf, rate):
    for p in MODE_PARAMS:
        cf.param.set_value(p, 0 if rate else 1)


rows, reason = [], 'normal exit'
cflib.crtp.init_drivers()
with SyncCrazyflie(hw['uri'], cf=Crazyflie(rw_cache=os.path.join(HERE, '..', 'cache'))) as scf:
    cf = scf.cf
    lg = LogConfig(name='att', period_in_ms=10)
    for v in ['stabilizer.roll', 'stabilizer.pitch', 'gyro.x', 'gyro.y']:
        lg.add_variable(v, 'float')
    cf.log.add_config(lg); lg.data_received_cb.add_callback(on_log); lg.start()
    lg2 = LogConfig(name='diag', period_in_ms=10)   # for failure analysis: motor outputs, controller outputs, battery
    for v, t in [('motor.m1', 'uint16_t'), ('motor.m2', 'uint16_t'), ('motor.m3', 'uint16_t'), ('motor.m4', 'uint16_t'),
                 ('controller.cmd_roll', 'float'), ('controller.cmd_pitch', 'float'), ('pm.vbat', 'float'), ('gyro.z', 'float')]:
        lg2.add_variable(v, t)
    cf.log.add_config(lg2); lg2.data_received_cb.add_callback(on_diag); lg2.start()
    set_mode(cf, rate=False)                       # start with the PID (angle mode)
    for _ in range(10):                            # zero commands to unlock the thrust
        cf.commander.send_setpoint(0, 0, 0, 0); time.sleep(0.02)
    while latest['v'] is None:
        time.sleep(0.01)

    mode, thrust, landing = 'PID', 0, False
    prev_cmd = np.zeros(2)
    t0 = time.perf_counter(); next_tick = t0; last_print = 0.0
    print('\nStart: PID mode, thrust 0.  Press W step by step until the tether is slightly tight.\n')
    try:
        while True:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(0.001, next_tick - now)); continue
            next_tick += 0.02
            # ---- key input ----
            while msvcrt.kbhit():
                k = msvcrt.getwch().lower()
                if k == ' ':
                    reason = 'emergency stop (Space)'; raise StopIteration
                elif k == 'h' and not landing:
                    thrust = min(args.thrust_max, max(thrust, args.h_thrust))
                elif k == 'w' and not landing:
                    thrust = min(args.thrust_max, thrust + args.thrust_step)
                elif k == 's':
                    thrust = max(0, thrust - args.thrust_step)
                elif k == 'd' and mode != 'DQN':
                    set_mode(cf, rate=True); mode = 'DQN'; prev_cmd = np.zeros(2)
                elif k == 't' and mode != 'RATE0':   # take-off: rate mode + zero rate command (no correction build-up on the ground)
                    set_mode(cf, rate=True); mode = 'RATE0'
                elif k == 'p' and mode != 'PID':
                    set_mode(cf, rate=False); mode = 'PID'
                elif k == 'q':
                    landing = True
            if landing:
                thrust = max(0, thrust - 150)        # 150 every ~0.02 s -> slow descent
                if thrust == 0:
                    reason = 'landed (Q)'; raise StopIteration
            # ---- sensors and auto stop ----
            with lock:
                roll, pitch, gx, gy = latest['v']; age = now - latest['t']; d = dict(diag)
            if age > args.max_age:
                reason = f'auto stop: sensor lost ({age * 1000:.0f} ms)'; raise StopIteration
            if not args.no_auto_stop and max(abs(roll), abs(pitch)) > args.max_tilt:
                reason = f'auto stop: tilt limit (roll {roll:.0f}°, pitch {pitch:.0f}°)'; raise StopIteration
            if not args.no_auto_stop and now - t0 > args.max_time:
                reason = f'auto stop: time limit {args.max_time:.0f} s'; raise StopIteration
            # ---- command ----
            cmd = np.zeros(2)
            if mode == 'DQN':
                o = np.clip(np.concatenate([np.array([roll, pitch]) / TILT_NORM, np.array([gx, gy]) / OMEGA_NORM,
                                            prev_cmd / RATE_MAX]), -5, 5).astype(np.float32)
                cmd = ACTIONS[int(model.predict(o, deterministic=True)[0])]
                prev_cmd = cmd
                cf.commander.send_setpoint(CS['roll'] * cmd[0], CS['pitch'] * cmd[1], 0.0, int(thrust))
            else:
                cf.commander.send_setpoint(0.0, 0.0, 0.0, int(thrust))
            rows.append(dict(t=round(now - t0, 3), mode=mode, thrust=thrust, roll=roll, pitch=pitch,
                             roll_rate=gx, pitch_rate=gy, cmd_roll=cmd[0], cmd_pitch=cmd[1], age_ms=age * 1000, **d))
            if now - last_print > 0.15:
                print(f'\r  [{mode}] thrust {thrust:5d}{" (landing)" if landing else ""}  roll {roll:6.1f}°  pitch {pitch:6.1f}°  '
                      f'cmd roll {cmd[0]:+4.0f} pitch {cmd[1]:+4.0f}  |  {now - t0:4.0f}s   ', end='', flush=True)
                last_print = now
    except (StopIteration, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            reason = 'emergency stop (Ctrl+C)'
    finally:
        for _ in range(5):
            cf.commander.send_stop_setpoint(); time.sleep(0.01)
        try:
            set_mode(cf, rate=False); lg.stop(); lg2.stop()
        except Exception as ex:
            print(f'\n(link lost, skipped restoring the mode: {ex})')

print(f'\n\nEnd: {reason}  (motors stopped, PID mode restored)')
if rows:
    out = os.path.join(HERE, 'results', time.strftime('hw_step4_%Y%m%d_%H%M%S.csv'))
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'Log: {out}')
