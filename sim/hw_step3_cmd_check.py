r"""
Hardware step 3: command sign check (without propellers! thrust is 0, so the motors do not spin)

Goal: check that a "+" rate command we send turns the drone in the simulator's "+" direction (right side down / nose down)

Method (compare firmware internal values only, without spinning the motors)
  1) Switch the firmware to rate mode (flightmode.stabModeRoll/Pitch = 0). Restored at the end
  2) Command test: send +30 and -30 deg/s roll/pitch commands and log the target rate the firmware received
     (controller.rollRate / pitchRate)
  3) Hand test: with a zero command, turn the drone by hand in the "sim +" direction and read the sign of the
     rate measured inside the firmware (controller.r_roll / r_pitch)
  4) The "target" in 2) and the "measurement" in 3) use the same convention inside the firmware, so comparing their signs
     tells whether "our + command = rotation in the sim + direction" -> saved to hw_signs.json as cmd_signs

Run (.venv-sim, in the sim folder):
    python hw_step3_cmd_check.py
"""
import json
import os
import threading
import time

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNS_FILE = os.path.join(HERE, 'hw_signs.json')
with open(SIGNS_FILE, encoding='utf-8') as f:
    hw = json.load(f)

LOG_VARS = ['controller.rollRate', 'controller.pitchRate', 'controller.r_roll', 'controller.r_pitch']
MODE_PARAMS = ['flightmode.stabModeRoll', 'flightmode.stabModePitch']

latest = {'v': None}
buf, buf_on = [], [False]
lock = threading.Lock()
cmd = [0.0, 0.0]          # command to send [roll, pitch] (deg/s); the sender thread keeps sending it
running = [True]


def on_log(ts, data, conf):
    v = [data[k] for k in LOG_VARS]
    with lock:
        latest['v'] = v
        if buf_on[0]:
            buf.append(v)


def sender(cf):
    """Keep sending the command at 50 Hz (if it stops, the firmware safety sets the command to 0). Thrust is always 0."""
    while running[0]:
        cf.commander.send_setpoint(cmd[0], cmd[1], 0.0, 0)
        time.sleep(0.02)


def collect(seconds):
    with lock:
        buf.clear(); buf_on[0] = True
    time.sleep(seconds)
    with lock:
        buf_on[0] = False
        return np.array(buf)


cflib.crtp.init_drivers()
with SyncCrazyflie(hw['uri'], cf=Crazyflie(rw_cache=os.path.join(HERE, '..', 'cache'))) as scf:
    cf = scf.cf
    # ---- check names (may differ between firmware versions) ----
    log_toc, par_toc = cf.log.toc.toc, cf.param.toc.toc
    missing = [v for v in LOG_VARS if v.split('.')[1] not in log_toc.get(v.split('.')[0], {})]
    missing += [p for p in MODE_PARAMS if p.split('.')[1] not in par_toc.get(p.split('.')[0], {})]
    if missing:
        print('[stop] names not found in the firmware:', missing)
        print('  controller log variables:', sorted(log_toc.get('controller', {}).keys()))
        print('  flightmode parameters:', sorted(par_toc.get('flightmode', {}).keys()))
        raise SystemExit(1)
    ctrl_type = cf.param.get_value('stabilizer.controller') if 'controller' in par_toc.get('stabilizer', {}) else '?'
    print(f'Connected. Controller type (stabilizer.controller) = {ctrl_type}  (should be 1 = PID)')
    print('Make sure the propellers are off. Thrust is always sent as 0.\n')
    old_modes = {p: cf.param.get_value(p) for p in MODE_PARAMS}

    lg = LogConfig(name='ctrl', period_in_ms=10)
    for v in LOG_VARS:
        lg.add_variable(v, 'float')
    cf.log.add_config(lg); lg.data_received_cb.add_callback(on_log); lg.start()

    th = threading.Thread(target=sender, args=(cf,), daemon=True)
    try:
        for p in MODE_PARAMS:
            cf.param.set_value(p, 0)      # 0 = rate mode
        time.sleep(0.3)
        print('Switched to rate mode:', {p: cf.param.get_value(p) for p in MODE_PARAMS})
        th.start(); time.sleep(0.5)

        # ---- 1) command test ----
        print('\n1) Command test: leave the drone still on the desk (about 6 s)')
        received = {}
        for name, c in [('roll +30', (30, 0)), ('roll -30', (-30, 0)), ('pitch +30', (0, 30)), ('pitch -30', (0, -30))]:
            cmd[0], cmd[1] = c
            time.sleep(0.5)
            X = collect(0.8)
            received[name] = (float(np.median(X[:, 0])), float(np.median(X[:, 1])))
            print(f'  sent {name:9s} -> target received by firmware: roll {received[name][0]:+7.1f}, pitch {received[name][1]:+7.1f} deg/s')
        cmd[0], cmd[1] = 0.0, 0.0

        # ---- 2) hand test ----
        print('\n2) Hand test (zero command). Lift the drone on the desk as in step 1.')
        hand = {}
        for name, idx, how in [('roll', 2, 'lift the left side (M3, M4) (= right side down, sim roll +)'),
                               ('pitch', 3, 'lift the rear (M2, M3) (= nose down, sim pitch +)')]:
            input(f'  [{name}] after Enter, within 2 s, {how}  > ')
            X = collect(2.0)
            v = X[:, idx]
            big = v[np.abs(v) > 0.1 * np.abs(v).max()] if len(v) else v
            hand[name] = float(np.sum(big))
            print(f'    sign of the {name} rate measured by the firmware: {"+" if hand[name] > 0 else "-"}  (sum {hand[name]:+.2f})')
            input('    Put the drone back down level and press Enter > ')
    finally:
        running[0] = False
        time.sleep(0.1)
        cf.commander.send_stop_setpoint()
        for p, v in old_modes.items():
            cf.param.set_value(p, int(v))
        time.sleep(0.3)
        lg.stop()
        print('\nRestored the original mode:', {p: cf.param.get_value(p) for p in MODE_PARAMS})

# ---- verdict ----
ok = True
rx_roll, rx_pitch = received['roll +30'][0], received['pitch +30'][1]
for name, val, other in [('roll', rx_roll, received['roll +30'][1]), ('pitch', rx_pitch, received['pitch +30'][0])]:
    if abs(abs(val) - 30) > 5 or abs(other) > 5:
        print(f'[warning] the {name} +30 command did not arrive as sent (received {val:+.1f}, other axis {other:+.1f}).'); ok = False
if min(abs(hand['roll']), abs(hand['pitch'])) < 1e-3:
    print('[warning] almost no rotation was measured in the hand test. Lift it more and run again.'); ok = False

cmd_signs = {'roll': int(np.sign(rx_roll) * np.sign(hand['roll'])),
             'pitch': int(np.sign(rx_pitch) * np.sign(hand['pitch']))}
print('\n=== Result: sign of the commands to send ===')
for k in ['roll', 'pitch']:
    print(f'  {k}: our "+" command = {"rotation in the sim + direction (send as is)" if cmd_signs[k] > 0 else "rotation in the sim - direction (flip the sign)"}')
if ok:
    hw['cmd_signs'] = cmd_signs
    hw['cmd_checked'] = time.strftime('%Y-%m-%d %H:%M')
    with open(SIGNS_FILE, 'w', encoding='utf-8') as f:
        json.dump(hw, f, indent=2)
    print(f'\nSaved: {SIGNS_FILE} (cmd_signs)')
else:
    print('\nNot saved because of the warning above.')
