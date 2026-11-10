"""Motor thrust test (bypasses the controller, motorPowerSet).
Put the drone on a kitchen scale (on top of a cup so the prop wash does not hit the scale).
Each step: Enter -> motor(s) spin for --secs seconds -> read the scale while it spins.
Usage: python hw_motor_test.py --power 50000"""
import argparse, json, os, time, threading
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--power', type=int, default=50000)
ap.add_argument('--secs', type=float, default=4.0)
args = ap.parse_args()
uri = json.load(open(os.path.join(HERE, 'hw_signs.json')))['uri']
vb = {'v': 0.0, 'min': 9.0}

def on_log(ts, data, conf):
    vb['v'] = data['pm.vbat']

cflib.crtp.init_drivers()
with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=os.path.join(HERE, '..', 'cache'))) as scf:
    cf = scf.cf
    lg = LogConfig(name='vb', period_in_ms=20); lg.add_variable('pm.vbat', 'float')
    cf.log.add_config(lg); lg.data_received_cb.add_callback(on_log); lg.start()
    def setm(vals):
        for i, v in enumerate(vals, 1):
            cf.param.set_value(f'motorPowerSet.m{i}', int(v))
    steps = [('M1', [1, 0, 0, 0]), ('M2', [0, 1, 0, 0]), ('M3', [0, 0, 1, 0]), ('M4', [0, 0, 0, 1]), ('all 4', [1, 1, 1, 1])]
    try:
        setm([0, 0, 0, 0]); cf.param.set_value('motorPowerSet.enable', 1); time.sleep(0.3)
        time.sleep(0.5)
        input(f'\nMotors stopped. Write down the scale reading (g) (battery {vb["v"]:.2f} V). Enter > ')
        for name, mask in steps:
            input(f'\n[{name}] spin at power {args.power} for {args.secs:.0f} s. Press Enter when ready > ')
            setm([args.power * m for m in mask]); t0 = time.time(); vmin = 9.0
            while time.time() - t0 < args.secs:
                vmin = min(vmin, vb['v']); time.sleep(0.05)
            setm([0, 0, 0, 0])
            print(f'   -> lowest battery while spinning {vmin:.2f} V.  Write down the scale reading while spinning.')
            time.sleep(1.0)
    finally:
        setm([0, 0, 0, 0]); cf.param.set_value('motorPowerSet.enable', 0); time.sleep(0.3)
        lg.stop()
print('\nDone. Report the reading at rest, the reading for each step, and the lowest battery voltage.')
