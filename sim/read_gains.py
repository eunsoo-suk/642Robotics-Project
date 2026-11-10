"""
Read the PID gains from the real Crazyflie and save them to sim/cf_gains.json.
Environment: the drone-control .venv (with cflib installed)
    .\.venv\Scripts\Activate.ps1
    python sim\read_gains.py
Just turn the drone on. The motors do not spin.
"""
import json
import os
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = 'radio://0/80/2M/E7E7E7E7E7'
GROUPS = ['pid_attitude', 'pid_rate']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cf_gains.json')

cflib.crtp.init_drivers()
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    cf = scf.cf
    toc = cf.param.toc.toc
    gains = {}
    for g in GROUPS:
        if g not in toc:
            print(f'[warning] parameter group {g} not found')
            continue
        gains[g] = {}
        for name in sorted(toc[g].keys()):
            v = cf.param.get_value(f'{g}.{name}')
            gains[g][name] = float(v)
            print(f'{g}.{name:<14} = {v}')
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(gains, f, indent=2)
    print(f'\nSaved: {OUT}')