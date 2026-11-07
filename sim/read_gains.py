"""
실기 Crazyflie에서 PID 게인을 읽어 sim/cf_gains.json으로 저장합니다.
실행 환경: 드론 제어용 .venv (cflib 설치된 환경)
    .\.venv\Scripts\Activate.ps1
    python sim\read_gains.py
드론은 켜기만 하면 됩니다. 모터는 돌지 않습니다.
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
            print(f'[경고] 파라미터 그룹 {g} 없음')
            continue
        gains[g] = {}
        for name in sorted(toc[g].keys()):
            v = cf.param.get_value(f'{g}.{name}')
            gains[g][name] = float(v)
            print(f'{g}.{name:<14} = {v}')
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(gains, f, indent=2)
    print(f'\n저장 완료: {OUT}')