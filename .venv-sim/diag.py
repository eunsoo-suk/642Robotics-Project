# 자동 진단: 드론은 책상 위에 두기만 (프로펠러 뺀 상태)
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

URI = 'radio://0/80/2M/E7E7E7E7E7'
WANT = ['controller.roll', 'controller.rollRate', 'controller.cmd_roll',
        'controller.actuatorThrust', 'motor.m1', 'supervisor.info', 'sys.canfly', 'sys.isFlying']
last = {}

cflib.crtp.init_drivers()
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    cf = scf.cf
    names = []
    for n in WANT:
        try:
            cf.log.toc.get_element_by_complete_name(n)
            names.append(n)
        except Exception:
            print('  (로그 변수 없음)', n)
    lg = LogConfig('diag', period_in_ms=100)
    for n in names[:6]:
        lg.add_variable(n)
    cf.log.add_config(lg)
    lg.data_received_cb.add_callback(lambda ts, d, c: last.update(d))
    lg.start()

    cf.param.set_value('flightmode.stabModeRoll', '1')
    cf.param.set_value('flightmode.stabModePitch', '1')
    cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.5)

    try:
        for label, roll_cmd in [('roll=0 명령', 0), ('roll=30 명령', 30)]:
            print(f'\n=== ANGLE 모드, {label}, 추력 20000 ===')
            t0 = time.time(); k = 0
            while time.time() - t0 < 2.5:
                cf.commander.send_setpoint(roll_cmd, 0, 0, 20000)
                time.sleep(0.02); k += 1
                if k % 25 == 0:
                    print('  ' + '  '.join(f'{n.split(".")[1]}={last.get(n, "-")}' if not isinstance(last.get(n), float)
                                           else f'{n.split(".")[1]}={last[n]:.1f}' for n in names[:6]))
    finally:
        cf.commander.send_stop_setpoint()
        lg.stop()
        print('\n종료')