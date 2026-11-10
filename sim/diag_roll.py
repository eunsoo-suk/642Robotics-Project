"""Debug the odd PID +roll behavior: log internal states after a +/- roll kick."""
import sys
import numpy as np
import pybullet as p
from attitude_env import AttitudeEnv

mag = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
env = AttitudeEnv()
for d in (+1, -1, 'p+', 'p-'):
    env.reset(options={'tilt_deg': [0, 0], 'rate_deg': [0, 0]})
    for _ in range(50):
        env.step_angle_pid()
    if d in (1, -1):
        env.apply_rate_kick(d * mag, 0); name = f'roll {d*mag:+.0f}'
    else:
        s = 1 if d == 'p+' else -1
        env.apply_rate_kick(0, s * mag); name = f'pitch {s*mag:+.0f}'
    print(f'\n=== kick {name} ===')
    print('   t  roll  pitch    yaw |   p     q     r | attI_r attI_p | rI_r  rI_p  rI_y |   z   | rpm min/max')
    for k in range(100):
        env.step_angle_pid()
        if k % 5 == 0 or k == 99:
            pos, quat, att, vel, gyro = env._state()
            yaw = np.degrees(p.getEulerFromQuaternion(quat)[2])
            c = env.ctrl
            rpm = env.aviary.last_clipped_action[0] if hasattr(env.aviary, 'last_clipped_action') else np.zeros(4)
            print(f'{1+(k+1)/50:4.2f} {att[0]:5.1f} {att[1]:6.1f} {yaw:6.1f} | {gyro[0]:5.0f} {gyro[1]:5.0f} {gyro[2]:5.0f} | '
                  f'{c.att[0].integ:6.2f} {c.att[1].integ:6.2f} | {c.rate[0].integ:5.2f} {c.rate[1].integ:5.2f} {c.rate[2].integ:6.2f} | '
                  f'{pos[2]:5.2f} | {rpm.min():.0f}/{rpm.max():.0f}')
env.close()
