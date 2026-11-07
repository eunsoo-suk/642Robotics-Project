import numpy as np, pybullet as p
from attitude_env import AttitudeEnv
from cf_controller import body_rates_deg
import sys
pk = float(sys.argv[1])
env = AttitudeEnv()
env.reset(options={'tilt_deg':[0,0],'rate_deg':[0,0]})
for _ in range(50): env.step_angle_pid()
env.apply_rate_kick(1000, pk)
did,cid=env.aviary.DRONE_IDS[0],env.aviary.CLIENT
for k in range(1000):
    pos,quat,att,vel,gyro=env._state()
    sp = env.ctrl.att[0].kp*(-att[0]) + env.ctrl.att[0].ki*env.ctrl.att[0].integ
    env._inner_step(None)
    if 480<=k<492 or k in (700,999):
        lin,w=p.getBaseVelocity(did,physicsClientId=cid)
        print(k, f'roll={att[0]:.3f} pitch={att[1]:.4f} p={gyro[0]:.2f} sp={sp:.2f} rI={env.ctrl.rate[0].integ:.2f} rpm={env.aviary.last_clipped_action[0].round(0)} vel={np.round(vel,3)}')
env.close()
