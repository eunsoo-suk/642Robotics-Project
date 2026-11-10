"""Compare gyro values with the actual attitude change (quaternion derivative), plus RPM per motor."""
import numpy as np
import pybullet as p
from attitude_env import AttitudeEnv
from cf_controller import body_rates_deg

env = AttitudeEnv()
env.reset(options={'tilt_deg': [0, 0], 'rate_deg': [0, 0]})
for _ in range(50):
    env.step_angle_pid()
env.apply_rate_kick(1000, 0)
did, cid = env.aviary.DRONE_IDS[0], env.aviary.CLIENT
prev = None
for k in range(1000):
    env._inner_step(rate_cmd=None)
    _, q = p.getBasePositionAndOrientation(did, physicsClientId=cid)
    _, w = p.getBaseVelocity(did, physicsClientId=cid)
    R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
    if prev is not None and k % 50 == 0:
        dR = prev.T @ R            # rotation change in the body frame
        w_fd = np.degrees(np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2 * 500)
        st = env.aviary._getDroneStateVector(0)
        print(f't={1+(k+1)/500:4.2f} roll={np.degrees(p.getEulerFromQuaternion(q)[0]):6.1f} '
              f'gyro(state)={body_rates_deg(st[3:7], st[13:16]).round(1)} '
              f'gyro(pyb)={np.degrees(R.T @ np.array(w)).round(1)} actual(fd)={w_fd.round(1)} '
              f'rpm={env.aviary.last_clipped_action[0].round(0) if hasattr(env.aviary,"last_clipped_action") else "-"}')
    prev = R
env.close()
