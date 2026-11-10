import numpy as np, sys
from attitude_env import AttitudeEnv
from cf_controller import cos_tilt

class LPF2:  # firmware filter.c lpf2p (Butterworth 2nd order)
    def __init__(s, fs, fc):
        fr = fs/fc; ohm = np.tan(np.pi/fr); c = 1+2*np.cos(np.pi/4)*ohm+ohm**2
        s.b0 = ohm**2/c; s.b1 = 2*s.b0; s.b2 = s.b0
        s.a1 = 2*(ohm**2-1)/c; s.a2 = (1-2*np.cos(np.pi/4)*ohm+ohm**2)/c
        s.d1 = s.d2 = None
    def __call__(s, x):
        if s.d1 is None: s.d1 = s.d2 = x/(s.b0+s.b1+s.b2)*0+x  # init to steady state
        d0 = x - s.d1*s.a1 - s.d2*s.a2
        y = d0*s.b0 + s.d1*s.b1 + s.d2*s.b2
        s.d2 = s.d1; s.d1 = d0
        return y

class Env(AttitudeEnv):
    def __init__(s, fc): super().__init__(); s.fc = fc
    def reset(s, **kw):
        s.f = [LPF2(500, s.fc) for _ in range(3)] if s.fc else None
        return super().reset(**kw)
    def _inner_step(s, rate_cmd):
        pos, quat, att, vel, gyro = s._state()
        if s.f:
            # steady-state init: on the first call d1 = d2 = x / (1 + a1 + a2)
            for i,f in enumerate(s.f):
                if f.d1 is None: f.d1 = f.d2 = gyro[i]/(1+f.a1+f.a2)
            gyro = np.array([s.f[i](gyro[i]) for i in range(3)])
        if rate_cmd is None: rate_cmd = s.ctrl.attitude_to_rate([0,0], att)
        rpm = s.ctrl.rate_to_rpm([rate_cmd[0], rate_cmd[1], 0], gyro, s.alt.thrust(pos[2], vel[2], cos_tilt(quat)))
        s.aviary.step(rpm.reshape(1,4))

for fc in (None, 80):
    env = Env(fc)
    print(f'--- gyro LPF: {fc} Hz ---')
    for mag in (300, 1000, 1250):
        res = []
        for d in range(0, 360, 45):
            env.reset(options={'tilt_deg':[0,0],'rate_deg':[0,0]})
            for _ in range(50): env.step_angle_pid()
            a = np.radians(d); env.apply_rate_kick(mag*np.cos(a), mag*np.sin(a))
            ps = []; fail = False
            for k in range(100):
                *_, term, _, info = env.step_angle_pid(); fail |= term
                ps.append(abs(info['roll'])+abs(info['pitch']))
            # oscillation size: step-to-step rate change during the last 0.2 s
            res.append(f'{d}:{"X" if fail else max(ps[-25:]):.1f}' if not fail else f'{d}:X')
        print(f'{mag:5d} max tilt in last 0.5 s ', ' '.join(res))
    env.close()
