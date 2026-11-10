"""
Copy of the Crazyflie firmware controller (controller_pid) for the simulator.

Structure (same layers as the firmware):
    [angle PID]  target angle (deg)      -> target rate (deg/s)      <- PID baseline / firmware angle mode
    [rate PID]   target rate (deg/s)     -> torque command (PWM)     <- the DQN gives this input directly (firmware rate mode)
    [mixer]      thrust + torque         -> 4 motor PWM -> RPM

Default gains are the firmware defaults. If sim/cf_gains.json (read from the real drone by read_gains.py) exists, it overrides them.
Frame: simulator body frame (roll = x axis, pitch = y axis rotation, positive torque = positive rotation).
        Sign matching with real cflib commands is checked separately in the hardware steps.
"""
import json
import os
import numpy as np
import pybullet as p

GAINS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cf_gains.json')

# firmware defaults (kp, ki, kd, i_limit)
DEFAULT_ATT = {'roll': (6.0, 3.0, 0.0, 20.0), 'pitch': (6.0, 3.0, 0.0, 20.0), 'yaw': (6.0, 1.0, 0.35, 360.0)}
DEFAULT_RATE = {'roll': (250.0, 500.0, 2.5, 33.3), 'pitch': (250.0, 500.0, 2.5, 33.3), 'yaw': (120.0, 16.7, 0.0, 166.7)}

# gym-pybullet-drones CF2X constants (same as DSLPIDControl)
PWM2RPM_SCALE = 0.2685
PWM2RPM_CONST = 4070.3
MIN_PWM, MAX_PWM = 0.0, 65535.0
MIXER = np.array([[-.5, -.5, -1.],
                  [-.5,  .5,  1.],
                  [ .5,  .5, -1.],
                  [ .5, -.5,  1.]])
INT16 = 32767.0


class PID:
    """Same as firmware pid.c: integral clamp + first-order low-pass filter on the derivative."""
    def __init__(self, kp, ki, kd, i_limit, dt, d_cutoff_hz=None):
        self.kp, self.ki, self.kd, self.i_limit, self.dt = kp, ki, kd, i_limit, dt
        if d_cutoff_hz:                     # only when the firmware filter is on (FiltEn=1)
            rc = 1.0 / (2 * np.pi * d_cutoff_hz)
            self.alpha = dt / (rc + dt)
        else:                               # real drone default: filter off
            self.alpha = 1.0
        self.reset()

    def reset(self):
        self.integ = 0.0
        self.prev_err = 0.0
        self.deriv = 0.0

    def update(self, err):
        self.integ = np.clip(self.integ + err * self.dt, -self.i_limit, self.i_limit)
        raw_d = (err - self.prev_err) / self.dt
        self.deriv += self.alpha * (raw_d - self.deriv)
        self.prev_err = err
        return self.kp * err + self.ki * self.integ + self.kd * self.deriv


class LPF2:
    """lpf2p from firmware filter.c (2nd-order Butterworth low-pass). Applied to the gyro."""
    def __init__(self, fs, fc):
        ohm = np.tan(np.pi * fc / fs)
        c = 1.0 + 2.0 * np.cos(np.pi / 4) * ohm + ohm ** 2
        self.b0 = ohm ** 2 / c
        self.b1, self.b2 = 2.0 * self.b0, self.b0
        self.a1 = 2.0 * (ohm ** 2 - 1.0) / c
        self.a2 = (1.0 - 2.0 * np.cos(np.pi / 4) * ohm + ohm ** 2) / c
        self.reset()

    def reset(self):
        self.d1 = self.d2 = None

    def __call__(self, x):
        if self.d1 is None:                     # initialize to steady state with the first value (avoids a jump at start)
            self.d1 = self.d2 = x / (1.0 + self.a1 + self.a2)
        d0 = x - self.d1 * self.a1 - self.d2 * self.a2
        y = d0 * self.b0 + self.d1 * self.b1 + self.d2 * self.b2
        self.d2, self.d1 = self.d1, d0
        return y


def _load_gains():
    att = {k: list(v) for k, v in DEFAULT_ATT.items()}
    rate = {k: list(v) for k, v in DEFAULT_RATE.items()}
    src = 'firmware defaults'
    filt = {'att': None, 'rate': None}
    if os.path.exists(GAINS_FILE):
        with open(GAINS_FILE, encoding='utf-8') as f:
            g = json.load(f)
        pa, pr = g.get('pid_attitude', {}), g.get('pid_rate', {})
        if pa.get('attFiltEn', 0):
            filt['att'] = pa.get('attFiltCut', 15.0)
        if pr.get('rateFiltEn', 0):
            filt['rate'] = pr.get('omxFiltCut', 30.0)
        for grp, table in (('pid_attitude', att), ('pid_rate', rate)):
            for axis in ('roll', 'pitch', 'yaw'):
                for i, term in enumerate(('kp', 'ki', 'kd')):
                    key = f'{axis}_{term}'
                    if key in g.get(grp, {}):
                        table[axis][i] = g[grp][key]
        src = GAINS_FILE
    return att, rate, filt, src


class CrazyflieController:
    def __init__(self, dt, gyro_lpf_hz=80.0):
        self.dt = dt
        att, rate, filt, self.gain_source = _load_gains()
        self.att = [PID(*att[a], dt, filt['att']) for a in ('roll', 'pitch')]
        self.rate = [PID(*rate[a], dt, filt['rate']) for a in ('roll', 'pitch', 'yaw')]
        # gyro LPF of the firmware IMU driver (sensors_bmi088): 80 Hz 2nd-order Butterworth. Without it, the rate D term causes a 2 ms oscillation
        self.gyro_lpf = [LPF2(1.0 / dt, gyro_lpf_hz) for _ in range(3)] if gyro_lpf_hz else None
        self.gyro_f = None                      # last filtered gyro value (same meaning as gyro in the real drone logs)

    def reset(self):
        for pid in self.att + self.rate:
            pid.reset()
        for f in self.gyro_lpf or []:
            f.reset()
        self.gyro_f = None

    def attitude_to_rate(self, att_sp_deg, att_deg):
        """angle PID: target roll/pitch (deg) -> target roll/pitch rate (deg/s)."""
        return np.array([self.att[i].update(att_sp_deg[i] - att_deg[i]) for i in range(2)])

    def rate_to_rpm(self, rate_sp_deg, gyro_deg, thrust_pwm):
        """rate PID + mixer: target rate (deg/s, roll/pitch/yaw) -> 4 motor RPMs."""
        if self.gyro_lpf:
            gyro_deg = np.array([self.gyro_lpf[i](gyro_deg[i]) for i in range(3)])
        self.gyro_f = np.asarray(gyro_deg, dtype=float)
        torque = np.array([self.rate[i].update(rate_sp_deg[i] - gyro_deg[i]) for i in range(3)])
        torque = np.clip(torque, -INT16, INT16)
        pwm = np.clip(thrust_pwm + MIXER @ torque, MIN_PWM, MAX_PWM)
        return PWM2RPM_SCALE * pwm + PWM2RPM_CONST


# ---------------- helpers ----------------
def hover_pwm(mass=0.027, kf=3.16e-10, g=9.8):
    rpm = np.sqrt(mass * g / (4 * kf))
    return (rpm - PWM2RPM_CONST) / PWM2RPM_SCALE


class AltitudeHold:
    """Altitude hold (PD), simulation only. Holds altitude separately so the DQN can focus on leveling."""
    KP, KD = 24700.0, 14800.0   # PWM/m, PWM/(m/s)  (designed for about 3 rad/s, damping 0.9)

    def __init__(self, z_ref=1.0):
        self.z_ref = z_ref
        self.base = hover_pwm()

    def thrust(self, z, vz, cos_tilt):
        u = self.base + self.KP * (self.z_ref - z) - self.KD * vz
        return u / max(cos_tilt, 0.5)


def body_rates_deg(quat_xyzw, ang_v_world):
    """PyBullet world-frame angular velocity -> body-frame rate (deg/s). Same frame as the real gyro."""
    R = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
    return np.degrees(R.T @ np.asarray(ang_v_world))


def cos_tilt(quat_xyzw):
    R = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
    return R[2, 2]


# ---------------- self test ----------------
if __name__ == '__main__':
    import argparse
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    ap = argparse.ArgumentParser()
    ap.add_argument('--gui', action='store_true')
    args = ap.parse_args()

    FREQ = 500
    INIT = [np.radians(20), np.radians(-10), 0.0]

    def run(mode):
        env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                         initial_xyzs=np.array([[0, 0, 1.0]]), initial_rpys=np.array([INIT]),
                         physics=Physics.PYB, pyb_freq=FREQ, ctrl_freq=FREQ, gui=args.gui)
        obs, _ = env.reset()
        ctrl, alt = CrazyflieController(1.0 / FREQ), AltitudeHold(1.0)
        print(f'\n=== {mode} mode (initial roll 20°, pitch -10°) / gains: {ctrl.gain_source} ===')
        for k in range(int(2.0 * FREQ)):
            s = obs[0]
            pos, quat, rpy, vel, angv = s[0:3], s[3:7], s[7:10], s[10:13], s[13:16]
            gyro = body_rates_deg(quat, angv)
            if mode == 'ANGLE':
                rp_rate = ctrl.attitude_to_rate([0.0, 0.0], np.degrees(rpy[:2]))
            else:
                rp_rate = np.zeros(2)            # the DQN keeps commanding 'zero rate'
            rpm = ctrl.rate_to_rpm([rp_rate[0], rp_rate[1], 0.0], gyro,
                                   alt.thrust(pos[2], vel[2], cos_tilt(quat)))
            obs, *_ = env.step(rpm.reshape(1, 4))
            if k % (FREQ // 4) == 0:
                print(f't={k/FREQ:4.2f}s  roll={np.degrees(rpy[0]):6.1f}°  pitch={np.degrees(rpy[1]):6.1f}°  '
                      f'z={pos[2]:5.2f}m  gyro=({gyro[0]:6.1f},{gyro[1]:6.1f})°/s')
        env.close()

    run('ANGLE')
    run('RATE')
    print('\nExpected: ANGLE returns roll/pitch to 0°, RATE stays near the initial tilt (only the rotation stops).')