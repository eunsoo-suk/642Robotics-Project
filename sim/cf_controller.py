"""
Crazyflie 펌웨어(controller_pid)를 시뮬에 복제한 제어기.

구조 (펌웨어와 동일한 계층):
    [angle PID]  목표 각도(deg)      -> 목표 회전속도(deg/s)   <- 2단계 PID 베이스라인 / 펌웨어 angle 모드
    [rate PID]   목표 회전속도(deg/s) -> 토크 명령(PWM 단위)   <- DQN이 이 입력을 직접 줌 (펌웨어 rate 모드)
    [mixer]      추력 + 토크          -> 모터 4개 PWM -> RPM

기본 게인은 펌웨어 기본값이며, sim/cf_gains.json(read_gains.py로 실기에서 읽은 값)이 있으면 그 값으로 덮어씁니다.
좌표계: 시뮬의 body frame 기준 (roll=x축, pitch=y축 회전, 양수 토크 = 양의 방향 회전).
        실기 cflib 명령과의 부호 매칭은 5단계(실기 전이)에서 따로 확인합니다.
"""
import json
import os
import numpy as np
import pybullet as p

GAINS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cf_gains.json')

# 펌웨어 기본값 (kp, ki, kd, i_limit)
DEFAULT_ATT = {'roll': (6.0, 3.0, 0.0, 20.0), 'pitch': (6.0, 3.0, 0.0, 20.0), 'yaw': (6.0, 1.0, 0.35, 360.0)}
DEFAULT_RATE = {'roll': (250.0, 500.0, 2.5, 33.3), 'pitch': (250.0, 500.0, 2.5, 33.3), 'yaw': (120.0, 16.7, 0.0, 166.7)}

# gym-pybullet-drones CF2X 상수 (DSLPIDControl과 동일)
PWM2RPM_SCALE = 0.2685
PWM2RPM_CONST = 4070.3
MIN_PWM, MAX_PWM = 0.0, 65535.0
MIXER = np.array([[-.5, -.5, -1.],
                  [-.5,  .5,  1.],
                  [ .5,  .5, -1.],
                  [ .5, -.5,  1.]])
INT16 = 32767.0


class PID:
    """펌웨어 pid.c 방식: 적분 클램프 + 미분 1차 저역통과 필터."""
    def __init__(self, kp, ki, kd, i_limit, dt, d_cutoff_hz=None):
        self.kp, self.ki, self.kd, self.i_limit, self.dt = kp, ki, kd, i_limit, dt
        if d_cutoff_hz:                     # 펌웨어 필터 켜짐(FiltEn=1)일 때만
            rc = 1.0 / (2 * np.pi * d_cutoff_hz)
            self.alpha = dt / (rc + dt)
        else:                               # 실기 기본값: 필터 꺼짐
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
    """펌웨어 filter.c의 lpf2p (2차 Butterworth 저역통과). 자이로 값에 적용."""
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
        if self.d1 is None:                     # 첫 값으로 정상상태 초기화 (시작 시 튐 방지)
            self.d1 = self.d2 = x / (1.0 + self.a1 + self.a2)
        d0 = x - self.d1 * self.a1 - self.d2 * self.a2
        y = d0 * self.b0 + self.d1 * self.b1 + self.d2 * self.b2
        self.d2, self.d1 = self.d1, d0
        return y


def _load_gains():
    att = {k: list(v) for k, v in DEFAULT_ATT.items()}
    rate = {k: list(v) for k, v in DEFAULT_RATE.items()}
    src = '펌웨어 기본값'
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
        # 펌웨어 IMU 드라이버(sensors_bmi088)의 자이로 LPF: 80Hz 2차 Butterworth. 없으면 rate D항 때문에 2ms 주기 진동 발생
        self.gyro_lpf = [LPF2(1.0 / dt, gyro_lpf_hz) for _ in range(3)] if gyro_lpf_hz else None
        self.gyro_f = None                      # 마지막으로 필터를 거친 자이로 값 (실기 로그의 gyro와 같은 의미)

    def reset(self):
        for pid in self.att + self.rate:
            pid.reset()
        for f in self.gyro_lpf or []:
            f.reset()
        self.gyro_f = None

    def attitude_to_rate(self, att_sp_deg, att_deg):
        """angle PID: 목표 roll/pitch(deg) -> 목표 roll/pitch 회전속도(deg/s)."""
        return np.array([self.att[i].update(att_sp_deg[i] - att_deg[i]) for i in range(2)])

    def rate_to_rpm(self, rate_sp_deg, gyro_deg, thrust_pwm):
        """rate PID + mixer: 목표 회전속도(deg/s, roll/pitch/yaw) -> 모터 RPM 4개."""
        if self.gyro_lpf:
            gyro_deg = np.array([self.gyro_lpf[i](gyro_deg[i]) for i in range(3)])
        self.gyro_f = np.asarray(gyro_deg, dtype=float)
        torque = np.array([self.rate[i].update(rate_sp_deg[i] - gyro_deg[i]) for i in range(3)])
        torque = np.clip(torque, -INT16, INT16)
        pwm = np.clip(thrust_pwm + MIXER @ torque, MIN_PWM, MAX_PWM)
        return PWM2RPM_SCALE * pwm + PWM2RPM_CONST


# ---------------- 보조 함수 ----------------
def hover_pwm(mass=0.027, kf=3.16e-10, g=9.8):
    rpm = np.sqrt(mass * g / (4 * kf))
    return (rpm - PWM2RPM_CONST) / PWM2RPM_SCALE


class AltitudeHold:
    """시뮬 전용 고도 유지(PD). DQN이 수평에만 집중하도록 고도는 따로 잡아줌."""
    KP, KD = 24700.0, 14800.0   # PWM/m, PWM/(m/s)  (약 3 rad/s, 감쇠 0.9로 설계)

    def __init__(self, z_ref=1.0):
        self.z_ref = z_ref
        self.base = hover_pwm()

    def thrust(self, z, vz, cos_tilt):
        u = self.base + self.KP * (self.z_ref - z) - self.KD * vz
        return u / max(cos_tilt, 0.5)


def body_rates_deg(quat_xyzw, ang_v_world):
    """PyBullet의 world frame 각속도 -> body frame 각속도(deg/s). 실기 자이로와 같은 기준."""
    R = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
    return np.degrees(R.T @ np.asarray(ang_v_world))


def cos_tilt(quat_xyzw):
    R = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
    return R[2, 2]


# ---------------- 자체 테스트 ----------------
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
        print(f'\n=== {mode} 모드 (초기 roll 20°, pitch -10°) / 게인: {ctrl.gain_source} ===')
        for k in range(int(2.0 * FREQ)):
            s = obs[0]
            pos, quat, rpy, vel, angv = s[0:3], s[3:7], s[7:10], s[10:13], s[13:16]
            gyro = body_rates_deg(quat, angv)
            if mode == 'ANGLE':
                rp_rate = ctrl.attitude_to_rate([0.0, 0.0], np.degrees(rpy[:2]))
            else:
                rp_rate = np.zeros(2)            # DQN이 '회전속도 0'만 계속 명령하는 상황
            rpm = ctrl.rate_to_rpm([rp_rate[0], rp_rate[1], 0.0], gyro,
                                   alt.thrust(pos[2], vel[2], cos_tilt(quat)))
            obs, *_ = env.step(rpm.reshape(1, 4))
            if k % (FREQ // 4) == 0:
                print(f't={k/FREQ:4.2f}s  roll={np.degrees(rpy[0]):6.1f}°  pitch={np.degrees(rpy[1]):6.1f}°  '
                      f'z={pos[2]:5.2f}m  gyro=({gyro[0]:6.1f},{gyro[1]:6.1f})°/s')
        env.close()

    run('ANGLE')
    run('RATE')
    print('\n기대 결과: ANGLE은 roll/pitch가 0°로 복귀, RATE는 초기 기울기 근처에서 유지(회전만 멈춤).')