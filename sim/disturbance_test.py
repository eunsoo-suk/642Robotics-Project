r"""
실험 3: 비행 중 순간 충격 테스트 (DQN vs PID)

시나리오 (에피소드 3초):
    0~1초  수평 호버링 (기울기 0, 회전 0)
    1초    순간 충격: 회전속도를 갑자기 더함 (손으로 툭 친 효과)
    1~3초  복구 관찰
충격: 크기 100~2000 deg/s (9단계) x 방향 24개(360도를 15도 간격, roll-pitch 평면) = 216개, 모든 제어기에 동일
    - 500 deg/s 이하는 펌웨어 rate PID(500Hz)가 20ms 안에 대부분 흡수해서 기울기가 10° 미만 (예비 실험)
    - 상한 2000 deg/s = Crazyflie 자이로(BMI088) 측정 한계, 이 이상은 모터 출력 한계로 어떤 제어기도 복구 불가

학습 범위: 학습 시 초기 회전속도가 축마다 ±100 deg/s (정사각형 범위)
    크기 <= 100        : 모든 방향이 학습 범위 안
    100 < 크기 <= 141  : 대각선은 안, 정면은 밖 (경계)
    크기 > 141          : 모든 방향이 학습 범위 밖

실행:
    python disturbance_test.py                         # PID + 최종 DQN 3개(seed 0,1,2)
    python disturbance_test.py --models a\best_model.zip b\best_model.zip
결과: results\disturbance_<날짜시간>\  (요약표, CSV, 그래프 3개)
"""
import argparse
import csv
import os
import time
import numpy as np
from attitude_env import AttitudeEnv, load_env_config

HERE = os.path.dirname(os.path.abspath(__file__))
def _latest_models(prefix='dqn_25a_sw0.05', seeds=(0, 1, 2)):
    """시드별로 가장 최근 학습 폴더 중 자이로 필터(gyro_lpf_hz) 설정이 있는 모델을 자동 선택."""
    import glob, json
    out = []
    for s in seeds:
        for d in sorted(glob.glob(os.path.join(HERE, 'results', f'{prefix}_s{s}_*')), reverse=True):
            cfg = os.path.join(d, 'env_config.json')
            if os.path.exists(os.path.join(d, 'best_model.zip')) and os.path.exists(cfg) \
                    and json.load(open(cfg, encoding='utf-8')).get('gyro_lpf_hz'):
                out.append(os.path.basename(d))
                break
    return out


DEFAULT_MODELS = _latest_models()

ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='*', default=None)
ap.add_argument('--mags', default='100,250,500,750,1000,1250,1500,1750,2000', help='충격 크기 (deg/s)')
ap.add_argument('--polar', default='1000,1250,1500,1750', help='방향별 그래프를 그릴 크기')
ap.add_argument('--example-mag', type=float, default=1250, help='예시 그래프의 충격 크기 (+roll 방향)')
ap.add_argument('--dirs', type=int, default=24, help='방향 개수 (360도를 균등 분할)')
ap.add_argument('--kick-time', type=float, default=1.0)
ap.add_argument('--band', type=float, default=5.0)
args = ap.parse_args()

MAGS = [float(x) for x in args.mags.split(',')]
THETAS = np.arange(args.dirs) * 360.0 / args.dirs
model_paths = args.models or [os.path.join(HERE, 'results', m, 'best_model.zip') for m in DEFAULT_MODELS]
for m in model_paths:
    if not os.path.exists(m):
        raise FileNotFoundError(m)
out = os.path.join(HERE, 'results', time.strftime('disturbance_%Y%m%d_%H%M'))
os.makedirs(out, exist_ok=True)


def range_label(mag):
    return '안' if mag <= 100 else ('경계' if mag <= 100 * np.sqrt(2) + 1e-6 else '밖')


def run_episode(env, model, mag, theta):
    DT = 1.0 / env.CTRL_HZ
    LAST = int(0.5 * env.CTRL_HZ)
    kick_step = int(round(args.kick_time * env.CTRL_HZ))
    u = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    obs, info = env.reset(seed=0, options={'tilt_deg': [0.0, 0.0], 'rate_deg': [0.0, 0.0]})
    hist, switches, prev, term, done, k = [], 0, None, False, False, 0
    while not done:
        if k == kick_step:
            env.apply_rate_kick(mag * u[0], mag * u[1])
            obs = env._obs()
        if model is None:
            obs, r, term, trunc, info = env.step_angle_pid()
        else:
            a = int(model.predict(obs, deterministic=True)[0])
            if k >= kick_step and prev is not None and a != prev:
                switches += 1
            prev = a
            obs, r, term, trunc, info = env.step(a)
        hist.append([info['roll'], info['pitch']])
        k += 1
        done = term or trunc
    h = np.array(hist)
    post = h[kick_step:]
    tilt = np.abs(post).max(axis=1)
    success = bool((not term) and len(tilt) >= LAST and tilt[-LAST:].max() < args.band)
    settle = np.nan
    if success:
        inside = tilt < args.band
        j = len(inside) - 1
        while j > 0 and inside[j - 1]:
            j -= 1
        settle = (j + 1) * DT
    proj = post @ u                      # 충격 방향으로 기운 정도 (+), 반대로 넘어간 정도 (-)
    return dict(success=int(success), tumbled=int(term), max_tilt=float(tilt.max()),
                settle_s=settle, overshoot=float(max(0.0, -proj.min())),
                switch_per_s=switches / max(len(post) * DT, 1e-9)), h


controllers = [('PID', None, {})]
from stable_baselines3 import DQN
for i, mp in enumerate(model_paths):
    controllers.append((f'DQN_{i}', DQN.load(mp), load_env_config(mp)))

rows, examples = [], {}
t0 = time.time()
for name, model, cfg in controllers:
    env = AttitudeEnv(**cfg)
    for mag in MAGS:
        for th in THETAS:
            m, h = run_episode(env, model, mag, th)
            m.update(controller=name, mag=mag, theta=th)
            rows.append(m)
            if mag == args.example_mag and th == 0:
                examples[name] = h
    env.close()
    print(f'  {name} 완료 ({time.time() - t0:.0f}s)')

with open(os.path.join(out, 'disturbance_results.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

dqn_names = [c[0] for c in controllers if c[0] != 'PID']


def stat(name, mag, key, success_only=False):
    v = np.array([r[key] for r in rows if r['controller'] == name and r['mag'] == mag
                  and (r['success'] or not success_only)], dtype=float)
    return np.nanmean(v) if len(v) and np.any(~np.isnan(v)) else np.nan


def dqn_stat(mag, key, success_only=False):
    v = np.array([stat(n, mag, key, success_only) for n in dqn_names])
    v = v[~np.isnan(v)]
    return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)


def fmt(x, unit='', d=1):
    return '   -  ' if np.isnan(x) else f'{x:.{d}f}{unit}'


print(f'\n충격 {len(MAGS)}단계 x 방향 {len(THETAS)}개 = {len(MAGS) * len(THETAS)}개 / 제어기, '
      f'DQN은 시드 {len(dqn_names)}개 평균±편차\n')
print(f'{"크기":>6} {"학습범위":>6} | {"성공률 PID":>10} {"DQN":>12} | {"최대기울기 PID":>13} {"DQN":>12} | '
      f'{"복귀시간 PID":>11} {"DQN":>13} | {"뒤집힘 PID":>9} {"DQN":>6} | {"명령변경 DQN":>11}')
summary = []
for mag in MAGS:
    ps, (ds, dss) = stat('PID', mag, 'success') * 100, [x * 100 for x in dqn_stat(mag, 'success')]
    pt, (dt, dts) = stat('PID', mag, 'max_tilt'), dqn_stat(mag, 'max_tilt')
    pr, (dr, drs) = stat('PID', mag, 'settle_s', True), dqn_stat(mag, 'settle_s', True)
    pf, (df, _) = stat('PID', mag, 'tumbled') * 100, [x * 100 for x in dqn_stat(mag, 'tumbled')]
    sw, sws = dqn_stat(mag, 'switch_per_s')
    summary.append((mag, ps, ds, dss, pt, dt, dts, pr, dr, drs))
    print(f'{mag:6.0f} {range_label(mag):>6} | {ps:9.0f}% {ds:6.0f}±{dss:3.0f}% | {pt:12.1f}° {dt:6.1f}±{dts:4.1f}° | '
          f'{fmt(pr, "s", 2):>11} {fmt(dr, "s", 2) + "±" + fmt(drs, "", 2):>13} | {pf:8.0f}% {df:5.0f}% | {sw:6.1f}±{sws:.1f}/s')

# ---------------- 그래프 ----------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S = np.array(summary)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
specs = [(1, 2, 3, 'Success rate (%)'), (4, 5, 6, 'Max tilt after kick (deg)'), (7, 8, 9, 'Recovery time (s, successes)')]
for a, (ip, idq, isd, lab) in zip(ax, specs):
    a.axvspan(0, 100, color='tab:green', alpha=0.08, label='trained range')
    a.axvspan(100, 141.4, color='tab:orange', alpha=0.10, label='boundary')
    a.plot(S[:, 0], S[:, ip], 'o-', color='tab:blue', label='PID (firmware)')
    a.errorbar(S[:, 0], S[:, idq], yerr=S[:, isd], fmt='s-', color='tab:red', capsize=3,
               label=f'DQN (mean±std, {len(dqn_names)} seeds)')
    a.set_xlabel('Kick magnitude (deg/s)'); a.set_ylabel(lab); a.grid(alpha=0.3)
ax[0].set_ylim(-5, 105); ax[0].legend(fontsize=8, loc='lower left')
fig.suptitle('Mid-flight rotational kick: DQN vs PID (24 directions per magnitude)')
fig.tight_layout(); fig.savefig(os.path.join(out, 'disturbance_summary.png'), dpi=120)

polar_mags = [m for m in (float(x) for x in args.polar.split(',')) if m in MAGS] or MAGS[-4:]
fig = plt.figure(figsize=(4 * len(polar_mags), 4.3))
th_closed = np.radians(np.append(THETAS, THETAS[0]))
for i, mag in enumerate(polar_mags):
    a = fig.add_subplot(1, len(polar_mags), i + 1, projection='polar')
    pid = [next(r['success'] for r in rows if r['controller'] == 'PID' and r['mag'] == mag and r['theta'] == th) for th in THETAS]
    dqn = [np.mean([next(r['success'] for r in rows if r['controller'] == n and r['mag'] == mag and r['theta'] == th)
                    for n in dqn_names]) for th in THETAS]
    a.plot(th_closed, np.append(pid, pid[0]), 'o-', color='tab:blue', ms=3, label='PID')
    a.plot(th_closed, np.append(dqn, dqn[0]), 's-', color='tab:red', ms=3, label='DQN (seed avg)')
    a.set_ylim(0, 1.05); a.set_title(f'{mag:.0f} deg/s', pad=12)
    a.set_xticks(np.radians([0, 90, 180, 270])); a.set_xticklabels(['+roll', '+pitch', '-roll', '-pitch'])
    if i == 0:
        a.legend(fontsize=8, loc='lower left', bbox_to_anchor=(-0.25, -0.15))
fig.suptitle('Success rate by kick direction (1 = recovered)')
fig.tight_layout(); fig.savefig(os.path.join(out, 'disturbance_polar.png'), dpi=120)

if examples:
    DTp = 1.0 / 50
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for name, h in examples.items():
        if name not in ('PID', 'DQN_0'):
            continue
        t = (np.arange(len(h)) + 1) * DTp
        lab = 'PID (firmware)' if name == 'PID' else 'DQN (seed 0)'
        ax[0].plot(t, h[:, 0], label=lab); ax[1].plot(t, h[:, 1], label=lab)
    for a, lab in zip(ax, ['roll (deg)', 'pitch (deg)']):
        a.axvline(args.kick_time, color='k', ls='--', lw=1); a.axhline(0, color='gray', lw=0.8)
        a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[0].legend(); ax[1].set_xlabel('time (s)')
    ax[0].set_title(f'Example: {args.example_mag:.0f} deg/s kick in +roll direction at t = 1 s')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'disturbance_example.png'), dpi=120)

print(f'\n결과 폴더: {out}')
print('  disturbance_summary.png  (크기별 성공률·최대기울기·복귀시간)')
print(f'  disturbance_polar.png    (방향별 성공률: {args.polar} deg/s)')
print(f'  disturbance_example.png  (예시: {args.example_mag:.0f} deg/s, +roll 방향)')
print('  disturbance_results.csv  (전체 결과)')