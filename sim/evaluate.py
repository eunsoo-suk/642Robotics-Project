r"""
DQN vs PID 비교 평가 (2단계 지표 포함, Koch et al. 방식 참고)

    python evaluate.py                                   # PID, 가만히만
    python evaluate.py --model results\dqn_xxx\best_model.zip

모든 제어기가 같은 초기 조건(시드 1000~)에서 시작합니다.
지표:
    점수        에피소드 보상 합 (0에 가까울수록 좋음)
    성공률      실패 없이 끝나고, 마지막 0.5초 동안 기울기 < 5°
    복귀시간    기울기가 5° 안에 들어온 뒤 끝까지 유지되기 시작한 시점 (s)
    오버슈트    반대 방향으로 넘어간 최대 각도 (°)
    정상오차    마지막 0.5초 평균 기울기 (°)
    명령변경    초당 행동이 바뀐 횟수 (떨림 지표, DQN만 의미 있음)
"""
import argparse
import csv
import os
import numpy as np
from attitude_env import AttitudeEnv, load_env_config

ap = argparse.ArgumentParser()
ap.add_argument('--model', default=None)
ap.add_argument('--episodes', type=int, default=50)
ap.add_argument('--band', type=float, default=5.0, help='복귀 판정 기준 (°)')
args = ap.parse_args()

env = AttitudeEnv(**load_env_config(args.model))
print(f'행동 수: {env.action_space.n}')
model = None
if args.model:
    from stable_baselines3 import DQN
    model = DQN.load(args.model)
out_dir = os.path.dirname(os.path.abspath(args.model)) if args.model else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(out_dir, exist_ok=True)
DT = 1.0 / env.CTRL_HZ
LAST = int(0.5 * env.CTRL_HZ)


def rollout(policy, seed):
    obs, info = env.reset(seed=seed)
    init = np.array([info['roll'], info['pitch']])
    hist, ret, switches, prev_a, term = [], 0.0, 0, None, False
    done = False
    while not done:
        if policy == 'pid':
            obs, r, term, trunc, info = env.step_angle_pid()
        else:
            a = env.ZERO_ACTION if policy == 'zero' else int(model.predict(obs, deterministic=True)[0])
            switches += int(prev_a is not None and a != prev_a)
            prev_a = a
            obs, r, term, trunc, info = env.step(a)
        ret += r
        hist.append([info['roll'], info['pitch']])
        done = term or trunc
    h = np.array(hist)
    tilt = np.abs(h).max(axis=1)
    inside = tilt < args.band
    settle = np.nan
    if not term and inside[-1]:
        k = len(inside) - 1
        while k > 0 and inside[k - 1]:
            k -= 1
        settle = (k + 1) * DT
    over = np.maximum(-np.sign(init) * h, 0).max()
    return dict(score=ret, failed=int(term),
                success=int((not term) and tilt[-LAST:].max() < args.band),
                settle_s=settle, overshoot_deg=over,
                ss_err_deg=tilt[-LAST:].mean(), switch_per_s=switches / (len(h) * DT),
                init_roll=init[0], init_pitch=init[1]), h


policies = ['zero', 'pid'] + (['dqn'] if model else [])
names = {'zero': '가만히', 'pid': 'PID(펌웨어)', 'dqn': 'DQN'}
rows, examples = [], {}
for pol in policies:
    for i in range(args.episodes):
        m, h = rollout(pol, 1000 + i)
        m['policy'] = pol; m['episode'] = i
        rows.append(m)
        if i == 0:
            examples[pol] = h

print(f'\n같은 초기 조건 {args.episodes}개 (기울기 ±{env.INIT_TILT:.0f}°, 각속도 ±{env.INIT_RATE:.0f}°/s)\n')
print(f'{"":<12}{"점수":>8}{"성공률":>8}{"복귀시간":>9}{"오버슈트":>9}{"정상오차":>9}{"명령변경":>9}')
for pol in policies:
    r = [x for x in rows if x['policy'] == pol]
    g = lambda k: np.array([x[k] for x in r], dtype=float)
    settle = g('settle_s')
    settle_txt = f'{np.nanmean(settle):6.2f}s' if np.any(~np.isnan(settle)) else '     -'
    print(f'{names[pol]:<12}{g("score").mean():8.1f}{100 * g("success").mean():7.0f}%'
          f'{settle_txt:>9}{g("overshoot_deg").mean():8.1f}°{g("ss_err_deg").mean():8.1f}°'
          f'{g("switch_per_s").mean():8.1f}/s')

csv_path = os.path.join(out_dir, 'eval_results.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
plot_names = {'zero': 'Zero-rate', 'pid': 'PID (firmware)', 'dqn': 'DQN'}
for pol, h in examples.items():
    t = np.arange(len(h)) * DT
    ax[0].plot(t, h[:, 0], label=plot_names[pol]); ax[1].plot(t, h[:, 1], label=plot_names[pol])
for a, lab in zip(ax, ['roll (deg)', 'pitch (deg)']):
    a.axhline(0, color='gray', lw=0.8); a.set_ylabel(lab); a.grid(alpha=0.3)
ax[1].set_xlabel('time (s)'); ax[0].legend(); ax[0].set_title('Example episode (same initial tilt)')
png = os.path.join(out_dir, 'eval_example.png')
fig.tight_layout(); fig.savefig(png, dpi=120)
print(f'\n상세 결과: {csv_path}\n예시 그래프: {png}')
env.close()