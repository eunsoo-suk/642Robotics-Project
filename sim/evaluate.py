r"""
DQN vs PID evaluation (step 2 metrics, following Koch et al.)

    python evaluate.py                                   # PID and zero-rate only
    python evaluate.py --model results\dqn_xxx\best_model.zip

All controllers start from the same initial conditions (seeds 1000+).
Metrics:
    score        sum of episode rewards (closer to 0 is better)
    success      no failure, and tilt < 5 deg during the last 0.5 s
    recovery     time when the tilt enters the 5 deg band and stays there until the end (s)
    overshoot    largest angle past level in the opposite direction (deg)
    ss error     mean tilt during the last 0.5 s (deg)
    switches     action changes per second (jitter measure, DQN only)
"""
import argparse
import csv
import os
import numpy as np
from attitude_env import AttitudeEnv, load_env_config

ap = argparse.ArgumentParser()
ap.add_argument('--model', default=None)
ap.add_argument('--episodes', type=int, default=50)
ap.add_argument('--band', type=float, default=5.0, help='recovery band (deg)')
args = ap.parse_args()

env = AttitudeEnv(**load_env_config(args.model))
print(f'Number of actions: {env.action_space.n}')
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
names = {'zero': 'zero-rate', 'pid': 'PID (fw)', 'dqn': 'DQN'}
rows, examples = [], {}
for pol in policies:
    for i in range(args.episodes):
        m, h = rollout(pol, 1000 + i)
        m['policy'] = pol; m['episode'] = i
        rows.append(m)
        if i == 0:
            examples[pol] = h

print(f'\n{args.episodes} identical initial conditions (tilt +/-{env.INIT_TILT:.0f}°, rate +/-{env.INIT_RATE:.0f}°/s)\n')
print(f'{"":<12}{"score":>8}{"success":>8}{"recovery":>9}{"overshoot":>9}{"ss err":>9}{"switches":>9}')
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
print(f'\nDetailed results: {csv_path}\nExample plot: {png}')
env.close()