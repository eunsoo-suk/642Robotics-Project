r"""
Experiment 3: sudden mid-flight kick test (DQN vs PID)

Scenario (3 s episode):
    0-1 s  level hover (tilt 0, rate 0)
    1 s    sudden kick: add angular rate at once (like a hand flick)
    1-3 s  watch the recovery
Kicks: size 100-2000 deg/s (9 sizes) x 24 directions (every 15 deg in the roll-pitch plane) = 216 cases, same for every controller
    - at 500 deg/s or less, the firmware rate PID (500 Hz) absorbs most of it within 20 ms, so the tilt stays under 10 deg (pilot test)
    - upper limit 2000 deg/s = range of the Crazyflie gyro (BMI088); beyond that, no controller can recover because of motor limits

Training range: initial rate during training is +/-100 deg/s per axis (a square region)
    size <= 100        : every direction is inside the training range
    100 < size <= 141  : diagonals inside, axis directions outside (edge)
    size > 141         : every direction is outside the training range

Run:
    python disturbance_test.py                         # PID + the 3 final DQN models (seeds 0, 1, 2)
    python disturbance_test.py --models a\best_model.zip b\best_model.zip
Output: results\disturbance_<date_time>\  (summary table, CSV, 3 plots)
"""
import argparse
import csv
import os
import time
import numpy as np
from attitude_env import AttitudeEnv, load_env_config

HERE = os.path.dirname(os.path.abspath(__file__))
def _latest_models(prefix='dqn_25a_sw0.05', seeds=(0, 1, 2)):
    """For each seed, pick the latest training folder whose model has the gyro filter (gyro_lpf_hz) setting."""
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
ap.add_argument('--mags', default='100,250,500,750,1000,1250,1500,1750,2000', help='kick sizes (deg/s)')
ap.add_argument('--polar', default='1000,1250,1500,1750', help='kick sizes for the per-direction plot')
ap.add_argument('--example-mag', type=float, default=1250, help='kick size for the example plot (+roll direction)')
ap.add_argument('--dirs', type=int, default=24, help='number of directions (equal split of 360 deg)')
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
    return 'in' if mag <= 100 else ('edge' if mag <= 100 * np.sqrt(2) + 1e-6 else 'out')


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
    proj = post @ u                      # tilt toward the kick direction (+), overshoot past level (-)
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
    print(f'  {name} done ({time.time() - t0:.0f}s)')

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


print(f'\nKicks: {len(MAGS)} sizes x {len(THETAS)} directions = {len(MAGS) * len(THETAS)} per controller, '
      f'DQN = mean +/- std over {len(dqn_names)} seeds\n')
print(f'{"size":>6} {"range":>6} | {"success PID":>11} {"DQN":>11} | {"max tilt PID":>12} {"DQN":>13} | '
      f'{"recovery PID":>12} {"DQN":>12} | {"flip PID":>9} {"DQN":>6} | {"switches DQN":>12}')
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

# ---------------- plots ----------------
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

print(f'\nOutput folder: {out}')
print('  disturbance_summary.png  (success rate, max tilt, recovery time by kick size)')
print(f'  disturbance_polar.png    (success rate by direction: {args.polar} deg/s)')
print(f'  disturbance_example.png  (example: {args.example_mag:.0f} deg/s, +roll direction)')
print('  disturbance_results.csv  (all results)')