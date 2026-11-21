r"""
Evaluation v2: DQN vs firmware PID with an explicit recovery definition and three disturbance types.
All controllers get exactly the same conditions (same seeds, same disturbances).

Recovery definition (defaults; all can be changed with the options below)
    timing reference t = 0 : tilt test -> episode start
                             kick test -> the moment of the kick
                             sustained test -> the moment the torque is RELEASED (the max tilt while
                             it acts is reported separately)
    recovered  : |roll| and |pitch| <= --band (5 deg) AND |roll rate| and |pitch rate| <= --rate-band (30 deg/s),
                 both held continuously for --hold (0.3 s)
    recovery time : time from t = 0 to the start of that holding window
    success    : recovered within --t-max (2 s) and no tumble (firmware rule: tilted > 60 deg for 1 s, or upside down)
    accuracy   : overshoot past level, steady-state error (mean tilt in the last 0.5 s), and recovery time
                 with a strict 2 deg band
Disturbances (in the roll-pitch plane, --dirs directions every 360/dirs deg, applied at t = 1 s of level hover)
    tilt      : random initial tilt +/-30 deg and rate +/-100 deg/s (seeds 1000+, same as evaluate.py)
    kick      : IMPULSIVE. Angular velocity changes at once by --kick-mags deg/s (duration: one step)
    sustained : SUSTAINED. Constant external torque for --dur s, size = --torque-fracs x reference torque
                (reference = all hover thrust moved to one side = 4 * F_hover * L / sqrt 2, about 7.4 mN m)

Run (.venv-sim, in the sim folder):
    python eval_v2.py                                    # PID + the 3 final DQN models, all tests
    python eval_v2.py --tests kick --models a\best_model.zip
    python eval_v2.py --motor-delay-ms 20                # same tests under a fixed sim-to-real condition
Output: results\eval_v2_<date_time>\  (trials.csv, summary.csv, summary.png)
"""
import argparse
import csv
import os
import time
import numpy as np
import pybullet as p
from attitude_env import AttitudeEnv, load_env_config

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = ['dqn_25a_sw0.05_prop_s0_20261107_142302', 'dqn_25a_sw0.05_prop_s1_20261107_174517',
         'dqn_25a_sw0.05_prop_s2_20261107_174607']
ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='*', default=None, help='DQN model zips (default: the 3 final models)')
ap.add_argument('--tests', default='tilt,kick,sustained')
ap.add_argument('--episodes', type=int, default=50, help='tilt test episodes')
ap.add_argument('--kick-mags', default='100,250,500,750,1000', help='impulsive kick sizes (deg/s)')
ap.add_argument('--torque-fracs', default='0.1,0.2,0.3,0.4,0.5', help='sustained torque sizes (x reference torque)')
ap.add_argument('--dur', type=float, default=0.5, help='sustained torque duration (s)')
ap.add_argument('--dirs', type=int, default=24)
ap.add_argument('--band', type=float, default=5.0)
ap.add_argument('--strict-band', type=float, default=2.0)
ap.add_argument('--rate-band', type=float, default=30.0)
ap.add_argument('--hold', type=float, default=0.3)
ap.add_argument('--t-max', type=float, default=2.0)
ap.add_argument('--episode-sec', type=float, default=4.0)
ap.add_argument('--out', default=None, help='output folder (default: results\\eval_v2_<date_time>)')
# fixed sim-to-real condition for every trial (0 = off)
ap.add_argument('--mass-scale', type=float, default=1.0)
ap.add_argument('--motor-delay-ms', type=float, default=0.0)
ap.add_argument('--motor-tau-ms', type=float, default=0.0)
ap.add_argument('--cmd-delay-ms', type=float, default=0.0)
ap.add_argument('--gyro-noise', type=float, default=0.0)
ap.add_argument('--dist-torque', type=float, default=0.0, help='random air disturbance std (x reference torque)')
ap.add_argument('--dist-tau', type=float, default=0.05, help='air disturbance correlation time (s)')
ap.add_argument('--thrust-limit', type=float, default=1.0, help='usable fraction of max motor command')
ap.add_argument('--motor-mismatch', type=float, default=0.0, help='thrust difference between motors (+/- fraction)')
ap.add_argument('--att-bias', type=float, default=0.0, help='attitude bias (+/- deg)')
ap.add_argument('--real-like', action='store_true',
                help='use the condition calibrated from the real flight logs (sysid): delay 10 ms, noise 10, '
                     'air disturbance 0.07 (0.05 s), motor limit 0.7, motor mismatch 0.25')
args = ap.parse_args()
if args.real_like:
    args.motor_delay_ms, args.gyro_noise, args.dist_torque, args.dist_tau = 10.0, 10.0, 0.07, 0.05
    args.thrust_limit, args.motor_mismatch = 0.7, 0.25

models = args.models or [os.path.join(HERE, 'results', m, 'best_model.zip') for m in FINAL]
for m in models:
    if not os.path.exists(m):
        raise FileNotFoundError(m)
TESTS = [t.strip() for t in args.tests.split(',') if t.strip()]
KICKS = [float(x) for x in args.kick_mags.split(',')]
FRACS = [float(x) for x in args.torque_fracs.split(',')]
THETAS = np.arange(args.dirs) * 360.0 / args.dirs
T_DIST = 1.0
COND = dict(motor_delay_ms=args.motor_delay_ms, motor_tau_ms=args.motor_tau_ms,
            cmd_delay_ms=args.cmd_delay_ms, gyro_noise_dps=args.gyro_noise, dist_torque=args.dist_torque,
            thrust_limit=args.thrust_limit, motor_mismatch=args.motor_mismatch, att_bias_deg=args.att_bias)
cond_txt = ('REAL-LIKE: ' if args.real_like else '') + ', '.join([f'{k}={v:g}' for k, v in COND.items() if v and not (k == 'thrust_limit' and v == 1)] + ([f'mass x{args.mass_scale:g}'] if args.mass_scale != 1 else [])) or 'nominal'
out = args.out or os.path.join(HERE, 'results', time.strftime('eval_v2_%Y%m%d_%H%M'))
os.makedirs(out, exist_ok=True)


def make_env(cfg):
    env = AttitudeEnv(**cfg, **COND, dist_tau_s=args.dist_tau, episode_sec=args.episode_sec)
    did, cid = env.aviary.DRONE_IDS[0], env.aviary.CLIENT
    info = p.getDynamicsInfo(did, -1, physicsClientId=cid)
    env._base = (info[0], list(info[2]))
    return env


def reset(env, seed, options=None):
    obs, info = env.reset(seed=seed, options=options)
    if args.mass_scale != 1.0:
        m0, i0 = env._base
        p.changeDynamics(env.aviary.DRONE_IDS[0], -1, mass=m0 * args.mass_scale,
                         localInertiaDiagonal=[i * args.mass_scale for i in i0], physicsClientId=env.aviary.CLIENT)
    return obs, info


def metrics(h, k_ref, tumbled, hz, sign_ref):
    """h: rows [roll, pitch, roll_rate, pitch_rate] per control step. k_ref: index of t = 0."""
    post = h[k_ref:]
    tilt = np.abs(post[:, :2]).max(axis=1)
    rate = np.abs(post[:, 2:4]).max(axis=1)
    H = max(1, int(round(args.hold * hz)))

    def rec_time(band):
        ok = (tilt <= band) & (rate <= args.rate_band)
        for i in range(len(ok) - H + 1):
            if ok[i:i + H].all():
                return i / hz
        return np.nan
    r, rs = rec_time(args.band), rec_time(args.strict_band)
    succ = (not tumbled) and not np.isnan(r) and r <= args.t_max
    proj = post[:, :2] @ sign_ref if sign_ref is not None else None       # + toward the disturbance, - past level
    return dict(success=int(succ), tumbled=int(tumbled), recovery_s=r if succ else np.nan,
                strict_recovery_s=rs if (succ and not np.isnan(rs) and rs <= args.t_max) else np.nan,
                max_tilt=float(tilt.max()),
                overshoot=float(max(0.0, -proj.min())) if proj is not None else np.nan,
                ss_err=float(np.abs(h[-int(0.5 * hz):, :2]).max(axis=1).mean()))


def run(env, model, test, level, theta, seed):
    hz = env.CTRL_HZ
    u = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    if test == 'tilt':
        obs, info = reset(env, seed)
        k_ref, init = 0, np.array([info['roll'], info['pitch']])
        sign_ref = init / (np.linalg.norm(init) + 1e-9)
    else:
        obs, info = reset(env, 0, {'tilt_deg': [0.0, 0.0], 'rate_deg': [0.0, 0.0]})
        k_dist = int(round(T_DIST * hz))
        k_off = k_dist + int(round(args.dur * hz))
        k_ref = k_dist if test == 'kick' else k_off
        sign_ref = u
    hist, sw, prev, k, done, term, max_during = [], 0, None, 0, False, False, 0.0
    while not done:
        if test == 'kick' and k == k_dist:
            env.apply_rate_kick(level * u[0], level * u[1]); obs = env._obs()
        if test == 'sustained':
            if k == k_dist:
                tq = level * env.torque_reference()
                env.set_external_torque(tq * u[0], tq * u[1])
            if k == k_off:
                env.set_external_torque(0.0, 0.0)
        if model is None:
            obs, r, term, trunc, info = env.step_angle_pid()
        else:
            a = int(model.predict(obs, deterministic=True)[0])
            if k >= k_ref and prev is not None and a != prev:
                sw += 1
            prev = a
            obs, r, term, trunc, info = env.step(a)
        hist.append([info['roll'], info['pitch'], info['roll_rate'], info['pitch_rate']])
        if test == 'sustained' and k_dist <= k < k_off:
            max_during = max(max_during, abs(info['roll']), abs(info['pitch']))
        k += 1
        done = term or trunc
    env.set_external_torque(0.0, 0.0)
    h = np.array(hist)
    m = metrics(h, min(k_ref, len(h) - 1), term, hz, sign_ref)
    m['switch_per_s'] = sw / max((len(h) - k_ref) / hz, 1e-9) if model is not None else np.nan
    m['max_tilt_during'] = max_during if test == 'sustained' else np.nan
    return m


from stable_baselines3 import DQN
controllers = [('PID', None, load_env_config(models[0]))] + \
              [(f'DQN_{i}', DQN.load(mp), load_env_config(mp)) for i, mp in enumerate(models)]
print(f'Condition: {cond_txt}.  Tests: {TESTS}.  Models: {[os.path.basename(os.path.dirname(m)) for m in models]}')
rows, t0 = [], time.time()
for name, model, cfg in controllers:
    env = make_env(cfg)
    if 'tilt' in TESTS:
        for i in range(args.episodes):
            rows.append(dict(controller=name, test='tilt', level=0, theta=np.nan, **run(env, model, 'tilt', 0, 0, 1000 + i)))
    for test, levels in (('kick', KICKS), ('sustained', FRACS)):
        if test in TESTS:
            for lv in levels:
                for th in THETAS:
                    rows.append(dict(controller=name, test=test, level=lv, theta=th, **run(env, model, test, lv, th, 0)))
    env.close()
    print(f'  {name} done ({time.time() - t0:.0f}s)', flush=True)

with open(os.path.join(out, 'trials.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---------------- summary: PID vs DQN (mean over models) ----------------
dqn_names = [c[0] for c in controllers if c[0] != 'PID']
KEYS = ['success', 'tumbled', 'recovery_s', 'strict_recovery_s', 'overshoot', 'ss_err', 'max_tilt', 'max_tilt_during', 'switch_per_s']


def stat(names, test, key, level=None):
    vals = []
    for n in names:
        v = np.array([r[key] for r in rows if r['controller'] == n and r['test'] == test and (level is None or r['level'] == level)], float)
        vals.append(np.nanmean(v) if np.any(~np.isnan(v)) else np.nan)
    vals = np.array(vals)
    return np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan


summary = []
for test, levels in (('tilt', [None]), ('kick', [None] + KICKS), ('sustained', [None] + FRACS)):
    if test not in TESTS:
        continue
    for lv in levels:
        for who, names in (('PID', ['PID']), ('DQN', dqn_names)):
            summary.append(dict(test=test, level='all' if lv is None else lv, controller=who,
                                **{k: stat(names, test, k, lv) for k in KEYS}))
with open(os.path.join(out, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)


def fmt(x, d=2, unit=''):
    return '   -  ' if x is None or np.isnan(x) else f'{x:.{d}f}{unit}'


print(f'\nRecovery: tilt <= {args.band:g} deg and rate <= {args.rate_band:g} deg/s held {args.hold:g} s, within {args.t_max:g} s.  '
      f'Condition: {cond_txt}.  DQN = mean of {len(dqn_names)} models.\n')
print(f'{"test":10s} {"level":>6s} {"ctrl":4s} | {"success":>7s} {"flips":>5s} | {"recovery":>8s} {"strict":>7s} | '
      f'{"overshoot":>9s} {"ss err":>6s} {"max tilt":>8s} {"during":>6s} | {"switch/s":>8s}')
for s in summary:
    print(f'{s["test"]:10s} {str(s["level"]):>6s} {s["controller"]:4s} | {fmt(s["success"] * 100, 0, "%"):>7s} {fmt(s["tumbled"] * 100, 0, "%"):>5s} | '
          f'{fmt(s["recovery_s"], 2, "s"):>8s} {fmt(s["strict_recovery_s"], 2, "s"):>7s} | {fmt(s["overshoot"], 1):>9s} '
          f'{fmt(s["ss_err"], 1):>6s} {fmt(s["max_tilt"], 1):>8s} {fmt(s["max_tilt_during"], 1):>6s} | {fmt(s["switch_per_s"], 1):>8s}')

# ---------------- plot ----------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
panels = [t for t in ('kick', 'sustained') if t in TESTS]
if panels:
    fig, ax = plt.subplots(2, len(panels), figsize=(6 * len(panels), 7), squeeze=False)
    for j, test in enumerate(panels):
        lv = KICKS if test == 'kick' else FRACS
        for who, names, col in (('PID (firmware)', ['PID'], 'tab:blue'), ('DQN (mean)', dqn_names, 'tab:red')):
            ax[0, j].plot(lv, [stat(names, test, 'success', l) * 100 for l in lv], 'o-', color=col, label=who)
            ax[1, j].plot(lv, [stat(names, test, 'recovery_s', l) for l in lv], 'o-', color=col, label=who)
        xl = 'kick size (deg/s)' if test == 'kick' else f'torque (x reference, {args.dur:g} s)'
        ax[0, j].set_title(f'{test}: success rate'); ax[0, j].set_ylim(-5, 105); ax[0, j].set_xlabel(xl)
        ax[1, j].set_title(f'{test}: recovery time (successes)'); ax[1, j].set_xlabel(xl); ax[1, j].set_ylabel('s')
        for a in ax[:, j]:
            a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f'Condition: {cond_txt}')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'summary.png'), dpi=120)
print(f'\nOutput: {out}  (trials.csv, summary.csv, summary.png)')
