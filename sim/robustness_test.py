r"""
Robustness check: compare the 3 final DQN models and the firmware PID on drone conditions different from training (no retraining, evaluation only)

Conditions (change one at a time, everything else as in training):
    mass          : 0.8, 0.9, 1.0, 1.1, 1.2x (inertia scaled by the same ratio: same shape, different weight)
    motor delay   : 0, 5, 10, 20 ms (motor commands applied late; affects both PID and DQN)
    command delay : only the DQN rate command arrives late (PC -> radio -> drone delay on the real drone; the firmware PID runs on the drone, so it is not affected)
Tests:
    A. recover from initial tilt : the same 50 starting conditions as evaluate.py (seeds 1000+)
    B. mid-flight kick           : 1000 deg/s, 24 directions (same as disturbance_test.py)

Run:
    python robustness_test.py --models a\best_model.zip b\best_model.zip c\best_model.zip
Output: results\robustness_<date_time>\  (robustness_results.csv, robustness_summary.png)
"""
import argparse
import collections
import csv
import os
import time

import numpy as np
import pybullet as p
from attitude_env import AttitudeEnv, load_env_config

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='+', required=True)
ap.add_argument('--masses', default='0.8,0.9,1.0,1.1,1.2')
ap.add_argument('--delays-ms', default='0,5,10,20')
ap.add_argument('--cmd-delays-ms', default='', help='DQN command delay conditions (e.g. 10,20,30,40). Empty = skip')
ap.add_argument('--episodes', type=int, default=50)
ap.add_argument('--kick', type=float, default=1000.0)
ap.add_argument('--dirs', type=int, default=24)
ap.add_argument('--band', type=float, default=5.0)
args = ap.parse_args()

MASSES = [float(x) for x in args.masses.split(',')]
DELAYS = [float(x) for x in args.delays_ms.split(',')]
# conditions that change one thing at a time: (name, mass scale, delay ms)
CMD_DELAYS = [float(x) for x in args.cmd_delays_ms.split(',') if x.strip()]
CONDS = [(f'mass x{m:g}', m, 0.0, 0.0) for m in MASSES] + [(f'delay {d:g} ms', 1.0, d, 0.0) for d in DELAYS if d > 0] + \
        [(f'cmd delay {d:g} ms', 1.0, 0.0, d) for d in CMD_DELAYS]


class PerturbedEnv(AttitudeEnv):
    """Evaluation env with a mass scale and a motor command delay (everything else as in training)."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.mass_scale, self.delay_steps, self.cmd_delay_steps = 1.0, 0, 0
        did, cid = self.aviary.DRONE_IDS[0], self.aviary.CLIENT
        info = p.getDynamicsInfo(did, -1, physicsClientId=cid)
        self.base_mass, self.base_inertia = info[0], list(info[2])
        self.buf = collections.deque()
        self.cmd_buf = collections.deque()

    def set_condition(self, mass_scale, delay_ms, cmd_delay_ms=0.0):
        self.mass_scale = mass_scale
        self.delay_steps = int(round(delay_ms / 1000.0 * self.SIM_HZ))
        self.cmd_delay_steps = int(round(cmd_delay_ms / 1000.0 * self.SIM_HZ))

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        did, cid = self.aviary.DRONE_IDS[0], self.aviary.CLIENT
        p.changeDynamics(did, -1, mass=self.base_mass * self.mass_scale,
                         localInertiaDiagonal=[i * self.mass_scale for i in self.base_inertia], physicsClientId=cid)
        self.buf.clear()
        self.cmd_buf.clear()
        self._last_cmd = np.zeros(2)
        return out

    def _inner_step(self, rate_cmd):
        pos, quat, att_deg, vel, gyro = self._state()
        if rate_cmd is None:
            rate_cmd = self.ctrl.attitude_to_rate([0.0, 0.0], att_deg)
        else:   # DQN command: reaches the firmware after the radio delay (keeps the previous command until then, 0 at first)
            self.cmd_buf.append(np.asarray(rate_cmd, dtype=float))
            if len(self.cmd_buf) > self.cmd_delay_steps:
                rate_cmd = self.cmd_buf.popleft()
            else:
                rate_cmd = getattr(self, '_last_cmd', np.zeros(2)) if self.cmd_delay_steps else self.cmd_buf.popleft()
            self._last_cmd = rate_cmd
        from cf_controller import cos_tilt
        rpm = self.ctrl.rate_to_rpm([rate_cmd[0], rate_cmd[1], 0.0], gyro,
                                    self.alt.thrust(pos[2], vel[2], cos_tilt(quat)))
        self.buf.append(rpm)
        applied = self.buf.popleft() if len(self.buf) > self.delay_steps else self.buf[0]
        self.aviary.step(np.asarray(applied).reshape(1, 4))


def act(env, model, obs):
    if model is None:
        return env.step_angle_pid()
    return env.step(int(model.predict(obs, deterministic=True)[0]))


def tilt_episode(env, model, seed):
    DT, LAST = 1.0 / env.CTRL_HZ, int(0.5 * env.CTRL_HZ)
    obs, info = env.reset(seed=seed)
    init = np.array([info['roll'], info['pitch']])
    hist, term, done = [], False, False
    while not done:
        obs, r, term, trunc, info = act(env, model, obs)
        hist.append([info['roll'], info['pitch']]); done = term or trunc
    h = np.array(hist); tilt = np.abs(h).max(axis=1); inside = tilt < args.band
    settle = np.nan
    if not term and inside[-1]:
        k = len(inside) - 1
        while k > 0 and inside[k - 1]:
            k -= 1
        settle = (k + 1) * DT
    return dict(success=int((not term) and tilt[-LAST:].max() < args.band), tumbled=int(term), settle_s=settle,
                overshoot=float(np.maximum(-np.sign(init) * h, 0).max()), ss_err=float(tilt[-LAST:].mean()))


def kick_episode(env, model, theta):
    DT, LAST = 1.0 / env.CTRL_HZ, int(0.5 * env.CTRL_HZ)
    kick_step = env.CTRL_HZ   # at t = 1 s
    u = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    obs, info = env.reset(seed=0, options={'tilt_deg': [0.0, 0.0], 'rate_deg': [0.0, 0.0]})
    hist, term, done, k = [], False, False, 0
    while not done:
        if k == kick_step:
            env.apply_rate_kick(args.kick * u[0], args.kick * u[1]); obs = env._obs()
        obs, r, term, trunc, info = act(env, model, obs)
        hist.append([info['roll'], info['pitch']]); k += 1; done = term or trunc
    post = np.array(hist)[kick_step:]; tilt = np.abs(post).max(axis=1)
    success = bool((not term) and len(tilt) >= LAST and tilt[-LAST:].max() < args.band)
    settle = np.nan
    if success:
        inside = tilt < args.band; j = len(inside) - 1
        while j > 0 and inside[j - 1]:
            j -= 1
        settle = (j + 1) * DT
    return dict(success=int(success), tumbled=int(term), settle_s=settle,
                overshoot=float(max(0.0, -(post @ u).min())), ss_err=float(tilt[-LAST:].mean()))


from stable_baselines3 import DQN
controllers = [('PID', None, load_env_config(args.models[0]))] + \
              [(f'DQN_{i}', DQN.load(m), load_env_config(m)) for i, m in enumerate(args.models)]
out = os.path.join(HERE, 'results', time.strftime('robustness_%Y%m%d_%H%M'))
os.makedirs(out, exist_ok=True)
thetas = np.arange(args.dirs) * 360.0 / args.dirs
rows, t0 = [], time.time()
for name, model, cfg in controllers:
    env = PerturbedEnv(**cfg)
    for cname, ms, dl, cdl in CONDS:
        env.set_condition(ms, dl, cdl)
        for i in range(args.episodes):
            rows.append(dict(controller=name, condition=cname, test='tilt', case=i, **tilt_episode(env, model, 1000 + i)))
        for th in thetas:
            rows.append(dict(controller=name, condition=cname, test='kick', case=float(th), **kick_episode(env, model, th)))
        print(f'  {name:6s} {cname:12s} done ({time.time() - t0:.0f}s)', flush=True)
    env.close()

with open(os.path.join(out, 'robustness_results.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def agg(name_filter, cond, test, key):
    v = np.array([r[key] for r in rows if name_filter(r['controller']) and r['condition'] == cond and r['test'] == test], dtype=float)
    return np.nanmean(v) if np.any(~np.isnan(v)) else np.nan


is_pid, is_dqn = (lambda n: n == 'PID'), (lambda n: n != 'PID')
print(f'\nDQN = mean of {len(args.models)} models.  A: {args.episodes} initial tilts,  B: {args.kick:g} deg/s kicks in {args.dirs} directions\n')
print(f'{"condition":<13}| {"A success PID/DQN":>17} | {"A recovery PID/DQN":>18} | {"A ss err PID/DQN":>17} | {"B success PID/DQN":>17} | {"B recovery PID/DQN":>18} | {"flips PID/DQN":>14}')
summary = []
for cname, *_ in CONDS:
    v = [agg(is_pid, cname, 'tilt', 'success') * 100, agg(is_dqn, cname, 'tilt', 'success') * 100,
         agg(is_pid, cname, 'tilt', 'settle_s'), agg(is_dqn, cname, 'tilt', 'settle_s'),
         agg(is_pid, cname, 'tilt', 'ss_err'), agg(is_dqn, cname, 'tilt', 'ss_err'),
         agg(is_pid, cname, 'kick', 'success') * 100, agg(is_dqn, cname, 'kick', 'success') * 100,
         agg(is_pid, cname, 'kick', 'settle_s'), agg(is_dqn, cname, 'kick', 'settle_s'),
         (agg(is_pid, cname, 'tilt', 'tumbled') + agg(is_pid, cname, 'kick', 'tumbled')) * 50,
         (agg(is_dqn, cname, 'tilt', 'tumbled') + agg(is_dqn, cname, 'kick', 'tumbled')) * 50]
    summary.append((cname, v))
    print(f'{cname:<13}| {v[0]:7.0f}% {v[1]:7.0f}% | {v[2]:6.2f}s {v[3]:6.2f}s | {v[4]:8.1f}° {v[5]:7.1f}° | '
          f'{v[6]:7.0f}% {v[7]:7.0f}% | {v[8]:6.2f}s {v[9]:6.2f}s | {v[10]:5.0f}% {v[11]:6.0f}%')
print('\nSuccess rate per model (A / B):')
for name, _, _ in controllers[1:]:
    print('  ' + name + ': ' + ',  '.join(f"{c}: {agg(lambda n: n == name, c, 'tilt', 'success') * 100:.0f}/{agg(lambda n: n == name, c, 'kick', 'success') * 100:.0f}%" for c, *_ in CONDS))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
labels = [s[0] for s in summary]; x = np.arange(len(labels)); V = np.array([s[1] for s in summary])
for a, (ip, idq, lab) in zip(ax, [(0, 1, 'A: success rate (%)'), (8, 9, f'B: recovery time after {args.kick:g} deg/s kick (s)'), (4, 5, 'A: residual tilt (deg)')]):
    a.bar(x - 0.2, V[:, ip], 0.4, label='PID (firmware)', color='tab:blue')
    a.bar(x + 0.2, V[:, idq], 0.4, label=f'DQN ({len(args.models)}-model mean)', color='tab:red')
    a.set_xticks(x); a.set_xticklabels(labels, rotation=35, ha='right', fontsize=8); a.set_title(lab, fontsize=10); a.grid(alpha=0.3, axis='y')
ax[0].legend(fontsize=8); fig.suptitle('Robustness to mass change and motor delay (no retraining)')
fig.tight_layout(); fig.savefig(os.path.join(out, 'robustness_summary.png'), dpi=120)
print(f'\nOutput folder: {out}')
