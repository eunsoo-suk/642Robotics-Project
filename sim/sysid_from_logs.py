r"""
Priority 2: calibrate the simulator with the real flight logs (system identification), then validate it.

1) Features of the REAL drone hovering under the firmware PID (from a hw_step4 flight log):
   gyro rate RMS (deg/s), tilt RMS (deg), and the main oscillation frequency (Hz).
2) The SAME firmware PID hovers in the simulator for every combination of
   motor delay x motor time constant x gyro noise x air disturbance x battery headroom (grid below). The same features are computed
   from the simulated gyro (after the firmware 80 Hz filter), sampled at 50 Hz like the log.
3) The combinations closest to the real drone are ranked (score = log error of RMS values + frequency error).
4) Validation: the old DQN (trained without delay/noise) flies in the best calibrated simulators.
   If the simulator is right, it should fail the way the real flight did
   (real: oscillation grew and the drone flipped after about 4.6 s, about 24 command changes per second).

Run (.venv-sim, in the sim folder):
    python sysid_from_logs.py
    python sysid_from_logs.py --delays 0,10,15,20,25 --taus 0,10,20,30,40 --noises 0,10,20,30
Output: results\sysid_<date_time>\  (grid.csv, best.json, spectrum.png)
"""
import argparse
import csv
import json
import os
import time
import numpy as np
from attitude_env import AttitudeEnv, load_env_config

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--log', default=os.path.join(HERE, 'results', 'hw_step4_20261109_031721.csv'))
ap.add_argument('--pid-window', default='5,11.7', help='time window (s) of the airborne PID hover in the log')
ap.add_argument('--dqn-window', default='11.82,16.4', help='time window (s) of the DQN flight before the flip')
ap.add_argument('--delays', default='0,5,10,15', help='motor delay grid (ms)')
ap.add_argument('--taus', default='0,10,20', help='motor time-constant grid (ms)')
ap.add_argument('--noises', default='10,20,30', help='raw gyro noise grid (deg/s)')
ap.add_argument('--dists', default='0,0.02,0.03,0.05,0.07', help='random air disturbance grid (x reference torque)')
ap.add_argument('--limits', default='1.0,0.8,0.7', help='usable motor command grid (battery sag)')
ap.add_argument('--mismatches', default='0.05', help='motor thrust mismatch grid (+/- fraction)')
ap.add_argument('--dist-taus', default='0.3', help='air disturbance correlation time grid (s); shorter = faster gusts')
ap.add_argument('--val-sec', type=float, default=10.0, help='validation flight length (s)')
ap.add_argument('--sim-sec', type=float, default=8.0)
ap.add_argument('--top', type=int, default=5)
ap.add_argument('--model', default=os.path.join(HERE, 'results', 'dqn_25a_sw0.05_prop_s1_20261107_174517', 'best_model.zip'),
                help='old DQN used for the validation step')
ap.add_argument('--val-runs', type=int, default=5)
args = ap.parse_args()
out = os.path.join(HERE, 'results', time.strftime('sysid_%Y%m%d_%H%M'))
os.makedirs(out, exist_ok=True)
HZ = 50.0


def features(rr, pr, roll, pitch):
    """RMS of rates and tilt (axis mean) and the main frequency of the rate signal (both axes)."""
    rr, pr = np.asarray(rr) - np.mean(rr), np.asarray(pr) - np.mean(pr)
    f = np.fft.rfftfreq(len(rr), 1 / HZ)
    spec = np.abs(np.fft.rfft(rr)) ** 2 + np.abs(np.fft.rfft(pr)) ** 2
    band = (f > 0.5) & (f < 20)
    return dict(rate_rms=float(np.sqrt((np.mean(rr ** 2) + np.mean(pr ** 2)) / 2)),
                tilt_rms=float(np.sqrt((np.var(roll) + np.var(pitch)) / 2)),
                freq=float(f[band][np.argmax(spec[band])]) if band.any() else np.nan), (f, spec)


# ---------------- 1) real drone ----------------
rows = list(csv.DictReader(open(args.log, encoding='utf-8')))
a, b = [float(x) for x in args.pid_window.split(',')]
seg = [r for r in rows if r['mode'] == 'PID' and a <= float(r['t']) <= b]
col = lambda s, k: [float(r[k]) for r in s]
real, real_spec = features(col(seg, 'roll_rate'), col(seg, 'pitch_rate'), col(seg, 'roll'), col(seg, 'pitch'))
a, b = [float(x) for x in args.dqn_window.split(',')]
dseg = [r for r in rows if r['mode'] == 'DQN' and a <= float(r['t']) <= b]
real_dqn, _ = features(col(dseg, 'roll_rate'), col(dseg, 'pitch_rate'), col(dseg, 'roll'), col(dseg, 'pitch'))
sw = sum((x['cmd_roll'], x['cmd_pitch']) != (y['cmd_roll'], y['cmd_pitch']) for x, y in zip(dseg, dseg[1:]))
real_dqn['switch_per_s'] = sw / max(len(dseg) / HZ, 1e-9)
real_dqn['flip_after_s'] = len(dseg) / HZ
print(f'Real PID hover : rate RMS {real["rate_rms"]:.1f} deg/s, tilt RMS {real["tilt_rms"]:.2f} deg, main frequency {real["freq"]:.1f} Hz')
print(f'Real DQN flight: rate RMS {real_dqn["rate_rms"]:.1f} deg/s, {real_dqn["switch_per_s"]:.0f} command changes/s, flipped after {real_dqn["flip_after_s"]:.1f} s\n')

cfg = load_env_config(args.model)


def sim_hover(model, delay, tau, noise, seed, sec, dist=0.0, limit=1.0, mismatch=0.0, dist_tau=0.3):
    env = AttitudeEnv(**cfg, motor_delay_ms=delay, motor_tau_ms=tau, gyro_noise_dps=noise, episode_sec=sec,
                      dist_torque=dist, thrust_limit=limit, motor_mismatch=mismatch, dist_tau_s=dist_tau)
    obs, info = env.reset(seed=seed, options={'tilt_deg': [0.0, 0.0], 'rate_deg': [0.0, 0.0]})
    rec, sw, prev, term, k = [], 0, None, False, 0
    while True:
        if model is None:
            obs, r, term, trunc, info = env.step_angle_pid()
        else:
            act = int(model.predict(obs, deterministic=True)[0]); sw += int(prev is not None and act != prev); prev = act
            obs, r, term, trunc, info = env.step(act)
        g = env.ctrl.gyro_f if env.ctrl.gyro_f is not None else np.array([info['roll_rate'], info['pitch_rate'], 0])
        rec.append([g[0], g[1], info['roll'], info['pitch']])
        k += 1
        if term or trunc:
            break
    env.close()
    return np.array(rec), term, sw / (k / HZ)


# ---------------- 2) grid search with the PID ----------------
D = [float(x) for x in args.delays.split(',')]
T = [float(x) for x in args.taus.split(',')]
N = [float(x) for x in args.noises.split(',')]
DI = [float(x) for x in args.dists.split(',')]
LI = [float(x) for x in args.limits.split(',')]
MI = [float(x) for x in args.mismatches.split(',')]
DT = [float(x) for x in args.dist_taus.split(',')]
grid, t0 = [], time.time()
for d in D:
    for tau in T:
        for n in N:
          for di in DI:
            for li in LI:
             for mi in MI:
              for dtau in DT:
                h, term, _ = sim_hover(None, d, tau, n, 0, args.sim_sec, di, li, mi, dtau)
                h = h[int(HZ):]                                     # drop the first second
                f, _ = features(h[:, 0], h[:, 1], h[:, 2], h[:, 3])
                score = abs(np.log((f['rate_rms'] + 1) / (real['rate_rms'] + 1))) + \
                        abs(np.log((f['tilt_rms'] + 0.1) / (real['tilt_rms'] + 0.1))) + \
                        abs(f['freq'] - real['freq']) / max(real['freq'], 0.5)
                grid.append(dict(motor_delay_ms=d, motor_tau_ms=tau, gyro_noise_dps=n, dist_torque=di, thrust_limit=li,
                                 motor_mismatch=mi, dist_tau_s=dtau,
                                 pid_tumbled=int(term), score=score, **f))
    print(f'  delay {d:g} ms done ({time.time() - t0:.0f}s)', flush=True)
with open(os.path.join(out, 'grid.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(grid[0].keys())); w.writeheader(); w.writerows(grid)
best = sorted([g for g in grid if not g['pid_tumbled']], key=lambda g: g['score'])[:args.top]
print(f'\nClosest simulators to the real PID hover (real: rate {real["rate_rms"]:.1f}, tilt {real["tilt_rms"]:.2f}, {real["freq"]:.1f} Hz):')
print(f'{"delay ms":>8} {"tau ms":>6} {"noise":>5} {"dist":>5} {"d_tau":>5} {"limit":>5} {"mism":>5} | {"rate RMS":>8} {"tilt RMS":>8} {"freq Hz":>7} | {"score":>5}')
for g in best:
    print(f'{g["motor_delay_ms"]:8g} {g["motor_tau_ms"]:6g} {g["gyro_noise_dps"]:5g} {g["dist_torque"]:5g} {g["dist_tau_s"]:5g} {g["thrust_limit"]:5g} {g["motor_mismatch"]:5g} | '
          f'{g["rate_rms"]:8.1f} {g["tilt_rms"]:8.2f} {g["freq"]:7.1f} | {g["score"]:5.2f}')

# ---------------- 3) validation with the old DQN ----------------
from stable_baselines3 import DQN
model = DQN.load(args.model)
print(f'\nValidation: old DQN ({os.path.basename(os.path.dirname(args.model))}) hovering {args.val_sec:g} s in each calibrated simulator')
print(f'(real flight: rate RMS {real_dqn["rate_rms"]:.0f} deg/s, {real_dqn["switch_per_s"]:.0f} changes/s, flipped after {real_dqn["flip_after_s"]:.1f} s)')
val = []
for g in best + [dict(motor_delay_ms=0, motor_tau_ms=0, gyro_noise_dps=0, dist_torque=0, thrust_limit=1.0,
                     motor_mismatch=0.0, dist_tau_s=0.3, label='nominal')]:
    flips, times, rms, sws = 0, [], [], []
    for s in range(args.val_runs):
        h, term, swr = sim_hover(model, g['motor_delay_ms'], g['motor_tau_ms'], g['gyro_noise_dps'], s, args.val_sec,
                                 g['dist_torque'], g['thrust_limit'], g['motor_mismatch'], g['dist_tau_s'])
        flips += int(term); times.append(len(h) / HZ); sws.append(swr)
        f, _ = features(h[int(HZ):, 0], h[int(HZ):, 1], h[int(HZ):, 2], h[int(HZ):, 3]) if len(h) > 2 * HZ else ({'rate_rms': np.nan}, None)
        rms.append(f['rate_rms'])
    v = dict(motor_delay_ms=g['motor_delay_ms'], motor_tau_ms=g['motor_tau_ms'], gyro_noise_dps=g['gyro_noise_dps'],
             dist_torque=g['dist_torque'], thrust_limit=g['thrust_limit'], motor_mismatch=g['motor_mismatch'], dist_tau_s=g['dist_tau_s'],
             dqn_flips=flips, runs=args.val_runs, mean_flight_s=float(np.mean(times)),
             dqn_rate_rms=float(np.nanmean(rms)), dqn_switch_per_s=float(np.mean(sws)))
    val.append(v)
    tag = ' (nominal sim)' if g.get('label') else ''
    print(f'  delay {v["motor_delay_ms"]:4g} tau {v["motor_tau_ms"]:4g} noise {v["gyro_noise_dps"]:4g} dist {v["dist_torque"]:4g}/{v["dist_tau_s"]:g}s limit {v["thrust_limit"]:4g} mism {v["motor_mismatch"]:4g}{tag}: '
          f'flipped {flips}/{args.val_runs}, mean flight {v["mean_flight_s"]:.1f} s, rate RMS {v["dqn_rate_rms"]:.0f}, '
          f'{v["dqn_switch_per_s"]:.0f} changes/s')
json.dump(dict(real_pid=real, real_dqn=real_dqn, best=best, validation=val), open(os.path.join(out, 'best.json'), 'w'), indent=2)

# ---------------- spectrum plot: real vs best simulator ----------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
g = best[0]
h, _, _ = sim_hover(None, g['motor_delay_ms'], g['motor_tau_ms'], g['gyro_noise_dps'], 0, args.sim_sec,
                    g['dist_torque'], g['thrust_limit'], g['motor_mismatch'], g['dist_tau_s'])
_, (fs, ss) = features(h[int(HZ):, 0], h[int(HZ):, 1], h[int(HZ):, 2], h[int(HZ):, 3])
hn, _, _ = sim_hover(None, 0, 0, 0, 0, args.sim_sec)
_, (fn, sn) = features(hn[int(HZ):, 0], hn[int(HZ):, 1], hn[int(HZ):, 2], hn[int(HZ):, 3])
fr, sr = real_spec
plt.figure(figsize=(8, 4))
plt.semilogy(fr, sr / len(fr), label='real drone (PID hover)')
plt.semilogy(fs, ss / len(fs), label=f'calibrated sim (delay {g["motor_delay_ms"]:g} ms, tau {g["motor_tau_ms"]:g} ms, noise {g["gyro_noise_dps"]:g}, dist {g["dist_torque"]:g}, limit {g["thrust_limit"]:g})')
plt.semilogy(fn, sn / len(fn) + 1e-3, label='original sim (no delay, no noise)', alpha=0.7)
plt.ylim(1e-1, None)
plt.xlabel('frequency (Hz)'); plt.ylabel('gyro power (roll + pitch)'); plt.xlim(0, 25); plt.grid(alpha=0.3); plt.legend(fontsize=8)
plt.title('Gyro spectrum while hovering under the firmware PID')
plt.tight_layout(); plt.savefig(os.path.join(out, 'spectrum.png'), dpi=130)
print(f'\nOutput: {out}  (grid.csv, best.json, spectrum.png)')
