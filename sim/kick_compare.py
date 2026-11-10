r"""
Kick test, DQN vs firmware PID (one case, for the Phase 1 report and video).

Scenario (same as disturbance_test.py):
    0-1 s  level hover
    1 s    sudden rotational kick (adds roll/pitch rate, like a hand flick)
    1-3 s  recovery
Both controllers get exactly the same kick.

Run:
    python kick_compare.py                          # 1000 deg/s, +roll, DQN seed 1
    python kick_compare.py --mag 750 --theta 45     # other size / direction
    python kick_compare.py --no-video               # figure only
Output: results\kick_compare_<time>\
    kick_compare.png   tilt vs time, both controllers
    kick_compare.mp4   side-by-side 3D view + live plot, slow motion
                       (needs: pip install imageio imageio-ffmpeg; otherwise saves kick_compare.gif)
"""
import argparse
import os
import time
import numpy as np
import pybullet as p
from attitude_env import AttitudeEnv, load_env_config

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.path.join(HERE, 'results', 'dqn_25a_sw0.05_prop_s1_20261107_174517', 'best_model.zip'))
ap.add_argument('--mag', type=float, default=1000.0, help='kick size (deg/s)')
ap.add_argument('--theta', type=float, default=0.0, help='kick direction (deg): 0 = +roll, 90 = +pitch')
ap.add_argument('--kick-time', type=float, default=1.0)
ap.add_argument('--band', type=float, default=5.0, help='recovered = tilt stays below this (deg)')
ap.add_argument('--no-video', action='store_true')
ap.add_argument('--slow', type=float, default=10.0, help='video slow-motion factor')
ap.add_argument('--window', type=float, default=0.9, help='video length after the kick (s, sim time)')
ap.add_argument('--cam-yaw', type=float, default=90.0, help='90 = look along the roll axis')
ap.add_argument('--cam-pitch', type=float, default=8.0)
ap.add_argument('--cam-dist', type=float, default=0.18)
args = ap.parse_args()

out = os.path.join(HERE, 'results', time.strftime('kick_compare_%Y%m%d_%H%M'))
os.makedirs(out, exist_ok=True)
FPS = 25
CAP_HZ = FPS * args.slow                     # capture rate in sim time (250 Hz for 10x slow motion)
W, H = 360, 300


def snapshot(env):
    cid, did = env.aviary.CLIENT, env.aviary.DRONE_IDS[0]
    pos, _ = p.getBasePositionAndOrientation(did, physicsClientId=cid)
    view = p.computeViewMatrixFromYawPitchRoll(pos, args.cam_dist, args.cam_yaw, args.cam_pitch, 0, 2, physicsClientId=cid)
    proj = p.computeProjectionMatrixFOV(50, W / H, 0.01, 10, physicsClientId=cid)
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)[2]
    return np.reshape(img, (H, W, 4))[:, :, :3].astype(np.uint8)


def run(model, cfg):
    env = AttitudeEnv(**cfg)
    sim_dt = 1.0 / env.SIM_HZ
    every = max(1, int(round(env.SIM_HZ / CAP_HZ)))
    t0, t1 = args.kick_time - 0.1, args.kick_time + args.window
    frames, k_inner = [], [0]
    orig = env._inner_step

    def inner(rate_cmd):                     # wrap the 500 Hz step to grab video frames
        orig(rate_cmd)
        k_inner[0] += 1
        t = k_inner[0] * sim_dt
        if not args.no_video and t0 <= t <= t1 and k_inner[0] % every == 0:
            _, _, att, _, _ = env._state()
            frames.append((t, snapshot(env), float(np.abs(att).max())))
    env._inner_step = inner

    u = np.array([np.cos(np.radians(args.theta)), np.sin(np.radians(args.theta))])
    kick_step = int(round(args.kick_time * env.CTRL_HZ))
    obs, info = env.reset(seed=0, options={'tilt_deg': [0.0, 0.0], 'rate_deg': [0.0, 0.0]})
    hist, done, k, term = [], False, 0, False
    while not done:
        if k == kick_step:
            env.apply_rate_kick(args.mag * u[0], args.mag * u[1])
            obs = env._obs()
        if model is None:
            obs, r, term, trunc, info = env.step_angle_pid()
        else:
            obs, r, term, trunc, info = env.step(int(model.predict(obs, deterministic=True)[0]))
        hist.append([(k + 1) / env.CTRL_HZ, info['roll'], info['pitch']])
        k += 1
        done = term or trunc
    env.close()
    h = np.array(hist)
    tilt = np.abs(h[kick_step:, 1:]).max(axis=1)          # same metric as disturbance_test.py
    last = int(0.5 * env.CTRL_HZ)
    ok = (not term) and tilt[-last:].max() < args.band
    settle = np.nan
    if ok:
        inside = tilt < args.band
        j = len(inside) - 1
        while j > 0 and inside[j - 1]:
            j -= 1
        settle = (j + 1) / env.CTRL_HZ
    return dict(t=h[:, 0], tilt=np.abs(h[:, 1:]).max(axis=1), max_tilt=float(tilt.max()),
                settle=settle, tumbled=bool(term), frames=frames)


from stable_baselines3 import DQN
res = {'PID (firmware)': run(None, load_env_config(args.model)),
       'DQN (ours)': run(DQN.load(args.model), load_env_config(args.model))}
COL = {'PID (firmware)': 'tab:blue', 'DQN (ours)': 'tab:red'}

print(f'\nKick {args.mag:.0f} deg/s, direction {args.theta:.0f} deg, at t = {args.kick_time:.1f} s')
for n, r in res.items():
    s = 'tumbled' if r['tumbled'] else (f"{r['settle']:.2f} s" if not np.isnan(r['settle']) else 'not recovered')
    print(f'  {n:15s} max tilt {r["max_tilt"]:5.1f} deg   recovery time {s}')

# ---------------- figure ----------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_plot(ax, cursor=None):
    for n, r in res.items():
        ax.plot(r['t'] - args.kick_time, r['tilt'], color=COL[n], lw=2,
                label=f"{n}: back within {args.band:.0f} deg in {r['settle']:.2f} s")
        if not np.isnan(r['settle']):
            ax.axvline(r['settle'], color=COL[n], ls=':', lw=1.2)
    ax.axhspan(0, args.band, color='tab:green', alpha=0.10)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.text(0.005, ax.get_ylim()[1] * 0.92 if cursor is not None else 1, ' kick', fontsize=9)
    ax.set_xlim(-0.1, args.window); ax.set_xlabel('time after kick (s)'); ax.set_ylabel('tilt (deg)')
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    if cursor is not None:
        ax.axvline(cursor, color='gray', lw=1.5)


fig, ax = plt.subplots(figsize=(7.5, 3.6))
draw_plot(ax)
ax.set_title(f'{args.mag:.0f} deg/s kick: tilt after the kick (lower and faster is better)')
fig.tight_layout(); fig.savefig(os.path.join(out, 'kick_compare.png'), dpi=150); plt.close(fig)
print(f'\nSaved {os.path.join(out, "kick_compare.png")}')

# ---------------- video ----------------
if not args.no_video:
    names = list(res)
    n_fr = min(len(res[n]['frames']) for n in names)
    fig = plt.figure(figsize=(9.6, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    axi = [fig.add_subplot(gs[0, i]) for i in range(2)]
    axp = fig.add_subplot(gs[1, :])
    imgs = []
    for a, n in zip(axi, names):
        imgs.append(a.imshow(res[n]['frames'][0][1])); a.axis('off')
        a.axhline(H / 2, color='k', ls='--', lw=0.8, alpha=0.5)   # level reference
    draw_plot(axp)
    cur = axp.axvline(0, color='gray', lw=1.5)
    fig.suptitle(f'Same {args.mag:.0f} deg/s kick at t = 0 s   ({args.slow:.0f}x slow motion)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    video = []
    for i in range(n_fr):
        t = res[names[0]]['frames'][i][0] - args.kick_time
        for a, im, n in zip(axi, imgs, names):
            _, frame, tl = res[n]['frames'][i]
            im.set_data(frame)
            a.set_title(f'{n}   tilt {tl:5.1f} deg', color=COL[n], fontsize=12)
        cur.set_xdata([t, t])
        fig.canvas.draw()
        video.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    video += [video[-1]] * FPS                       # hold the last frame for 1 s
    plt.close(fig)
    try:
        import imageio.v2 as imageio
        path = os.path.join(out, 'kick_compare.mp4')
        imageio.mimsave(path, video, fps=FPS, macro_block_size=8)
    except Exception as e:
        from PIL import Image
        path = os.path.join(out, 'kick_compare.gif')
        ims = [Image.fromarray(v) for v in video[::2]]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(2000 / FPS), loop=0)
        print(f'  (mp4 needs "pip install imageio imageio-ffmpeg"; saved a GIF instead: {e.__class__.__name__})')
    print(f'Saved {path}')
