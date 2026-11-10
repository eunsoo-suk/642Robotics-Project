r"""
Watch a policy in the simulator's 3D view.

    python watch.py --policy pid            # angle PID baseline
    python watch.py --policy zero           # zero rate (do nothing)
    python watch.py --policy random         # random actions
    python watch.py --policy dqn --model results\dqn_xxx\best_model.zip   # trained DQN
Options: --episodes 5  --slow 2.0 (2x slower)
"""
import argparse
import time
import numpy as np
import pybullet as p
from attitude_env import AttitudeEnv, load_env_config

ap = argparse.ArgumentParser()
ap.add_argument('--policy', default='pid', choices=['pid', 'zero', 'random', 'dqn'])
ap.add_argument('--model', default=None)
ap.add_argument('--episodes', type=int, default=5)
ap.add_argument('--slow', type=float, default=1.0)
args = ap.parse_args()

env = AttitudeEnv(gui=True, **load_env_config(args.model))
cid = env.aviary.CLIENT
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
model = None
if args.policy == 'dqn':
    from stable_baselines3 import DQN
    model = DQN.load(args.model)

rng = np.random.default_rng(0)
for ep in range(args.episodes):
    obs, info = env.reset(seed=100 + ep)
    print(f'\n[episode {ep+1}] initial tilt roll={info["roll"]:.1f}°, pitch={info["pitch"]:.1f}°')
    ret, done, text = 0.0, False, -1
    while not done:
        t0 = time.time()
        pos = env.aviary._getDroneStateVector(0)[0:3]
        p.resetDebugVisualizerCamera(0.6, 45, -20, pos.tolist(), physicsClientId=cid)
        if args.policy == 'pid':
            obs, r, term, trunc, info = env.step_angle_pid()
        else:
            if args.policy == 'zero':
                a = env.ZERO_ACTION
            elif args.policy == 'random':
                a = int(rng.integers(env.action_space.n))
            else:
                a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
        ret += r
        done = term or trunc
        msg = f'{args.policy}  roll {info["roll"]:5.1f}  pitch {info["pitch"]:5.1f}  score {ret:6.1f}'
        text = p.addUserDebugText(msg, [pos[0], pos[1], pos[2] + 0.15], [1, 0, 0], 1.2,
                                  replaceItemUniqueId=text if text >= 0 else -1, physicsClientId=cid)
        time.sleep(max(0.0, args.slow / env.CTRL_HZ - (time.time() - t0)))
    print(f'  end: score {ret:.1f}, final roll={info["roll"]:.1f}°, pitch={info["pitch"]:.1f}°' + ('  (failed)' if term else ''))
env.close()