r"""
Step 3: DQN training (stable-baselines3)

    python train_dqn.py --steps 100000          # short check (does the score go up?)
    python train_dqn.py --steps 500000          # full training

Output: results\dqn_<date_time>\
    best_model.zip        model with the best evaluation score
    final_model.zip       last model
    checkpoints\          intermediate models (to compare training progress, replay with watch.py)
    evaluations.npz       periodic evaluation log
    tb\                   TensorBoard logs (if tensorboard is installed)

Settings follow Mnih et al. 2015 (replay buffer, target network, epsilon decay, Huber loss, gamma 0.99),
          Hwangbo et al. 2017 (MLP 64x2)
"""
import argparse
import importlib.util
import json
import os
import time

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from attitude_env import AttitudeEnv

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=100_000, help='total training steps (1 step = 0.02 s of flight)')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--eval-freq', type=int, default=10_000)
ap.add_argument('--eval-episodes', type=int, default=20)
ap.add_argument('--out', default=None, help='output folder (default: results\\dqn_<n_actions>a_<date_time>)')
ap.add_argument('--levels', default='-90,0,90',
                help='rate levels per axis (deg/s). e.g. -90,-30,0,30,90 -> 25 actions')
ap.add_argument('--w-switch', type=float, default=0.05, help='command-change penalty weight (reduces jitter)')
ap.add_argument('--switch-mode', default='binary', choices=['binary', 'prop'],
                help="command-change penalty: binary (fixed penalty on any change) / prop (proportional to the change)")
args = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
levels = [float(x) for x in args.levels.split(',')]
n_act = len(levels) ** 2
out = args.out or os.path.join(here, 'results', time.strftime(f'dqn_{n_act}a_sw{args.w_switch:g}' + ('_prop' if args.switch_mode == 'prop' else '') + f'_s{args.seed}_%Y%m%d_%H%M%S'))
os.makedirs(out, exist_ok=True)
env_cfg = {'rate_levels': levels, 'tilt_norm_deg': 60.0, 'w_switch': args.w_switch, 'switch_mode': args.switch_mode, 'gyro_lpf_hz': 80.0}   # save the env settings used in training
with open(os.path.join(out, 'env_config.json'), 'w', encoding='utf-8') as f:
    json.dump(env_cfg, f, indent=2)   # used by evaluate.py / watch.py to build the env with the same actions
has_tb = importlib.util.find_spec('tensorboard') is not None

env = Monitor(AttitudeEnv(**env_cfg), filename=os.path.join(out, 'train_monitor'))
eval_env = Monitor(AttitudeEnv(**env_cfg))
eval_env.reset(seed=12345)   # fix the evaluation initial conditions so runs can be compared

model = DQN(
    'MlpPolicy', env,
    policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=5_000,          # collect random experience for the first 5k steps
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,   # target network update interval
    exploration_fraction=0.3,       # epsilon 1.0 -> 0.05 over the first 30% of training
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10,
    tensorboard_log=os.path.join(out, 'tb') if has_tb else None,
    seed=args.seed,
    verbose=0,
)

callbacks = [
    EvalCallback(eval_env, n_eval_episodes=args.eval_episodes, eval_freq=args.eval_freq,
                 best_model_save_path=out, log_path=out, deterministic=True, verbose=1),
    CheckpointCallback(save_freq=max(args.steps // 5, 1), save_path=os.path.join(out, 'checkpoints'),
                       name_prefix='dqn'),
]

print(f'Output folder: {out}')
print(f'Actions: {levels} deg/s per axis -> {n_act},  command-change penalty {args.w_switch} ({args.switch_mode}),  seed {args.seed}')
print(f'Training steps: {args.steps:,}  (about {args.steps / 50 / 60:.0f} min of flight time)')
print('Reference scores: zero-rate ~ -38,  PID ~ -3.4   (closer to 0 is better)')
if has_tb:
    print(f'Training curve (in another PowerShell):  tensorboard --logdir "{os.path.join(out, "tb")}"')
else:
    print('TensorBoard not installed: evaluation scores are shown in the log below only (pip install tensorboard)')
print()

t0 = time.time()
model.learn(total_timesteps=args.steps, callback=callbacks, log_interval=10)
model.save(os.path.join(out, 'final_model'))
print(f'\nDone: {(time.time() - t0) / 60:.1f} min')
print(f'Evaluate:   python evaluate.py --model "{os.path.join(out, "best_model.zip")}"')
print(f'Replay:     python watch.py --policy dqn --model "{os.path.join(out, "best_model.zip")}"')
env.close(); eval_env.close()