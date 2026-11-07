r"""
3단계: DQN 학습 (stable-baselines3)

    python train_dqn.py --steps 100000          # 짧은 확인용 (점수가 올라가는지)
    python train_dqn.py --steps 500000          # 본 학습

결과: results\dqn_<날짜시간>\
    best_model.zip        평가 점수가 가장 좋았던 모델
    final_model.zip       마지막 모델
    checkpoints\          중간 모델 (학습 과정 비교용, watch.py로 재생)
    evaluations.npz       주기적 평가 기록
    tb\                   TensorBoard 로그 (tensorboard 설치 시)

설정 근거: Mnih et al. 2015 (replay buffer, target network, epsilon 감소, Huber loss, gamma 0.99),
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
ap.add_argument('--steps', type=int, default=100_000, help='총 학습 스텝 (1스텝 = 0.02초 비행)')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--eval-freq', type=int, default=10_000)
ap.add_argument('--eval-episodes', type=int, default=20)
ap.add_argument('--out', default=None, help='결과 폴더 (기본: results\\dqn_<행동수>a_날짜시간)')
ap.add_argument('--levels', default='-90,0,90',
                help='축별 회전속도 단계(deg/s). 예: -90,-30,0,30,90 -> 25개 행동')
ap.add_argument('--w-switch', type=float, default=0.05, help='명령 변경 감점 가중치 (떨림 억제)')
args = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
levels = [float(x) for x in args.levels.split(',')]
n_act = len(levels) ** 2
out = args.out or os.path.join(here, 'results', time.strftime(f'dqn_{n_act}a_sw{args.w_switch:g}_s{args.seed}_%Y%m%d_%H%M%S'))
os.makedirs(out, exist_ok=True)
env_cfg = {'rate_levels': levels, 'tilt_norm_deg': 60.0, 'w_switch': args.w_switch, 'gyro_lpf_hz': 80.0}   # 학습 때 쓴 환경 설정을 함께 저장
with open(os.path.join(out, 'env_config.json'), 'w', encoding='utf-8') as f:
    json.dump(env_cfg, f, indent=2)   # evaluate.py / watch.py가 같은 행동 설정으로 환경을 만들 때 사용
has_tb = importlib.util.find_spec('tensorboard') is not None

env = Monitor(AttitudeEnv(**env_cfg), filename=os.path.join(out, 'train_monitor'))
eval_env = Monitor(AttitudeEnv(**env_cfg))
eval_env.reset(seed=12345)   # 학습 중 평가 초기 조건을 모든 실행에서 동일하게 고정 (실행 간 비교 가능)

model = DQN(
    'MlpPolicy', env,
    policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=5_000,          # 처음 5천 스텝은 무작위로 경험만 모음
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,   # 타깃 네트워크 동기화 주기
    exploration_fraction=0.3,       # 전체의 30% 동안 epsilon 1.0 -> 0.05
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

print(f'결과 폴더: {out}')
print(f'행동: 축별 {levels} deg/s -> {n_act}개,  명령 변경 감점 {args.w_switch},  seed {args.seed}')
print(f'학습 스텝: {args.steps:,}  (비행 시간으로 약 {args.steps / 50 / 60:.0f}분 분량)')
print('기준점: 가만히 ≈ -38,  PID ≈ -3.4   (0에 가까울수록 좋음)')
if has_tb:
    print(f'학습 곡선 보기 (다른 PowerShell):  tensorboard --logdir "{os.path.join(out, "tb")}"')
else:
    print('TensorBoard 미설치: 평가 점수는 아래 로그로만 표시됩니다 (pip install tensorboard 로 설치 가능)')
print()

t0 = time.time()
model.learn(total_timesteps=args.steps, callback=callbacks, log_interval=10)
model.save(os.path.join(out, 'final_model'))
print(f'\n완료: {(time.time() - t0) / 60:.1f}분 소요')
print(f'평가:   python evaluate.py --model "{os.path.join(out, "best_model.zip")}"')
print(f'재생:   python watch.py --policy dqn --model "{os.path.join(out, "best_model.zip")}"')
env.close(); eval_env.close()