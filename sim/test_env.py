"""
AttitudeEnv 동작 확인 (1단계 완료 조건).
실행: .venv-sim 활성화 후  python sim\test_env.py
"""
import numpy as np
from stable_baselines3.common.env_checker import check_env
from attitude_env import AttitudeEnv

N_EP = 20


def run(env, policy, name):
    rets, lens, final_tilt, fails = [], [], [], 0
    rng = np.random.default_rng(0)
    for ep in range(N_EP):
        obs, info = env.reset(seed=ep)
        ret, done = 0.0, False
        while not done:
            if policy == 'pid':
                obs, r, term, trunc, info = env.step_angle_pid()
            else:
                a = env.ZERO_ACTION if policy == 'zero' else int(rng.integers(env.action_space.n))
                obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc
        rets.append(ret); lens.append(env.steps)
        final_tilt.append(max(abs(info['roll']), abs(info['pitch'])))
        fails += int(term)
    print(f'{name:<22} 평균보상 {np.mean(rets):7.1f}   평균길이 {np.mean(lens):5.1f}/{env.MAX_STEPS}   '
          f'최종기울기 {np.mean(final_tilt):5.1f}°   실패 {fails}/{N_EP}')


if __name__ == '__main__':
    env = AttitudeEnv()
    print('[1] Gym 규격 검사 (check_env)')
    check_env(env, warn=True)
    print('    통과\n')
    print(f'행동 수: {env.action_space.n},  관측 차원: {env.observation_space.shape},  '
          f'에피소드: {env.MAX_STEPS}스텝 ({env.CTRL_HZ}Hz)\n')
    print('[2~4] 정책별 비교 (각 20 에피소드, 같은 초기 조건)')
    run(env, 'zero',   '[2] 회전속도 0 (가만히)')
    run(env, 'random', '[3] 무작위')
    run(env, 'pid',    '[4] angle PID (베이스라인)')
    env.close()
    print('\n기대: [2] 기울기 유지(복귀 못 함)  [3] 실패 많음  [4] 최종기울기 ~0°, 보상 가장 높음')