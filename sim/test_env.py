"""
Check that AttitudeEnv works (pass condition for step 1).
Run: activate .venv-sim, then  python sim\test_env.py
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
    print(f'{name:<22} mean return {np.mean(rets):7.1f}   mean length {np.mean(lens):5.1f}/{env.MAX_STEPS}   '
          f'final tilt {np.mean(final_tilt):5.1f}°   failures {fails}/{N_EP}')


if __name__ == '__main__':
    env = AttitudeEnv()
    print('[1] Gym API check (check_env)')
    check_env(env, warn=True)
    print('    passed\n')
    print(f'Actions: {env.action_space.n},  observation shape: {env.observation_space.shape},  '
          f'episode: {env.MAX_STEPS} steps ({env.CTRL_HZ} Hz)\n')
    print('[2-4] Compare policies (20 episodes each, same initial conditions)')
    run(env, 'zero',   '[2] zero rate (do nothing)')
    run(env, 'random', '[3] random')
    run(env, 'pid',    '[4] angle PID (baseline)')
    env.close()
    print('\nExpected: [2] tilt stays (no recovery)  [3] many failures  [4] final tilt ~0°, highest return')