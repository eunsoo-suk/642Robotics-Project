# Learning to Stay Level: DQN-Based Attitude Recovery for the Crazyflie 2.1

MSML642 Robotics course project. Team: Eunsoo Suk, Fnu Hardik, Abigail Basener.

When a small drone is pushed or bumped, it tilts and can flip over. This project trains a
**deep Q-network (DQN)** that brings a Crazyflie 2.1 back to level (roll = pitch = 0) as fast as
possible, using only the onboard IMU, and compares it with the default PID controller in the
Crazyflie firmware. The policy is trained in simulation and then moved to the real drone step by step.

## How it works

The DQN does not drive the motors directly. Every 0.02 s (50 Hz) it picks a target rotation speed
for roll and pitch. The Crazyflie's own rate controller (500 Hz) turns that target into motor
commands. This keeps the learned part small and keeps the firmware's safety checks active, so the
same policy runs in simulation and on the real drone.

```
IMU -> state (6 numbers) -> DQN policy (50 Hz, learned) -> rate PID (500 Hz, firmware) -> mixer -> 4 motors
 ^                                                                                                   |
 +---------------------------------------- new attitude --------------------------------------------+
```

| RL element | Design |
|---|---|
| State (6) | roll, pitch, roll rate, pitch rate, previous command (roll, pitch), all normalized |
| Action (25) | target rate per axis from {-90, -30, 0, +30, +90} deg/s, for roll x pitch |
| Reward (per step, in [-1, 0]) | minus tilt, minus rotation speed, minus a penalty proportional to the command change |
| Failure | the firmware's tumble rule: tilted over 60 deg for 1 s, or upside down for 0.1 s |
| Training | random initial tilt (+/-30 deg) and rate (+/-100 deg/s), 500k steps, 3 seeds |

To make the simulator behave like the real drone, the firmware rate PID is rewritten in Python
(`cf_controller.py`) with the gains read from our drone (kp 250, ki 500, kd 2.5), and the
firmware's 80 Hz gyro low-pass filter is added.

## Results (simulation, mean of 3 DQN seeds)

| Test | Metric | DQN | Firmware PID |
|---|---|---|---|
| Start from a random tilt (50 cases) | Success rate | 100% | 100% |
| | Recovery time | **0.18 s** | 0.20 s |
| | Overshoot | **0.6 deg** | 1.3 deg |
| | Leftover tilt | 0.9 deg | **0.5 deg** |
| | Command changes | 1.8 / s | - |
| Kick while hovering, 100-1000 deg/s, 24 directions (120 trials) | Success rate | 100% (no flips) | 100% |
| | Mean recovery time | **0.09 s** | 0.14 s |
| | Recovery time at 1000 deg/s | **0.20 s** | 0.43 s |
| | Overshoot | **1.4 deg** | 2.1 deg |
| | Command changes | 9.7 / s | - |

Robustness (1000 deg/s kicks): with mass +/-20% both stay at 100% (20% heavier: PID 0.62 s, DQN
0.23 s). DQN command delay is fine up to 30 ms. With 20 ms motor lag the DQN drops to about 60%
while the PID stays at 100%; this is the main weakness and the focus of the next step.

Side-by-side video of one kick (DQN vs PID, 10x slow motion):
https://drive.google.com/file/d/1uuz_r76RBcZPLYME3w4G3TXwoDv6CSdp/view?usp=sharing

## Repository layout

```
sim/
  attitude_env.py           Gymnasium environment (DQN -> firmware rate PID -> PyBullet CF2X)
  cf_controller.py          Python copy of the firmware PID (angle + rate), mixer, gyro filter
  cf_gains.json             PID gains read from the real drone (read_gains.py)
  train_dqn.py              DQN training (stable-baselines3)
  evaluate.py               DQN vs PID from 50 random initial tilts
  disturbance_test.py       mid-flight kick test, 9 sizes x 24 directions
  robustness_test.py        mass change, motor delay, command delay
  kick_compare.py           one kick, DQN vs PID: figure + side-by-side video
  watch.py                  watch a policy in the PyBullet 3D view
  test_env.py               environment sanity check
  read_gains.py             read PID gains from the real drone
  hw_step1_sensor_check.py  real drone step 1: sensor sign check (no propellers)
  hw_step2_dry_run.py       real drone step 2: live DQN loop, commands not sent (no propellers)
  hw_step3_cmd_check.py     real drone step 3: command sign check (thrust 0)
  hw_step4_tether_flight.py real drone step 4: flight, PID take-off then switch to DQN
  hw_motor_test.py          spin each motor directly (hardware check)
  hw_signs.json             sign conventions found in steps 1 and 3
  diag*.py                  debugging scripts used during development
  results/                  trained models, evaluation results, flight logs
gym-pybullet-drones/        simulator (utiasDSL/gym-pybullet-drones)
```

## Setup

Python 3.12 on Windows was used. From the repository root:

```powershell
python -m venv .venv-sim
.\.venv-sim\Scripts\Activate.ps1
pip install -e gym-pybullet-drones
pip install stable-baselines3 torch gymnasium pybullet matplotlib cflib
pip install imageio imageio-ffmpeg     # only for the kick_compare.py video
cd sim
```

For the real drone: Crazyflie 2.1, Crazyradio PA, and cflib. The default radio address is
`radio://0/80/2M/E7E7E7E7E7`.

## Usage (run inside `sim/`)

Simulation:

```powershell
python test_env.py                                     # environment check
python train_dqn.py --levels=-90,-30,0,30,90 --switch-mode prop --steps 500000 --seed 0
python evaluate.py --model results\<run>\best_model.zip
python disturbance_test.py --models a\best_model.zip b\best_model.zip c\best_model.zip
python robustness_test.py --models a\best_model.zip b\best_model.zip c\best_model.zip --cmd-delays-ms 10,20,30,40
python kick_compare.py                                 # 1000 deg/s kick, figure + mp4
python watch.py --policy dqn --model results\<run>\best_model.zip
```

Note: `--levels` needs the `=` sign because the value starts with a minus. The final models are:

```
results\dqn_25a_sw0.05_prop_s0_20261107_142302\best_model.zip
results\dqn_25a_sw0.05_prop_s1_20261107_174517\best_model.zip   (default for the real-drone scripts)
results\dqn_25a_sw0.05_prop_s2_20261107_174607\best_model.zip
```

Real drone (in this order; steps 1-3 run without propellers):

```powershell
python read_gains.py                 # once: save the drone's PID gains to cf_gains.json
python hw_step1_sensor_check.py      # sensor signs -> hw_signs.json
python hw_step2_dry_run.py           # live DQN loop, nothing sent
python hw_step3_cmd_check.py         # command signs -> hw_signs.json
python hw_step4_tether_flight.py --no-auto-stop --h-thrust 34000 --thrust-max 50000
```

Step 4 keys: Space = emergency stop, W/S = thrust up/down, H = jump to take-off thrust,
T = take-off mode (rate mode, zero rate command), D = DQN, P = PID, Q = land.
Every flight is logged to `results\hw_step4_<date_time>.csv` (attitude, rates, commands, motors,
battery) at 100 Hz.

Sign conventions found on our drone: the logged pitch angle (`stabilizer.pitch`) is inverted
compared with the simulator; roll, both gyro rates, and the rate commands match.

## Status and next steps

- Simulation: complete (training, evaluation, kick and robustness tests).
- Real drone: sensor, timing, and command checks pass, and the drone flies under the firmware PID.
  In the first DQN flight the policy kept the tilt within 7 deg for about 4.6 s, then an
  oscillation grew and the drone flipped. The logs point to motor lag and vibration noise.
- Next: retrain with random motor lag (0-30 ms), measured gyro noise, and sensor delay; check
  in simulation (goal: >= 95% success with 20 ms motor lag); then tethered flights and push tests.

## References

- W. Koch, R. Mancuso, R. West, A. Bestavros, "Reinforcement Learning for UAV Attitude Control,"
  ACM Transactions on Cyber-Physical Systems, 2019.
- J. Panerati et al., "Learning to Fly: a Gym Environment with PyBullet Physics for Reinforcement
  Learning of Multi-agent Quadcopter Control," IROS 2021.
- V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, 2015.
- Bitcraze, crazyflie-firmware: https://github.com/bitcraze/crazyflie-firmware
