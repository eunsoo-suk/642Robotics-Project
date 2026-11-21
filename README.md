# Learning to Stay Level: DQN-Based Attitude Recovery for the Crazyflie 2.1

MSML642 Robotics course project. Team: Eunsoo Suk, Fnu Hardik, Abigail Basener.

When a small drone is pushed or bumped, it tilts and can flip over. This project trains a
**deep Q-network (DQN)** that brings a Crazyflie 2.1 back to level (roll = pitch = 0) as fast as
possible, using only the onboard IMU, and compares it with the default PID controller in the
Crazyflie firmware. The policy is trained in simulation, compared with the PID under identical disturbances, and the simulator is calibrated with a real flight log to study the sim-to-real gap.

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

## Recovery definition and tests

Recovery = tilt <= 5 deg **and** angular rate <= 30 deg/s, held for 0.3 s, within 2 s and without a
tumble. Recovery time runs from the disturbance (t = 0) to the start of that 0.3 s window. A strict
version uses a 2 deg band.

| Test | Disturbance |
|---|---|
| Initial tilt | 50 random starting tilts and rates |
| Kick (impulse) | sudden 100-1000 deg/s spin after 1 s of hover, 24 directions (every 15 deg) |
| Sustained push | constant torque of 0.1-0.5 x reference (~7.4 mN m, all hover thrust on one side) for 0.5 s, 24 directions |

## Results (clean simulator, mean of 3 DQN seeds)

| Test | Metric | Firmware PID | DQN |
|---|---|---|---|
| All tests | Success / flips | 100% / 0% | 100% / 0% |
| Initial tilt | Recovery time | **0.23 s** | 0.26 s |
| | Overshoot | 1.5 deg | **0.2 deg** |
| | Leftover tilt | **0.3 deg** | 0.9 deg |
| Kick 1000 deg/s | Recovery time (5 deg) | 0.47 s | **0.44 s** |
| | Strict recovery (2 deg) | 1.06 s | **0.63 s** |
| | Overshoot | 5.1 deg | **2.3 deg** |
| Kick 250 deg/s | Recovery time | **0.01 s** | 0.09 s |
| Sustained 0.5 x ref. | Recovery time | 0.61 s | **0.40 s** |
| | Max tilt while pushed | 9.5 deg | **4.8 deg** |

The DQN is better for large disturbances (about half the overshoot); the PID is faster for small ones
and more precise near level. With 20% more mass both stay at 100%; with a 20 ms motor delay both
keep oscillating (the real delay is only 5-10 ms).

## Sim-to-real: calibrating the simulator with flight logs

The real DQN flight flipped after 4.6 s. Following the reality-gap lecture (base set, envelope of
noise), `sysid_from_logs.py` searches 972 simulator settings (motor delay, motor response time, gyro
noise, turbulence strength and speed, battery headroom, motor-to-motor thrust difference) and keeps
those whose PID hover matches the real log (gyro-rate RMS, tilt RMS, vibration frequency).

- Best match: motor delay 5-10 ms, gyro noise 10-20 deg/s, turbulence 0.07 x ref., headroom 70%
  (weak battery), motor mismatch up to 25%. Real 41 deg/s / 2.8 deg / 4.3 Hz vs sim 32-53 deg/s /
  2.3-2.7 deg / 4.3-5.9 Hz.
- Validation: the original DQN flips in 3 of 5 calibrated runs, matching the real failure. The flip
  needs a 25% weaker motor, which agrees with the log (motor M2 stuck at maximum command).
- Healthy battery (realistic sim): the DQN keeps the drone about 2x more level (mean tilt 1.8-2.9 vs
  4.0-6.7 deg) and tumbles less under a 0.5 push (4.2% vs 12.5%). Weak battery: both tumble 17-46%.

Retraining with domain randomization (two attempts, `--motor-delay-ms= --gyro-noise= ...`) did not
beat the original model, so the Nov 7 models remain final.

Side-by-side video (DQN vs PID, 10x slow motion): `sim/results/MSML642_Phase2_video.mp4`

## Repository layout

```
sim/
  attitude_env.py           Gymnasium environment (DQN -> firmware rate PID -> PyBullet CF2X)
  cf_controller.py          Python copy of the firmware PID (angle + rate), mixer, gyro filter
  cf_gains.json             PID gains read from the real drone (read_gains.py)
  train_dqn.py              DQN training (stable-baselines3)
  evaluate.py               DQN vs PID from 50 random initial tilts
  disturbance_test.py       mid-flight kick test, 9 sizes x 24 directions
  robustness_test.py        mass change, motor delay, command delay (Phase 1)
  eval_v2.py                final evaluation: new recovery definition, kicks, sustained pushes, realistic conditions
  sysid_from_logs.py        calibrate the simulator with a real flight log (972-setting search)
  kick_compare.py           one kick or sustained push, DQN vs PID: figure + side-by-side video
  watch.py                  watch a policy in the PyBullet 3D view
  test_env.py               environment sanity check
  read_gains.py             read PID gains from the real drone
  hw_step1_sensor_check.py  real drone step 1: sensor sign check (no propellers)
  hw_step2_dry_run.py       real drone step 2: live DQN loop, commands not sent (no propellers)
  hw_step3_cmd_check.py     real drone step 3: command sign check (thrust 0)
  hw_step4_tether_flight.py real drone step 4: flight, PID take-off then switch to DQN
  hw_motor_test.py          spin each motor directly (hardware check)
  cam_*.py                  webcam + ArUco marker position tracking (experimental, not used in results)
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
pip install opencv-contrib-python      # only for the experimental cam_*.py scripts
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
python robustness_test.py --models a\best_model.zip b\best_model.zip c\best_model.zip
python kick_compare.py                                 # 1000 deg/s kick, figure + mp4
python kick_compare.py --disturbance sustained --frac 0.5 [--realistic]
python eval_v2.py                                      # final comparison, clean simulator
python eval_v2.py --real-like                          # calibrated realistic simulator
python eval_v2.py --mass-scale 1.2                     # or --motor-delay-ms 20
python sysid_from_logs.py                              # simulator calibration from the flight log
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

## Status

- Simulation: complete. Final comparison under the recovery definition above, plus robustness and
  realistic (calibrated) conditions.
- Real drone: sensor, timing and command checks pass and the drone flies under the firmware PID. The
  DQN flight flipped after 4.6 s; calibration traced this mainly to the hardware (weak battery,
  uneven motors). Hardware testing was stopped after a frame part broke.
- Known limits: DQN command jitter (7-15 changes/s clean, ~40/s under turbulence) and ~1 deg leftover
  tilt. Possible next steps: a smoothness-focused reward or output filter, and a hybrid controller
  (DQN for large tilts, PID near level).

## References

- W. Koch, R. Mancuso, R. West, A. Bestavros, "Reinforcement Learning for UAV Attitude Control,"
  ACM Transactions on Cyber-Physical Systems, 2019.
- J. Panerati et al., "Learning to Fly: a Gym Environment with PyBullet Physics for Reinforcement
  Learning of Multi-agent Quadcopter Control," IROS 2021.
- V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, 2015.
- Bitcraze, crazyflie-firmware: https://github.com/bitcraze/crazyflie-firmware
