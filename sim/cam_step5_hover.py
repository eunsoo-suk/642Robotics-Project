r"""
Webcam positioning, step 5: take off and hold position with the webcam (propellers ON).

The webcam position is sent to the drone 30 times per second (as in step 4). The firmware's
own position controller (high-level commander) takes off, holds the drone above the take-off
spot, and lands.

Safety
    Space or Esc (in the video window) : emergency stop, motors off at once
    q                                  : land now
    w / s                              : target height up / down by --step-m (moves at --vspeed)
    auto: marker lost for more than --lost-s seconds          -> land
    auto: more than --fence-m away from the origin, or above --max-height -> motors off
Before flying: steps 3/4 done and directions checked (forward x+, left y+, up z+).
Clear 1 m around the drone, keep hands and face away.

Run (.venv-sim, in the sim folder):
    python cam_step5_hover.py --cam 1 --marker-mm 30
    python cam_step5_hover.py --cam 1 --marker-mm 30 --vspeed 0.1 --step-m 0.05    # slower up/down
Every run is logged to results\cam_hover_<date_time>.csv
"""
import argparse
import csv
import json
import os
import time
import cv2
import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import cam_tracker as ct

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--cam', type=int, default=1)
ap.add_argument('--marker-mm', type=float, default=30.0)
ap.add_argument('--height', type=float, default=0.4, help='hover height above the take-off spot (m)')
ap.add_argument('--hover-s', type=float, default=0.0, help='hover time before auto landing (s); 0 = until q')
ap.add_argument('--vspeed', type=float, default=0.2, help='vertical speed for take-off and height changes (m/s)')
ap.add_argument('--step-m', type=float, default=0.05, help='height change per w/s press (m)')
ap.add_argument('--max-height', type=float, default=1.0, help='motors off above this height (m)')
ap.add_argument('--thrust-base', type=int, default=43000, help='approximate hover thrust (firmware posCtlPid.thrustBase)')
ap.add_argument('--lost-s', type=float, default=0.5, help='land if the marker is lost this long (s)')
ap.add_argument('--fence-m', type=float, default=0.5, help='motors off if this far from the origin (m)')
args = ap.parse_args()
uri = json.load(open(os.path.join(HERE, 'hw_signs.json')))['uri']

trk = ct.MarkerTracker(cam=args.cam, marker_mm=args.marker_mm)
est = {'x': np.nan, 'y': np.nan, 'z': np.nan, 'yaw': np.nan, 'vbat': np.nan,
       'm1': 0, 'm2': 0, 'm3': 0, 'm4': 0}


def on_est(ts, d, conf):
    est.update({k.split('.')[-1]: v for k, v in d.items()})


def say(msg):
    print(f'\n  {msg}', flush=True)


cflib.crtp.init_drivers()
print(f'Connecting: {uri}')
rows, reason = [], 'normal end'
with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=os.path.join(HERE, '..', 'cache'))) as scf:
    cf = scf.cf
    hlc = cf.high_level_commander
    cf.param.set_value('stabilizer.estimator', '2')           # Kalman filter (uses the webcam position)
    cf.param.set_value('stabilizer.controller', '1')          # PID
    cf.param.set_value('flightmode.stabModeRoll', '1')        # angle mode (in case an old script left rate mode)
    cf.param.set_value('flightmode.stabModePitch', '1')
    try:
        cf.param.set_value('commander.enHighLevel', '1')     # older firmware only; newer firmware has it always on
    except Exception:
        pass
    cf.param.set_value('posCtlPid.thrustBase', str(args.thrust_base))
    lg = LogConfig(name='est', period_in_ms=20)
    for v in ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z', 'stabilizer.yaw', 'pm.vbat']:
        lg.add_variable(v, 'float')
    lg2 = LogConfig(name='mot', period_in_ms=20)
    for v in ['motor.m1', 'motor.m2', 'motor.m3', 'motor.m4']:
        lg2.add_variable(v, 'uint16_t')
    for g in (lg, lg2):
        cf.log.add_config(g); g.data_received_cb.add_callback(on_est); g.start()
    try:
        cf.platform.send_arming_request(True)                 # needed on newer firmware, harmless otherwise
    except Exception:
        pass

    print('Connected. Put the drone at the take-off spot, then press o in the video window to set the origin.')
    print('Then press t to take off.  w/s = height up/down, q = land, Space/Esc = emergency stop.')
    target_z, t_climb = args.height, args.height / args.vspeed
    phase, t_phase, last_seen, next_send, t0, last_print = 'wait', 0.0, time.time(), time.time(), time.time(), 0
    try:
        while True:
            now = time.time()
            pose = trk.get()
            seen = pose is not None and now - pose[0] < 0.1
            if seen:
                last_seen = now
            if phase != 'wait' and now >= next_send:
                next_send = now + 1 / 30
                if seen:
                    cf.extpos.send_extpos(pose[1], pose[2], pose[3])
            # ---- flight phases ----
            if phase == 'takeoff' and now - t_phase > t_climb + 0.5:
                phase, t_phase = 'hover', now
                say(f'hovering at {target_z:.2f} m' + (f' for {args.hover_s:.0f} s' if args.hover_s > 0 else ' (q to land)'))
            elif phase == 'hover' and args.hover_s > 0 and now - t_phase > args.hover_s:
                hlc.land(0.0, max(2.0, target_z / args.vspeed)); phase, t_phase = 'land', now; say('landing')
            elif phase == 'land' and now - t_phase > max(2.5, target_z / args.vspeed + 0.5):
                hlc.stop(); reason = 'landed'; break
            # ---- safety ----
            if phase in ('takeoff', 'hover'):
                if now - last_seen > args.lost_s:
                    hlc.land(0.0, max(2.0, target_z / args.vspeed)); phase, t_phase = 'land', now; say('marker lost -> landing')
                elif seen and (np.hypot(pose[1], pose[2]) > args.fence_m or pose[3] > args.max_height):
                    reason = f'fence: x {pose[1]:+.2f} y {pose[2]:+.2f} z {pose[3]:+.2f} m -> motors off'; break
            # ---- display and log ----
            with trk.lock:
                frame = None if trk.frame is None else trk.frame.copy()
            if frame is not None:
                if seen:
                    t1 = f'cam x {pose[1]*100:+5.1f} y {pose[2]*100:+5.1f} z {pose[3]*100:+5.1f} cm'
                elif trk.cam_pose is not None and trk.R0 is None:
                    t1 = 'marker OK - press o to set the origin'
                else:
                    t1 = 'marker not seen'
                t2 = f'est x {est["x"]*100:+5.1f} y {est["y"]*100:+5.1f} z {est["z"]*100:+5.1f} cm  target z {target_z*100:.0f}  {phase.upper()}  bat {est["vbat"]:.2f} V'
                cv2.putText(frame, t1, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 160, 0), 2)
                cv2.putText(frame, t2, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 0, 0), 2)
                cv2.imshow('step 5 hover (o origin, t take off, q land, SPACE stop)', cv2.resize(frame, (trk.W // 2, trk.H // 2)))
            if phase != 'wait':
                rows.append(dict(t=round(now - t0, 3), phase=phase, seen=int(seen),
                                 cam_x=pose[1] if seen else '', cam_y=pose[2] if seen else '', cam_z=pose[3] if seen else '',
                                 cam_yaw=pose[4] if seen else '', **est))
            if now - last_print > 0.3:
                print(f'\r  [{phase}] {t1 if frame is not None else ""}   est z {est["z"]*100:+5.1f} cm   ', end='', flush=True)
                last_print = now
            key = cv2.waitKey(1) & 0xFF
            if key in (ord(' '), 27):
                reason = 'emergency stop'; break
            if key == ord('q') and phase in ('takeoff', 'hover'):
                hlc.land(0.0, max(2.0, target_z / args.vspeed)); phase, t_phase = 'land', now; say('landing (q)')
            if key in (ord('w'), ord('s')) and phase == 'hover':
                dz = args.step_m if key == ord('w') else -args.step_m
                target_z = float(np.clip(target_z + dz, 0.15, args.max_height - 0.15))
                hlc.go_to(0.0, 0.0, target_z, 0.0, max(abs(dz) / args.vspeed, 0.2))
                say(f'target height {target_z*100:.0f} cm')
            if key == ord('o') and phase == 'wait':
                if trk.set_origin():
                    phase = 'ready'
                    cf.param.set_value('kalman.resetEstimation', '1'); time.sleep(0.1)
                    cf.param.set_value('kalman.resetEstimation', '0')
                    say('origin set, estimator reset. Press t to take off.')
                else:
                    say('marker not seen')
            if key == ord('t') and phase == 'ready':
                if not seen or abs(est['x']) > 0.05 or abs(est['y']) > 0.05:
                    say('estimate not settled yet (or marker not seen); wait a second and press t again')
                else:
                    hlc.takeoff(target_z, t_climb); phase, t_phase = 'takeoff', now; say(f'taking off ({args.vspeed:.2f} m/s)')
    except KeyboardInterrupt:
        reason = 'emergency stop (Ctrl+C)'
    finally:
        try:
            hlc.stop()
        except Exception:
            pass
        for _ in range(5):
            cf.commander.send_stop_setpoint(); time.sleep(0.01)
        for g in (lg, lg2):
            g.stop()
        trk.stop()
        cv2.destroyAllWindows()

print(f'\n\nEnd: {reason}')
if rows:
    out = os.path.join(HERE, 'results', time.strftime('cam_hover_%Y%m%d_%H%M%S.csv'))
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    hv = [r for r in rows if r['phase'] == 'hover' and r['seen']]
    if hv:
        xy = np.array([[r['cam_x'], r['cam_y']] for r in hv], float)
        z = np.array([r['cam_z'] for r in hv], float)
        print(f'Hover: horizontal error mean {np.hypot(*xy.T).mean()*100:.1f} cm, max {np.hypot(*xy.T).max()*100:.1f} cm; '
              f'height {z.mean()*100:.1f} +/- {z.std()*100:.1f} cm')
    print(f'Log: {out}')
