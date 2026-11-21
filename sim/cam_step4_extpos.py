r"""
Webcam positioning, step 4: send the camera position to the drone (motors OFF).

The drone's onboard Kalman filter receives the webcam position (x, y, z) about 30 times
per second. This script compares the camera position with the drone's own estimate.
If they match while you carry the drone around, the drone is ready to hold position (step 5).
No thrust is ever sent: the motors do not spin.

Setup: same as step 3 (webcam above, marker on top of the drone). Calibration is optional.
Run (.venv-sim, in the sim folder):
    python cam_step4_extpos.py --cam 1 --marker-mm 30
Steps: put the drone at the take-off spot, press o (sets the origin and resets the drone's
estimator), then carry the drone around slowly. Keys: o = origin + reset, q = quit
"""
import argparse
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
args = ap.parse_args()
uri = json.load(open(os.path.join(HERE, 'hw_signs.json')))['uri']

trk = ct.MarkerTracker(cam=args.cam, marker_mm=args.marker_mm)
print('Camera calibration:', 'cam_calib.json' if os.path.exists(ct.CALIB_FILE) else
      f'none, using the webcam spec ({ct.DEFAULT_DIAG_FOV_DEG:.0f} deg field of view); cm values are approximate')
est = {'x': np.nan, 'y': np.nan, 'z': np.nan, 'yaw': np.nan}


def on_est(ts, d, conf):
    est.update(x=d['stateEstimate.x'], y=d['stateEstimate.y'], z=d['stateEstimate.z'], yaw=d['stabilizer.yaw'])


def reset_estimator(cf):
    cf.param.set_value('kalman.resetEstimation', '1'); time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0'); time.sleep(0.5)


cflib.crtp.init_drivers()
print(f'Connecting: {uri}')
with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=os.path.join(HERE, '..', 'cache'))) as scf:
    cf = scf.cf
    cf.param.set_value('stabilizer.estimator', '2')        # 2 = Kalman filter (uses external position)
    lg = LogConfig(name='est', period_in_ms=50)
    for v in ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z', 'stabilizer.yaw']:
        lg.add_variable(v, 'float')
    cf.log.add_config(lg); lg.data_received_cb.add_callback(on_est); lg.start()
    print('Connected. Motors stay off. Put the drone at the take-off spot and press o in the video window.')
    origin, sent, lost, errs, last_print, t1 = False, 0, 0, [], 0, ''
    next_send = time.time()
    try:
        while True:
            pose = trk.get()
            now = time.time()
            if origin and now >= next_send:
                next_send = now + 1 / 30
                if pose is not None and now - pose[0] < 0.1:
                    cf.extpos.send_extpos(pose[1], pose[2], pose[3])
                    sent += 1
                    errs.append(np.linalg.norm(np.array(pose[1:4]) - np.array([est['x'], est['y'], est['z']])))
                else:
                    lost += 1
            with trk.lock:
                frame = None if trk.frame is None else trk.frame.copy()
            if frame is not None:
                if pose is None:
                    t1 = 'marker not seen' if trk.cam_pose is None else 'press o to set origin'
                else:
                    t1 = f'camera   x {pose[1]*100:+6.1f}  y {pose[2]*100:+6.1f}  z {pose[3]*100:+6.1f} cm  yaw {pose[4]:+5.0f}'
                t2 = f'estimate x {est["x"]*100:+6.1f}  y {est["y"]*100:+6.1f}  z {est["z"]*100:+6.1f} cm  yaw {est["yaw"]:+5.0f}'
                cv2.putText(frame, t1, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 160, 0), 2)
                cv2.putText(frame, t2, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 0, 0), 2)
                cv2.putText(frame, f'FPS {trk.fps:.0f}  sent {sent}  lost {lost}', (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow('step 4 extpos (o origin, q quit)', cv2.resize(frame, (trk.W // 2, trk.H // 2)))
            if now - last_print > 0.3 and origin:
                e = np.mean(errs[-15:]) * 100 if errs else np.nan
                print(f'\r  {t1} | estimate {est["x"]*100:+6.1f} {est["y"]*100:+6.1f} {est["z"]*100:+6.1f} | diff {e:4.1f} cm   ', end='', flush=True)
                last_print = now
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('o'):
                if trk.set_origin():
                    reset_estimator(cf)
                    origin, errs = True, []
                    print('\n  origin set, drone estimator reset')
                else:
                    print('\n  marker not seen, cannot set origin')
    finally:
        lg.stop()
        trk.stop()
        cv2.destroyAllWindows()
if errs:
    e = np.array(errs[30:] or errs) * 100
    print(f'\nCamera vs drone estimate: mean {e.mean():.1f} cm, 95% {np.percentile(e, 95):.1f} cm.  '
          f'Positions sent {sent}, missed {lost} ({100 * lost / max(sent + lost, 1):.0f}%).')
    print('Good if the mean is below about 3 cm and missed is below 10%.')
