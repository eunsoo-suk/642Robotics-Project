r"""
Webcam positioning, step 3: track the marker on the drone and check the coordinates.
No radio and no motors: the drone is only carried by hand.

Setup: webcam about 1 m ABOVE the drone, looking straight down.
Marker (3 cm) taped flat on top of the drone, printed TOP edge toward the drone FRONT
(front = side opposite the blue LEDs).

A) Quick calibration (OPTIONAL: without it the webcam spec is used and cm values are approximate):
   Lay the drone flat on the floor (or a box) and measure the distance from the camera lens
   to the marker with a tape measure, e.g. 100 cm. Then run
       python cam_step3_track.py --cam 1 --marker-mm 30 --focal-dist-cm 100
   and press f when the marker is detected. This saves cam_calib.json.

B) Coordinate check:
       python cam_step3_track.py --cam 1 --marker-mm 30
   Put the drone at the take-off spot and press o (set origin). Then move it by hand:
       forward -> x +,   left -> y +,   up -> z +,   turn left (counter-clockwise) -> yaw +
Keys: o = set origin, f = quick calibration, q = quit
"""
import argparse
import os
import time
import cv2
import cam_tracker as ct

ap = argparse.ArgumentParser()
ap.add_argument('--cam', type=int, default=1)
ap.add_argument('--marker-mm', type=float, default=30.0, help='measured width of the black square (mm)')
ap.add_argument('--focal-dist-cm', type=float, default=None, help='measured camera-to-marker distance for the quick calibration')
args = ap.parse_args()

trk = ct.MarkerTracker(cam=args.cam, marker_mm=args.marker_mm)
print(f'Camera {args.cam}: {trk.W}x{trk.H}.  Calibration: {"cam_calib.json" if os.path.exists(ct.CALIB_FILE) else "webcam spec (approximate)"}')
print('Keys: o = set origin, f = quick calibration, q = quit')
last = 0
while True:
    with trk.lock:
        frame = None if trk.frame is None else trk.frame.copy()
        side = trk.side_px
    if frame is None:
        time.sleep(0.01); continue
    pose = trk.get()
    if trk.K is None:
        txt = 'no calibration: press f (needs --focal-dist-cm)'
    elif trk.R0 is None:
        txt = 'press o at the take-off spot to set the origin'
    elif pose is None:
        txt = 'marker not seen'
    else:
        _, x, y, z, yaw = pose
        txt = f'x {x*100:+6.1f} cm  y {y*100:+6.1f} cm  z {z*100:+6.1f} cm  yaw {yaw:+6.1f} deg'
    cv2.putText(frame, f'FPS {trk.fps:.0f}   marker {"%.0f px" % side if side else "-"}', (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(frame, txt, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 160, 0), 2)
    cv2.imshow('step 3 tracking (o origin, f calib, q quit)', cv2.resize(frame, (trk.W // 2, trk.H // 2)))
    if time.time() - last > 0.2:
        print(f'\r  FPS {trk.fps:4.1f}  {txt}        ', end='', flush=True); last = time.time()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('o'):
        print('\n  origin set' if trk.set_origin() else '\n  marker not seen, cannot set origin')
    if key == ord('f'):
        if args.focal_dist_cm is None or side is None:
            print('\n  need --focal-dist-cm and a detected marker')
        else:
            fx = side * (args.focal_dist_cm / 100.0) / (args.marker_mm / 1000.0)
            ct.save_quick_calib(fx, trk.W, trk.H, args.cam)
            trk.K, trk.dist = ct.load_calib(trk.W, trk.H)
            print(f'\n  quick calibration saved: fx = {fx:.0f} px (cam_calib.json)')
trk.stop()
cv2.destroyAllWindows()
print()
