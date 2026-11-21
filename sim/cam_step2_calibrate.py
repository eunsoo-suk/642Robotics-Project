r"""
Webcam positioning, step 2: camera calibration (so distances in meters are correct).

1) Saves a checkerboard image (cam_checkerboard.png) to print or show on a screen.
2) Hold the checkerboard in front of the webcam. When it is found, colored lines appear.
   Press SPACE to take a shot. Take about 15 shots from different angles and distances
   (tilted left/right/up/down, near and far, in the corners of the view).
3) Press c to calculate. The result is saved to cam_calib.json (used by the next steps).

Run (.venv-sim, in the sim folder):
    python cam_step2_calibrate.py                       # first: only saves the checkerboard image
    python cam_step2_calibrate.py --cam 1 --square-mm 25
--square-mm: the measured width of ONE square of your printed checkerboard (use a ruler).
Keys: SPACE = take shot, c = calculate and save, q = quit
"""
import argparse
import json
import os
import time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--cam', type=int, default=1)
ap.add_argument('--square-mm', type=float, default=None, help='measured size of one square (mm); without it, only the board image is saved')
ap.add_argument('--width', type=int, default=1280)
ap.add_argument('--height', type=int, default=720)
args = ap.parse_args()

COLS, ROWS = 9, 6                      # inner corners of the checkerboard (10 x 7 squares)

# ---- 1) checkerboard image ----
sq = 100
board = np.zeros(((ROWS + 1) * sq, (COLS + 1) * sq), np.uint8)
for r in range(ROWS + 1):
    for c in range(COLS + 1):
        if (r + c) % 2 == 0:
            board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255
board = cv2.copyMakeBorder(board, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
cv2.imwrite(os.path.join(HERE, 'cam_checkerboard.png'), board)
print('Checkerboard saved: cam_checkerboard.png (10 x 7 squares). Print it flat, or show it full screen on a monitor.')
if args.square_mm is None:
    raise SystemExit('Measure one square with a ruler, then run again with --square-mm <mm>.')

# ---- 2) capture ----
objp = np.zeros((ROWS * COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * args.square_mm / 1000.0   # meters
obj_pts, img_pts = [], []
cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
if not cap.isOpened():
    raise SystemExit(f'Could not open camera {args.cam}.')
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
size, flash = None, 0
print('SPACE = take shot, c = calculate, q = quit')
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    found, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), cv2.CALIB_CB_FAST_CHECK)
    view = frame.copy()
    if found:
        cv2.drawChessboardCorners(view, (COLS, ROWS), corners, found)
    cv2.putText(view, f'shots: {len(obj_pts)} (aim for 15)   ' + ('board FOUND - press SPACE' if found else 'board not found'),
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0) if found else (0, 0, 255), 2)
    if flash > 0:
        view[:] = 255 - view; flash -= 1
    cv2.imshow('calibration (SPACE shot, c calculate, q quit)', view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        raise SystemExit('Quit without saving.')
    if key == ord(' ') and found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj_pts.append(objp); img_pts.append(corners); flash = 2
        print(f'  shot {len(obj_pts)}')
    if key == ord('c'):
        if len(obj_pts) < 8:
            print('  need at least 8 shots'); continue
        break
cap.release()
cv2.destroyAllWindows()

# ---- 3) calibrate ----
rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
out = os.path.join(HERE, 'cam_calib.json')
json.dump({'camera': args.cam, 'width': size[0], 'height': size[1], 'rms_px': rms,
           'K': K.tolist(), 'dist': dist.ravel().tolist(), 'shots': len(obj_pts),
           'date': time.strftime('%Y-%m-%d %H:%M')}, open(out, 'w'), indent=2)
print(f'\nReprojection error: {rms:.2f} px  (below 1.0 is good, below 0.5 is very good)')
print(f'Focal length: fx {K[0, 0]:.0f} px, fy {K[1, 1]:.0f} px   center ({K[0, 2]:.0f}, {K[1, 2]:.0f})')
print(f'Saved: {out}')
