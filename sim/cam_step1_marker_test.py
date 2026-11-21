r"""
Webcam positioning, step 1: make the marker and check that the webcam can see it.

1) Saves an ArUco marker image (cam_marker_id0.png) to print.
2) Opens the webcam and shows a live view. A green box and the marker ID appear when
   the marker is found. FPS and detection rate are shown at the top.

Run (.venv-sim, in the sim folder):
    pip install opencv-contrib-python
    python cam_step1_marker_test.py              # camera 0
    python cam_step1_marker_test.py --cam 1      # try 1 if 0 is the wrong camera
Press q in the video window to quit, s to save a snapshot (cam_snapshot.png).
Red outlines = square shapes found but not decoded as a marker.
"""
import argparse
import os
import time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--cam', type=int, default=0)
ap.add_argument('--id', type=int, default=0, help='marker ID')
args = ap.parse_args()

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# ---- 1) marker image for printing (white border so it is easy to cut out) ----
img = cv2.aruco.generateImageMarker(DICT, args.id, 400)
img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
path = os.path.join(HERE, f'cam_marker_id{args.id}.png')
cv2.imwrite(path, img)
print(f'Marker saved: {path}')
print('Print it so that the BLACK square is 3.0 cm wide, and measure it with a ruler.')

# ---- 2) live detection ----
det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise SystemExit(f'Could not open camera {args.cam}. Try --cam 1.')
print(f'Camera {args.cam}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}. Press q to quit.')

t0, n, hits, fps = time.time(), 0, 0, 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = det.detectMarkers(gray)
    mirrored = False
    if ids is None:                                   # a mirrored image cannot be decoded: try flipping it
        c2, i2, _ = det.detectMarkers(cv2.flip(gray, 1))
        if i2 is not None:
            mirrored = True
            w = gray.shape[1]
            corners = tuple(np.array([[[w - 1 - x, y] for x, y in cc[0]]], dtype=np.float32) for cc in c2)
            ids = i2
    n += 1
    for r in rejected:
        cv2.polylines(frame, [r.astype(np.int32)], True, (0, 0, 255), 1)
    if mirrored:
        cv2.putText(frame, 'MIRRORED image: turn off mirror mode in the webcam settings', (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if ids is not None and args.id in ids.flatten():
        hits += 1
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        c = corners[list(ids.flatten()).index(args.id)][0]
        side_px = np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])
        cv2.putText(frame, f'marker side {side_px:.0f} px', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    if time.time() - t0 >= 1.0:
        fps = n / (time.time() - t0)
        rate = 100.0 * hits / max(n, 1)
        t0, n, hits = time.time(), 0, 0
        print(f'\r  FPS {fps:4.1f}   detected {rate:5.1f}% of frames   ', end='', flush=True)
    cv2.putText(frame, f'FPS {fps:.1f}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('marker test (q = quit)', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite(os.path.join(HERE, 'cam_snapshot.png'), frame)
        print('\n  snapshot saved: cam_snapshot.png')
cap.release()
cv2.destroyAllWindows()
print()
