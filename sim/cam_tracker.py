"""
Webcam marker tracker used by the cam_step3/4/5 scripts.

A camera above the drone looks down at an ArUco marker taped flat on top of the drone,
with the printed TOP edge of the marker pointing to the drone's FRONT.

Output frame (same as the Crazyflie world frame when the origin is set at take-off):
    x = forward, y = left, z = up (meters), yaw = rotation about z (deg, counter-clockwise +)
The origin and axes are taken from the marker pose when set_origin() is called.
"""
import json
import os
import threading
import time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, 'cam_calib.json')
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


DEFAULT_DIAG_FOV_DEG = 110.0     # NexiGo N60 spec. Only sets the cm scale; directions are correct anyway.


def default_K(width, height, diag_fov_deg=DEFAULT_DIAG_FOV_DEG):
    """Camera matrix from the spec field of view (no calibration). Distances may be off by ~10-20%."""
    f = (np.hypot(width, height) / 2) / np.tan(np.radians(diag_fov_deg / 2))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], float)


def load_calib(width, height):
    """Camera matrix and distortion from cam_calib.json (scaled to the resolution), or the spec-based default."""
    if not os.path.exists(CALIB_FILE):
        return default_K(width, height), np.zeros(5)
    c = json.load(open(CALIB_FILE))
    K = np.array(c['K'], float)
    s = width / c['width']
    K[:2] *= s
    return K, np.array(c.get('dist', [0, 0, 0, 0, 0]), float)


def save_quick_calib(fx, width, height, cam):
    """Calibration from one measured distance: focal length only, image center, no distortion."""
    K = [[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]]
    json.dump({'camera': cam, 'width': width, 'height': height, 'K': K, 'dist': [0, 0, 0, 0, 0],
               'method': 'quick (one distance)', 'date': time.strftime('%Y-%m-%d %H:%M')},
              open(CALIB_FILE, 'w'), indent=2)


class MarkerTracker:
    def __init__(self, cam=1, marker_mm=30.0, marker_id=0, width=1920, height=1080):
        self.cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))   # faster frames at high resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise SystemExit(f'Could not open camera {cam}.')
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cam, self.id, self.s = cam, marker_id, marker_mm / 1000.0
        self.K, self.dist = load_calib(self.W, self.H)
        self.det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
        h = self.s / 2
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)  # ArUco corner order
        self.R0 = self.p0 = None            # origin pose (camera frame)
        self.lock = threading.Lock()
        self.frame = None                   # last frame with drawings (for display)
        self.side_px = None                 # marker size in pixels (for the quick calibration)
        self.pose = None                    # (t, x, y, z, yaw) in the output frame, None if not seen
        self.cam_pose = None                # (R, p) in the camera frame
        self.fps, self._n, self._t = 0.0, 0, time.time()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def set_origin(self):
        with self.lock:
            if self.cam_pose is None:
                return False
            self.R0, self.p0 = self.cam_pose
            return True

    def get(self):
        with self.lock:
            return self.pose

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            t = time.time()
            corners, ids, _ = self.det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cam_pose, pose, side = None, None, None
            if ids is not None and self.id in ids.flatten():
                c = corners[list(ids.flatten()).index(self.id)]
                side = float(np.mean([np.linalg.norm(c[0][i] - c[0][(i + 1) % 4]) for i in range(4)]))
                cv2.aruco.drawDetectedMarkers(frame, [c], np.array([[self.id]]))
                if self.K is not None:
                    ok, rvec, tvec = cv2.solvePnP(self.obj, c[0], self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        R, _ = cv2.Rodrigues(rvec)
                        p = tvec.ravel()
                        cam_pose = (R, p)
                        cv2.drawFrameAxes(frame, self.K, self.dist, rvec, tvec, self.s)
                        if self.R0 is not None:
                            # marker frame: x = right, y = top of print (= drone front), z = up
                            d = self.R0.T @ (p - self.p0)          # position in the origin marker frame
                            Rr = self.R0.T @ R                     # rotation relative to the origin
                            yaw = np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0]))
                            pose = (t, d[1], -d[0], d[2], yaw)     # -> x forward, y left, z up
            self._n += 1
            if t - self._t >= 1.0:
                self.fps, self._n, self._t = self._n / (t - self._t), 0, t
            with self.lock:
                self.frame, self.side_px, self.cam_pose, self.pose = frame, side, cam_pose, pose
