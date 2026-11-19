#!/usr/bin/env python3
"""pose_global.py - global pose of the Tello camera/drone from ArUco markers with a KNOWN MAP (no EKF yet).

READ-ONLY: this script never takes off, never sends rc and never lands. It only reads video + telemetry, so you can
CARRY the drone by hand (propellers off the table, battery in) in front of the hoop and check the geometry.

For every visible mapped marker i:   p_C^W = p_Mi^W + R_WMi * (-R_cmᵀ t_cm),   R_WC = R_WMi * R_cmᵀ
(R_cm, t_cm come from solvePnP: p_camera = R_cm * p_marker + t_cm).  World frame: X crossing direction, Y left, Z up.
Marker frame (OpenCV): x right, y up, z out of the marker towards the viewer (= the marker normal).

Outputs two CSV files in --log-dir:  pose_<stamp>_frames.csv (fused pose + full telemetry per frame)
and pose_<stamp>_markers.csv (raw tvec/rvec and the pose from every single marker): the input for the EKF later.

Usage:
  python3 pose_global.py --map aruco_map_circulo.yaml --calibration tello_calibration.npz            # live camera, hand-held
  python3 pose_global.py --map aruco_map_circulo.yaml --video flight.mp4                              # offline
  python3 pose_global.py --selftest                                                                  # math check, no drone
Keys: Q = quit.   Yaw convention here: 0 = +X, POSITIVE = counter-clockwise (left). The Tello telemetry yaw is POSITIVE CLOCKWISE.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ map
def R_from_normal_yaw(phi_deg):
    """Rotation marker->world for a VERTICAL marker whose normal points towards yaw phi (0 = +X, CCW)."""
    p = math.radians(phi_deg)
    n = np.array([math.cos(p), math.sin(p), 0.0])          # marker z (out of the face)
    x = np.array([-math.sin(p), math.cos(p), 0.0])         # marker x (right, as seen by a viewer in front of it)
    y = np.array([0.0, 0.0, 1.0])                          # marker y (up)
    return np.column_stack([x, y, n])


def load_map(path):
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit('PyYAML missing: pip install pyyaml') from exc
    with open(path, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f)
    mk = {}
    for k, v in d['markers'].items():
        R = R_from_normal_yaw(float(v['normal_yaw_deg']))
        tilt = math.radians(float(v.get('tilt_deg', 0.0)))      # rotation about the marker's own x axis (leans the face up/down)
        ct, st = math.cos(tilt), math.sin(tilt)
        R = R @ np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        mk[int(k)] = {'p': np.array(v['position'], float), 'R': R,
                      'size': float(v.get('size', 0.115)), 'measured': bool(v.get('measured', False))}
    return mk


# ------------------------------------------------------------------ vision
def load_calibration(path):
    with np.load(path) as d:
        K = np.asarray(d['camera_matrix'], float)
        dist = np.asarray(d['dist_coeffs'], float)
        size = tuple(int(v) for v in np.asarray(d['image_size']).reshape(-1)) if 'image_size' in d.files else None
    return K, dist, size


def make_detector():
    ar = cv2.aruco
    dic = ar.getPredefinedDictionary(ar.DICT_4X4_50)
    prm = ar.DetectorParameters() if hasattr(ar, 'DetectorParameters') else ar.DetectorParameters_create()
    return ar.ArucoDetector(dic, prm) if hasattr(ar, 'ArucoDetector') else (dic, prm)


def detect_corners(gray, det):
    if isinstance(det, tuple):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, det[0], parameters=det[1])
    else:
        corners, ids, _ = det.detectMarkers(gray)
    return {} if ids is None else {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.flatten())}


def solve_marker(pixels, size, K, dist):
    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, pixels, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or tvec[2, 0] <= 0:
        return None
    rep, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((rep.reshape(4, 2) - pixels) ** 2, axis=1))))
    return rvec.reshape(3), tvec.reshape(3), rms


# ------------------------------------------------------------------ geometry
def camera_pose_world(m, rvec, tvec):
    """Camera position (3,) and rotation R_WC (columns = camera x right, y down, z forward in W) from ONE marker."""
    R_cm, _ = cv2.Rodrigues(rvec)
    p_cam_in_marker = -R_cm.T @ tvec
    p_w = m['p'] + m['R'] @ p_cam_in_marker
    R_wc = m['R'] @ R_cm.T
    return p_w, R_wc


def yaw_pitch_deg(R_wc):
    f = R_wc[:, 2]                                        # camera forward in W
    yaw = math.degrees(math.atan2(f[1], f[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))   # + = looking UP
    return yaw, pitch


def fuse(estimates):
    """Weighted mean of per-marker estimates (weight 1/distance^2). estimates: list of (p_w, yaw_deg, pitch_deg, dist)."""
    w = np.array([1.0 / max(e[3], 0.2) ** 2 for e in estimates])
    w /= w.sum()
    p = sum(wi * e[0] for wi, e in zip(w, estimates))
    yaw = math.degrees(math.atan2(sum(wi * math.sin(math.radians(e[1])) for wi, e in zip(w, estimates)),
                                  sum(wi * math.cos(math.radians(e[1])) for wi, e in zip(w, estimates))))
    pitch = float(sum(wi * e[2] for wi, e in zip(w, estimates)))
    return p, yaw, pitch


def process_frame(frame_bgr, det, K, dist, mmap):
    """Returns (per_marker list of dicts, fused (p, yaw, pitch) or None, unmapped ids)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    per, est, unmapped = [], [], []
    for i, px in detect_corners(gray, det).items():
        if i not in mmap:
            unmapped.append(i)
            continue
        sol = solve_marker(px, mmap[i]['size'], K, dist)
        if sol is None:
            continue
        rvec, tvec, rms = sol
        if rms > 2.5:
            continue
        p_w, R_wc = camera_pose_world(mmap[i], rvec, tvec)
        yaw, pitch = yaw_pitch_deg(R_wc)
        d = float(np.linalg.norm(tvec))
        per.append({'id': i, 'rvec': rvec, 'tvec': tvec, 'rms': rms, 'p': p_w, 'yaw': yaw, 'pitch': pitch, 'dist': d, 'px': px})
        est.append((p_w, yaw, pitch, d))
    return per, (fuse(est) if est else None), unmapped


def scale_K(K, calib_size, shape):
    Ks = K.copy()
    if calib_size:
        h, w = shape[:2]
        Ks[0, :] *= w / calib_size[0]
        Ks[1, :] *= h / calib_size[1]
    return Ks


# ------------------------------------------------------------------ logging
TELE_KEYS = ('pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'agx', 'agy', 'agz', 'tof', 'h', 'baro', 'bat')


class Logger:
    def __init__(self, log_dir, stamp):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.f1 = open(Path(log_dir) / f'pose_{stamp}_frames.csv', 'w', newline='')
        self.f2 = open(Path(log_dir) / f'pose_{stamp}_markers.csv', 'w', newline='')
        self.w1, self.w2 = csv.writer(self.f1), csv.writer(self.f2)
        self.w1.writerow(['t', 'frame', 'n_markers', 'ids', 'x', 'y', 'z', 'yaw_ccw_deg', 'cam_pitch_deg', 'spread_xy_m'] +
                         ['tele_' + k for k in TELE_KEYS])
        self.w2.writerow(['t', 'frame', 'id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rms_px', 'dist', 'x_w', 'y_w', 'z_w', 'yaw_ccw_deg', 'cam_pitch_deg'])
        self.n = 0

    def write(self, t, per, fused, tele):
        self.n += 1
        spread = ''
        if len(per) >= 2:
            xy = np.array([e['p'][:2] for e in per])
            spread = round(float(np.max(np.linalg.norm(xy - xy.mean(axis=0), axis=1))), 3)
        tel = [tele.get(k, '') for k in TELE_KEYS]
        if fused:
            p, yaw, pitch = fused
            self.w1.writerow([round(t, 3), self.n, len(per), ';'.join(str(e['id']) for e in per),
                              *np.round(p, 3), round(yaw, 2), round(pitch, 2), spread, *tel])
        else:
            self.w1.writerow([round(t, 3), self.n, 0, '', '', '', '', '', '', '', *tel])
        for e in per:
            self.w2.writerow([round(t, 3), self.n, e['id'], *np.round(e['tvec'], 4), *np.round(e['rvec'], 4), round(e['rms'], 2),
                              round(e['dist'], 3), *np.round(e['p'], 3), round(e['yaw'], 2), round(e['pitch'], 2)])

    def close(self):
        self.f1.close()
        self.f2.close()


def overlay(frame, per, fused, unmapped, lines_extra=()):
    for e in per:
        cv2.polylines(frame, [e['px'].astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, f"ID{e['id']} {e['dist']:.2f}m", tuple(e['px'][0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    if fused:
        p, yaw, pitch = fused
        txt = f'W: x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:+.2f} m  yaw={yaw:+.1f} deg  cam pitch={pitch:+.1f} deg'
    else:
        txt = 'NO MAPPED MARKER'
    cv2.putText(frame, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
    y = 52
    for e in per:
        cv2.putText(frame, f"  ID{e['id']}: x={e['p'][0]:+.2f} y={e['p'][1]:+.2f} z={e['p'][2]:+.2f} yaw={e['yaw']:+.1f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
        y += 20
    if unmapped:
        cv2.putText(frame, f'not in map: {unmapped}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 128, 255), 1)
    return frame


# ------------------------------------------------------------------ modes
def run_live(args, K, dist, csize, mmap):
    from djitellopy import Tello
    import logging
    drone = Tello()
    for lg in (getattr(Tello, 'LOGGER', None), logging.getLogger('djitellopy')):
        if lg is not None:
            lg.setLevel(logging.WARNING)
    drone.connect()
    print(f'Battery {drone.get_battery()}%  (READ-ONLY: no takeoff, no rc)')
    drone.streamoff()
    drone.streamon()
    reader = drone.get_frame_read()
    det = make_detector()
    log = Logger(args.log_dir, time.strftime('%Y%m%d_%H%M%S'))
    t0, last_print = time.time(), 0.0
    try:
        while True:
            fr = reader.frame
            if fr is None:
                time.sleep(0.02)
                continue
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            Ks = scale_K(K, csize, bgr.shape)
            per, fused, unm = process_frame(bgr, det, Ks, dist, mmap)
            try:
                tele = drone.get_current_state()
            except Exception:
                tele = {}
            t = time.time() - t0
            log.write(t, per, fused, tele)
            if fused and t - last_print > 0.5:
                last_print = t
                p, yaw, pitch = fused
                print(f't={t:6.1f}s ids={[e["id"] for e in per]} x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:+.2f} yaw={yaw:+.1f} pitch={pitch:+.1f}')
            cv2.imshow('pose_global (Q = quit)', overlay(bgr, per, fused, unm))
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        log.close()
        cv2.destroyAllWindows()
        try:
            drone.streamoff()
        except Exception:
            pass
        print(f'logs in {args.log_dir}')


def run_video(args, K, dist, csize, mmap):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = make_detector()
    log = Logger(args.log_dir, time.strftime('%Y%m%d_%H%M%S'))
    n = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            Ks = scale_K(K, csize, bgr.shape)
            per, fused, unm = process_frame(bgr, det, Ks, dist, mmap)
            log.write(n / fps, per, fused, {})
            n += 1
            cv2.imshow('pose_global offline (Q = quit)', overlay(bgr, per, fused, unm))
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        log.close()
        cap.release()
        cv2.destroyAllWindows()


def selftest():
    """Math round-trip, no drone: render the hoop markers from a known camera pose and recover it."""
    mmap = {0: {'p': np.array([0, 0, 1.055]), 'R': R_from_normal_yaw(180), 'size': 0.115, 'measured': False},
            1: {'p': np.array([0, 0, 1.605]), 'R': R_from_normal_yaw(180), 'size': 0.115, 'measured': False}}
    K = np.array([[914.0, 0, 482], [0, 916.0, 380], [0, 0, 1]])
    dist = np.zeros(5)
    true_p = np.array([-1.0, 0.12, 1.30])
    yaw, pitch = math.radians(8.0), math.radians(-10.0)          # looking 8 deg to the left, 10 deg DOWN
    fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_wc = np.column_stack([right, down, fwd])
    est = []
    for i, m in mmap.items():
        h = m['size'] / 2
        corners_m = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        pw = m['p'] + (m['R'] @ corners_m.T).T                  # corners in W
        pc = (R_wc.T @ (pw - true_p).T).T                       # corners in camera
        px = (K @ pc.T).T
        px = (px[:, :2] / px[:, 2:3]).astype(np.float32)
        rvec, tvec, rms = solve_marker(px, m['size'], K, dist)
        p, R = camera_pose_world(m, rvec, tvec)
        y, pt = yaw_pitch_deg(R)
        print(f'marker {i}: pos error {np.linalg.norm(p - true_p) * 100:.2f} cm, yaw {y:+.2f} (true +8.00), pitch {pt:+.2f} (true -10.00)')
        est.append((p, y, pt, float(np.linalg.norm(tvec))))
    p, y, pt = fuse(est)
    print(f'fused: {np.round(p, 3)} yaw {y:+.2f} pitch {pt:+.2f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map', default='aruco_map_circulo.yaml')
    ap.add_argument('--calibration', default='tello_calibration.npz')
    ap.add_argument('--video', default=None, help='offline: video file instead of the live Tello camera')
    ap.add_argument('--log-dir', default='logs')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    mmap = load_map(args.map)
    unmeasured = [i for i, m in mmap.items() if not m['measured']]
    if unmeasured:
        print(f'WARNING: markers with measured=false (provisional map): {unmeasured}')
    K, dist, csize = load_calibration(args.calibration)
    (run_video if args.video else run_live)(args, K, dist, csize, mmap)


if __name__ == '__main__':
    main()