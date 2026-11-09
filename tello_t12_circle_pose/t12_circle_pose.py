#!/usr/bin/env python3
"""T1.2: ground-only ArUco pose and circle midpoint validation."""
import argparse
import csv
import datetime as dt
from pathlib import Path
import time

import cv2
import numpy as np

MARKER_SIZE_M = 0.115
EXPECTED_SPACING_M = 0.55
CIRCLE_IDS = (0, 1)


def load_calibration(path):
    with np.load(path) as data:
        K = data['camera_matrix'].astype(np.float64)
        D = data['dist_coeffs'].astype(np.float64)
        size = tuple(map(int, data['image_size'].reshape(-1)))
    return K, D, size


def solve_marker(corners, K, D):
    h = MARKER_SIZE_M / 2
    # OpenCV marker corners are top-left, top-right, bottom-right, bottom-left.
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    success, rvec, tvec = cv2.solvePnP(obj, corners.reshape(4, 2).astype(np.float32), K, D,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success:
        return None
    return rvec, tvec.reshape(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', default='tello_calibration.npz')
    parser.add_argument('--output', default='circle_pose_log.csv')
    args = parser.parse_args()
    K, D, calib_size = load_calibration(args.calibration)
    print(f'Calibration loaded: {args.calibration}; image size={calib_size}; marker={MARKER_SIZE_M} m')
    print('GROUND ONLY. No takeoff/land/RC. Q=quit, S=screenshot.')
    if input('Drone on ground? Type SOLO: ').strip().upper() != 'SOLO':
        print('Cancelled.'); return

    from djitellopy import Tello
    tello = Tello()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params) if hasattr(cv2.aruco, 'ArucoDetector') else None
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = ['timestamp', 'id', 'tx_m', 'ty_m', 'tz_m', 'range_m', 'reprojection_px',
               'both_visible', 'spacing_m', 'center_x_m', 'center_y_m', 'center_z_m']
    connected = False
    try:
        tello.connect()
        connected = True
        tello.streamoff()
        tello.streamon()
        reader = tello.get_frame_read()
        with out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            while True:
                frame = reader.frame
                if frame is None:
                    time.sleep(0.03)
                    continue
                img = frame.copy()
                height, width = img.shape[:2]
                if (width, height) != calib_size:
                    cv2.putText(img, 'RESOLUTION MISMATCH - no pose', (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    cv2.imshow('T1.2 Circle Pose - ground only', img)
                    if (cv2.waitKey(1) & 0xff) in (ord('q'), 27): break
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(gray)
                else:
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
                poses = {}
                if ids is not None:
                    for marker_id, corner in zip(ids.flatten(), corners):
                        marker_id = int(marker_id)
                        if marker_id not in CIRCLE_IDS:
                            continue
                        result = solve_marker(corner, K, D)
                        if result is None: continue
                        rvec, tvec = result
                        poses[marker_id] = (rvec, tvec, corner)
                        pts = corner.reshape(4, 2).astype(int)
                        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                        cv2.drawFrameAxes(img, K, D, rvec, tvec.reshape(3, 1), 0.06)
                        x, y = np.mean(pts, axis=0).astype(int)
                        cv2.putText(img, f'ID {marker_id} Z={tvec[2]:.2f}m', (max(5,x-75), max(20,y-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)
                both = all(i in poses for i in CIRCLE_IDS)
                spacing = None
                center = None
                if both:
                    p0 = poses[0][1]; p1 = poses[1][1]
                    spacing = float(np.linalg.norm(p1-p0))
                    center = (p0+p1)/2
                    # Camera coords: +X right, +Y down, +Z forward.
                    msg = f'MIDPOINT X={center[0]:+.2f} Y={center[1]:+.2f} Z={center[2]:.2f} m'
                    cv2.putText(img, msg, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 0), 2)
                    cv2.putText(img, f'ID separation: {spacing:.3f}m (expected 0.550m)',
                                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 0), 2)
                    # This midpoint is NOT yet a validated safe passage target.
                else:
                    cv2.putText(img, 'Need BOTH IDs 0 and 1 for midpoint', (12, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
                cv2.putText(img, 'GROUND ONLY | Q quit | S screenshot', (12, height-18),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
                now = dt.datetime.now().isoformat(timespec='milliseconds')
                for marker_id, (rvec, tvec, corner) in poses.items():
                    h = MARKER_SIZE_M/2
                    obj = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], np.float32)
                    reproj, _ = cv2.projectPoints(obj, rvec, tvec.reshape(3,1), K, D)
                    err = float(np.sqrt(np.mean(np.sum((reproj.reshape(4,2)-corner.reshape(4,2))**2, axis=1))))
                    writer.writerow(dict(timestamp=now, id=marker_id, tx_m=tvec[0], ty_m=tvec[1],
                                         tz_m=tvec[2], range_m=np.linalg.norm(tvec), reprojection_px=err,
                                         both_visible=int(both), spacing_m=spacing if both else '',
                                         center_x_m=center[0] if both else '',
                                         center_y_m=center[1] if both else '',
                                         center_z_m=center[2] if both else ''))
                cv2.imshow('T1.2 Circle Pose - ground only', img)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), 27): break
                if key == ord('s'):
                    name = 'circle_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
                    cv2.imwrite(name, img)
                    print('Saved', name)
    except KeyboardInterrupt:
        print('\nInterrupted; cleaning up.')
    finally:
        cv2.destroyAllWindows()
        if connected:
            try: tello.streamoff()
            except Exception as exc: print('streamoff warning:', exc)
        print('CSV:', out)

if __name__ == '__main__':
    main()
