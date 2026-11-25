#!/usr/bin/env python3
"""T1.0: ground-only ArUco pose viewer for DJI Tello; never sends flight commands."""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from djitellopy import Tello


def detect(gray, dictionary, params):
    if hasattr(cv2.aruco, 'ArucoDetector'):
        return cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)[:2]
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids


def main():
    ap = argparse.ArgumentParser(description='T1.0 ground-only Tello ArUco detection and pose')
    ap.add_argument('--size', type=float, default=0.115, help='Black-border to black-border size in metres')
    ap.add_argument('--dict', default='DICT_4X4_50', help='OpenCV ArUco dictionary')
    ap.add_argument('--calibration', type=Path, help='NPZ containing camera_matrix and dist_coeffs')
    ap.add_argument('--log', type=Path, default=Path('t10_detections.csv'))
    args = ap.parse_args()
    if args.size <= 0: ap.error('--size must be positive')
    if not hasattr(cv2, 'aruco'): ap.error('OpenCV aruco missing; install opencv-contrib-python')
    if not hasattr(cv2.aruco, args.dict): ap.error('Unknown dictionary')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    params = cv2.aruco.DetectorParameters()
    camera_matrix = dist_coeffs = None
    if args.calibration:
        with np.load(args.calibration) as data:
            camera_matrix = np.asarray(data['camera_matrix'], dtype=np.float64)
            dist_coeffs = np.asarray(data['dist_coeffs'], dtype=np.float64)
        if camera_matrix.shape != (3, 3): ap.error('camera_matrix must be 3x3')
    else:
        print('NO CALIBRATION: IDs and pixel centers only; no metric pose or distance.')
    print('GROUND ONLY. No takeoff, land or RC commands. Q=quit, S=save image.')
    if input('Drone on ground? Type SOLO: ').strip() != 'SOLO': return
    drone = Tello()
    streaming = False
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['unix_time','id','cx_px','cy_px','tx_m','ty_m','tz_m','reprojection_error_px'])
        try:
            drone.connect()
            drone.streamoff()  # reset previous stream session
            drone.streamon()
            streaming = True
            reader = drone.get_frame_read()
            last_print = {}
            while True:
                frame = reader.frame
                if frame is None:
                    time.sleep(.05)
                    continue
                frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids = detect(gray, dictionary, params)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    for c, marker_id in zip(corners, ids.flatten()):
                        pts = c.reshape(4, 2)
                        center = pts.mean(axis=0)
                        tx = ty = tz = err = ''
                        if camera_matrix is not None:
                            half = args.size / 2
                            obj = np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]], dtype=np.float64)
                            ok, rvec, tvec = cv2.solvePnP(obj, pts.astype(np.float64), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                            if ok:
                                tx,ty,tz = map(float,tvec.reshape(3))
                                projected,_ = cv2.projectPoints(obj,rvec,tvec,camera_matrix,dist_coeffs)
                                err = float(np.sqrt(np.mean(np.sum((projected.reshape(4,2)-pts)**2,axis=1))))
                                cv2.drawFrameAxes(frame,camera_matrix,dist_coeffs,rvec,tvec,args.size/2)
                        writer.writerow([f'{time.time():.6f}',int(marker_id),round(float(center[0]),2),round(float(center[1]),2),tx,ty,tz,err])
                        label = f'ID {marker_id} ({center[0]:.0f},{center[1]:.0f})'
                        if tz != '': label += f' Z={tz:.2f}m'
                        cv2.putText(frame,label,(int(center[0]),max(18,int(center[1])-12)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),2)
                        now=time.monotonic()
                        if now-last_print.get(int(marker_id),0)>1:
                            print(label)
                            last_print[int(marker_id)]=now
                cv2.imshow('T1.0 ArUco ground-only',frame)
                key=cv2.waitKey(1)&0xff
                if key in (ord('q'),27): break
                if key==ord('s'):
                    name=f't10_frame_{int(time.time())}.png'
                    cv2.imwrite(name,frame)
                    print('Saved',name)
        finally:
            cv2.destroyAllWindows()
            if streaming:
                try: drone.streamoff()
                except Exception as exc: print('streamoff:',exc)
            try: drone.end()
            except Exception as exc: print('end:',exc)
    print('Saved',args.log)

if __name__ == '__main__':
    main()
