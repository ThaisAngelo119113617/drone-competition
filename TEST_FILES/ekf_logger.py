"""Hand-carry logger for the EKF. NO flight commands are sent.

Keys in the video window:
  1-9  = "I am standing on checkpoint N right now" (hold still ~3 s, press once)
  0    = free note marker (e.g. start/end of a run)
  Q/ESC = stop
"""
import csv
import time
from datetime import datetime

import cv2

from tello_io import TelloIO
from detect_aruco import create_detector
from part_2 import load_calibration, marker_poses, ALL_IDS, HERE


def main():
    K, D = load_calibration()
    detect = create_detector('DICT_4X4_50')

    log_dir = HERE / 'logs'
    log_dir.mkdir(exist_ok=True)
    path = log_dir / ('ekf_hand_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')

    io = TelloIO(log_path=None)
    try:
        io.connect(video=True)
        time.sleep(2.0)
        print('[LOGGER] gravando em', path)
        print('[LOGGER] 1-9 = checkpoint, 0 = nota, Q = sair. NENHUM comando de voo.')
        t0 = time.monotonic()
        with path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'kind', 'id', 'cam_x', 'cam_y', 'cam_z',
                        'vgx', 'vgy', 'vgz', 'yaw', 'pitch', 'roll', 'h', 'tof', 'bat'])
            while True:
                frame = io.frame()
                if frame is None or frame.size == 0:
                    cv2.waitKey(30)
                    continue
                t = round(time.monotonic() - t0, 3)
                frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                poses, corners, ids = marker_poses(gray, detect, K, D)
                poses = {i: p for i, p in poses.items() if i in ALL_IDS}

                s = io.t.get_current_state() or {}
                w.writerow([t, 'state', '', '', '', '',
                            *[s.get(k, '') for k in ('vgx', 'vgy', 'vgz', 'yaw', 'pitch',
                                                     'roll', 'h', 'tof', 'bat')]])
                for mid, (x, y, z) in poses.items():
                    w.writerow([t, 'marker', mid, round(x, 4), round(y, 4), round(z, 4),
                                '', '', '', '', '', '', '', '', ''])

                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                txt = (f"t={t:.1f} vgx={s.get('vgx')} vgy={s.get('vgy')} yaw={s.get('yaw')} "
                       f"ids={sorted(poses)}")
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow('EKF logger (sem voo)', frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if ord('0') <= key <= ord('9'):
                    cp = key - ord('0')
                    w.writerow([t, 'checkpoint', cp, '', '', '', '', '', '', '', '', '', '', '', ''])
                    f.flush()
                    print(f'[CHECKPOINT] {cp} em t={t:.1f}s', flush=True)
    finally:
        cv2.destroyAllWindows()
        io.close()
        print('[LOGGER] fim:', path)


if __name__ == '__main__':
    main()