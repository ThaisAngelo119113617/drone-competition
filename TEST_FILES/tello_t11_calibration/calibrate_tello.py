import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from djitellopy import Tello

PATTERN = (9, 6)  # 10x7 squares -> 9x6 inner corners
SQUARE_M = 0.020

def main():
    p = argparse.ArgumentParser(description='Tello camera checkerboard calibration, ground only')
    p.add_argument('--output', default='calibration_output')
    p.add_argument('--min-images', type=int, default=15)
    args = p.parse_args()
    output = Path(args.output)
    images = output / 'captures'
    images.mkdir(parents=True, exist_ok=True)
    print('GROUND ONLY: no takeoff/land/rc. 10x7 squares, 2cm each; 9x6 inner corners.')
    if input('Tello on ground? Type SOLO: ').strip() != 'SOLO':
        return
    tello = Tello()
    obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_M
    object_points, image_points = [], []
    image_size = None
    stream_on = False
    try:
        tello.connect()
        try:
            tello.streamoff()
        except Exception:
            pass
        tello.streamon()
        stream_on = True
        reader = tello.get_frame_read()
        print('Move/tilt the board. SPACE=save detected board; C=calibrate; Q=quit.')
        while True:
            frame = reader.frame
            if frame is None or frame.size == 0:
                cv2.waitKey(30)
                continue
            frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            if image_size is not None and size != image_size:
                print('ERROR: video resolution changed; stopping to avoid mixed calibration data')
                break
            found, corners = cv2.findChessboardCorners(gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            refined = None
            if found:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(frame, PATTERN, refined, True)
            cv2.putText(frame, f'Found: {found} | saved: {len(object_points)} | SPACE save | C calibrate | Q quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow('Tello T1.1 calibration', frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key == 32:
                if not found:
                    print('No complete checkerboard detected; not saved')
                    continue
                image_size = size
                object_points.append(obj.copy())
                image_points.append(refined.copy())
                filename = images / f'board_{len(object_points):03d}.png'
                cv2.imwrite(str(filename), gray)
                print(f'Saved {filename} ({len(object_points)} views)')
            if key in (ord('c'), ord('C')):
                if len(object_points) < args.min_images:
                    print(f'Need at least {args.min_images} valid views; have {len(object_points)}')
                    continue
                rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
                per_view = []
                for o, im, r, t in zip(object_points, image_points, rvecs, tvecs):
                    projected, _ = cv2.projectPoints(o, r, t, K, dist)
                    per_view.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-im.reshape(-1,2))**2, axis=1)))))
                np.savez(output / 'tello_calibration.npz', camera_matrix=K, dist_coeffs=dist, image_size=np.array(image_size), rms=np.array(rms), per_view_errors=np.array(per_view), square_size_m=np.array(SQUARE_M), pattern_inner_corners=np.array(PATTERN))
                with open(output / 'per_view_errors.csv', 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['view', 'reprojection_rmse_px'])
                    w.writerows((i+1, round(e, 4)) for i, e in enumerate(per_view))
                print(f'CALIBRATION SAVED: {output / "tello_calibration.npz"}')
                print(f'Views={len(object_points)} resolution={image_size} RMS={rms:.4f} px')
                print('K=\n', K, '\nDistortion=', dist.ravel())
                print('Check per_view_errors.csv; large RMS or outliers mean you should recapture varied sharp views.')
    except KeyboardInterrupt:
        print('\nInterrupted. Calibration not saved unless C was pressed.')
    finally:
        cv2.destroyAllWindows()
        if stream_on:
            try:
                tello.streamoff()
            except Exception as e:
                print('streamoff warning:', e)
        try:
            tello.end()
        except Exception as e:
            print('end warning:', e)

if __name__ == '__main__':
    main()
