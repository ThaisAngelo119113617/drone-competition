# T1.0 — Tello ArUco pose viewer (ground only)

Uses `DICT_4X4_50` and marker black-border side length **0.115 m** by default. No takeoff, land, emergency, or RC commands.

## Setup

Activate your existing venv with `djitellopy`, `numpy` and OpenCV with `cv2.aruco` support.

```bash
python3 t10_pose.py
```

Type `SOLO`. Q or Esc quits, S saves a screenshot. CSV records detections and pixel centers.

## Metric pose

Without a real camera calibration, metric translation and orientation are **not estimated**. To enable them, supply an NPZ with `camera_matrix` (3x3) and `dist_coeffs` (OpenCV distortion coefficients):

```bash
python3 t10_pose.py --calibration tello_camera.npz
```

Calibration must match the actual Tello video resolution and stream settings. OpenCV `solvePnP` translation `tx,ty,tz` is expressed in the camera optical coordinate frame (x right, y down, z forward), not drone/world coordinates. Check corner ordering and calibration accuracy before using for control.

The circle center cannot be determined from marker IDs alone: measure each marker pose relative to the hoop center or calibrate the obstacle geometry separately. Never infer a safe crossing trajectory solely from an ArUco center.
