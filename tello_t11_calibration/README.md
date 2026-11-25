# T1.1 — Tello camera calibration (ground only)

Checkerboard: 10 columns × 7 rows of complete squares, each 20 mm. OpenCV detects 9 × 6 **internal corners**.

## Setup

Connect the computer to Tello Wi-Fi. Keep the drone on the ground; this program sends no takeoff, land or RC commands. Install dependencies in your virtualenv:

```bash
pip install djitellopy opencv-python numpy
python3 calibrate_tello.py
```

Type `SOLO`. Show the entire checkerboard to the camera, in focus, with good lighting. Press **SPACE** to save a view only when the board is detected. Capture 20–30 distinct views with different tilt angles, distances, and positions across the frame (including edges and corners). Avoid saving many near-identical images. The full board must remain visible.

Press **C** to calibrate after at least 15 captures. Output: `calibration_output/tello_calibration.npz` (K, distortion, image size, RMS, per-view errors) and `calibration_output/per_view_errors.csv`. Press **Q** to exit. Ctrl+C is also handled.

Use the same camera video resolution and image processing when applying calibration to ArUco pose estimation. A low overall RMS alone does not prove metric accuracy: inspect per-view errors and validate against a known marker distance.

Note: the Tello camera is fixed; move or tilt the **board**, not the drone. Keep the board flat. If the camera preview never shows `Found: True`, confirm it is an ordinary alternating black-white checkerboard with exactly 10×7 squares and no obscured corners.
