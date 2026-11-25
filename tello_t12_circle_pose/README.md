# T1.2 — Circle ArUco 3D pose (ground only)

Run from this directory, connected to Tello Wi-Fi, with the drone **on the ground**:

```bash
python3 t12_circle_pose.py
```

Requires `djitellopy`, `opencv-contrib-python`, `numpy`. Type `SOLO` to confirm ground-only operation. Press `S` for screenshot, `Q` to quit; Ctrl+C is also handled. No flight commands are sent.

Calibration: included `tello_calibration.npz` from your T1.1 run (960×720). Do not resize frames without adjusting K. Marker size: **0.115 m**, black edge to black edge. Dictionary `DICT_4X4_50`. Only IDs **0 (bottom)** and **1 (top)** are used. Other obstacles' IDs are ignored.

Each marker pose is camera-relative: +X right, +Y down, +Z forward. The CSV includes translation and marker reprojection error. When both IDs appear, it also reports their 3D center-to-center spacing (expected 0.55 m) and midpoint. **The midpoint is only a candidate circle center**: it assumes the two markers are symmetric about the true opening center and share the relevant plane. It is not a safe waypoint until the opening geometry, plane orientation, and drone clearance are measured and validated. When only one marker is visible, no center is inferred.

Test: record with both IDs visible, then at a tape-measured camera-to-marker distance; move the camera sideways/tilt while remaining on ground. Send CSV and screenshot for analysis. The distance shown is relative to the *camera optical center*, not the drone center. A ground-only camera position may not match flight geometry.

Known caveat: the checkerboard calibration had an outlier view; confirm distance against tape before relying on metric estimates. No EKF is implemented yet; this test supplies observations for a later filter.
