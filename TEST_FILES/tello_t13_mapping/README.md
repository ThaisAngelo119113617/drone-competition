# T1.3 — Mapping of circle (IDs 00/01) and rear obstacle (ID 03)

GROUND ONLY. This program does not take off, land, or send RC commands.

## Run

Connect laptop to Tello Wi-Fi, activate the existing Python virtualenv, and run:

```bash
python3 t13_mapping.py
```

Type `SOLO`, point the camera toward the hoop and rear ID 03, press `S` for a screenshot and `Q` to quit. `Ctrl+C` also closes the video stream. The script creates `t13_map.csv`.

Uses the user's camera calibration for 960x720, ArUco DICT_4X4_50, 0.115m printed black-border size. IDs 0/1 are bottom/top of hoop and separated by 0.55m. ID 03 is behind the hoop. The 0.75m stopping goal is a *future* reference, not an automatic stop or movement command.

The estimated midpoint of IDs 0 and 1 is a proxy for the opening center, valid only if the markers are symmetric about the opening. The ID 03 coordinates in the hoop frame are estimated per frame, and must be validated while moving the camera in ground-only tests. The local frame orientation uses marker normals and assumes both markers are mounted in similar orientation on the same plane. `id3_circle_depth_m` should be positive for a marker behind the hoop; if negative, inspect marker mounting and orientation. `id3_z_m` is optical-axis depth, not Euclidean range (`id3_range_m`).

Quality gates: keep all three markers in view, inspect reprojection RMS, compare ID0–ID1 distance with 0.55m and ID03 depth with tape measurements. Validate free clearance through the hoop and clearance from the rear bar; never command blind forward flight from a single marker measurement. EKF and flight control are **not** implemented in this test.
