# TEL310 – Tello Drone Challenge (NMBU)

**Repository:** https://github.com/ThaisAngelo119113617/drone-competition

Autonomous DJI/Ryze Tello mission using ArUco markers, a known marker map and EKF localization
(Probabilistic Robotics, TEL310).

The repository has several folders. **Everything needed to run the drone challenge is in
`DRONE_CHALLENGE/`.** Run all commands from inside that folder.

```bash
git clone https://github.com/ThaisAngelo119113617/drone-competition.git
cd drone-competition/DRONE_CHALLENGE
```

---

## Mission overview

One flight, in this order:

1. **Hoop**: align with the hoop (markers ID 0, 1, 10, 12), fly through it and approach ID 3.
2. **Vertical slalom**: climb over the top bar (ID 2, 17), descend to the gap (ID 5), fly back
   through the gap, go down and cross the low gap (ID 3, 4), then approach ID 8.
3. **Weave between the poles**: purple (ID 8 bottom / ID 5 top), blue (ID 6), green (ID 7),
   then fly to the plant (ID 13).
4. **Plant lap**: fly around the plant past ID 13, 9, 11 and 16, then go to ID 18 and ID 19.
5. **Final tour**: ID 19, then ID 23 (left), back to ID 19, then ID 39 (up), back to ID 19, then **land**.

Localization:
- `ekf_tello.py`: EKF with known correspondences (state x, y, z, yaw) for the hoop and the
  slalom, using the map `aruco_map_circulo_calibrado.yaml`. By default it runs in "shadow mode":
  it estimates and logs the position but does not steer. It only steers with `--ekf-control`.
- `mission_all_5.py`: a second EKF (state x, y, θ, range/bearing updates) for the weave and the
  plant. Its marker maps are written in the code (`WV_MAP`, `plant_marker_map()`).

---

## Folder structure

```
drone-competition/
├── README.md                          # this file
├── DRONE_CHALLENGE/                    # <-- THE CHALLENGE: everything needed to fly the mission
│   ├── mission_all_5.py               # MAIN PROGRAM: run this
│   ├── ekf_tello.py                   # EKF for the hoop and the slalom (imported by mission_all_5.py)
│   ├── pose_global.py                 # map loader and pose geometry (imported by ekf_tello.py)
│   ├── aruco_map_circulo_calibrado.yaml   # marker map for the hoop and the slalom
│   ├── tello_calibration.npz          # camera calibration (camera_matrix, dist_coeffs, image_size)
│   ├── safety.py                      # older helper, NOT used by mission_all_5.py
│   └── logs/                          # flight logs (new ones are added on every run)
├── TEST_FILES/                        # development and test scripts (earlier versions, labs t06–t19,
│                                      # part_2.py weave, ekf_replay.py, hover_test.py): NOT needed to run the mission
└── side_notes_for_what_we_did/        # notes on the methods and formulas used
```

**The file names in `DRONE_CHALLENGE/` must stay exactly as above.** `mission_all_5.py` imports
`ekf_tello` and `pose_global`. 
