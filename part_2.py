# Standard library
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path


# Third-party
import cv2
import numpy as np

# our own files
from tello_io import TelloIO, FlightState
from safety_windows import Safety
from detect_aruco import create_detector


"""
this part is for the purple pole that has 2 ArUco one on the bottom and one on top

the goal for this part is to fly the drom from the bottom ArUco straigt to the top one 
"""
#course setup
ID_LOW = 8           
ID_TOP = 5         
MARKER_SIZE_M = 0.115  # black square only
 
# tuning
APPROACH_DIST_M = 0.70   # how far in front of the pole we stop (camera -> marker)
TOL_M = 0.08             # "close enough" for x, y and distance
STABLE_FRAMES = 15       # must stay inside tolerance this many loops 
K_P = 0.4                # rc units per cm of error (P controller)
MAX_RC = 20              # max |rc| we ever send (Tello range is -100..100)
CLIMB_RC = 20            # upward speed while searching for the top marker
MAX_CLIMB_CM = 180       # stop climbing above this height ('h' telemetry)
LOST_LAND_S = 2.0        # marker lost longer than this during APPROACH -> land
START_WAIT_S = 5.0       # after takeoff: wait this long to see ID_LOW, else land
CLIMB_TIMEOUT_S = 10.0
HOLD_S = 3.0             # hover in front of top marker, then land (test only)
LOOP_HZ = 20
 
HERE = Path(__file__).parent
 
 
# vision
def load_calibration():
    d = np.load(HERE / 'tello_calibration.npz')
    return d['camera_matrix'], d['dist_coeffs']
 
 
_h = MARKER_SIZE_M / 2
# Same corner order as ArUco: top-left, top-right, bottom-right, bottom-left
OBJ_PTS = np.array([[-_h, _h, 0], [_h, _h, 0], [_h, -_h, 0], [-_h, -_h, 0]], dtype=np.float32)
 
 
def marker_poses(gray, detect, K, D):
    """Return {id: (x, y, z)} in metres, camera frame: x right, y DOWN, z forward."""
    corners, ids, _ = detect(gray)
    poses = {}
    if ids is None:
        return poses, corners, ids
    for mid, c in zip(ids.flatten(), corners):
        ok, _rvec, tvec = cv2.solvePnP(OBJ_PTS, c.reshape(4, 2).astype(np.float32), K, D,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            poses[int(mid)] = tuple(float(v) for v in tvec.flatten())
    return poses, corners, ids
 
 
# control
def clamp(v, lim=MAX_RC):
    return int(max(-lim, min(lim, v)))
 
 
def center_on(pose, target_dist=APPROACH_DIST_M):
    """P controller: marker straight ahead, at camera height, target_dist away.
    Returns (lr, fb, ud) and whether we are within tolerance."""
    x, y, z = pose
    ex, ey, ez = x, y, z - target_dist          # metres
    lr = clamp(K_P * ex * 100)                  # marker to the right -> fly right
    ud = clamp(-K_P * ey * 100)                 # marker below (y>0) -> go down
    fb = clamp(K_P * ez * 100)                  # too far -> fly forward
    done = abs(ex) < TOL_M and abs(ey) < TOL_M and abs(ez) < TOL_M
    return lr, fb, ud, done
 
 
# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='no takeoff, only print commands')
    opts = ap.parse_args()
    if ID_TOP is None and not opts.dry_run:
        raise SystemExit('Set ID_TOP before flying.')
 
    K, D = load_calibration()
    detect = create_detector('DICT_4X4_50')
 
    log_dir = HERE / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / ('purple_pole_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')
 
    io = TelloIO(log_path=None)
    safety = Safety(io.t, max_height_cm=200, min_battery=20,
                    stall_stop_s=1, stall_land_s=3, max_mission_s=60)
    started = airborne_possible = False
 
    with log_path.open('w', newline='') as f:
        log = csv.writer(f)
        log.writerow(['t', 'stage', 'ids', 'x', 'y', 'z', 'lr', 'fb', 'ud', 'h_cm', 'tof_cm', 'bat'])
        try:
            io.connect(video=True)
            time.sleep(2.0)  # let the video stream start
            if io.t.get_battery() < 40 and not opts.dry_run:
                print('[ABORT] bateria < 40%')
                return
 
            if not opts.dry_run:
                if input('Area livre, drone a 2.2 m virado para o poste roxo? Digite VOAR: ').strip() != 'VOAR':
                    return
                safety.start()
                started = True
                airborne_possible = True
                try:
                    io.takeoff_once()
                except Exception as exc:
                    print('[TAKEOFF INCERTO]', exc, flush=True)
                if io.state != FlightState.FLYING:
                    safety.request_land('(takeoff incerto)')
                else:
                    safety.arm_watchdog()
 
            stage = 'APPROACH'
            t_stage = time.monotonic()
            last_seen = None
            stable = 0
            period = 1.0 / LOOP_HZ
 
            while not safety.abort.is_set():
                t_loop = time.monotonic()
                safety.heartbeat()
 
                frame = io.frame()
                if frame is None or frame.size == 0:
                    io.hover()
                    time.sleep(period)
                    continue
                frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                poses, corners, ids = marker_poses(gray, detect, K, D)
                # Only trust the purple-pole markers; ignore the floor mat etc.
                poses = {i: p for i, p in poses.items() if i in (ID_LOW, ID_TOP)}
 
                s = io.t.get_current_state() or {}
                h = s.get('h', 0)
                lr = fb = ud = 0
                used = None
 
                if safety.hover_requested.is_set():
                    stage_note = 'HOVER(key/watchdog)'
                elif stage == 'APPROACH':
                    if ID_LOW in poses:
                        last_seen = t_loop
                        used = poses[ID_LOW]
                        lr, fb, ud, ok = center_on(used)
                        stable = stable + 1 if ok else 0
                        if stable >= STABLE_FRAMES:
                            stage, t_stage, stable = 'CLIMB', t_loop, 0
                            print('[STAGE] CLIMB', flush=True)
                    else:
                        stable = 0
                        if last_seen is None:   # never seen yet since takeoff
                            lost_for, limit = t_loop - t_stage, START_WAIT_S
                        else:
                            lost_for, limit = t_loop - last_seen, LOST_LAND_S
                        if lost_for > limit and not opts.dry_run:
                            safety.request_land(f'(ID {ID_LOW} perdido)')
                elif stage == 'CLIMB':
                    if ID_TOP in poses:
                        used = poses[ID_TOP]
                        lr, fb, ud, ok = center_on(used)
                        stable = stable + 1 if ok else 0
                        if stable >= STABLE_FRAMES:
                            stage, t_stage, stable = 'HOLD', t_loop, 0
                            print('[STAGE] HOLD', flush=True)
                    else:
                        ud = CLIMB_RC
                        if ID_LOW in poses:            # keep distance/sideways while it is visible
                            used = poses[ID_LOW]
                            lr, fb, _, _ = center_on(used)
                        if isinstance(h, (int, float)) and h > MAX_CLIMB_CM:
                            ud = 0
                            if not opts.dry_run:
                                safety.request_land(f'(h={h} sem ID {ID_TOP})')
                        if t_loop - t_stage > CLIMB_TIMEOUT_S and not opts.dry_run:
                            safety.request_land('(timeout CLIMB)')
                elif stage == 'HOLD':
                    if ID_TOP in poses:
                        used = poses[ID_TOP]
                        lr, fb, ud, _ = center_on(used)
                    if t_loop - t_stage > HOLD_S:
                        if opts.dry_run:
                            stage = 'APPROACH'   # loop the test on the ground
                        else:
                            safety.request_land('(fim do teste roxo)')
 
                if not safety.hover_requested.is_set():
                    stage_note = stage
                if safety.hover_requested.is_set():
                    lr = fb = ud = 0
                if not opts.dry_run:
                    io.rc(lr=lr, fb=fb, ud=ud, yaw=0)
 
                xyz = used if used else ('', '', '')
                log.writerow([round(t_loop, 3), stage_note, list(poses), *[round(v, 3) if v != '' else '' for v in xyz],
                              lr, fb, ud, h, s.get('tof'), s.get('bat')])
 
                # ---- display (also lets you press the safety keys in the video window)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                txt = f'{stage_note} lr={lr} fb={fb} ud={ud} h={h}'
                if used:
                    txt += f'  x={used[0]:+.2f} y={used[1]:+.2f} z={used[2]:.2f}'
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('purple pole', frame) #puple_pole
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if opts.dry_run and key in (ord('q'), 27):
                        break
                    safety.handle_key(chr(key))
                if opts.dry_run and t_loop % 0.5 < period:
                    print('[DRY]', txt, flush=True)
 
                time.sleep(max(0.0, period - (time.monotonic() - t_loop)))
 
        except KeyboardInterrupt:
            safety.request_land('(Ctrl+C)')
        except Exception as exc:
            print('[ERRO]', repr(exc), flush=True)
            safety.request_land('(erro)')
        finally:
            if safety.emergency_requested.is_set():
                io.mark_emergency()
            elif airborne_possible and not io.land_latched:
                io.land_once()
            if started:
                safety.stop()
            cv2.destroyAllWindows()
            io.close()
            print('[FINAL] log=', log_path, 'state=', io.state.value, flush=True)
            if airborne_possible:
                print('[ATENCAO] Confira visualmente que o drone pousou.', flush=True)
 
 
if __name__ == '__main__':
    main()
