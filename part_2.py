import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from tello_io import TelloIO, FlightState
from safety_windows import Safety
from detect_aruco import create_detector
from ekf_replay import EKF, wrap, body_speeds, MAX_RANGE_M

ID_LOW = 8
ID_MID = 5
ID_TOP = 17
ID_BLUE = 6
ID_GREEN = 7
ALL_IDS = {ID_LOW, ID_MID, ID_TOP, ID_BLUE, ID_GREEN}
MARKER_SIZE_M = 0.115  # black square only

# Local frame: origin = start point, X forward, Y left(+) / right(-)
MAP = {
    ID_LOW:   (0.75, 0.0),
    ID_MID:   (0.75, 0.0),
    ID_TOP:   (0.75, 0.0),
    ID_BLUE:  (2.35, 0.0),
    ID_GREEN: (2.86, 0.0),
}

# (X, Y, height_cm or None, marker to look at)
WAYPOINTS = [
    (0.45, -0.40, None, ID_TOP),
    (1.15, -0.40, 60,   ID_BLUE),
    (1.95,  0.40, 60,   ID_BLUE),
    (2.70,  0.40, 60,   ID_GREEN),
    (2.86,  0.40, 60,   ID_GREEN),
]

APPROACH_DIST_M = 0.70
TOL_M = 0.08
STABLE_FRAMES = 15
K_P = 0.4
MAX_RC = 20
MIN_RC = 8          # smallest rc value that actually moves the Tello
CLIMB_RC = 20
MAX_CLIMB_CM = 180
SIGMA0 = np.diag([0.05**2, 0.05**2, np.radians(5)**2])
K_WP = 0.5
WP_RC = 15
WP_TOL_M = 0.10
WP_STABLE = 5
K_YAW = 1.0
K_H = 0.5
SIGMA_HOVER_M = 0.35
SIGMA_LAND_M = 0.60
SIGMA_WAIT_S = 4.0
LOST_LAND_S = 2.0
START_WAIT_S = 5.0
CLIMB_TIMEOUT_S = 10.0
WP_TIMEOUT_S = 60.0
MIN_HEIGHT_CM = 30
HOLD_S = 3.0

LOOP_HZ = 20

HERE = Path(__file__).parent


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


def clamp(v, lim=MAX_RC):
    return int(max(-lim, min(lim, v)))


def p_cmd(err_m, deadband_m=TOL_M / 2):
    """P controller with a minimum command: K_P * error, but at least MIN_RC
    when we are further away than deadband_m (otherwise the Tello doesn't move)."""
    u = K_P * err_m * 100
    if abs(err_m) > deadband_m and abs(u) < MIN_RC:
        u = MIN_RC if u > 0 else -MIN_RC
    return clamp(u)


def center_on(pose, target_dist=APPROACH_DIST_M, check_y=True):
    x, y, z = pose
    ex, ey, ez = x, y, z - target_dist
    lr = p_cmd(ex)          # marker to the right -> fly right
    ud = p_cmd(-ey)         # marker below (y>0) -> go down
    fb = p_cmd(ez)          # too far -> fly forward
    done = abs(ex) < TOL_M and abs(ez) < TOL_M and (not check_y or abs(ey) < TOL_M)
    return lr, fb, ud, done


class Mission:
    def __init__(self, dry_run=False):
        self.dry = dry_run
        self.land_reason = None
        self.ekf = EKF([0.0, 0.0, 0.0], SIGMA0)
        self.last_t = None
        self.last_yaw = None
        self.last_meas = {}
        self.n_used = self.n_rej = 0
        self.wp = 0
        self.unsure_since = None
        self.seen_once = False
        self.goto('APPROACH', time.monotonic())

    def goto(self, stage, t):
        self.stage, self.t_stage, self.stable, self.last_seen = stage, t, 0, t
        extra = f' -> WP {self.wp + 1}/{len(WAYPOINTS)} {WAYPOINTS[self.wp][:2]}' if stage == 'WAYPOINTS' else ''
        print(f'[STAGE] {stage}{extra}', flush=True)

    def land(self, why):
        if not self.dry and self.land_reason is None:
            self.land_reason = why

    def check_lost(self, t, mid):
        if t - self.last_seen > LOST_LAND_S:
            self.land(f'(ID {mid} perdido em {self.stage})')

    def debounce(self, ok, need=STABLE_FRAMES):
        # count up when inside tolerance; one noisy frame only costs 2, not everything
        self.stable = self.stable + 1 if ok else max(0, self.stable - 2)
        return self.stable >= need

    def run_ekf(self, poses, s, t):
        vgx, vgy, yaw = s.get('vgx'), s.get('vgy'), s.get('yaw')
        if all(isinstance(v, (int, float)) for v in (vgx, vgy, yaw)):
            if self.last_t is not None and t > self.last_t:
                dt = t - self.last_t
                v_f, v_l = body_speeds(vgx, vgy, yaw)
                w = -wrap(np.radians(yaw - self.last_yaw)) / dt  # Tello yaw is clockwise
                self.ekf.predict(v_f, v_l, w, dt)
            self.last_t, self.last_yaw = t, yaw
        for mid, (cx, cy, cz) in poses.items():
            if mid not in MAP or np.hypot(cx, cz) > MAX_RANGE_M:
                continue
            if self.last_meas.get(mid) == (cx, cz):  # same frame again
                continue
            self.last_meas[mid] = (cx, cz)
            if self.ekf.update(cx, cz, MAP[mid]):
                self.n_used += 1
            else:
                self.n_rej += 1

    def fly_to(self, X_w, Y_w, look_id=None):
        x, y, th = self.ekf.mu
        ex, ey = X_w - x, Y_w - y
        e_fwd = np.cos(th) * ex + np.sin(th) * ey
        e_left = -np.sin(th) * ex + np.cos(th) * ey
        fb = clamp(K_WP * e_fwd * 100, WP_RC)
        lr = clamp(-K_WP * e_left * 100, WP_RC)
        th_target = 0.0
        if look_id is not None:
            mx, my = MAP[look_id]
            th_target = float(np.arctan2(my - y, mx - x))
        yaw_rc = clamp(K_YAW * np.degrees(wrap(th - th_target)))
        return lr, fb, yaw_rc, float(np.hypot(ex, ey))

    def update(self, poses, s, t):
        self.run_ekf(poses, s, t)
        lr = fb = ud = yaw_rc = 0
        used = None
        h, tof = s.get('h', 0), s.get('tof')
        st = self.stage

        if st == 'APPROACH':
            if ID_LOW in poses:
                self.seen_once, self.last_seen = True, t
                used = poses[ID_LOW]
                lr, fb, ud, ok = center_on(used, check_y=False)
                if self.debounce(ok):
                    self.goto('CLIMB', t)
            else:
                self.stable = 0
                if not self.seen_once:
                    if t - self.t_stage > START_WAIT_S:
                        self.land(f'(ID {ID_LOW} nunca visto)')
                else:
                    self.check_lost(t, ID_LOW)

        elif st == 'CLIMB':
            if ID_TOP in poses:
                used = poses[ID_TOP]
                lr, fb, ud, ok = center_on(used)
                if self.debounce(ok):
                    self.wp = 0
                    self.goto('WAYPOINTS', t)
            else:
                ud = CLIMB_RC
                for mid in (ID_MID, ID_LOW):
                    if mid in poses:
                        used = poses[mid]
                        lr, fb, _, _ = center_on(used)
                        break
                if isinstance(h, (int, float)) and h > MAX_CLIMB_CM:
                    ud = 0
                    self.land(f'(h={h} sem ID {ID_TOP})')
                if t - self.t_stage > CLIMB_TIMEOUT_S:
                    self.land('(timeout CLIMB)')

        elif st == 'WAYPOINTS':
            X_w, Y_w, h_cm, look_id = WAYPOINTS[self.wp]
            lr, fb, yaw_rc, dist = self.fly_to(X_w, Y_w, look_id)
            if h_cm is not None and isinstance(tof, (int, float)) and 20 < tof < 300:
                ud = clamp(K_H * (h_cm - tof))
            sig = self.ekf.sigma_xy()
            if sig > SIGMA_LAND_M:
                self.land(f'(EKF perdido: sigma={sig:.2f} m)')
            elif sig > SIGMA_HOVER_M:
                lr = fb = 0
                self.unsure_since = self.unsure_since or t
                if t - self.unsure_since > SIGMA_WAIT_S:
                    self.land(f'(EKF inseguro {SIGMA_WAIT_S:.0f} s: sigma={sig:.2f} m)')
            else:
                self.unsure_since = None
            if self.debounce(dist < WP_TOL_M, WP_STABLE):
                x, y, _ = self.ekf.mu
                print(f'[WP] {self.wp + 1} reached: EKF=({x:+.2f}, {y:+.2f}) sigma={sig:.2f}', flush=True)
                if self.wp + 1 < len(WAYPOINTS):
                    self.wp += 1
                    self.goto('WAYPOINTS', t)
                else:
                    self.goto('HOLD', t)
            elif t - self.t_stage > WP_TIMEOUT_S:
                self.land(f'(timeout WP {self.wp + 1})')

        elif st == 'HOLD':
            X_w, Y_w, _, look_id = WAYPOINTS[-1]
            lr, fb, yaw_rc, _ = self.fly_to(X_w, Y_w, look_id)
            if t - self.t_stage > HOLD_S:
                self.land('(fim da parte 2)')
                if self.dry:
                    self.__init__(dry_run=True)

        return lr, fb, ud, yaw_rc, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='no takeoff, only print commands')
    opts = ap.parse_args()

    K, D = load_calibration()
    detect = create_detector('DICT_4X4_50')

    log_dir = HERE / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / ('part2_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')

    io = TelloIO(log_path=None)
    safety = Safety(io.t, max_height_cm=200, min_battery=20,
                    stall_stop_s=1, stall_land_s=3, max_mission_s=120)
    started = airborne_possible = False

    with log_path.open('w', newline='') as f:
        log = csv.writer(f)
        log.writerow(['t', 'stage', 'wp', 'ids', 'x', 'y', 'z', 'lr', 'fb', 'ud', 'yaw_rc',
                      'ekf_x', 'ekf_y', 'ekf_th_deg', 'ekf_sigma', 'vgx', 'vgy', 'yaw',
                      'h_cm', 'tof_cm', 'bat'])
        try:
            io.connect(video=True)
            time.sleep(2.0)
            if io.t.get_battery() < 40 and not opts.dry_run:
                print('[ABORT] bateria < 40%')
                return

            if not opts.dry_run:
                if input('Area livre, drone no ponto de partida virado para o poste roxo? Digite VOAR: ').strip() != 'VOAR':
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

            mission = Mission(dry_run=opts.dry_run)
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
                poses = {i: p for i, p in poses.items() if i in ALL_IDS}

                s = io.t.get_current_state() or {}
                h, tof = s.get('h', 0), s.get('tof')

                if safety.hover_requested.is_set():
                    mission.run_ekf(poses, s, t_loop)
                    lr = fb = ud = yaw_rc = 0
                    used = None
                    stage_note = 'HOVER(key/watchdog)'
                else:
                    lr, fb, ud, yaw_rc, used = mission.update(poses, s, t_loop)
                    stage_note = mission.stage
                    if mission.land_reason:
                        safety.request_land(mission.land_reason)

                if isinstance(tof, (int, float)) and 20 < tof < MIN_HEIGHT_CM:
                    ud = max(ud, 0)
                if not opts.dry_run:
                    io.rc(lr=lr, fb=fb, ud=ud, yaw=yaw_rc)

                ex, ey, eth = mission.ekf.mu
                sig = mission.ekf.sigma_xy()
                xyz = [round(v, 3) for v in used] if used else ['', '', '']
                log.writerow([round(t_loop, 3), stage_note, mission.wp, sorted(poses), *xyz,
                              lr, fb, ud, yaw_rc, round(ex, 3), round(ey, 3),
                              round(np.degrees(eth), 1), round(sig, 3),
                              s.get('vgx'), s.get('vgy'), s.get('yaw'), h, tof, s.get('bat')])

                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                txt1 = f'{stage_note} wp={mission.wp + 1} lr={lr} fb={fb} ud={ud} yaw={yaw_rc} h={h} tof={tof}'
                txt2 = f'EKF x={ex:+.2f} y={ey:+.2f} th={np.degrees(eth):+.0f} sigma={sig:.2f}  ids={sorted(poses)}'
                cv2.putText(frame, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(frame, txt2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow('part 2', frame)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if opts.dry_run and key in (ord('q'), 27):
                        break
                    safety.handle_key(chr(key))
                if opts.dry_run and t_loop % 0.5 < period:
                    print('[DRY]', txt1, '|', txt2, flush=True)

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