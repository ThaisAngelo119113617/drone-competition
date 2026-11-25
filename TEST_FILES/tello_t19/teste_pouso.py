"""MISSION ALL 4 (hoop T2.0 -> vertical slalom T2.1 -> weave -> plant lap -> ID18 -> land in front of ID19, in ONE flight). Built from t20 + t21 (see MERGE NOTES below).

T1.9 CROSS - supervised hoop-center alignment for the Tello. Auto-align is lateral/vertical plus a small forward/back hold of the distance to the hoop (--hold-z, 0 = off; no yaw); F,Y runs one supervised forward move.

Architecture (why it differs from T1.7):
  * GUI/vision loop (main thread) NEVER sends blocking commands and never calls input().
    Takeoff is confirmed inside the window (T, then Y within 3 s).
  * Commander thread owns ALL ordinary Tello commands (takeoff, rc, land). It uses fire-and-forget
    UDP sends and decides success from TELEMETRY, not from ACKs (ACKs are FIFO in djitellopy and can be stale).
  * EMERGENCY (E or K) is sent directly from the key handler / terminal thread; it never waits for any lock.
  * Escalating watchdog: GUI stalled >0.5 s -> rc zero; >3 s -> the commander lands the drone by itself.
    The watchdog never acts as a plain keep-alive.
  * takeoff is sent ONCE, never retried. land is retried at most 3 times, only when telemetry shows no descent.
  * If this process dies, the Tello's own 15 s failsafe lands it (observed in T1.7).

Keys (OpenCV window focus, or the terminal): E or K = EMERGENCY MOTOR CUT (drone falls) | P or L = land
  SPACE = cancel auto/climb and hover | T then Y = takeoff | G = auto alignment on/off | Q/ESC = land and quit
"""
import argparse
import atexit
import csv
import logging
import math
import select
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import termios
    import tty
except ImportError:  # non-POSIX
    termios = tty = None

try:                                    # EKF in SHADOW mode (logs only; never drives the drone). Needs ekf_tello.py + pose_global.py + the map next to this file.
    import TEST_FILES.tello_t19.ekf_tello as _ekf_mod
    from pose_global import load_map as _ekf_load_map
except Exception as _ekf_err:           # missing files/modules: the mission runs exactly as before
    _ekf_mod = None
    _ekf_import_err = _ekf_err

ZERO = (0, 0, 0, 0)
LOG = logging.getLogger('t18')
AIRBORNE_STATES = ('TAKING_OFF', 'FLYING', 'CROSSING', 'UNCERTAIN', 'LANDING')


class Cfg:
    def __init__(self, **kw):
        self.height_source = 'tof'
        self.target_height = 130
        self.max_seconds = 35.0
        self.min_battery_takeoff = 50
        self.min_battery_flight = 20
        self.max_height = 280           # cm abort limit (was 250: flight 155926 drifted up to tof 249 after the top bar, 1 cm from the cut)
        self.rc_hz = 20
        self.rc_limit = 25
        self.setpoint_ttl = 0.3
        self.gui_stall_zero = 0.5
        self.gui_stall_land = 3.0
        self.tele_stale_zero = 2.5
        self.tele_stale_land = 4.0
        self.video_stale_zero = 0.7
        self.video_stale_land = 2.0
        self.takeoff_ack_timeout = 20.0
        self.liftoff_wait = 6.0
        self.liftoff_alt = 30.0
        self.liftoff_rise = 20.0        # altitude must exceed the pre-takeoff reading by this much
        self.land_attempts = 3
        self.land_window = 3.0
        self.land_descent_cm = 8.0
        self.land_settle = 12.0
        self.uncertain_retries = 3      # automatic extra landing routines when landing is not confirmed
        self.uncertain_gap = 5.0
        self.tele_fresh_ground = 0.6    # telemetry older than this never counts as ground evidence
        self.jump_max_m = 0.25          # pose jump between consecutive frames treated as outlier
        self.single_zmin, self.single_zmax = 0.45, 1.3  # one marker: centre extrapolated, trusted only in this range
        self.pair_zmin, self.pair_zmax = 0.45, 2.5
        self.guide_id, self.guide_size = None, 0.115   # centre guide marker (OFF by default; enable with --guide-id)
        self.guide_off = (0.0, 0.0, 0.0)               # hoop centre relative to the guide, in the guide's own x/y/z (z = towards the camera)
        self.guide_zmin, self.guide_zmax = 0.4, 1.6
        self.guide_cam_off = None                      # (dx,dy,dz) m = guide - hoop centre in the camera frame, from the 'guide vs pair centre' log
        self.center_alpha = 0.35        # low-pass on the centre (smooths switches between 1-marker and 2-marker estimates)
        self.kp, self.kd = 100.0, 30.0   # PD on the centre error (rc per metre, rc per m/s)
        self.rc_auto_max, self.rc_auto_min = 20, 8
        self.y_offset = 0.0             # target y (m, camera frame, + = down); shifts where the red cross is held
        self.cross_beyond = 0.60        # how far past the hoop plane the forward move goes (m)
        self.hoop_height_cm = 133       # measured hoop-centre height (tof units); crossing needs the drone within +-cross_h_tol of it
        self.cross_h_tol = 10
        self.hold_z = 1.0               # hold this distance to the hoop plane (m) against the Tello forward drift; 0 = off
        self.hold_z_tol = 0.10
        self.rc_fb_limit = 15           # max |rc forward/back| the Commander will ever send
        # ---- autonomous mission (T2.0): takeoff -> climb -> align -> cross -> approach ID3 -> land
        self.mission = True
        # ---- slalom part 1 (T2.1): ID3 -> climb past ID2 -> fly OVER the top bar (ID17 as range) -> descend to the gap (ID5 height)
        self.slalom = True
        self.ekf_shadow = True          # run the EKF alongside (log only; no --ekf-control in the slalom yet)
        self.ekf_map = 'aruco_map_slalom.yaml'
        self.ekf_vg_gain = 1.2
        self.ekf_control = False        # --ekf-control: the EKF holds x/y/z in DESCEND (and keeps going when ID5/ID17 drop out); off = old behaviour
        self.ekf_max_s = 0.15           # m: EKF sigma (x and z) above this -> ignore the EKF
        self.ekf_hold_db = 0.08         # m: DESCEND x dead band
        self.ekf_hold_y_db = 0.06       # m: DESCEND y dead band
        self.back_ekf_margin = 0.22     # m: BACK (EKF) stops this much before low_dist (the drone coasts a few cm)
        self.slalom_ids = (2, 3, 4, 5, 8, 17)
        self.mk_h = {4: 0.70, 3: 1.30, 2: 1.85, 5: 1.50, 17: 1.82}   # marker centre heights above the floor (m)
        self.start_dist = 0.85          # m from the frame (ID3); accepted band start_dist-0.20 .. +0.15
        self.back_lr_big = 20           # rc lr cap in BACK when |y - frame line| > 0.10 m (12 gave ~0 net effect against the drift in flight 161945)
        self.over_lost_tof = 225        # cm: in OVER with ID17 lost and tof above this, descend gently (the Tello drifts up after the top bar)
        self.over_lost_ud = -20         # rc ud for that descent (|ud|<=15 does not move the Tello in free air)
        self.over_height = 2.15         # m: height for flying over the top bar (bar top ~1.9)
        self.pole_gap = 1.20            # m between the horizontal frame and the vertical poles
        self.pole_min = 0.40            # m: closer than this to the poles while descending -> land (was 0.50: aborted a good flight)
        # ---- height safety (flight 150052: DESCEND ended 11 cm below the gap centre, BACK entered the middle bar at tof 10 cm and stalled)
        self.tof_z_offset = 0.065       # m: tof reads this much ABOVE the marker/camera height (144020: +0.064, 150052: +0.070)
        self.descend_vz_max = 0.04      # m/s: DESCEND only ends when the vertical speed is below this ...
        self.descend_settle = 1.0       # s: ... and the height has been in the band this long (old _settled(1.0) behaviour)
        self.descend_ud_min = -15       # rc: max descent speed near the gap (old marker descent: clip(90*err, -15, 20); EKF version was -25)
        self.back_min_z = 1.45          # m: do not enter the middle bar lower than this (checked while tof still sees the floor)
        self.back_gate_floor = 1.20     # m: tof above this = floor below (no bar under the drone yet)
        self.back_gate_max = 8.0        # s: gate held longer than this -> land (free air in front of the frame)
        self.past = 0.45                # m past the frame before descending
        self.gap_height = 1.50          # m: middle of the gap between the top and the middle bar
        self.back = True                # after the descent: fly BACK through the gap
        self.back_dist = 0.80           # m in front of the frame (ID3 side) where the back pass stops
        self.back_fb = 10               # rc backward speed
        self.yaw_hold = True            # hold the heading seen at the start (marker yaw), self-checked
        self.yaw_limit = 15             # max |rc yaw|
        self.low_dist = 0.90           # m in front of the frame while going down to the low gap (ID3 and ID4 must overlap in view)
        self.cross2 = True              # after ID3 is re-acquired: go down to the low gap (between ID4 and ID3), cross it, find ID8
        self.low_height = 0.95          # m: middle of the low gap (ID4 bar 0.70 m, ID3 bar 1.30 m)
        self.cross2_fb = 12             # rc forward through the low gap
        self.clear2 = 0.60              # m beyond the frame plane before the drone starts to line up with ID8
        self.final_dist = 0.70          # m from ID8 where the mission ends
        self.turn = False               # no yaw: ID3 is seen from behind the frame while backing
        self.over_fb = 14               # rc forward while flying over
        self.over_boost = 14            # rc forward while the top bar is under the drone (tof < 120 cm) and for 1.5 s after: the Tello firmware
                                        # misreads the close bar as fast motion and brakes/recoils (flights 114437, 115818); 14 = no boost
        self.lost_land = 6.0            # s without the needed reference marker -> land
        self.h_gate = 0.45              # m: a marker whose implied height differs from tof/last estimate by more is a false detection
        self.phase_timeout = {'ALIGN3': 30.0, 'UP': 35.0, 'OVER': 20.0, 'DESCEND': 30.0, 'BACK': 45.0, 'TURN': 25.0, 'FIND3': 20.0, 'DOWN2': 35.0, 'CROSS2': 30.0, 'APPR8': 25.0, 'W_APPROACH': 20.0, 'W_CLIMB': 20.0, 'W_WP': 60.0, 'W_PLANT': 25.0, 'W_HOLD': 10.0, 'P_WP': 40.0, 'P_CORNER': 25.0}
        self.min_hoop_markers = 3       # hoop markers that must be visible to start the automatic crossing
        self.align_timeout = 45.0       # s: cannot align/cross within this -> land
        self.guide_stop = 0.85          # m: stop when the horizontal distance to ID3 is <= this
        self.guide_min = 0.55           # m: closer than this to ID3 -> land
        self.approach_fb = 12           # rc forward while approaching ID3
        self.approach_timeout = 10.0    # s
        self.guide_lost_land = 3.0      # s without ID3 during the approach -> land
        self.done_hover_s = 2.0         # hover after reaching the ID3 distance, then land
        self.cam_pitch_deg = 10.0       # Tello camera looks DOWN by ~10 deg: rotate the centre into a drone-level frame
        self.cross_min_z, self.cross_max_z = 0.70, 1.30
        self.cross_align_s = 2.0        # aligned this long before F is accepted
        self.cross_timeout = 12.0
        self.cross_hover_s = 0.6        # hover after the forward move; the mission then continues
        self.min_battery_cross = 40
        self.layout = None              # {id: (right_m, up_m)}; built in main from --separation/--hsep/--hids
        self.axis = 'v'                 # marker layout: v = ID0 below/ID1 above, h = ID0 left/ID1 right
        self.rc_slew = 6                # max rc increase per video frame (decreases are immediate)
        self.marker_size = 0.115
        self.separation = 0.55
        # ---- merged mission: hoop first, then the vertical slalom (the hoop's ID3 approach hands over to ALIGN3)
        self.circle = True              # True: start with the hoop (ALIGN -> CROSS -> APPROACH -> HANDOFF); False: start at ALIGN3 (old t21)
        self.stop_after = None          # 'hoop': land after the ID3 approach (old t20 behaviour)
        self.handoff_hover_s = 1.0      # hover after the ID3 approach before ALIGN3 takes over
        self.start_weave = False        # --start-at weave: take off and run only the part_2 sequence
        self.weave = True               # after the vertical slalom: weave between the poles (blue, green) and stop at the plant (part_2.py)
        self.weave_hover_s = 0.3        # hover at the end of the vertical slalom before the weave starts
        self.plant_chain = True          # after the weave: plant lap -> ID18 -> ID19, land there (--no-plant = land at the plant as before)
        self.record = False             # --record: save the camera video (raw + annotated) next to the logs
        self.start_id18 = False         # --start-at id18: take off at plant point 11 (C16_13), see ID18, approach, slide to ID19, align, land
        self.start_plant = False        # --start-at plant: take off PLANT_START_DIST from the plant centre, line up on ID13, fly the lap (PLANT_* constants)
        self.plant_stop_after = 'lap'   # --plant-stop-after: land once this stage is done (align, 13, 09, 11, 16, lap, 18, 19)
        # hoop EKF-control parameters (from t20)
        self.cross_min_x = 0.40         # EKF x must exceed this (m) after the forward, else one top-up forward
        self.ekf_x_stop = 0.55          # m: APPROACH ends when EKF x >= this
        self.rc_yaw_limit = 15
        self.ekf_yaw_tol = 3.0          # deg: CROSS only if |EKF heading| <= this
        self.ekf_kyaw = 1.1
        self.ekf_yaw_dead = 1.5
        self.ekf_lat_bias = 0.06
        self.ekf_y_gate = 0.18
        self.ekf_lat_dead = 0.10
        self.ekf_lat_max = 0.08
        self.ekf_max_sx = 0.15          # m: EKF x sigma above this -> ignore the EKF in ALIGN/APPROACH
        for k, v in kw.items():
            setattr(self, k, v)


# ----------------------------------------------------------------------------- telemetry
def valid_alt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < 400 else None


def alt_of(state, source):
    return valid_alt(state.get(source))


def alt_any(state):
    vals = [a for a in (valid_alt(state.get('h')), valid_alt(state.get('tof'))) if a is not None]
    return max(vals) if vals else None


def on_ground_state(state):
    tof, h = state.get('tof'), state.get('h')
    if tof is None or h is None:
        return False
    if tof <= 25 and h <= 30:
        return True
    return tof >= 6000 and h <= 10   # tof out of range (6553) right at the ground: seen when landing on some surfaces


class Telemetry:
    """Local copy of the Tello state stream (djitellopy's get_current_state is a local dict read, not a UDP query)."""

    def __init__(self, drone):
        self.drone = drone
        self.lock = threading.Lock()
        self.state = {}
        self._sig = None
        self._t_change = time.monotonic()

    def poll(self):
        try:
            s = dict(self.drone.get_current_state() or {})
        except Exception:
            return
        sig = tuple(sorted(s.items()))
        with self.lock:
            if sig != self._sig:
                self._sig = sig
                self._t_change = time.monotonic()
            self.state = s

    def snapshot(self):
        with self.lock:
            return dict(self.state), time.monotonic() - self._t_change


# ----------------------------------------------------------------------------- commander
def st_tof_str(app):
    try:
        return app.tele.snapshot()[0].get('tof')
    except Exception:
        return '?'


class Commander(threading.Thread):
    """Single owner of ordinary flight commands. Emergency is the only command sent from other threads."""

    def __init__(self, drone, tele, cfg, ev=None):
        super().__init__(daemon=True, name='commander')
        self.drone, self.tele, self.cfg = drone, tele, cfg
        self.ev = ev or (lambda m: LOG.info(m))
        self.state = 'GROUNDED'
        self._sp = ZERO
        self._sp_t = 0.0
        self.gui_beat = time.monotonic()
        self.takeoff_request = threading.Event()
        self.land_request = threading.Event()
        self.land_reason = ''
        self.emergency = threading.Event()
        self.stop_evt = threading.Event()
        self.takeoff_sent = False
        self.t_flying = None
        self._ground_since = None
        self._esc = ''
        self.counts = {'takeoff': 0, 'land': 0, 'rc': 0, 'emergency': 0}
        self._unc_n = 0
        self._land_end = 0.0
        self.cross_request = threading.Event()
        self.cross_cm = 0
        self.cross_kind = 'forward'
        self.cross_done_t = 0.0

    # ---- called from other threads
    def beat(self):
        self.gui_beat = time.monotonic()

    def set_setpoint(self, rc):
        self._sp = tuple(rc)
        self._sp_t = time.monotonic()

    def request_takeoff(self):
        if self.state == 'GROUNDED' and not self.takeoff_sent and not self.emergency.is_set():
            self.takeoff_request.set()
            return True
        return False

    def request_cross(self, cm):
        if self.state == 'FLYING' and not self.emergency.is_set() and not self.land_request.is_set():
            self.cross_cm = int(cm)
            self.cross_kind = 'forward'
            self.cross_request.set()
            return True
        return False

    def request_turn(self, deg):
        """Blocking closed-loop yaw ('cw N'). Same safety loop as the forward move."""
        if self.state == 'FLYING' and not self.emergency.is_set() and not self.land_request.is_set():
            self.cross_cm = int(deg)
            self.cross_kind = 'cw'
            self.cross_request.set()
            return True
        return False

    def request_land(self, reason='operator'):
        if self.state in AIRBORNE_STATES and not self.emergency.is_set():
            self.land_reason = reason
            self.land_request.set()
            return True
        return False

    def emergency_cut(self, why=''):
        """Immediate motor cut. Never blocks, never takes a lock."""
        if self.emergency.is_set():
            return
        self.emergency.set()
        self.state = 'EMERGENCY'
        self.ev(f'EMERGENCY MOTOR CUT requested ({why}) - drone will fall')

        def _send():
            for _ in range(3):  # UDP may drop one
                try:
                    self.drone.send_command_without_return('emergency')
                    self.counts['emergency'] += 1
                except Exception as exc:
                    self.ev(f'emergency send failed: {exc!r}')
                time.sleep(0.05)
        threading.Thread(target=_send, daemon=True, name='emergency-send').start()

    def stop(self):
        self.stop_evt.set()

    # ---- helpers (commander thread only)
    def _resp_list(self):
        try:
            return self.drone.get_own_udp_object()['responses']
        except Exception:
            return []

    def _pop_resp(self):
        lst = self._resp_list()
        try:
            raw = lst.pop(0)
        except (IndexError, AttributeError):
            return None
        return raw.decode('utf-8', 'replace').rstrip('\r\n') if isinstance(raw, (bytes, bytearray)) else str(raw)

    def _drain(self, why):
        while True:
            r = self._pop_resp()
            if r is None:
                return
            self.ev(f'stale/unexpected response drained before {why}: {r!r}')

    def _ground_stable(self):
        st, age = self.tele.snapshot()
        if age <= self.cfg.tele_fresh_ground and on_ground_state(st):
            if self._ground_since is None:
                self._ground_since = time.monotonic()
            return time.monotonic() - self._ground_since >= 1.0
        self._ground_since = None
        return False

    def _note_esc(self, text):
        if text != self._esc:
            self._esc = text
            if text:
                self.ev(f'WATCHDOG: {text}')

    # ---- main loop
    def run(self):
        period = 1.0 / self.cfg.rc_hz
        while not self.stop_evt.wait(period):
            try:
                self.tele.poll()
                if self.emergency.is_set():
                    self.state = 'EMERGENCY'
                    continue
                st = self.state
                if st == 'GROUNDED' and self.takeoff_request.is_set() and not self.takeoff_sent:
                    self._takeoff()
                elif st == 'FLYING':
                    if self.cross_request.is_set() and not self.land_request.is_set():
                        self._cross()
                    else:
                        self.cross_request.clear()
                        self._fly_tick()
                elif st == 'UNCERTAIN':
                    if self.land_request.is_set():
                        self._unc_n = 0
                        self._land(self.land_reason or 'operator retry')
                    elif self._unc_n < self.cfg.uncertain_retries and \
                            time.monotonic() - self._land_end > self.cfg.uncertain_gap:
                        self._unc_n += 1
                        self._land(f'automatic retry {self._unc_n}/{self.cfg.uncertain_retries} (landing not confirmed)')
                    elif self._unc_n == self.cfg.uncertain_retries and self._esc != 'gave up':
                        self._esc = 'gave up'
                        self.ev('LANDING STILL NOT CONFIRMED after automatic retries. No more commands are sent. '
                                'Verify physically; press P to try again; Tello 15 s failsafe is the backstop.')
            except Exception as exc:
                self.ev(f'COMMANDER ERROR {exc!r}')
                if self.state in ('FLYING', 'TAKING_OFF') and not self.emergency.is_set():
                    try:
                        self._land('commander error')
                    except Exception as exc2:
                        self.ev(f'COMMANDER land after error failed: {exc2!r}')

    def _takeoff(self):
        cfg = self.cfg
        self.takeoff_sent = True
        self.state = 'TAKING_OFF'
        self._drain('takeoff')
        st0, _ = self.tele.snapshot()
        base = alt_any(st0) or 0.0   # pre-takeoff reading: a real takeoff must rise clearly above it
        t0 = time.monotonic()
        self.counts['takeoff'] += 1
        self.ev('TAKEOFF sent (single attempt, never retried)')
        self.drone.send_command_without_return('takeoff')
        ack = t_ack = err_t = None
        hist = deque()  # (t, alt) while above liftoff_alt: FLYING only when altitude has SETTLED (takeoff manoeuvre over)
        while not self.stop_evt.is_set():
            time.sleep(0.05)
            now = time.monotonic()
            if self.emergency.is_set():
                self.state = 'EMERGENCY'
                return
            self.tele.poll()
            if self.land_request.is_set():
                self.ev('land requested during takeoff')
                self._land(self.land_reason)
                return
            gui_age = now - self.gui_beat
            if gui_age > cfg.gui_stall_land:
                self.ev(f'WATCHDOG: GUI stalled {gui_age:.1f}s during takeoff -> landing')
                self.state = 'UNCERTAIN'
                self._land(f'GUI stalled {gui_age:.1f}s during takeoff')
                return
            r = self._pop_resp()
            if r is not None:
                self.ev(f'takeoff-phase response: {r!r}')
                if r.lower() == 'ok' and ack is None:
                    ack, t_ack = r, now
                elif r.lower() != 'ok' and err_t is None:
                    err_t = now
            st, age = self.tele.snapshot()
            a = alt_any(st)
            if a is not None and a >= cfg.liftoff_alt and age < 1.0:
                hist.append((now, a))
                while hist and now - hist[0][0] > 1.2:
                    hist.popleft()
            else:
                hist.clear()  # glitch (e.g. h=0) or stale telemetry: start the evidence over
            if len(hist) >= 2 and hist[-1][0] - hist[0][0] >= 1.0 and \
                    max(v for _, v in hist) - min(v for _, v in hist) <= 8 and \
                    min(v for _, v in hist) - base >= cfg.liftoff_rise:
                self.state = 'FLYING'
                self.t_flying = now
                self.ev(f'takeoff VERIFIED by telemetry: settled at alt={a:.0f}cm (ack={ack!r})')
                return
            if t_ack is not None:
                deadline = t_ack + cfg.liftoff_wait
            elif err_t is not None:
                deadline = err_t + 3.0
            else:
                deadline = t0 + cfg.takeoff_ack_timeout
            if now > deadline:
                self.ev('TAKEOFF NOT VERIFIED by telemetry -> UNCERTAIN, running safe landing routine')
                self.state = 'UNCERTAIN'
                self._land('takeoff not verified')
                return

    def _cross(self):
        """One closed-loop 'forward N' (Tello odometry), then hover and land. Never retried."""
        cfg = self.cfg
        self.cross_request.clear()
        cm = max(20, min(500, self.cross_cm))
        self.state = 'CROSSING'
        self._sp = ZERO
        turn = self.cross_kind == 'cw'
        text = f'cw {cm}' if turn else f'forward {cm}'
        self.ev(f'CROSSING: {text} (single attempt)')
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
            self.counts['rc'] += 1
        except Exception as exc:
            self.ev(f'zero rc before forward failed: {exc!r}')
        time.sleep(0.3)
        self._drain('forward')
        self.drone.send_command_without_return(text)
        t0 = time.monotonic()
        result = None

        def abort(reason):
            self.ev(f'CROSS ABORT: {reason} -> stop + land')
            try:
                self.drone.send_rc_control(0, 0, 0, 0)   # 'stop' is unknown on the standard Tello
            except Exception:
                pass
            time.sleep(0.3)
            self._land(reason)

        while time.monotonic() - t0 < cfg.cross_timeout:
            time.sleep(0.05)
            if self.emergency.is_set():
                self.state = 'EMERGENCY'
                return
            self.tele.poll()
            _, tage = self.tele.snapshot()
            if self.land_request.is_set():
                return abort(self.land_reason or 'operator')
            if time.monotonic() - self.gui_beat > cfg.gui_stall_land:
                return abort('GUI stalled during crossing')
            if tage > cfg.tele_stale_land:
                return abort('telemetry lost during crossing')
            r = self._pop_resp()
            if r is not None:
                self.ev(f'forward response: {r!r}')
                result = r.lower()
                break
        if result != 'ok':
            return abort('forward not confirmed' if result is None else f'forward error {result!r}')
        self.ev(f'{text} done (ok): hovering, mission continues')
        t1 = time.monotonic()
        while time.monotonic() - t1 < cfg.cross_hover_s:
            time.sleep(0.05)
            if self.emergency.is_set():
                self.state = 'EMERGENCY'
                return
            self.tele.poll()
            if self.land_request.is_set():
                break
        if self.land_request.is_set():
            return self._land(self.land_reason or 'operator')
        self.cross_done_t = time.monotonic()
        self._sp = ZERO
        self.state = 'FLYING'

    def _fly_tick(self):
        cfg = self.cfg
        now = time.monotonic()
        st, tage = self.tele.snapshot()
        gui_age = now - self.gui_beat
        reason = None
        bat = st.get('bat')
        a = alt_any(st)
        if self.land_request.is_set():
            reason = self.land_reason or 'operator'
        elif gui_age > cfg.gui_stall_land:
            reason = f'GUI stalled {gui_age:.1f}s'
        elif tage > cfg.tele_stale_land:
            reason = f'telemetry lost {tage:.1f}s'
        elif isinstance(bat, (int, float)) and bat < cfg.min_battery_flight:
            reason = f'battery {bat}%'
        elif a is not None and a > cfg.max_height:
            reason = f'height {a:.0f}cm > {cfg.max_height}'
        elif self.t_flying is not None and now - self.t_flying > cfg.max_seconds:
            reason = f'max flight time {cfg.max_seconds:.0f}s'
        if reason:
            self._land(reason)
            return
        sp = self._sp
        zero_why = ''
        if gui_age > cfg.gui_stall_zero:
            zero_why = f'GUI stalled {gui_age:.1f}s -> rc zero'
        elif now - self._sp_t > cfg.setpoint_ttl:
            zero_why = 'setpoint stale -> rc zero'
        elif tage > cfg.tele_stale_zero:
            zero_why = f'telemetry stale {tage:.1f}s -> rc zero'
        self._note_esc(zero_why)
        lim = cfg.rc_limit
        clip = lambda v: int(max(-lim, min(lim, v)))
        # defense in depth: forward/back limited to rc_fb_limit; this script never commands yaw
        fbl = cfg.rc_fb_limit
        yl = cfg.yaw_limit
        rc = ZERO if zero_why else (clip(sp[0]), int(max(-fbl, min(fbl, sp[1]))), clip(sp[2]), int(max(-yl, min(yl, sp[3]))))
        self.drone.send_rc_control(*rc)
        self.counts['rc'] += 1

    def _land(self, reason):
        try:
            self._land_impl(reason)
        finally:
            self._land_end = time.monotonic()
            if self.state == 'LANDED':
                self._unc_n = 0

    def _land_impl(self, reason):
        cfg = self.cfg
        self._ground_since = None  # need fresh evidence gathered from now on
        self.land_request.clear()
        self.state = 'LANDING'
        self._sp = ZERO
        self.ev(f'LANDING: {reason}')
        try:
            if self.counts['rc'] > 0:  # only if rc was ever sent in this flight
                self.drone.send_rc_control(0, 0, 0, 0)  # the Tello keeps the LAST rc: stop any climb/drift first
            if self.counts['rc'] > 0:
                self.counts['rc'] += 1
        except Exception as exc:
            self.ev(f'zero rc before land failed: {exc!r}')
        time.sleep(0.3)  # no land datagram right behind an rc datagram
        for attempt in range(1, cfg.land_attempts + 1):
            if self.emergency.is_set():
                self.state = 'EMERGENCY'
                return
            self.tele.poll()
            st, _ = self.tele.snapshot()
            if self._ground_stable():
                self.state = 'LANDED'
                self.ev('on the ground by telemetry (no further land needed)')
                return
            self._drain(f'land#{attempt}')
            a0 = alt_any(st)
            self.counts['land'] += 1
            self.ev(f'LAND sent (attempt {attempt}/{cfg.land_attempts}) alt0={a0}')
            self.drone.send_command_without_return('land')
            deadline = time.monotonic() + cfg.land_window
            descending = False
            while time.monotonic() < deadline:
                time.sleep(0.05)
                if self.emergency.is_set():
                    self.state = 'EMERGENCY'
                    return
                self.tele.poll()
                st, _ = self.tele.snapshot()
                r = self._pop_resp()
                if r is not None:
                    self.ev(f'land-phase response: {r!r}')
                a = alt_any(st)
                if self._ground_stable():
                    self.state = 'LANDED'
                    self.ev('LANDED (telemetry: on ground). Verify physically.')
                    return
                if not descending and a0 is not None and a is not None and a0 - a >= cfg.land_descent_cm:
                    descending = True
                    deadline = time.monotonic() + cfg.land_settle
                    self.ev('descent observed in telemetry')
            if descending:
                self.state = 'UNCERTAIN'
                self.ev('descent seen but ground NOT confirmed: verify physically')
                return
        self.state = 'UNCERTAIN'
        self.ev('LAND NOT CONFIRMED after all attempts. No more rc is sent: the Tello 15 s failsafe is the last '
                'backstop. Use the Tello app / press P to retry / E only if unavoidable.')


# ----------------------------------------------------------------------------- vision
def load_calibration(path):
    with np.load(path) as d:
        K = np.asarray(d['camera_matrix'], dtype=float)
        dist = np.asarray(d['dist_coeffs'], dtype=float)
        size = tuple(int(v) for v in np.asarray(d['image_size']).reshape(-1)) if 'image_size' in d.files else None
    return K, dist, size


def make_detector():
    ar = cv2.aruco
    dictionary = ar.getPredefinedDictionary(ar.DICT_4X4_50)
    params = ar.DetectorParameters() if hasattr(ar, 'DetectorParameters') else ar.DetectorParameters_create()
    return ar.ArucoDetector(dictionary, params) if hasattr(ar, 'ArucoDetector') else (dictionary, params)


def detect(frame, det, K, dist, marker_size, extra=None, ids=(0, 1)):
    """Returns (obs, raw_ids, rejected). Observations keep only IDs 0/1 with good reprojection error."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if isinstance(det, tuple):
        corners, found, _ = cv2.aruco.detectMarkers(gray, det[0], parameters=det[1])
    else:
        corners, found, _ = det.detectMarkers(gray)
    allowed = set(int(i) for i in ids)
    obs, rejected = {}, []
    raw_ids = [] if found is None else [int(i) for i in found.flatten()]
    if found is None:
        return obs, raw_ids, rejected
    extra = extra or {}   # extra marker ids -> their own size in metres (e.g. the small centre guide)
    for cr, id_ in zip(corners, raw_ids):
        if id_ not in allowed and id_ not in extra:
            continue
        half = extra.get(id_, marker_size) / 2
        obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        pixels = cr.reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(obj, pixels, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or tvec[2, 0] <= 0:
            rejected.append((id_, 'pnp'))
            continue
        reproj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        rms = float(np.sqrt(np.mean(np.sum((reproj.reshape(4, 2) - pixels) ** 2, axis=1))))
        if rms > 2.5:
            rejected.append((id_, f'rms={rms:.1f}'))
            continue
        obs[id_] = {'xyz': tvec.reshape(3), 'rvec': rvec.reshape(3), 'corners': pixels, 'rms': rms}
        cv2.polylines(frame, [pixels.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, f'ID {id_}', tuple(pixels[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
    return obs, raw_ids, rejected


def make_layout(vsep=0.55, hsep=0.50, hids=(10, 12)):
    """Hoop-centre-relative position (right, up) in metres of every hoop marker, seen from the FRONT (drone side).
    ID0 below / ID1 above; hids = (LEFT id, RIGHT id). hsep <= 0 disables the horizontal pair."""
    lay = {0: (0.0, -vsep / 2), 1: (0.0, vsep / 2)}
    if hsep and hsep > 0:
        lay[hids[0]] = (-hsep / 2, 0.0)
        lay[hids[1]] = (hsep / 2, 0.0)
    return lay


def hoop_center(obs, separation, guide=None, guide_off=(0.0, 0.0, 0.0), axis='v', layout=None, guide_cam_off=None):
    """Hoop centre in camera coordinates (X right, Y down, Z forward) from ANY subset of the hoop markers.
    >=2 markers: mean (opposite markers cancel the rotation term); 1 marker: extrapolated with its own rotation.
    Marker local +x is right and +y is up (upright marker)."""
    if layout is None:
        layout = {0: (0.0, -separation / 2), 1: (0.0, separation / 2)} if axis == 'v' else \
                 {0: (-separation / 2, 0.0), 1: (separation / 2, 0.0)}
    vis = [i for i in layout if i in obs]
    if len(vis) >= 2:
        for k, i in enumerate(vis):
            for j in vis[k + 1:]:
                oi, oj = np.array(layout[i]), np.array(layout[j])
                exp = float(np.linalg.norm(oi - oj))
                d = obs[i]['xyz'] - obs[j]['xyz']
                if not 0.7 * exp <= float(np.linalg.norm(d)) <= 1.3 * exp:
                    return None, 'INCONSISTENT_PAIR', False
                cam = np.array([oi[0] - oj[0], -(oi[1] - oj[1]), 0.0])   # expected direction in camera axes
                if float(d @ cam) <= 0.0:
                    return None, f'LAYOUT_MISMATCH_{i}_{j}', False       # e.g. left/right ids swapped
        pts = [obs[i]['xyz'] for i in vis]
        mo = np.mean([layout[i] for i in vis], axis=0)
        if float(np.linalg.norm(mo)) < 1e-6:
            return np.mean(pts, axis=0), 'MULTI', True
        ests = []
        for i in vis:
            R, _ = cv2.Rodrigues(obs[i]['rvec'])
            ests.append(obs[i]['xyz'] - R @ np.array([layout[i][0], layout[i][1], 0.0]))
        return np.mean(ests, axis=0), 'MULTI', True
    if guide is not None and guide in obs:
        if guide_cam_off is not None:   # calibrated: centre = guide - (guide - centre) measured in the camera frame, no rotation noise
            return obs[guide]['xyz'] - np.array(guide_cam_off, float), 'GUIDE', False
        R, _ = cv2.Rodrigues(obs[guide]['rvec'])
        return obs[guide]['xyz'] + R @ np.array(guide_off, float), 'GUIDE', False
    if vis:
        i = vis[0]
        R, _ = cv2.Rodrigues(obs[i]['rvec'])
        return obs[i]['xyz'] - R @ np.array([layout[i][0], layout[i][1], 0.0]), f'ID{i}_ONLY', False
    return None, 'NO_MARKERS', False


def project(point, K, dist):
    if point is None or point[2] <= 0:
        return None
    px, _ = cv2.projectPoints(np.asarray(point, dtype=np.float64).reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)
    return tuple(int(v) for v in np.rint(px.reshape(2)))


# ----------------------------------------------------------------------------- weave + plant (ported from the partner's part_2.py)
# Local frame of part_2: origin 0.75 m in front of the purple pole (ID8 low / ID5 top), X forward, Y left(+).
# World frame of this mission: purple pole at (2.58, 0.13)  ->  local = world - (1.83, 0.13).
WV_ORIGIN = (1.83, 0.13)
WV_ID_LOW, WV_ID_MID, WV_ID_BLUE, WV_ID_GREEN, WV_ID_PLANT = 8, 5, 6, 7, 13
WV_IDS = (WV_ID_LOW, WV_ID_MID, WV_ID_BLUE, WV_ID_GREEN, WV_ID_PLANT)
WV_NAMES = {8: 'purple bottom', 5: 'purple top', 6: 'blue', 7: 'green', 13: 'plant'}
WV_MAP = {8: (0.75, 0.0), 5: (0.75, 0.0), 6: (1.60, 0.0), 7: (2.11, 0.0), 13: (3.90, -0.10)}
# (X, Y, height_cm or None, marker to look at)
WV_WAYPOINTS = [
    (0.45, -0.40, None, 5),     # beside purple: RIGHT side
    (1.10, -0.40, None, 6),     # 0.35 m past purple: RIGHT
    (1.25,  0.40, None, 6),     # crossed behind purple: LEFT side
    (1.85,  0.40, None, 7),     # between blue and green: LEFT
    (1.85, -0.40, None, 13),    # in FRONT of green: RIGHT
    (2.45, -0.40, None, 13),    # past green with the pole on the drone's LEFT, then the plant
]
WV_TOL_M, WV_STABLE_FRAMES, WV_KP, WV_MAX_RC, WV_MIN_RC, WV_MIN_UD_RC = 0.08, 15, 0.4, 20, 8, 15
WV_APPROACH_DIST, WV_PLANT_DIST = 0.70, 0.60
WV_CLIMB_RC, WV_MAX_CLIMB_CM = 20, 180
WV_K_WP, WV_WP_RC, WV_WP_TOL, WV_WP_STABLE, WV_K_YAW, WV_YAW_RC, WV_TURN_FIRST, WV_MIN_WP_RC = 0.5, 15, 0.10, 5, 1.0, 20, 15, 10
WV_SIGMA_HOVER, WV_SIGMA_LAND, WV_SIGMA_WAIT, WV_LOST_LAND = 0.35, 0.60, 4.0, 2.0
WV_START_WAIT, WV_SEARCH_UP_RC, WV_SEARCH_MAX_CM, WV_CLIMB_TIMEOUT, WV_WP_TIMEOUT = 10.0, 15, 150, 15.0, 60.0
WV_HOLD_S, WV_PLANT_TIMEOUT, WV_MIN_HEIGHT_CM = 1.0, 20.0, 30
WV_Q_XY, WV_Q_TH = 0.08, math.radians(3)
WV_SIGMA_R0, WV_SIGMA_R1, WV_SIGMA_PHI, WV_GATE, WV_MAX_RANGE = 0.03, 0.03, math.radians(1.5), 9.21, 3.0


def _wv_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _wv_body_speeds(vgx, vgy, yaw_deg):
    yw = np.radians(yaw_deg)
    fwd = 0.1 * (vgx * np.cos(yw) + vgy * np.sin(yw))
    right = 0.1 * (-vgx * np.sin(yw) + vgy * np.cos(yw))
    return fwd, -right


class WvEKF:
    """part_2 / ekf_replay.py EKF: state (x, y, theta) in the local weave frame, range-bearing updates (known correspondences)."""

    def __init__(self, mu0, Sigma0, q_th=WV_Q_TH):
        self.mu, self.Sigma = np.array(mu0, float), np.array(Sigma0, float)
        self.q_th = q_th                                   # heading process noise (rad/sqrt(s)); the weave keeps the default
        self.last_update = None

    def predict(self, v_fwd, v_left, w, dt):
        x, y, th = self.mu
        c, s_ = np.cos(th), np.sin(th)
        self.mu = np.array([x + (v_fwd * c - v_left * s_) * dt, y + (v_fwd * s_ + v_left * c) * dt, _wv_wrap(th + w * dt)])
        G = np.array([[1, 0, -(v_fwd * s_ + v_left * c) * dt], [0, 1, (v_fwd * c - v_left * s_) * dt], [0, 0, 1]])
        R = np.diag([WV_Q_XY ** 2 * dt, WV_Q_XY ** 2 * dt, self.q_th ** 2 * dt])
        self.Sigma = G @ self.Sigma @ G.T + R

    def update(self, cam_x, cam_z, landmark):
        r_meas = np.hypot(cam_x, cam_z)
        z = np.array([r_meas, np.arctan2(-cam_x, cam_z)])
        mx, my = landmark
        dx, dy = mx - self.mu[0], my - self.mu[1]
        q = dx * dx + dy * dy
        z_hat = np.array([np.sqrt(q), _wv_wrap(np.arctan2(dy, dx) - self.mu[2])])
        H = np.array([[-dx / np.sqrt(q), -dy / np.sqrt(q), 0], [dy / q, -dx / q, -1]])
        Q = np.diag([(WV_SIGMA_R0 + WV_SIGMA_R1 * r_meas) ** 2, WV_SIGMA_PHI ** 2])
        S = H @ self.Sigma @ H.T + Q
        nu = z - z_hat
        nu[1] = _wv_wrap(nu[1])
        nis = float(nu @ np.linalg.solve(S, nu))
        self.last_update = {'accepted': nis <= WV_GATE, 'nis': nis,
                            'range_innovation_m': float(nu[0]),
                            'bearing_innovation_deg': float(np.degrees(nu[1])),
                            'gate': WV_GATE,
                            'reason': 'accepted' if nis <= WV_GATE else 'mahalanobis_gate'}
        if nis > WV_GATE:
            return False
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ nu
        self.mu[2] = _wv_wrap(self.mu[2])
        self.Sigma = (np.eye(3) - K @ H) @ self.Sigma
        return True

    def sigma_xy(self):
        return float(np.sqrt(self.Sigma[0, 0] + self.Sigma[1, 1]))


def _wv_clamp(v, lim=WV_MAX_RC):
    return int(max(-lim, min(lim, v)))


def _wv_pcmd(err_m, deadband_m=WV_TOL_M / 2, min_rc=WV_MIN_RC):
    u = WV_KP * err_m * 100
    if abs(err_m) > deadband_m and abs(u) < min_rc:
        u = min_rc if u > 0 else -min_rc
    return _wv_clamp(u)


def _plant_recovery_yaw(last_x):
    """V16: yaw toward the side where the marker was last seen, strong enough to rotate."""
    if abs(last_x) < 0.02:
        return 0
    mag = max(PLANT_STOP_RECOVERY_MIN_YAW, min(abs(_wv_pcmd(last_x)), PLANT_STOP_RECOVERY_MAX_YAW))
    return int(mag if last_x > 0 else -mag)


def _wv_center_on(pose, target_dist=WV_APPROACH_DIST, check_y=True):
    x, y, z = pose
    ex, ey, ez = x, y, z - target_dist
    lr, ud, fb = _wv_pcmd(ex), _wv_pcmd(-ey, min_rc=WV_MIN_UD_RC), _wv_pcmd(ez)
    return lr, fb, ud, abs(ex) < WV_TOL_M and abs(ez) < WV_TOL_M and (not check_y or abs(ey) < WV_TOL_M)


# ----------------------------------------------------------------------------- plant lap test (--start-at plant)
# EVERY tunable of the plant test is a constant HERE (edit this block, not the command line).
# Frame = the WvEKF frame: X forward (= the take-off heading, nose towards ID13), Y LEFT, theta CCW. Origin = PLANT_C.
# Lap (clockwise seen from above): ID13 (front) -> ID09 (left) -> ID11 (back) -> ID16 (right) -> ID13, always facing the plant centre.
# Then (stages 18 / 19): slide to the RIGHT past the plant, line up on ID18, then ID19 (behind the plant), and land aligned with ID19.
PLANT_C = (0.0, 0.0)            # plant CENTRE in the WvEKF frame ((0,0) = standalone test; chained after the weave: its weave-local coordinates)
PLANT_R = 1.10                  # m: orbit radius (plant centre -> drone) at every marker stop; 0.85 m from the nominal ID13 marker with PLANT_MK_OFF = 0.25
PLANT_START_DIST = 1.10         # m: where the drone is placed on the floor (distance to the plant centre), nose towards ID13
PLANT_HOVER_S = 1.0             # s: hover at every marker stop (to be shortened later)
BACK_UD_FF = 10                 # rc: constant down-bias during the vertical-slalom BACK pass (backward flight makes the Tello climb)
FULL_MAX_S = 540.0              # s: whole mission (hoop -> slalom -> weave -> plant -> ID19); the plant leg alone is ~155 s. Watch the battery (min_battery_flight)
PLANT_MAX_S = 90.0              # s: flight-time limit of the lap test, counted from take-off (--max-seconds overrides it)
PLANT_MAX_S_LAP = 180.0         # V29: the full lap took 144 s to reach ID13 in the 16:31 flight; 150 s would cut it
PLANT_SLIDE19_RC = 14        # V35: lateral speed (left) after the ID18 stop
PLANT_SLIDE19_X_TOL = 0.15   # m: ID19 counts as 'seen' when its camera x is within this
PLANT_ID19_DIST = 0.49          # m: camera distance to ID19 where it stops, then lands. EMPIRICAL: the camera says 0.57-0.61 but the drone ends 75-78 cm from the board after landing (18:xx and 15:28 flights, +0.155-0.17 m), so 0.45 -> ~60 cm on the floor. Tune with --start-at id18
PLANT_ID19_X_BIAS = -0.01        # m: EMPIRICAL lateral offset: it ends 7-10 cm to the RIGHT of ID19 although the camera says ~0-2 cm. +0.07 = aim with ID19 7 cm right of the image centre (drone 7 cm left). 0 = off
PLANT_ID19_X_TOL = 0.04         # m: lateral centring tolerance on ID19
PLANT_ID19_Z_TOL = 0.04         # m: distance tolerance (farther than the target)
PLANT_ID19_Z_TOL_NEAR = 0.07    # m: tolerance when CLOSER than the target (16:57 flight: rc=8 coasts ~6 cm, so a +-4 cm band made it hunt forward/back for 9 s)
PLANT_ID19_MAX_RC = 12          # rc cap of the final alignment
PLANT_ID19_STABLE = 3           # consecutive frames inside the tolerances before starting the settle hover
PLANT_ID19_SETTLE_S = 0.7       # s: hover with rc=0 (kills the coast: 15:52 flight kept sliding ~6 cm forward and ~4 cm left after 'aligned'); lands only if STILL inside the tolerances afterwards
PLANT_ID19_NEAR_M = 0.10        # m: closer than this to the target the rc is capped at PLANT_ID19_NEAR_RC (less momentum)
PLANT_ID19_NEAR_RC = 8
PLANT_ID19_YAW_TOL = 6.0        # deg: final rotation tolerance relative to the ID19 board (0 = square with the board)
PLANT_ID19_YAW_GAIN = 0.6       # rc yaw per degree of error
PLANT_ID19_MAX_YAW = 12         # rc yaw cap of the final alignment
PLANT_ID19_EXIT_K = 1.4         # hysteresis: while settling, only 'drifted out' beyond tol*EXIT_K (16:27 flight: noise cancelled the settle 4x)
PLANT_ID19_YAW_MED = 5          # median of the last N yaw samples (frame-to-frame flicker +-3..6 deg)
PLANT_ID19_LEAD_S = 0.6         # s: the Tello coasts ~0.6 s after rc=0 (15:52 flight: +5 cm left, +7 cm forward); decisions use position + velocity*LEAD
PLANT_ID19_MIN_RC = 8           # rc below this does not move the Tello (16:10 flight: rc=-5 for 8 s, z stuck at 0.38 m)
PLANT_ID19_MAX_S = 35.0         # s: land anyway after this long
PLANT_ID19_LOST_S = 3.0         # s without ID19 -> land
PLANT_SLIDE19_MAX_S = 8.0    # s: slide time cap (~0.5 m), then land anyway
PLANT_ID18_MIN_Z = 0.50       # m: closer than this to ID18 -> back away
PLANT_ID18_MAX_OVER = 0.15     # m: tolerance beyond 0.65 m
PLANT_ID18_LATERAL = True    # ID18 centred by lateral slide (lr), not by yaw; the search turn (ID18 not visible) stays
PLANT_ID18_MAX_LR = 12
PLANT_ID18_SEARCH_YAW = 12   # V34: rc yaw (right) while ID18 is not visible
PLANT_ID18_CM = 110   # V32: ToF height after C16_13 (17:20 flight: ID18 sat 0.34 m above the camera, left the image at z~0.99 m)
PLANT_ID18_VIA = (0.15, -0.70)   # V31: via (plant frame) between C16_13 and the ID18 stand: ~0.47 m from ID16 (max 0.40-0.50 m)
PLANT_MAX_S_FINAL = 150.0       # s: limit when the run goes on to ID18/ID19 (lap + final leg do not fit in 90 s)
PLANT_FLY_H = 0.90              # m: PROVISIONAL camera-height target; verify camera-to-ToF offset and marker heights physically
PLANT_TAKEOFF_CM = int(round((PLANT_FLY_H + 0.065) * 100))   # tof target of the CLIMB (tof reads ~6.5 cm above the camera = tof_z_offset)
PLANT_MK_OFF = 0.25             # m: nominal offsets for IDs 13/11/16; ID9 has separate provisional measured coordinates
PLANT_ID9_STOP_Y = 0.95          # m: V17 EKF y of the ID9 stop (was PLANT_R=1.10); target range to ID9 = hypot(0.115, 0.95-0.196) = 0.76 m
PLANT_ID9_XY = (-0.115, +0.196) # m: PROVISIONAL from joint ID13/ID9 observations; must be physically validated
PLANT_X_FINAL = 1.30            # m: plant centre -> plane of ID18/ID19, behind the plant. 1.55 m is the ID13 -> ID19 distance (logs 13:08, X19-X13 = 1.552), minus the 0.25 m ID13 offset
PLANT_Y_FINAL = 0.0             # m: lateral position of ID19 relative to the plant axis (+ = left)   [CONFIRM]
PLANT_DY_AUX = -0.52            # m: ID18 relative to ID19 seen from the front (negative = to the RIGHT)
PLANT_AUX_DIST = 0.65           # m: stand-off from ID18
PLANT_LAND_X = 1.00             # m: landing point = this far from the plant centre (behind it), aligned with ID19
PLANT_SIDE = 0.85               # m: lateral clearance (right of the plant axis) while sliding past the plant towards ID18
PLANT_VIA_STEP_DEG = 30         # via points every 30 deg between two markers
PLANT_WP_RC, PLANT_MIN_WP_RC = 15, 15        # rc speed along the lap: cap / minimum (rc 10 did not move the drone in the 13:54 test: vgx=vgy=0 for 8 s, EKF frozen)
PLANT_TURN_FIRST = 15.0         # deg: above this heading error the drone only turns (= WV_TURN_FIRST, unchanged from the first plant version)
PLANT_VIA_TOL, PLANT_STOP_TOL = 0.15, 0.10   # m: a via point is passed / a stop point is reached inside this radius (EKF position)
PLANT_YAW_DB, PLANT_YAW_MIN_RC = 3.0, 8      # deg: heading dead band, and the smallest rc yaw sent outside of it
PLANT_UD_DB = 0.06              # m: vertical dead band while a marker is in view
PLANT_TOF_DB = 15               # cm: with no marker in view the height is held on tof only outside +-this
PLANT_R_MIN = 0.45              # m: EKF distance to the plant centre below this -> land (markers sit 0.25 m from the centre; was 0.50, tripped by one noisy fix)
PLANT_LOST_S = 5.0             # s without ANY plant marker in view -> land (09 -> 11 is a ~8 s blind arc: ID11 is edge-on to the drone)
PLANT_START_WAIT = 10.0         # s: the start alignment on ID13 must be done within this, else land
PLANT_STOP_MAX_S = 6.0          # s: warning only; never abort merely because alignment takes longer
PLANT_SEE_EXTRA_S = 3.0         # s: a stop whose marker is not in view goes on after this long
PLANT_VG_SCALE = 1.6            # x: the Tello vgx/vgy (integer dm/s) under-report the real speed in the plant mode. 14:03 flight, markers vs integral of vg: 0.26 m vs 0.00 m (rc 8-9), 0.40 m vs 0.21 m and 0.31 m vs 0.18 m (rc 15)
PLANT_X_MAX = PLANT_X_FINAL - 0.14   # V17: 1.16 (was 1.08). 14:14 flight: ID11 fix snapped EKF x +0.89 -> +1.08 in one update and tripped the guard. VERIFY the physical clearance to the ID18/ID19 board.
_PLANT_X_MAX_V15 = PLANT_X_FINAL - 0.22   # V15: 1.08 (was 1.00; the ID11 stop sits at x=+0.95 and the corner/via overshoot reaches ~+1.0). VERIFY the physical clearance to the ID18/ID19 board first.
_PLANT_X_MAX_OLD = PLANT_X_FINAL - 0.30   # m: lap stages only: EKF x beyond this (the ID18/ID19 board is at PLANT_X_FINAL) -> land
# Visual stop: maintain clearance instead of closing to the geometric 0.60 m.
PLANT_VISUAL_STOP_M = 0.85       # m: conservative camera stand-off; do not chase 0.75 m
PLANT_VISUAL_DIST_TOL = 0.10     # m: accept z <= 0.95 m, do not chase exact distance
PLANT_VISUAL_X_TOL = 0.060      # m: marker horizontally centred
PLANT_VISUAL_STABLE_FRAMES = 12 # consecutive frames before stationary dwell
PLANT_VISUAL_MAX_FB = 8        # forward correction cap; NEVER back away at a stop
PLANT_VISUAL_MAX_YAW = 10      # slow visual-only yaw
PLANT_VISUAL_LOG_S = 0.6
# Keep the next stop's target in the camera while closing the last segment.
PLANT_STOP_TRACK_RADIUS_M = 0.48  # waypoint distance for camera-priority yaw
PLANT_STOP_TRACK_X_DB_M = 0.045
# V21: ID16 is approached diagonally from V11_16 (camera range already in band while still ~0.4 m
# lateral from the waypoint). Freeze translation only when the waypoint is actually close.
PLANT_TRACK_FREEZE_MAX_DIST_BY_ID = {13: 0.0, 16: 0.0, 18: 0.0}   # V22: never freeze ID16 (15:15 flight hung at 0.14 m from waypoint); V29: same for the ID13 lap stop (16:33 flight froze 0.45 m off, seeing ID13 at z=0.85 from the side)
PLANT_STOP_TOL_BY_ID = {16: 0.12}               # V22: arrival radius for ID16 stop (EKF jitter ~0.1 m at sigma 0.13)
PLANT_STOP_RECOVERY_S = 3.5       # V16: yaw-only recovery after last target observation (was 1.4 s)
PLANT_STOP_RECOVERY_MAX_YAW = 18  # V16: was 8 (rc<=10 barely rotates the Tello)
PLANT_STOP_RECOVERY_MIN_YAW = 14  # V16: floor so the drone really rotates toward the last-seen side
PLANT_CAPTURE_FRAMES = 5        # consecutive frames with the NEXT stop's marker in view (inside the view limit) -> stop travelling and line up on it at once (13:58 flight: ID9 was in view for ~0.6 s below 60 deg, then lost while the nose was still turning)
PLANT_CAPTURE_DIST = 1.0        # m: ... and closer than this
PLANT_MAX_VIEW_AUX_DEG = 72.0   # deg: same limit for ID18/ID19, seen from ~2 m (13:49 flight: their position stayed smooth up to ~72 deg, jumped beyond ~75)
PLANT_ALIGN_VIEW_DEG = 88.0     # deg: limit for LINING UP / early capture only (lvl): the lateral and range readings were smooth even at 85-89 deg (14:23 ID9)
PLANT_MAX_VIEW_DEG = 75.0       # deg (was 60: at 14:23 ID9 was seen in 94 frames at 0.5-0.65 m, 73-89 deg, smooth tx/tz, and ALL were rejected -> EKF blind, sigma 0.42 -> landed; ID11 68-70 deg also rejected): a plant marker seen more edge-on than this is ignored (its pose is unreliable; at ~85 deg it dragged the EKF 0.15 m, 13:00 tests)
PLANT_Q_TH_DEG = 1.5            # deg/sqrt(s): heading process noise of the plant EKF (weave uses 3.0; 1.5 drifted less in sim)
PLANT_IDS = (13, 9, 11, 16, 18, 19)
WV_NAMES.update({9: 'plant left', 11: 'plant back', 16: 'plant right', 18: 'aux (right of 19)', 19: 'final'})


def plant_marker_map():
    """Marker centres (x, y) in the WvEKF frame."""
    cx, cy = PLANT_C
    return {13: (cx - PLANT_MK_OFF, cy), 9: (cx + PLANT_ID9_XY[0], cy + PLANT_ID9_XY[1]), 11: (cx + PLANT_MK_OFF, cy), 16: (cx, cy - PLANT_MK_OFF),
            19: (cx + PLANT_X_FINAL, cy + PLANT_Y_FINAL), 18: (cx + PLANT_X_FINAL, cy + PLANT_Y_FINAL + PLANT_DY_AUX)}


# SQUARE ROUTE: d is the distance from plant centre to each FACE midpoint.
# Corners use independent +/-PLANT_CORNER_D coordinates.
# The corner heading is continuously directed to the plant centre (diagonal 45 deg).
PLANT_CORNER_HOLD_S = 0.8
PLANT_CORNER_ENTER_TOL = 0.15
PLANT_CORNER_EXIT_TOL = 0.20
PLANT_CORNER_BAD_FRAMES = 3
PLANT_CORNER_YAW_ENTER_DEG = 8.0
PLANT_CORNER_YAW_EXIT_DEG = 12.0
PLANT_CORNER_TIMEOUT_S = 20.0  # bounded observation window
PLANT_CORNER_NEED_BOTH = False  # at least one accepted EKF fix; both corner markers must be visibly observed
PLANT_CORNER_D = 0.85  # independent corner coordinate magnitude; stops remain at PLANT_R=1.10
# A corner needs two *accepted EKF corrections*, recent and individually measured.
PLANT_CORNER_FIX_AGE_S = 3.0  # accepted corrections may be sequential, not same-frame
PLANT_CORNER_MAX_SIGMA_M = 0.20
PLANT_STOP_MAX_SIGMA_M = 0.20
PLANT_STOP_YAW_TOL_DEG = 12.0
PLANT_STOP_VIEW_TOL_DEG = 65.0
# The marker-range threshold is a configurable heuristic, not verified box clearance.
# Reduced from 0.85 m to 0.80 m for the supervised limited test; physical clearance remains unverified.
PLANT_ALIGN_MIN_Z_M = 0.45        # m: closest ID13 range accepted by the initial 'align' stop (the weave ends at ~0.50-0.56 m; 18:10 flight z=0.56 OK, 18:30 flight z=0.50 stuck with the old 0.55)
PLANT_ALIGN_RETREAT_MARGIN = 0.02 # m: the 'align' stop only backs away while z < PLANT_ALIGN_MIN_Z_M + this
PLANT_STOP_MIN_Z_M = 0.55  # V18: was 0.62 (z=0.60 at the ID11 stop in the 14:27 flight would have failed range_ok). V15: was 0.75; the ID11 stop (x=+0.95) is nominally 0.70 m from ID11, so z=0.70 must count as inside the band
PLANT_STOP_WP_TOL_M = 0.12
PLANT_RANGE_MAP_CONFLICT_M = 0.18  # legacy threshold (not a valid comparison of unlike distances)
PLANT_GEOM_RANGE_RESIDUAL_M = 0.20  # provisional map-vs-camera horizontal range consistency threshold
PLANT_GEOM_MIN_MARKER_RANGE_M = 0.62  # V15: was 0.80, which aborted at 13:24 with the ID9 pass at 0.77 m, and also aborts the 0.70 m ID11 stop. NOT a box-clearance guarantee
PLANT_GEOM_MIN_RANGE_BY_ID = {11: 0.52}  # V18: ID11 stop is nominally 0.70 m and the camera showed 0.61-0.73 m; 0.62 aborted at 14:27 (predicted_nearest 0.62). Others keep 0.62.
def _plant_min_marker_range(mid):
    return PLANT_GEOM_MIN_RANGE_BY_ID.get(mid, PLANT_GEOM_MIN_MARKER_RANGE_M)
PLANT_STOP_RECOVER_TOL_M = 0.18  # use EKF to recover position before visual fine tuning



def plant_corner_gate(now, start, bad, dist, yaw_error_deg, sigma, fixes_ok):
    """Pure corner confirmation with 15/20 cm hysteresis and 3 bad-frame grace.

    Returns (start, bad, hold_position, confirmed). The final confirmation
    requires a currently valid measurement, not merely elapsed time.
    """
    entering = (dist <= PLANT_CORNER_ENTER_TOL and
                yaw_error_deg <= PLANT_CORNER_YAW_ENTER_DEG and
                sigma <= PLANT_CORNER_MAX_SIGMA_M and fixes_ok)
    maintaining = (dist <= PLANT_CORNER_EXIT_TOL and
                   yaw_error_deg <= PLANT_CORNER_YAW_EXIT_DEG and
                   sigma <= PLANT_CORNER_MAX_SIGMA_M and fixes_ok)
    if start is None:
        if entering:
            start, bad = now, 0
            return start, bad, True, False
        return None, 0, False, False
    if maintaining:
        bad = 0
        return start, bad, True, now - start >= PLANT_CORNER_HOLD_S
    bad += 1
    if bad <= PLANT_CORNER_BAD_FRAMES and dist <= PLANT_CORNER_EXIT_TOL:
        return start, bad, True, False
    return None, 0, False, False


PLANT_CORNER_D_C1116 = 0.70   # V24: C11_16 only (was 0.85): ranges to ID11/ID16 ~0.83 m instead of ~1.04 m


def plant_points():
    """13 -> corner(13/09) -> 09 -> corner(09/11) -> 11 ->
       corner(11/16) -> 16 -> corner(16/13) -> 13.
       Includes V13_09 and V09_11 via points after their respective corners.
       Output keeps original tuple shape; corner has its own state P_CORNER.
       Stops and corners are never skipped by early marker capture.
    """
    cx, cy = PLANT_C
    d = PLANT_R
    k = PLANT_CORNER_D
    c = (cx, cy)
    stand = d - PLANT_MK_OFF
    return [
        ('stop', cx-d, cy, 13, c, stand, '13'),
        ('corner', cx-k, cy+k, (13, 9), c, 0.0, 'C13_09'),
        # Intermediate waypoint: first move outward in +Y before crossing toward ID9.
        # Uses the existing 'via' handling (15 cm tolerance, no stop/hold).
        ('via', cx-0.40, cy+1.05, 9, c, 0.0, 'V13_09'),
        # V17: ID9 stop 0.15 m closer (EKF y = cy+0.95 -> camera z ~0.76-0.80 m; was cy+d=1.10 -> z 0.83-0.93)
        ('stop', cx, cy+PLANT_ID9_STOP_Y, 9, (cx + PLANT_ID9_XY[0], cy + PLANT_ID9_XY[1]), stand, '09'),
        ('corner', cx+k, cy+k, (9, 11), c, 0.0, 'C09_11'),
        # V12-style VIA after the second corner; remain within PLANT_X_MAX=1.00 m.
        ('via', cx+0.93, cy+0.40, 11, c, 0.0, 'V09_11'),
        # Conservative ID11 stop: nominal 0.70 m from ID11 at x=+0.25 m.
        # Original +1.10 m stop violates the existing x=+1.00 m safety guard.
        ('stop', cx+0.95, cy, 11, c, 0.70, '11'),
        ('corner', cx+PLANT_CORNER_D_C1116, cy-PLANT_CORNER_D_C1116, (11, 16), c, 0.0, 'C11_16'),
        # V20: VIA mirrored from V13_09 (-0.40,+1.05) -> (+0.40,-1.05): comes in from the corner side, ID16 seen at ~27 deg instead of edge-on; min range to ID16 0.89 m
        ('via', cx+0.40, cy-1.05, 16, c, 0.0, 'V11_16'),
        ('stop', cx, cy-d, 16, c, stand, '16'),
        ('corner', cx-k, cy-k, (16, 13), c, 0.0, 'C16_13'),
        ('stop', cx-d, cy, 13, c, stand, 'lap'),
    ]


# ----------------------------------------------------------------------------- GUI / supervision
class App:
    """One call to step() per video frame. Never blocks on the drone."""

    def __init__(self, cfg, K, dist, calib_size, cmd=None, tele=None, rgb=True, csv_path=None):
        self.cfg, self.K, self.dist, self.calib_size = cfg, K, dist, calib_size
        self.cmd, self.tele, self.rgb = cmd, tele, rgb
        self.det = make_detector()
        self.events = []
        self.t_start = time.monotonic()
        self._keys = deque()
        self._last_raw = None
        self._t_new = None
        self._n_new = 0
        self._K_cache = {}
        self._prev_state = None
        self.done = False
        self.quit_req = False
        self.quit_t = 0.0
        self.auto = False
        self.height_done = False
        self.height_fail = False
        self.climb_t0 = 0.0
        self.ok_since = None
        self.good = 0
        self.confirm_until = 0.0
        self.confirm_t0 = 0.0
        self.last_rc = ZERO
        self._prev_c, self._prev_t, self._jumps, self.n_jump_rejected = None, 0.0, 0, 0
        self._act_x = self._act_y = False
        self._aligned_since, self._aligned_said = None, False
        self._video_age = 1e9
        self._cf, self._cf_t, self._last_source, self.z_ok = None, 0.0, 'NO_MARKERS', False
        self._gd_t = 0.0
        self._v, self._cf_prev, self._vel_t, self._vel_n = np.zeros(3), None, 0.0, -1
        self.confirm_kind, self._cross_cm = 'takeoff', 0
        self.phase, self._phase_t, self._last_want = 'CLIMB', 0.0, 0.0
        self._g_lost_t, self._ap, self._n_hoop, self._f3 = None, None, 0, None
        self._topup_done, self._ek_gate_t, self._ek_lat_t = False, 0.0, 0.0
        self._ek_ref, self._ek_ctl_said = None, False
        self.paused = False
        self._lv, self._h_est, self._h_t, self._h_ref = {}, None, 0.0, None
        self._mt, self._ok_t, self._lost_t, self._x17_ref = 0.0, None, None, 0.0
        self._off, self._tgt = {}, None
        self._turn_req, self._turn_t = False, 0.0
        self._cr_last, self._cr_prev, self._cr_v = None, None, 0.15
        self._a8, self._a8_v, self._hstep, self._ang = None, 0.08, 0.0, {}
        self._yb, self._ye, self._ye_id, self._yrc, self._yaw_on = {}, None, None, 0, True
        self._yc = None
        self._d2_ok, self._d2_t = False, None
        self._gap8 = None
        self.max_loop_ms = 0.0
        self._ek = self._ek_w = self._ek_file = self._ek_f = self._ek_m = None
        self._ek_ms = self._ek_cnt = 0
        self._ek_init_said = False
        self._ek_x = self._ek_y = self._ek_z = self._ek_sx = self._ek_sz = self._ek_psi = self._ek_spsi = None
        self._ek_x_t = 0.0
        self._wv = None                 # weave EKF (local frame), created in _weave_begin
        self._wv_t = self._wv_yaw = None
        self._wv_meas, self._wv_seen = {}, {}
        self._wv_i, self._wv_stable, self._wv_last_seen = 0, 0, 0.0
        self._wv_seen_once, self._wv_low, self._wv_unsure, self._wv_t0 = False, None, None, 0.0
        self._wv_ids, self._wv_map = WV_IDS, WV_MAP     # markers / map used by _wv_step (the plant mode swaps them in _plant_begin)
        self._plant_on = False
        self._pl_pts, self._pl_i, self._pl_cur = [], 0, None
        self._pl_hold_t, self._pl_stop_seen, self._pl_seen_t, self._pl_end, self._pl_log_t = None, False, 0.0, False, 0.0
        self._pl_cap = 0
        self._ek_geo = None             # (frame_x, pole_x) from the EKF map
        self._ek_fy = None
        self._bk_hold_t = None
        self._bk_end_t = None
        self._fy_t0, self._fy_psi0, self._fy_bad = None, 0.0, False
        self._ey_t0, self._ey_psi0, self._ey_bad = None, 0.0, False
        self._eh_ref, self._eh_t = None, 0.0
        self.last_center = None
        self.last_status = ''
        self._csv = self._writer = None
        self._rows = 0
        if csv_path:
            self._csv = open(csv_path, 'w', newline='')
            self._writer = csv.writer(self._csv)
            self._writer.writerow(['t', 'state', 'height_done', 'height_fail', 'auto', 'video_age_s', 'tele_age_s',
                                   'h', 'tof', 'bat', 'ids', 'raw_ids', 'source', 'x_m', 'y_m', 'z_m',
                                   'px', 'py', 'rc_lr', 'rc_fb', 'rc_ud', 'rc_yaw', 'loop_ms', 'phase', 'h_est', 'h_ref', 'f_frame', 'f17', 'f5', 'x17', 'yaw_mk', 'yaw_err'])
            # ---- EKF offline log (log only). Same format as pose_global.py / ekf_tello.py
            _base = str(csv_path)[:-4] if str(csv_path).endswith('.csv') else str(csv_path)
            self._csv_base = _base
            self._ek_f = open(_base + '_ekf_frames.csv', 'w', newline='')
            self._ek_m = open(_base + '_ekf_markers.csv', 'w', newline='')
            self._ekw_f, self._ekw_m = csv.writer(self._ek_f), csv.writer(self._ek_m)
            self._ekw_f.writerow(['t', 'frame', 'n_markers', 'ids', 'state', 'phase', 'tele_age_s', 'tele_pitch', 'tele_roll', 'tele_yaw',
                                  'tele_vgx', 'tele_vgy', 'tele_vgz', 'tele_agx', 'tele_agy', 'tele_agz', 'tele_tof', 'tele_h', 'tele_baro',
                                  'tele_bat', 'rc_lr', 'rc_fb', 'rc_ud', 'rc_yaw'])
            self._ekw_m.writerow(['t', 'frame', 'id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rms_px', 'dist'])
            self._ek_n = 0
            self._pl_diag_f = open(_base + '_plant_updates.csv', 'w', newline='')
            self._pl_diag_w = csv.writer(self._pl_diag_f)
            self._pl_diag_w.writerow(['t','phase','id','cam_x','cam_z','map_x','map_y','accepted','nis','gate','innovation_range_m','innovation_bearing_deg','reason'])

        # ---- EKF shadow mode (reads the same observations/telemetry; its output is only logged and drawn, never used for control)
        if csv_path and cfg.ekf_shadow:
            if _ekf_mod is None:
                LOG.warning(f'EKF shadow OFF (import failed: {_ekf_import_err}). Mission unaffected.')
                if cfg.ekf_control:
                    LOG.warning('EKF control OFF too (no EKF): the hoop CROSS is NOT gated by the EKF')
                    cfg.ekf_control = False
            else:
                try:
                    # slalom: the filter starts from the frame markers (ID2/3/4, +poles), ID3 yaw is used (it is the main marker here),
                    # and tof is NOT used (over the bars it reads the bar, not the floor)
                    if cfg.circle:
                        # hoop first (t20 settings): starts from the hoop markers, tof ON, ID3 yaw off. HANDOFF switches to the slalom settings.
                        _a = _ekf_mod.default_args(vg_gain=cfg.ekf_vg_gain, init_ids='0,1,10,12', init_min=3, fuse_exclude='3',
                                                   use_id3_full=False, use_tof=True)
                    else:
                        _a = _ekf_mod.default_args(vg_gain=cfg.ekf_vg_gain, init_ids='2,3,4,5,17', init_min=2, fuse_exclude='',
                                                   use_id3_full=True, use_tof=False)
                    self._ek = _ekf_mod.OnlineEKF(_ekf_load_map(cfg.ekf_map), _a)
                    _mp = self._ek.mp
                    if all(i in _mp for i in (2, 4, 17)):
                        self._ek_geo = (0.5 * (float(_mp[2]['p'][0]) + float(_mp[4]['p'][0])), float(_mp[17]['p'][0]))
                        self._ek_fy = 0.5 * (float(_mp[2]['p'][1]) + float(_mp[4]['p'][1]))   # y of the frame markers (line through the gap)
                    elif cfg.ekf_control:
                        LOG.warning('EKF control OFF: the map has no ID2/ID4/ID17 (needs the slalom map)')
                        cfg.ekf_control = False
                    self._ek_file = open(_base + '_ekf_state.csv', 'w', newline='')
                    self._ek_w = csv.writer(self._ek_file)
                    self._ek_w.writerow(['t', 'x', 'y', 'z', 'psi_deg', 'sx', 'sy', 'sz', 'spsi_deg', 'n_acc', 'vx', 'vy', 'vz', 'vpsi_deg',
                                         'state', 'phase', 'ek_ms'])
                except Exception as _e:
                    self._ek = None
                    LOG.warning(f'EKF shadow OFF ({_e}). Mission unaffected.')
                    if cfg.ekf_control:
                        LOG.warning('EKF control OFF too (no EKF)')
                        cfg.ekf_control = False

    def ev(self, msg):
        self.events.append((time.monotonic() - self.t_start, msg))
        LOG.info(msg)

    # ---- keys
    def critical(self, ch):
        """Handled immediately, from any thread."""
        if ch in ('e', 'k'):
            if self.cmd:
                self.cmd.emergency_cut(f'key {ch}')
            return True
        if ch in ('p', 'l'):
            if self.cmd and not self.cmd.request_land(f'operator key {ch}'):
                self.ev(f'land key ignored: state={self.cmd.state}')
            return True
        return False

    def feed_key(self, key):
        if key is None or key < 0 or key == 255:
            return
        ch = 'q' if key == 27 else (chr(key).lower() if key < 255 else '')
        self.feed_char(ch)

    def feed_char(self, ch):
        if not ch or self.critical(ch):
            return
        self._keys.append(ch)

    def preflight_problems(self, now, live, st, tage):
        problems = []
        if not live:
            problems.append('video not live')
        elif self._video_age > self.cfg.video_stale_zero:
            problems.append(f'video stale {self._video_age:.1f}s')
        bat = st.get('bat')
        if not isinstance(bat, (int, float)):
            problems.append('no telemetry')
        elif bat < self.cfg.min_battery_takeoff:
            problems.append(f'battery {bat}% < {self.cfg.min_battery_takeoff}%')
        if tage > 1.0 and st:
            problems.append('telemetry stale')
        if not self.cmd or self.cmd.state != 'GROUNDED' or self.cmd.takeoff_sent:
            problems.append('already flown / not grounded: restart the script for another flight')
        return problems

    def cross_problems(self, now, tage):
        cfg, p = self.cfg, []
        st, _ = self.tele.snapshot() if self.tele else ({}, 0)
        if not self.cmd or self.cmd.state != 'FLYING':
            p.append('not flying')
        if not self.auto:
            p.append('auto align is OFF (press G first)')
        elif self._aligned_since is None or now - self._aligned_since < cfg.cross_align_s:
            p.append(f'not aligned within 5 cm for {cfg.cross_align_s:.1f} s yet')
        if self._cf is None or not self.z_ok or now - self._cf_t > 0.5:
            p.append('no fresh valid centre')
        elif not cfg.cross_min_z <= float(self._cf[2]) <= cfg.cross_max_z:
            p.append(f'Z={float(self._cf[2]):.2f} m outside {cfg.cross_min_z}..{cfg.cross_max_z}')
        a = alt_of(st, 'tof')
        if a is None:
            p.append('tof invalid')
        elif abs(a - cfg.hoop_height_cm) > cfg.cross_h_tol:
            p.append(f'height tof={a:.0f} cm, hoop centre is {cfg.hoop_height_cm} +-{cfg.cross_h_tol} cm')
        if self._n_hoop < cfg.min_hoop_markers:
            p.append(f'only {self._n_hoop} hoop markers visible (need {cfg.min_hoop_markers})')
        if cfg.mission and (self._f3 is None or not 1.6 <= self._f3 <= 2.8):
            p.append('ID3 not visible or its distance is implausible')
        bat = st.get('bat')
        if isinstance(bat, (int, float)) and bat < cfg.min_battery_cross:
            p.append(f'battery {bat}%')
        if tage > 1.0 or self._video_age > cfg.video_stale_zero:
            p.append('stale telemetry/video')
        if cfg.ekf_control:
            e = self._ek_align_state(now)
            if e is None:
                p.append('EKF not valid yet (heading unknown)')
            else:
                x, y, psi = e
                d = max(0.0, -x)
                y_pred = y + math.tan(math.radians(psi)) * d + cfg.ekf_lat_bias
                if abs(psi) > cfg.ekf_yaw_tol:
                    p.append(f'EKF heading {psi:+.1f} deg (> {cfg.ekf_yaw_tol:.1f}): the forward would drift {100 * d * math.tan(math.radians(psi)):+.0f} cm sideways')
                if abs(y_pred) > cfg.ekf_y_gate:
                    p.append(f'EKF predicts y={y_pred:+.2f} m at the hoop plane (> {cfg.ekf_y_gate:.2f})')
        return p

    def _handle(self, ch, now, live, st, tage):
        if self.cfg.mission and ch == ' ':
            self.paused = not self.paused
            self.ev('PAUSED: rc zero, hovering (SPACE again to resume)' if self.paused else 'RESUMED')
            return
        if self.cfg.mission and ch in ('f', 'g'):
            self.ev('key disabled in the slalom mission (T,Y start | SPACE pause | P/L land | E/K emergency | Q quit)')
            return
        if ch == ' ':
            self.auto = False
            if self.cmd and self.cmd.state == 'FLYING' and not self.height_done:
                self.height_fail = True
            self.ev('SPACE: auto/climb cancelled, hover')
        elif ch == 't':
            probs = self.preflight_problems(now, live, st, tage)
            if probs:
                self.ev('takeoff blocked: ' + '; '.join(probs))
            else:
                self.confirm_kind = 'takeoff'
                self.confirm_t0, self.confirm_until = now, now + 3.0
                self.ev('CONFIRM TAKEOFF: press Y within 3 s (N cancels). Area clear? Observer ready?')
        elif ch == 'f':
            probs = self.cross_problems(now, tage)
            if probs:
                self.ev('CROSS blocked: ' + '; '.join(probs))
            else:
                self._cross_cm = int(round((float(self._cf[2]) + self.cfg.cross_beyond) * 100))
                self.confirm_kind, self.confirm_t0, self.confirm_until = 'cross', now, now + 3.0
                self.ev(f'CONFIRM CROSSING: forward {self._cross_cm} cm through the hoop, then land. Press Y within 3 s (N cancels)')
        elif ch == 'y':
            if now < self.confirm_until and now - self.confirm_t0 >= 0.3 and self.confirm_kind == 'cross':
                self.confirm_until = 0.0
                probs = self.cross_problems(now, tage)
                if probs:
                    self.ev('CROSS blocked: ' + '; '.join(probs))
                elif self.cmd and self.cmd.request_cross(self._cross_cm):
                    self.auto = False
                    self.ev('crossing confirmed')
            elif now < self.confirm_until and now - self.confirm_t0 >= 0.3:
                self.confirm_until = 0.0
                if self.cmd and self.cmd.request_takeoff():
                    self.ev('takeoff confirmed')
            else:
                self.ev('Y ignored (no pending confirmation)')
        elif ch == 'n':
            self.confirm_until = 0.0
        elif ch == 'g':
            if self.auto:
                self.auto = False
                self.ev('AUTO OFF')
            else:
                self._want_auto = True
        elif ch == 'q':
            if not self.quit_req:
                self.quit_req, self.quit_t = True, now
                if self.cmd and self.cmd.state in AIRBORNE_STATES:
                    self.cmd.request_land('quit requested')
                self.ev('quit requested')

    _want_auto = False

    # ------------------------------------------------------------------ slalom part 1
    def _csv_refs(self):
        fx = self._frame_xf()
        return (round(fx[1], 3) if fx else '', round(self._lv[17][2], 3) if 17 in self._lv else '',
                round(self._lv[5][2], 3) if 5 in self._lv else '', round(self._lv[17][0], 3) if 17 in self._lv else '',
                next((round(self._ang[i], 1) for i in (4, 3, 2, 8, 5, 17) if i in self._ang), ''),
                '' if self._ye is None else round(self._ye, 1))

    def _filter_lv(self, now, st):
        """Drop false detections (the floor mat uses the same dictionary): the implied drone height must agree with tof
        (before the bar, when tof is valid) or with the last marker-based height."""
        ref = None
        tof = valid_alt(st.get('tof'))
        if self.phase in ('CLIMB', 'ALIGN3', 'UP', 'DOWN2') and tof is not None:
            ref = tof / 100.0
        elif self._h_ok(now):
            ref = self._h_est
        else:
            for i in list(self._lv):      # no reference at all: far markers are not trusted
                if i in self.cfg.mk_h and self._lv[i][2] > 3.2:
                    del self._lv[i]
        if ref is None:
            return
        for i in list(self._lv):
            if i in self.cfg.mk_h and (self._lv[i][2] > 3.2 or abs(self.cfg.mk_h[i] + self._lv[i][1] - ref) > self.cfg.h_gate):
                del self._lv[i]

    def _learn_offsets(self):
        """While a frame marker (3/2/4) and a pole marker (5/17) are visible together, learn the pole marker's lateral/depth
        offset relative to the frame. Later the poles alone can stand in for the frame."""
        base = self._ref((3, 2, 4))
        if base is None:
            return
        bx, bf = base[1], base[3]
        if self.phase in ('DOWN2', 'CROSS2') and 8 in self._lv and bf < 1.1:
            g = self._lv[8][2] - bf          # depth of ID8 behind the frame plane
            self._gap8 = g if self._gap8 is None else self._gap8 + 0.15 * (g - self._gap8)
        for i in (5, 17):
            if i in self._lv:
                dx, df = self._lv[i][0] - bx, self._lv[i][2] - bf
                o = self._off.get(i)
                self._off[i] = (dx, df) if o is None else (o[0] + 0.2 * (dx - o[0]), o[1] + 0.2 * (df - o[1]))

    def _frame_xf(self):
        """(x, f, id) of the frame plane from a frame marker, else from a pole marker via the learned offset."""
        r = self._ref((3, 2, 4))
        if r:
            return r[1], r[3], r[0]
        for i in (5, 17):
            if i in self._lv and i in self._off:
                return self._lv[i][0] - self._off[i][0], self._lv[i][2] - self._off[i][1], i
        return None

    def _update_height(self, now, st=None):
        """Drone height from the markers (NEVER tof: over the bars tof reads the bar). Uses the marker nearest the image centre.
        When the reference marker changes (ID3 -> ID4 ...) the step between the two estimates is absorbed and decays, so the
        controller does not see a jump."""
        best = None
        for i, (x, yl, f) in self._lv.items():
            if i in self.cfg.mk_h and (best is None or abs(yl) < abs(best[1])):
                best = (i, yl, self.cfg.mk_h[i] + yl)
        if best:
            fresh = self._h_est is not None and now - self._h_t <= 0.6
            if fresh and best[0] != self._h_ref:
                self._hstep = (self._h_est - best[2]) if abs(self._h_est - best[2]) < 0.20 else 0.0
            self._hstep *= 0.97
            self._h_ref = best[0]
            raw = best[2] + self._hstep
            self._h_est = raw if not fresh else self._h_est + 0.4 * (raw - self._h_est)
            self._h_t = now
        elif st is not None and self.phase in ('ALIGN3', 'UP', 'DOWN2') and self._h_est is not None:
            # no marker in view (hand-over gap): in front of the frame tof is the true height, so bridge with it
            tof = valid_alt(st.get('tof'))
            if tof is not None and abs(tof / 100.0 - self._h_est) < 0.30 and now - self._h_t < 3.0:
                self._h_est += 0.4 * (tof / 100.0 - self._h_est)
                self._h_ref = 'tof'
                self._h_t = now

    def _yaw_learn(self):
        """Per-marker reference of the marker yaw while the drone is at its start heading (CLIMB/ALIGN3/UP)."""
        if self.phase not in ('CLIMB', 'ALIGN3', 'UP'):
            return
        for i in (3, 4, 2, 5, 17):
            if i in self._ang and i in self._lv and self._lv[i][2] < 2.8:
                a = self._ang[i]
                self._yb[i] = a if i not in self._yb else self._yb[i] + 0.05 * (a - self._yb[i])

    def _yaw_add(self, now, flying, fresh, rc):
        """Adds rc yaw to hold the starting heading. Sign: marker yaw > 0 = nose turned clockwise relative to the marker normal,
        so the correction is counter-clockwise (negative). Self-check: if the error grows while correcting, yaw control is disabled."""
        cfg = self.cfg
        act = cfg.yaw_hold and self._yaw_on and flying and fresh and not self.paused
        if act:
            ph = self.phase
            if ph == 'BACK':
                fx = self._frame_xf()      # never turn while inside/next to the frame (+-0.5 m of its plane)
                act = fx is not None and not (-0.50 < fx[1] < 0.35)
            elif ph == 'DOWN2':
                act = not self._d2_ok      # only while lining up in front of ID3, before the descent and the crossing
            elif ph == 'APPR8':
                act = 8 in self._lv and self._lv[8][2] > 0.55
            else:
                act = ph == 'ALIGN3'
        cur = None
        if self.phase == 'APPR8':
            if 8 in self._ang and 8 in self._lv:
                cur = (8, self._ang[8])          # line up with ID8 itself: its normal is the reference
        else:
            for i in (3, 4, 2, 5, 17):
                if i in self._ang and i in self._yb and i in self._lv and self._lv[i][2] < 3.0:
                    cur = (i, self._ang[i] - self._yb[i])
                    break
        if cur:
            self._ye = cur[1] if (self._ye is None or self._ye_id != cur[0]) else self._ye + 0.35 * (cur[1] - self._ye)
            self._ye_id = cur[0]
        else:
            self._ye = None
        y = 0
        if act and self._ye is not None and abs(self._ye) > 3.5:
            y = int(np.clip(-1.2 * self._ye, -cfg.yaw_limit, cfg.yaw_limit))
            if abs(y) < 8:
                y = 8 if y > 0 else -8
        self._yrc = int(np.clip(y, self._yrc - 5, self._yrc + 5))
        if self._yrc != 0 and self._ye is not None:
            if self._yc is None:
                self._yc = (now, abs(self._ye))
            elif now - self._yc[0] > 1.5:
                if abs(self._ye) > self._yc[1] + 2.0:
                    self._yaw_on, self._yrc = False, 0
                    self.ev(f'YAW HOLD DISABLED: error grew from {self._yc[1]:.1f} to {abs(self._ye):.1f} deg while correcting (sign wrong?)')
                self._yc = None
        else:
            self._yc = None
        return (rc[0], rc[1], rc[2], self._yrc if self._yrc != 0 else rc[3])

    def _h_ok(self, now):
        return self._h_est is not None and now - self._h_t < 0.6

    def _slew(self, rc):
        prev, lim = self.last_rc, self.cfg.rc_slew
        return tuple(int(v) if abs(v) <= abs(prev[i]) else int(np.clip(v, prev[i] - lim, prev[i] + lim)) for i, v in enumerate(rc))

    def _lctl(self, x, db=0.05, scale=1.0, minv=8):
        if abs(x) <= db:
            return 0
        v = int(np.clip(scale * self.cfg.kp * x, -12, 12))
        return v if abs(v) >= minv else (minv if x > 0 else -minv)

    def _vctl(self, target, fast=False, db=0.04, minv=8):
        err = target - self._h_est
        if abs(err) < db:
            return 0
        v = int(np.clip(110 * err, -20, 20)) if fast else int(np.clip(90 * err, -15, 20))
        return v if abs(v) >= minv else (minv if err > 0 else -minv)

    def _tofdown(self, target):
        """Blind height hold: only ever DESCENDS toward target using tof. A low tof reading means a bar below, so it is never
        used to climb (the Tello drifts up when it flies forward/back; the bars are above and below the gap)."""
        try:
            st = self.tele.snapshot()[0]
        except Exception:
            return 0
        tof = valid_alt(st.get('tof'))
        if tof is None:
            return 0
        err = target - tof / 100.0
        if err > -0.04 or err < -0.45:
            return 0
        return int(np.clip(110 * err, -15, -8))

    @staticmethod
    def _fband(f, lo, hi):
        return 10 if f > hi else (-10 if f < lo else 0)

    def _set_phase(self, name, now):
        self.phase, self._phase_t, self._ok_t, self._lost_t = name, now, None, None
        self._eh_ref = None
        self._ey_t0 = None
        self._bar_t = -9.0
        self._vz_prev, self._vz = None, 0.0
        self._bk_bar, self._bk_gate_t, self._bk_gate_said = False, None, False
        self.ev(f'MISSION: -> {name}')

    def _abort(self, why):
        self.ev(f'MISSION: {why} -> landing')
        self.cmd.request_land(why)
        return ZERO

    def _lost(self, now, what):
        self._lost_t = self._lost_t or now
        if now - self._lost_t > self.cfg.lost_land:
            return self._abort(f'{what} lost for {self.cfg.lost_land:.0f} s')
        return ZERO   # hover while waiting for the marker to come back

    def _settled(self, now, rc, secs):
        if rc[0] == rc[1] == rc[2] == 0:
            self._ok_t = self._ok_t or now
            return now - self._ok_t >= secs
        self._ok_t = None
        return False

    def _ref(self, ids):
        for i in ids:
            if i in self._lv:
                return (i,) + self._lv[i]
        return None

    def _ek_state(self, now):
        """(x, y, z, psi_deg) from the EKF if fresh and confident, else None."""
        if self._ek is None or self._ek_x is None or now - self._ek_x_t > 0.5:
            return None
        if self._ek_sx is None or self._ek_sx > self.cfg.ekf_max_s or self._ek_sz > self.cfg.ekf_max_s or self._ek_spsi > 3.0:
            return None
        return self._ek_x, self._ek_y, self._ek_z, self._ek_psi

    def _tof_cm(self):
        try:
            return valid_alt(self.tele.snapshot()[0].get('tof')) if self.tele else None
        except Exception:
            return None

    def _z_safe(self, now, ek_z):
        """Conservative drone height (camera frame): the LOWEST of the EKF z, the marker height (h_est) and tof minus its offset.
        tof is used only when it sees the floor (> back_gate_floor): over a bar it reads the bar. Flight 150052: EKF z 1.46 while
        markers and tof said 1.39."""
        cands = [ek_z]
        tof = self._tof_cm()
        if tof is not None and tof / 100.0 > self.cfg.back_gate_floor:
            cands.append(tof / 100.0 - self.cfg.tof_z_offset)
        if self._h_ok(now):
            cands.append(self._h_est)
        zc = min(cands)
        if ek_z - zc > 0.04 and now - getattr(self, '_zs_t', -9.0) > 3.0:
            self._zs_t = now
            self.ev(f'height check: EKF z={ek_z:.2f} but lowest reading is {zc:.2f} m (tof-offset/markers) -> using {zc:.2f}')
        return zc

    def _vz_est(self, now, z):
        pv = getattr(self, '_vz_prev', None)
        if pv is not None and 0.02 < now - pv[0] < 0.5:
            self._vz = getattr(self, '_vz', 0.0) + 0.15 * ((z - pv[1]) / (now - pv[0]) - getattr(self, '_vz', 0.0))
        self._vz_prev = (now, z)
        return getattr(self, '_vz', 0.0)

    def _descend_ekf(self, now, ek):
        """DESCEND with the EKF holding x, y and z. Used only with --ekf-control and a confident EKF; any problem falls back
        to the marker logic. Markers ID5/ID17 are still a safety check but no longer needed to keep going."""
        cfg = self.cfg
        x, y, z, _ = ek
        x0, y0 = self._eh_ref
        frame_x, pole_x = self._ek_geo
        r = self._ref((5, 17))
        if r is not None and r[3] < cfg.pole_min:
            return self._abort(f'too close to the poles ({r[3]:.2f} m)')
        if pole_x - x < cfg.pole_min:
            return self._abort(f'too close to the poles (EKF {pole_x - x:.2f} m)')
        self._lost_t = None
        ey, ex = y - y0, x - x0                      # + = drifted LEFT / FORWARD since the stop point
        lr = self._lctl(ey, db=cfg.ekf_hold_y_db)    # left of the line -> go right (+)
        fb = 0
        if ex > cfg.ekf_hold_db and x - frame_x > 0.30:
            fb = -8                                  # drifted forward: push back, never closer than 0.30 m to the frame bar
        elif ex < -cfg.ekf_hold_db and pole_x - x > 0.90:
            fb = 8                                   # fell short: creep forward
        if fb < 0 and now - self._eh_t > 2.0:
            self._eh_t = now
            self.ev(f'EKF-control DESCEND: drifted {100 * ex:+.0f} cm forward, pushing back (x={x:+.2f}, poles {pole_x - x:.2f} m ahead)')
        zc = self._z_safe(now, z)
        vz = self._vz_est(now, zc)
        err = cfg.gap_height - zc
        ud = 0 if abs(err) < 0.07 else int(np.clip(90 * err, cfg.descend_ud_min, 20))
        if ud and abs(ud) < 11:
            ud = 11 if err > 0 else -11
        out = (lr, fb, ud, self._ek_yaw_cmd(now, ek[3]))
        # done when the HEIGHT has been in the band for 0.7 s; the x/y corrections must not keep resetting this timer
        # (flight 112729: the descent reached 1.56 m at 16 s but the fb/lr pushes kept re-arming the 1 s settle -> 20 s timeout)
        if abs(err) < 0.07 and abs(vz) < cfg.descend_vz_max:
            self._ok_t = self._ok_t or now
            if now - self._ok_t >= cfg.descend_settle:
                self.ev(f'MISSION: gap height reached and settled (z={zc:.2f} m, EKF z={z:.2f}, vz={vz:+.3f} m/s, x={x:+.2f}, y={y:+.2f})')
                self._set_phase('BACK' if cfg.back else 'END', now)
                return ZERO
        else:
            self._ok_t = None
        return self._slew(out)

    def _back_ekf(self, now, ek, fx):
        """BACK pass with the EKF: backs up all the way to the DOWN2 line-up point (frame low_dist ahead) in ONE go, holding the
        y line of the start of the pass and the gap height (the Tello climbs when it flies backwards: flight 113149 rose 1.55 -> 1.75 m
        and DOWN2 then kept backing up at that height to x=0.30 m, 0.30 m from the hoop). Keeps going if ID5/ID17 drop out."""
        cfg = self.cfg
        x, y, z, _ = ek
        frame_x = self._ek_geo[0]
        fr = frame_x - x                     # > 0: frame ahead of the drone (same meaning as f)
        stop = cfg.low_dist - cfg.back_ekf_margin
        zc = self._z_safe(now, z)
        # lateral reference = the frame line (y of ID2/ID4 from the map), not the entry y (DESCEND ends 0.08-0.21 m off and the Tello drifts)
        _fy = self._ek_fy if getattr(self, '_ek_fy', None) is not None else self._eh_ref[1]
        ey, zerr = y - _fy, cfg.gap_height - zc

        def ud_hold(e):
            if abs(e) < 0.04:
                return 0
            u = int(np.clip(150 * e, -30, 20))
            return u if abs(u) >= 10 else (10 if e > 0 else -10)
        if abs(ey) > 0.35 or abs(zerr) > 0.40:
            return self._abort(f'back pass off course (EKF y drift={ey:+.2f} m, dz={zerr:+.2f} m)')
        if fr >= stop:
            if fx is None:
                return self._lost(now, f'frame markers at the end of the back pass (EKF: frame {fr:.2f} m ahead)')
            self.ev(f'MISSION: back pass done by EKF, frame {fr:.2f} m ahead (x={x:+.2f}, y={y:+.2f}, z={z:.2f})')
            self._set_phase('TURN' if cfg.turn else 'FIND3', now)
            return ZERO
        self._lost_t = None
        # ENTRY GATE (the old marker BACK only moved when 'ok'): do not enter the middle bar too low. Only while tof still sees the
        # floor (free air in front of the bar, where hovering is harmless); once a bar has been under the drone the gate is off.
        tofc = self._tof_cm()
        if tofc is not None and tofc / 100.0 < cfg.back_gate_floor:
            self._bk_bar = True
        if not self._bk_bar and tofc is not None and zc < cfg.back_min_z:
            self._bk_gate_t = self._bk_gate_t or now
            if not self._bk_gate_said:
                self._bk_gate_said = True
                self.ev(f'BACK entry gate: z={zc:.2f} m < {cfg.back_min_z:.2f} (EKF z={z:.2f}, tof {tofc:.0f} cm) -> hold and climb before the bar')
            if now - self._bk_gate_t > cfg.back_gate_max:
                return self._abort(f'BACK entry gate: height {zc:.2f} m stayed below {cfg.back_min_z:.2f} for {cfg.back_gate_max:.0f} s')
            lr_g = 0 if abs(ey) <= 0.05 else (8 if ey > 0 else -8)
            return self._slew((lr_g, 0, int(np.clip(150 * (cfg.gap_height - zc), 12, 25)), 0))
        if self._bk_gate_t is not None:
            self.ev(f'BACK entry gate released after {now - self._bk_gate_t:.1f} s (z={zc:.2f} m)')
            self._bk_gate_t = None
        # STRAIGHT LINE: never stops (fb=0 near the frame bar made the Tello drift: flight 133755 sat 3 s over the bar at y=+0.25
        # and was cut). Constant slow fb; y is only nudged back gently (dead band 8 cm, max 8, off near the bar) and z is held with ud.
        # FIND3 re-centres afterwards.
        up = -zerr                            # > 0: above the gap height
        near = -0.45 < fr < 0.25
        fb = -min(cfg.back_fb, 8) if (up > 0.07 or fr > stop - 0.20) else -cfg.back_fb   # full speed over the frame bars (flights 144020/144333 at -8 crawled at 0.03 m/s)
        if abs(zerr) < 0.04:
            ud = 0
        else:
            ud = int(np.clip(150 * zerr, -30, 20))
            if abs(ud) < 10:
                ud = 10 if zerr > 0 else -10
        # lateral stays ON the whole pass (flight 143416: 'near' switched it off from the first cycle, the Tello drifted left
        # 0.10 -> 0.46 m at 0.4 m/s and touched the frame). Dead band 5 cm, rc 8..12, gentle: the ref is the entry y, so no big jump.
        if abs(ey) <= 0.05:
            lr = 0
        else:
            _cap = cfg.back_lr_big if abs(ey) > 0.10 else 12
            lr = int(np.clip(cfg.kp * ey, -_cap, _cap))
            if abs(lr) < 8:
                lr = 8 if ey > 0 else -8
        if abs(ey) > 0.15:
            fb = max(fb, -6)                  # well off the line: slower while it comes back (never 0: a hover over the bar drifts)
        return self._slew((lr, fb, ud, 0))

    def _ek_yaw_cmd(self, now, psi):
        """rc yaw to bring the EKF heading back to 0 (crossing direction). EKF psi is CCW+, rc yaw is CW+: psi>0 -> yaw>0.
        Self-check: if |psi| grows by more than 2 deg while correcting for 2 s the correction is disabled for good."""
        if self._ey_bad or abs(psi) <= 3.0:
            self._ey_t0 = None
            return 0
        if self._ey_t0 is None:
            self._ey_t0, self._ey_psi0 = now, psi
        elif now - self._ey_t0 > 2.0 and abs(psi) > abs(self._ey_psi0) + 2.0:
            self._ey_bad = True
            self.ev(f'EKF heading correction disabled: psi grew from {self._ey_psi0:+.1f} to {psi:+.1f} deg while correcting (sign wrong?)')
            return 0
        return int(np.copysign(max(8, min(12, abs(psi) * 1.2)), psi))

    def _find3_recenter(self, now, ek):
        """FIND3 without ID2/3/4 in view (flight 123108: stopped at y=+0.25, nose 6 deg left, landed after 6 s): move to the frame
        line and turn the nose back to the crossing direction using the EKF, then the markers come back into view."""
        cfg = self.cfg
        x, y, z, psi = ek
        if self._fy_t0 is None:
            self._fy_t0, self._fy_psi0 = now, psi
            self.ev(f'FIND3: ID2/3/4 not in view -> EKF line-up (y={y:+.2f} -> frame line {self._ek_fy:+.2f}, heading {psi:+.1f} deg)')
        ey = y - self._ek_fy
        lr = self._lctl(ey, db=0.04, scale=1.0, minv=8)
        yaw = 0
        if abs(psi) > 3.0 and not self._fy_bad:
            yaw = int(np.copysign(max(8, min(12, abs(psi) * 1.2)), psi))   # EKF psi is CCW+, rc yaw is CW+: psi>0 -> yaw>0
            if now - self._fy_t0 > 2.0 and abs(psi) > abs(self._fy_psi0) + 2.0:
                self._fy_bad, yaw = True, 0
                self.ev(f'FIND3: heading grew from {self._fy_psi0:+.1f} to {psi:+.1f} deg while correcting -> EKF yaw correction disabled')
        err = cfg.gap_height - z
        ud = 0 if abs(err) < 0.05 else int(np.clip(110 * err, -25, 20))
        if ud and abs(ud) < 10:
            ud = 10 if err > 0 else -10
        return self._slew((lr, 0, ud, yaw))

    def _mission(self, now, flying, cstate, obs, fresh, rc, tage):
        cfg = self.cfg
        dt = (now - self._mt) if self._mt else 0.0
        self._mt = now
        self._wv_step(now, obs)          # weave EKF predict/update (no-op until the weave starts)
        if self.paused:
            self._phase_t += dt          # the phase clock does not run while paused
            return ZERO
        ph = self.phase
        if ph == 'CLIMB':
            if flying and self.height_done:
                if cfg.start_plant:
                    self._plant_begin(now)
                elif cfg.start_weave:
                    self._weave_begin(now, chained=False)
                elif cfg.circle:
                    self._set_phase('ALIGN', now)
                    self._want_auto = True
                else:
                    self._set_phase('ALIGN3', now)
            return rc
        if ph in ('ALIGN', 'CROSS', 'APPROACH', 'DONE'):
            return self._hoop_mission(now, flying, cstate, obs, fresh, rc, tage)
        if not flying or not fresh:
            return ZERO
        if ph in cfg.phase_timeout and now - self._phase_t > cfg.phase_timeout[ph]:
            return self._abort(f'phase {ph} timeout')
        hok = self._h_ok(now)
        band = (cfg.start_dist - 0.20, cfg.start_dist + 0.15)
        if ph == 'ALIGN3':
            fx = self._frame_xf()
            if fx is None or not hok:
                return self._lost(now, 'frame markers (ID3/2/4)')
            self._lost_t = None
            x, f, _ = fx
            out = (self._lctl(x), self._fband(f, *band), self._vctl(cfg.mk_h[3]), 0)
            if self._settled(now, out, 1.0):
                self._set_phase('UP', now)
                return ZERO
            return self._slew(out)
        if ph == 'UP':
            fx = self._frame_xf()
            if fx is None or not hok:
                return self._lost(now, 'frame/pole markers')
            self._lost_t = None
            x, f, _ = fx
            out = (self._lctl(x), self._fband(f, *band), self._vctl(cfg.over_height, db=0.07), 0)
            # done when the HEIGHT is in the band (+-7 cm) and the drone is roughly in line for 0.7 s. The small x/fb corrections
            # must not re-arm the timer: flight 113933 hovered at 2.05-2.11 m for 20 s (target 2.15, db 0.04) and hit the UP timeout.
            if abs(cfg.over_height - self._h_est) < 0.07 and abs(x) < 0.15:
                self._ok_t = self._ok_t or now
            else:
                self._ok_t = None
            if self._ok_t is not None and now - self._ok_t >= 0.7:
                if 17 in self._lv and 1.4 <= self._lv[17][2] <= 2.9:
                    self._x17_ref = self._lv[17][0]
                    gap = self._off[17][1] if 17 in self._off else cfg.pole_gap
                    self._tgt = gap - cfg.past
                    self.ev(f'MISSION: at {self._h_est:.2f} m, ID17 {self._lv[17][2]:.2f} m ahead, frame-to-pole gap '
                            f'{gap:.2f} m ({"learned" if 17 in self._off else "configured"}); stop target ID17 at {self._tgt:.2f} m')
                    self._set_phase('OVER', now)
                    return ZERO
                if now - self._ok_t > 6.0:
                    return self._abort('ID17 not seen from the top')
            return self._slew(out)
        if ph == 'OVER':
            if 17 not in self._lv:
                r = self._lost(now, 'ID17')      # keeps the lost timer / abort
                try:
                    _tl = valid_alt(self.tele.snapshot()[0].get('tof')) if self.tele else None
                except Exception:
                    _tl = None
                if r is ZERO and _tl is not None and _tl > cfg.over_lost_tof:
                    if now - getattr(self, '_ol_said', -9.0) > 3.0:
                        self._ol_said = now
                        self.ev(f'OVER: ID17 lost and tof {_tl:.0f} cm > {cfg.over_lost_tof} -> descending gently (ud {cfg.over_lost_ud}) until ID17 is back')
                    return self._slew((0, 0, cfg.over_lost_ud, 0))
                return r
            self._lost_t = None
            x17, yl17, f17 = self._lv[17]
            if f17 < 0.55:
                return self._abort(f'too close to ID17 ({f17:.2f} m)')
            left = f17 - self._tgt
            if left <= 0.04:
                self.ev(f'MISSION: ID17 at {f17:.2f} m -> over the bar, stop')
                self._set_phase('DESCEND', now)
                return ZERO
            ok_h = hok and self._h_est > cfg.over_height - 0.10
            ok_x = abs(x17 - self._x17_ref) < 0.20
            speed = cfg.over_fb if left > 0.30 else max(8, int(cfg.over_fb * left / 0.30))   # slow down near the stop point
            try:
                _tof = valid_alt(self.tele.snapshot()[0].get('tof')) if self.tele else None
            except Exception:
                _tof = None
            if _tof is not None and _tof < 120 and self._h_est is not None and self._h_est > 1.8:
                self._bar_t = now                # the top bar is right under the drone
            if left > 0.30 and now - getattr(self, '_bar_t', -9.0) < 1.5 and cfg.over_boost > speed:
                speed = cfg.over_boost
            vud = self._vctl(cfg.over_height) if hok else 0
            if vud < 0 and _tof is not None and _tof < 120 and self._h_est is not None and self._h_est > 1.8:
                vud = 0                          # the top bar is right under the drone (flight 134815 pushed down at tof 14-50 cm and was cut)
            out = (self._lctl(x17 - self._x17_ref), speed if (ok_h and ok_x) else 0, vud, 0)
            return self._slew(out)
        if ph == 'DESCEND':
            ek = self._ek_state(now) if (cfg.ekf_control and self._ek_geo is not None) else None
            if ek is not None and self._eh_ref is None:
                self._eh_ref, self._eh_t = (ek[0], ek[1]), 0.0
                self.ev(f'EKF-control DESCEND: hold x={ek[0]:+.2f} y={ek[1]:+.2f} (frame at x={self._ek_geo[0]:.2f}, poles at x={self._ek_geo[1]:.2f}), '
                        f'descend to {cfg.gap_height:.2f} m by EKF z')
            if now - self._phase_t < 1.5:
                # let the forward momentum die, but keep the height (the Tello drifts up when it stops)
                return self._slew((0, 0, self._vctl(cfg.over_height, db=0.06) if hok else 0, 0))
            if ek is not None and self._eh_ref is not None:
                return self._descend_ekf(now, ek)
            r = self._ref((5, 17))
            if r is None or not hok:
                # short marker dropout (seen: 2 x ~1.5 s while descending, video fresh): keep going DOWN on tof (beyond the frame tof reads
                # the floor), no sideways/forward moves, instead of hovering and drifting; a long loss still lands (lost_land)
                self._lost_t = self._lost_t or now
                tof = valid_alt(self.tele.snapshot()[0].get('tof')) if self.tele else None
                if now - self._lost_t <= 3.0 and tof is not None and tof / 100.0 > cfg.gap_height + 0.06:
                    return self._slew((0, 0, -10, 0))
                return self._lost(now, 'ID5/ID17')
            self._lost_t = None
            i, x, yl, f = r
            if f < cfg.pole_min:
                return self._abort(f'too close to the poles ({f:.2f} m)')
            lr = self._lctl(self._lv[17][0] - self._x17_ref) if 17 in self._lv else 0
            # the Tello drifts FORWARD while descending (seen: 0.35-0.39 m in 12 s with rc fb = 0): push back gently when the poles come closer
            # than the stop point, but never so far back that the drone gets within 0.30 m of the frame plane (the bar is behind)
            gap = self._off[17][1] if 17 in self._off else cfg.pole_gap
            if f < self._tgt - 0.08 and f < gap - 0.30:
                fb = -8
            else:
                fb = 8 if f > self._tgt + 0.15 else 0   # fell short: creep forward
            out = (lr, fb, self._vctl(cfg.gap_height, db=0.07, minv=11), 0)
            if self._settled(now, out, 1.0):
                self.ev(f'MISSION: gap height reached (H={self._h_est:.2f} m)')
                self._set_phase('BACK' if cfg.back else 'END', now)
                return ZERO
            return self._slew(out)
        if ph == 'BACK':
            # camera looks at the poles; frame plane position comes from ID5/17 minus the learned offset
            fx = self._frame_xf()
            ek = self._ek_state(now) if (cfg.ekf_control and self._ek_geo is not None) else None
            if ek is not None and self._eh_ref is None:
                _ry = ek[1]
                self._eh_ref, self._eh_t, self._bk_hold_t = (ek[0], _ry), 0.0, None
                self.ev(f'EKF-control BACK: entry y={_ry:+.2f}, lateral ref = frame line y={(self._ek_fy if getattr(self, "_ek_fy", None) is not None else _ry):+.2f}, from x={ek[0]:+.2f}; stop with the frame {cfg.low_dist - cfg.back_ekf_margin:.2f} m ahead')
            if ek is not None and self._eh_ref is not None:
                return self._back_ekf(now, ek, fx)
            if fx is None or not hok:
                return self._lost(now, 'ID5/ID17 (back pass)')
            self._lost_t = None
            x, f, src = fx                       # f > 0: frame still AHEAD of the drone (start side reached), f < 0: frame behind
            if f >= cfg.back_dist or (src in (3, 2, 4) and f > 0.30):
                self.ev(f'MISSION: back pass done, seen ID{src}, frame {f:.2f} m ahead (H={self._h_est:.2f} m)')
                self._set_phase('TURN' if cfg.turn else 'FIND3', now)
                return ZERO
            herr = cfg.gap_height - self._h_est
            near = -0.45 < f < 0.25              # drone is inside/next to the frame plane
            if abs(x) > 0.30 or abs(herr) > 0.30:
                return self._abort(f'back pass off course (x={x:+.2f} m, dh={herr:+.2f} m)')
            ok = abs(x) < (0.12 if near else 0.20) and abs(herr) < (0.10 if near else 0.15)
            ud_b = self._vctl(cfg.gap_height, fast=True, db=0.05)
            if ok and herr > -0.12:
                ud_b = min(ud_b, -BACK_UD_FF)        # the Tello climbs when it flies backwards (18:08 flight, battery 100%: 1.51 -> 1.82 m in 4 s, abort): lean down while backing
            out = (self._lctl(x, db=0.08, scale=0.6, minv=6), -cfg.back_fb if ok else 0, ud_b, 0)
            return self._slew(out)
        if ph == 'TURN':
            if not self._turn_req:
                if now - self._phase_t < 1.5:
                    return ZERO                  # let the drone settle before the yaw
                if self.cmd and self.cmd.request_turn(180):
                    self._turn_req, self._turn_t = True, time.monotonic()
                    self.ev('MISSION: yaw 180 (cw 180) to look for ID3')
                return ZERO
            if self.cmd and self.cmd.cross_done_t > self._turn_t:
                self._turn_req = False
                self._set_phase('FIND3', now)
            return ZERO
        if ph == 'FIND3':
            r = self._ref((3, 2, 4))
            if r is None:
                ek = self._ek_state(now) if (cfg.ekf_control and self._ek_geo is not None and self._ek_fy is not None) else None
                if ek is not None and now - self._phase_t < 9.0:
                    return self._find3_recenter(now, ek)      # ID2/3/4 not in view: line up on the frame line by EKF before giving up
                return self._lost(now, 'ID3')
            i, x, yl, f = r
            hid = cfg.mk_h[i] + yl
            self.ev(f'MISSION: ID{i} re-acquired: {f:.2f} m ahead, x={x:+.2f} m, height from marker {hid:.2f} m '
                    f'(tof {st_tof_str(self)})')
            self._d2_ok, self._d2_t = False, None
            self._set_phase('DOWN2' if cfg.cross2 else 'END', now)
            return ZERO
        if ph == 'DOWN2':
            fx = self._frame_xf()
            if fx is None:
                if self._d2_ok and hok:
                    # short hand-over gap (ID3 above the view, ID4 not yet in it): keep descending on tof, no lateral moves
                    self._lost_t = self._lost_t or now
                    if now - self._lost_t <= 3.0:
                        return self._slew((0, 0, self._vctl(cfg.low_height, fast=True), 0))
                return self._lost(now, 'frame markers (descent to the low gap)')
            if not hok:
                return self._lost(now, 'height (descent to the low gap)')
            self._lost_t = None
            x, f, _ = fx
            dband = (cfg.low_dist - 0.10, cfg.low_dist + 0.10)
            if not self._d2_ok:
                # first line up with ID3 (side + heading + distance), only then go down
                good = abs(x) < 0.06 and dband[0] <= f <= dband[1] + 0.05 and (not self._yaw_on or self._ye is None or abs(self._ye) < 4.5)
                self._d2_t = (self._d2_t or now) if good else None
                if self._d2_t and now - self._d2_t >= 0.6:
                    self._d2_ok = True
                    self.ev(f'MISSION: aligned with ID3 before the descent (x={x:+.2f} m, yaw err={self._ye if self._ye is not None else float("nan"):+.1f} deg)')
                return self._slew((self._lctl(x, db=0.04), self._fband(f, *dband), min(0, self._vctl(1.45)), 0))   # no upward drift while lining up
            out = (self._lctl(x), self._fband(f, *dband), self._vctl(cfg.low_height, fast=True), 0)
            # done when the HEIGHT is in the band (+-7 cm) and the drone is in line / at the right distance for 0.7 s; the small x/fb
            # corrections must not keep re-arming the old 'all rc zero for 1 s' timer (same bug that timed out DESCEND and UP)
            if abs(cfg.low_height - self._h_est) < 0.07 and abs(x) < 0.12 and dband[0] - 0.05 <= f <= dband[1] + 0.05:
                self._ok_t = self._ok_t or now
            else:
                self._ok_t = None
            if self._ok_t is not None and now - self._ok_t >= 0.7:
                self.ev(f'MISSION: low gap height reached (H={self._h_est:.2f} m, frame {f:.2f} m ahead, x={x:+.2f} m)')
                self._cr_last, self._cr_prev, self._cr_v = None, None, 0.15
                self._set_phase('CROSS2', now)
                return ZERO
            return self._slew(out)
        if ph == 'CROSS2':
            r = self._ref((4, 3, 2))
            if r:
                i, x, yl, f = r
                if self._cr_prev and self._cr_prev[2] == i and 0.05 < now - self._cr_prev[0] < 0.5:
                    v = (self._cr_prev[1] - f) / (now - self._cr_prev[0])      # measured closing speed (m/s)
                    if 0.02 < v < 0.4:
                        self._cr_v += 0.15 * (v - self._cr_v)
                self._cr_prev = (now, f, i)
                if f > 0.15:
                    self._cr_last = (now, f)
                self._lost_t = None
                herr = (cfg.low_height - self._h_est) if hok else 0.0
                if abs(x) > 0.30 or abs(herr) > 0.22:
                    return self._abort(f'crossing off course (x={x:+.2f} m, dh={herr:+.2f} m)')
                ok = abs(x) < 0.12 and abs(herr) < 0.12
                # flight 124558: the height sagged to 0.76 m (bar at 0.70 m) while moving forward and the slow vertical gain needed ~5 s
                # to recover: faster vertical gain, and the EKF heading is straightened while still 0.55 m or more before the frame plane
                yaw = 0
                ek = self._ek_state(now) if (cfg.ekf_control and self._ek_geo is not None) else None
                if ek is not None and self._ek_geo[0] - ek[0] > 0.55:
                    yaw = self._ek_yaw_cmd(now, ek[3])
                return self._slew((self._lctl(x), cfg.cross2_fb if ok else 0, self._vctl(cfg.low_height, fast=True, db=0.04) if hok else 0, yaw))
            if self._cr_last is None:
                return self._lost(now, 'frame markers (ID4/3/2)')
            t0, f0 = self._cr_last
            if f0 > 0.75:
                return self._lost(now, 'frame markers (ID4/3/2)')      # vanished while still far away: wait, do not go blind
            if self._gap8 is not None and 8 in self._lv:
                # ID8 is a fixed point behind the frame: how far we are past the frame plane = gap8 - f8 (no speed guess needed)
                f8 = self._lv[8][2]
                beyond = self._gap8 - f8
                need = min(cfg.clear2, max(0.25, self._gap8 - cfg.final_dist - 0.10))
                trav = beyond
                passed = beyond >= need
                if f8 < 0.50:
                    return self._abort(f'too close to ID8 while crossing ({f8:.2f} m)')
            else:
                trav = min(self._cr_v, 0.10) * (now - t0)                # fallback: dead reckoning since the last sighting
                passed = trav >= f0 + cfg.clear2
            if 8 in self._lv and passed:
                self.ev(f'MISSION: {trav:.2f} m beyond the frame (ID8 {self._lv[8][2]:.2f} m ahead, frame-to-ID8 gap '
                        f'{"%.2f" % self._gap8 if self._gap8 is not None else "unknown"} m)')
                self._a8, self._a8_v = None, 0.08
                self._set_phase('APPR8', now)
                return ZERO
            if self._gap8 is None and trav > f0 + 2.2:
                return self._abort('ID8 not found after the crossing')
            # blind: straight ahead; height only pulled DOWN by tof. Flight 123831 drifted 0.34 m to the left here with the nose 7-12 deg
            # off: steer sideways on ID8 (fixed point behind the frame) or, without it, on the EKF y; turn the nose back with the EKF heading.
            lat, yaw = 0, 0
            if 8 in self._lv and self._lv[8][2] < 2.8:
                lat = self._lctl(self._lv[8][0], db=0.05, scale=1.0, minv=8)
            ek = self._ek_state(now) if (cfg.ekf_control and self._ek_geo is not None and self._ek_fy is not None) else None
            if ek is not None:
                if lat == 0 and 8 not in self._lv:
                    lat = self._lctl(ek[1] - self._ek_fy, db=0.08, scale=1.0, minv=8)
                if abs(ek[0] - self._ek_geo[0]) > 0.50:
                    yaw = self._ek_yaw_cmd(now, ek[3])      # never turn inside/next to the frame (+-0.5 m of its plane)
            return self._slew((lat, cfg.cross2_fb, self._tofdown(cfg.low_height), yaw))
        if ph == 'APPR8':
            tgt = cfg.final_dist
            if 8 in self._lv:
                x, yl, f = self._lv[8]
                if f < 0.45:
                    return self._abort(f'too close to ID8 ({f:.2f} m)')
                if self._a8 and 0.05 < now - self._a8[0] < 0.5:
                    v = (self._a8[1] - f) / (now - self._a8[0])
                    if 0.02 < v < 0.3:
                        self._a8_v += 0.15 * (v - self._a8_v)
                self._a8 = (now, f)
                self._lost_t = None
                fb = (12 if f - tgt > 0.45 else 10 if f - tgt > 0.25 else 8) if f > tgt + 0.06 else (-8 if f < tgt - 0.08 else 0)
                yaw_ok = (not self._yaw_on) or self._ye is None or abs(self._ye) < 4.0
                out = (self._lctl(x, db=0.04), fb, 0, 0)
                if yaw_ok and self._settled(now, out, 1.0):
                    self.ev(f'MISSION: ID8 at {f:.2f} m, x={x:+.2f} m -> hover')
                    self._set_phase('END', now)
                    return ZERO
                return self._slew(out)
            # ID8 is below the camera's view when close (marker lower than the drone): finish by dead reckoning
            if self._a8 is None or self._a8[1] > 1.1:
                return self._lost(now, 'ID8')
            t0, f0 = self._a8
            rem = f0 - tgt - self._a8_v * (now - t0)
            if now - t0 > 8.0:
                return self._abort('ID8 out of view for 8 s')
            if rem > 0.03:
                return self._slew((0, 8, 0, 0))
            self.ev(f'MISSION: ID8 left the view at {f0:.2f} m; dead-reckoned to ~{tgt:.2f} m (v={self._a8_v:.2f} m/s) -> hover')
            self._set_phase('END', now)
            return ZERO
        if ph == 'END':
            if cfg.weave and cfg.stop_after != 'slalom':
                if now - self._phase_t > cfg.weave_hover_s:
                    self._weave_begin(now, chained=True)
            elif now - self._phase_t > 3.0:
                self.cmd.request_land('slalom complete')
            return ZERO
        if ph.startswith('W_'):
            return self._weave_mission(now, obs)
        if ph.startswith('P_'):
            return self._plant_mission(now, obs)
        return rc

    # ------------------------------------------------------------------ weave + plant (part_2.py of the partner)
    def _weave_begin(self, now, chained):
        """Start the weave. chained: we come from the vertical slalom (hovering ~0.70 m before ID8, so local x ~ 0.05) -> skip the
        ID8 approach and climb to ID5; otherwise (--start-at weave) do the part_2 sequence: APPROACH ID8 -> CLIMB -> waypoints."""
        cfg = self.cfg
        cfg.slalom, cfg.guide_id, cfg.circle = True, None, False
        cfg.slalom_ids = tuple(sorted(set(cfg.slalom_ids) | {6, 7, 13}))
        x0, y0 = 0.05, 0.0
        e = self._ek_align_state(now) if chained else None
        if e is not None and abs(e[0] - WV_ORIGIN[0] - x0) < 0.5 and abs(e[1] - WV_ORIGIN[1]) < 0.4:
            x0, y0 = e[0] - WV_ORIGIN[0], e[1] - WV_ORIGIN[1]
        self._wv = WvEKF([x0, y0, 0.0], np.diag([0.05 ** 2, 0.05 ** 2, np.radians(5) ** 2]))
        self._wv_t = self._wv_yaw = None
        self._wv_meas, self._wv_seen = {}, {}
        self._wv_i, self._wv_stable, self._wv_last_seen, self._wv_t0 = 0, 0, now, now
        self._wv_seen_once, self._wv_low, self._wv_unsure = False, None, None
        self._yb, self._ye, self._ye_id, self._yrc, self._yc = {}, None, None, 0, None
        self.ev(f'WEAVE start ({"chained after the slalom" if chained else "standalone"}): local x={x0:+.2f} y={y0:+.2f} (world {x0 + WV_ORIGIN[0]:+.2f}, {y0 + WV_ORIGIN[1]:+.2f}); '
                f'{len(WV_WAYPOINTS)} waypoints then the plant ID{WV_ID_PLANT}')
        self._set_phase('W_CLIMB' if chained else 'W_APPROACH', now)

    def _wv_step(self, now, obs):
        """part_2 run_ekf: predict with the Tello speeds, update with the weave markers. Called EVERY loop (also paused / not fresh)."""
        if self._wv is None:
            return {}
        try:
            s = self.tele.snapshot()[0] if self.tele else {}
        except Exception:
            s = {}
        poses = {i: tuple(float(v) for v in o['xyz']) for i, o in obs.items() if i in self._wv_ids and self._pl_view_ok(o, i)}
        vgx, vgy, yaw = s.get('vgx'), s.get('vgy'), s.get('yaw')
        if all(isinstance(v, (int, float)) for v in (vgx, vgy, yaw)):
            if self._wv_t is not None and now > self._wv_t:
                dt = now - self._wv_t
                v_f, v_l = _wv_body_speeds(vgx, vgy, yaw)
                if self._plant_on:
                    v_f, v_l = v_f * PLANT_VG_SCALE, v_l * PLANT_VG_SCALE
                self._wv.predict(v_f, v_l, -_wv_wrap(np.radians(yaw - self._wv_yaw)) / dt, dt)
            self._wv_t, self._wv_yaw = now, yaw
        for mid, (cx, cy, cz) in poses.items():
            if mid not in self._wv_seen:
                self._wv_seen[mid] = (now - self._wv_t0, float(np.hypot(cx, cz)))
                self.ev(f'WEAVE SEEN ID{mid} ({WV_NAMES.get(mid, "?")}) at {np.hypot(cx, cz):.2f} m, t={now - self._wv_t0:.1f} s, phase={self.phase}')
            if np.hypot(cx, cz) > WV_MAX_RANGE or self._wv_meas.get(mid) == (cx, cz):
                continue
            self._wv_meas[mid] = (cx, cz)
            accepted = self._wv.update(cx, cz, self._wv_map[mid])
            if self._plant_on:
                diag = self._wv.last_update or {}
                self._pl_update_counts.setdefault(mid, [0, 0])[0 if accepted else 1] += 1
                if getattr(self, '_pl_diag_w', None) is not None:
                    mx, my = self._wv_map[mid]
                    self._pl_diag_w.writerow([round(now-self._wv_t0,3), self.phase, mid,
                        round(cx,4), round(cz,4), mx, my, int(accepted),
                        round(diag.get('nis',float('nan')),3), WV_GATE,
                        round(diag.get('range_innovation_m',float('nan')),4),
                        round(diag.get('bearing_innovation_deg',float('nan')),3),
                        diag.get('reason','unknown')])
            if accepted and self._plant_on:
                self._pl_fix_times[mid] = now
        return poses

    def _wv_debounce(self, ok, need=WV_STABLE_FRAMES):
        self._wv_stable = self._wv_stable + 1 if ok else max(0, self._wv_stable - 2)
        return self._wv_stable >= need

    def _wv_goto(self, name, now):
        self._wv_stable, self._wv_last_seen = 0, now
        self._set_phase(name, now)

    def _wv_check_lost(self, now, mid):
        if now - self._wv_last_seen > WV_LOST_LAND:
            return self._abort(f'weave: ID{mid} lost in {self.phase}')
        return None

    def _wv_fly_to(self, X_w, Y_w, look_id=None):
        x, y, th = self._wv.mu
        ex, ey = X_w - x, Y_w - y
        e_fwd = np.cos(th) * ex + np.sin(th) * ey
        e_left = -np.sin(th) * ex + np.cos(th) * ey
        fb, lr = WV_K_WP * e_fwd * 100, -WV_K_WP * e_left * 100
        speed = float(np.hypot(fb, lr))
        if speed > 0 and np.hypot(ex, ey) > WV_WP_TOL / 2 and speed < WV_MIN_WP_RC:
            fb, lr = fb * WV_MIN_WP_RC / speed, lr * WV_MIN_WP_RC / speed
        fb, lr = _wv_clamp(fb, WV_WP_RC), _wv_clamp(lr, WV_WP_RC)
        th_target = 0.0
        if look_id is not None:
            mx, my = WV_MAP[look_id]
            th_target = float(np.arctan2(my - y, mx - x))
        head_err = np.degrees(_wv_wrap(th - th_target))
        yaw_rc = _wv_clamp(WV_K_YAW * head_err, WV_YAW_RC)
        if abs(head_err) > WV_TURN_FIRST:
            lr = fb = 0
        return lr, fb, yaw_rc, float(np.hypot(ex, ey))

    def _weave_mission(self, now, obs):
        """Port of part_2.Mission.update (stages APPROACH, CLIMB, WAYPOINTS, PLANT, HOLD) -> rc (lr, fb, ud, yaw)."""
        if self._wv is None:
            return self._abort('weave phase without weave state')
        poses = {i: tuple(float(v) for v in o['xyz']) for i, o in obs.items() if i in WV_IDS}   # camera frame, as in part_2
        lr = fb = ud = yaw_rc = 0
        tof = valid_alt(self.tele.snapshot()[0].get('tof')) if self.tele else None
        try:
            h = self.tele.snapshot()[0].get('h', 0)
        except Exception:
            h = 0
        ph = self.phase
        if ph == 'W_APPROACH':
            if WV_ID_LOW in poses:
                self._wv_seen_once, self._wv_last_seen = True, now
                self._wv_low = poses[WV_ID_LOW]
                lr, fb, ud, ok = _wv_center_on(self._wv_low, check_y=False)
                if self._wv_debounce(ok):
                    self._wv_goto('W_CLIMB', now)
            else:
                self._wv_stable = 0
                if not self._wv_seen_once:
                    if tof is not None and 20 < tof < WV_SEARCH_MAX_CM:
                        ud = WV_SEARCH_UP_RC
                    if now - self._phase_t > WV_START_WAIT:
                        return self._abort(f'weave: ID{WV_ID_LOW} never seen')
                else:
                    x_l, y_l, _ = self._wv_low
                    if abs(y_l) >= abs(x_l):
                        ud = WV_MIN_RC if y_l < 0 else -WV_MIN_RC
                        if tof is not None and tof >= WV_SEARCH_MAX_CM:
                            ud = min(ud, 0)
                    else:
                        lr = WV_MIN_RC if x_l > 0 else -WV_MIN_RC
                    r = self._wv_check_lost(now, WV_ID_LOW)
                    if r is not None:
                        return r
        elif ph == 'W_CLIMB':
            if WV_ID_MID in poses:
                self._wv_last_seen = now
                lr, fb, ud, ok = _wv_center_on(poses[WV_ID_MID])
                if self._wv_debounce(ok):
                    self._wv_i = 0
                    self.ev('WEAVE: centred on ID5 -> waypoints')
                    self._wv_goto('W_WP', now)
            else:
                ud = WV_CLIMB_RC
                if WV_ID_LOW in poses:
                    lr, fb, _, _ = _wv_center_on(poses[WV_ID_LOW])
                if isinstance(h, (int, float)) and h > WV_MAX_CLIMB_CM:
                    return self._abort(f'weave: h={h} and no ID{WV_ID_MID}')
        elif ph == 'W_WP':
            X_w, Y_w, h_cm, look_id = WV_WAYPOINTS[self._wv_i]
            lr, fb, yaw_rc, dist = self._wv_fly_to(X_w, Y_w, look_id)
            if h_cm is not None and tof is not None and 20 < tof < 300:
                ud = _wv_clamp(0.5 * (h_cm - tof))
            elif h_cm is None and look_id in poses:
                ud = _wv_pcmd(-poses[look_id][1], min_rc=WV_MIN_UD_RC)
            sig = self._wv.sigma_xy()
            if sig > WV_SIGMA_LAND:
                return self._abort(f'weave: EKF lost (sigma={sig:.2f} m)')
            elif sig > WV_SIGMA_HOVER:
                lr = fb = 0
                self._wv_unsure = self._wv_unsure or now
                if now - self._wv_unsure > WV_SIGMA_WAIT:
                    return self._abort(f'weave: EKF unsure {WV_SIGMA_WAIT:.0f} s (sigma={sig:.2f} m)')
            else:
                self._wv_unsure = None
            if self._wv_debounce(dist < WV_WP_TOL, WV_WP_STABLE):
                x, y, _ = self._wv.mu
                self.ev(f'WEAVE WP {self._wv_i + 1}/{len(WV_WAYPOINTS)} reached: EKF local=({x:+.2f}, {y:+.2f}) sigma={sig:.2f}')
                if self._wv_i + 1 < len(WV_WAYPOINTS):
                    self._wv_i += 1
                    self._wv_goto('W_WP', now)
                else:
                    self._wv_goto('W_PLANT', now)
        elif ph == 'W_PLANT':
            if WV_ID_PLANT in poses:
                self._wv_last_seen = now
                lr, fb, ud, ok = _wv_center_on(poses[WV_ID_PLANT], target_dist=WV_PLANT_DIST)
                if self._wv_debounce(ok):
                    x, y, _ = self._wv.mu
                    self.ev(f'WEAVE: in front of the plant, EKF local=({x:+.2f}, {y:+.2f})')
                    self._wv_goto('W_HOLD', now)
            else:
                self._wv_stable = max(0, self._wv_stable - 2)
                mx, my = WV_MAP[WV_ID_PLANT]
                lr, fb, yaw_rc, _ = self._wv_fly_to(mx - WV_PLANT_DIST, my, WV_ID_PLANT)
                if now - self._wv_last_seen > WV_LOST_LAND:
                    return self._abort(f'weave: ID{WV_ID_PLANT} lost in PLANT')
        elif ph == 'W_HOLD':
            if WV_ID_PLANT in poses:
                lr, fb, ud, _ = _wv_center_on(poses[WV_ID_PLANT], target_dist=WV_PLANT_DIST)
            if now - self._phase_t > WV_HOLD_S:
                seen = ', '.join(f'ID{m}={"OK" if m in self._wv_seen else "NOT SEEN"}' for m in WV_IDS)
                if self.cfg.plant_chain:
                    self.ev(f'WEAVE complete ({seen}) -> plant lap')
                    self._plant_begin(now, chained=True)
                    return ZERO
                self.ev(f'WEAVE complete -> landing near the plant ({seen})')
                self.cmd.request_land('mission complete (weave)')
                return ZERO
        if tof is not None and 20 < tof < WV_MIN_HEIGHT_CM:
            ud = max(ud, 0)
        return self._slew((lr, fb, ud, yaw_rc))

    # ------------------------------------------------------------------ plant lap (--start-at plant)
    def _plant_begin(self, now, chained=False):
        """The drone was placed PLANT_START_DIST from the plant centre, nose towards ID13. Own WvEKF with the plant map (frame: see PLANT_C)."""
        cfg = self.cfg
        cfg.slalom, cfg.guide_id, cfg.circle = True, None, False
        cfg.slalom_ids = tuple(sorted(set(cfg.slalom_ids) | set(PLANT_IDS))) if chained else tuple(PLANT_IDS)
        self._wv_ids, self._wv_map = set(PLANT_IDS), plant_marker_map()
        cx, cy = PLANT_C
        sx, sy, th0 = cx - PLANT_START_DIST, cy, 0.0
        if chained:
            # handoff from the weave (it stops WV_PLANT_DIST in front of ID13): same pose in the plant frame (ID13 = (cx-PLANT_MK_OFF, cy), nose +X)
            wx, wy, wth = (float(v) for v in self._wv.mu)
            sx, sy, th0 = cx - PLANT_MK_OFF - (WV_MAP[WV_ID_PLANT][0] - wx), cy + (wy - WV_MAP[WV_ID_PLANT][1]), wth
        self._plant_on = True
        self._wv = WvEKF([sx, sy, th0], np.diag([0.08 ** 2, 0.08 ** 2, np.radians(6) ** 2]), q_th=np.radians(PLANT_Q_TH_DEG))
        self._wv_t = self._wv_yaw = None
        self._wv_meas, self._wv_seen = {}, {}
        self._wv_i, self._wv_stable, self._wv_last_seen, self._wv_t0 = 0, 0, now, now
        self._wv_seen_once, self._wv_low, self._wv_unsure = False, None, None
        self._pl_pts, self._pl_i = plant_points(), 0
        if cfg.plant_stop_after == '19':
            raise ValueError('Square test supports only align/09/11/16/lap/18; ID19 stage disabled')
        if cfg.plant_stop_after == '18':
            # V31: stop the lap at C16_13 (point 11), slide right past the plant and align with ID18 at PLANT_AUX_DIST
            _cx, _cy = PLANT_C
            self._pl_pts = self._pl_pts[:11] + [   # V32: no via; C16_13 -> ID18 stand in a straight line, nose on ID18
                ('stop', _cx + PLANT_X_FINAL - PLANT_AUX_DIST, _cy + PLANT_Y_FINAL + PLANT_DY_AUX, 18,
                 (_cx + PLANT_X_FINAL, _cy + PLANT_Y_FINAL + PLANT_DY_AUX), PLANT_AUX_DIST, '18')]
        self._pl_cur = ('stop', sx, sy, 13, (cx, cy), PLANT_START_DIST - PLANT_MK_OFF, 'align')   # start alignment = a stop on ID13 at the start stand-off
        self._pl_hold_t, self._pl_stop_seen, self._pl_seen_t, self._pl_end, self._pl_log_t = None, False, now, False, 0.0
        self._pl_cap = 0
        self._pl_fix_times = {}
        self._pl_update_counts = {}
        self._pl_visual_log_t = 0.0
        self._pl_align_warned = False
        self._pl_corner_t = None
        self._pl_corner_bad = 0
        self._pl_vis18 = False   # V33: ID18 leg flown by the camera only (no EKF waypoint, no EKF guards)
        self._pl_target_last = {}  # id -> (time, horizontal camera offset)
        self._pl_track_log_t = 0.0
        self.ev(f'PLANT start{" (chained after the weave)" if chained else ""}: EKF at ({sx - cx:+.2f},{sy - cy:+.2f}) from the centre, nose +X; stop_R={PLANT_R:.2f}m corner_xy=±{PLANT_CORNER_D:.2f}m, hover {PLANT_HOVER_S:.1f} s, '
                f'{len(self._pl_pts)} route points, stop after "{cfg.plant_stop_after}", flight limit {cfg.max_seconds:.0f} s')
        if cfg.start_id18 and not chained:
            # Start at plant point 11 (C16_13), nose +X: skip ID13/lap and go straight to the ID18 visual leg (same code as after the corner)
            sx, sy, th0 = cx - PLANT_CORNER_D + 0.09, cy - PLANT_CORNER_D, 0.0   # logged corner pose: (-0.76, -0.85)
            self._wv = WvEKF([sx, sy, th0], np.diag([0.15 ** 2, 0.15 ** 2, np.radians(10) ** 2]), q_th=np.radians(PLANT_Q_TH_DEG))
            self._wv_t = self._wv_yaw = None
            self._pl_i = len(self._pl_pts) - 1
            self._pl_cur = self._pl_pts[self._pl_i]
            self._pl_vis18 = True
            self.ev(f'PLANT start at ID18 leg: EKF ({sx - cx:+.2f},{sy - cy:+.2f}) nose +X, search/approach ID18 -> slide -> ID19 -> land')
        self._set_phase('P_STOP', now)

    def _pl_view_ok(self, o, mid=None, align=False):
        """Plant mode only: False when the marker is seen more edge-on than PLANT_MAX_VIEW_DEG (angle between its normal and the camera axis)."""
        if not self._plant_on:
            return True
        try:
            n = cv2.Rodrigues(np.asarray(o['rvec'], float))[0][:, 2]
            return math.degrees(math.acos(min(1.0, abs(float(n[2]))))) <= (PLANT_ALIGN_VIEW_DEG if align else PLANT_MAX_VIEW_AUX_DEG if mid in (18, 19) else PLANT_MAX_VIEW_DEG)
        except Exception:
            return True

    def _pl_level(self, obs):
        """Plant markers in the drone-level frame (x right, y DOWN, f forward): the camera pitch is removed, so y = 0 = marker at the camera height."""
        th = math.radians(self.cfg.cam_pitch_deg)
        out = {}
        for i, o in obs.items():
            if i in PLANT_IDS and self._pl_view_ok(o, i, align=True):
                x, y, z = (float(v) for v in o['xyz'])
                out[i] = (x, z * math.sin(th) + y * math.cos(th), z * math.cos(th) - y * math.sin(th))
        return out

    def _pl_ud(self, lvl, tof):
        """Hold a fixed ToF height; visual marker Y is diagnostic, not a second height setpoint."""
        if tof is None or not 20 < tof < 280:
            return 0  # never climb blindly on stale/missing ToF
        h_goal = PLANT_ID18_CM if (self.cfg.plant_stop_after == '18' and self._pl_i >= 11) else PLANT_TAKEOFF_CM
        err_cm = h_goal - tof
        if abs(err_cm) <= 3:
            return 0
        u = max(-12, min(12, round(0.8 * err_cm)))
        if err_cm > 10:  # sinking (battery sag / backward pitch): stronger climb, rc 12 was not enough in the 16:05 flight
            u = min(28, round(1.4 * err_cm))
        if abs(u) < 8:
            u = 8 if u > 0 else -8
        return int(u)

    def _pl_fly_to(self, X, Y, look_xy):
        """EKF position control towards (X, Y) with the nose on look_xy (None = +X). Same law as _wv_fly_to, but the speed is capped on the VECTOR
        (a tangential move keeps a constant speed along the arc) and the heading has a dead band and a minimum rc. -> (lr, fb, yaw, dist)"""
        x, y, th = self._wv.mu
        ex, ey = X - x, Y - y
        dist = float(np.hypot(ex, ey))
        e_fwd = np.cos(th) * ex + np.sin(th) * ey
        e_left = -np.sin(th) * ex + np.cos(th) * ey
        fb, lr = WV_K_WP * e_fwd * 100, -WV_K_WP * e_left * 100
        speed = float(np.hypot(fb, lr))
        if speed > 0 and dist > WV_WP_TOL / 2:
            tgt = min(max(speed, PLANT_MIN_WP_RC), PLANT_WP_RC)
            fb, lr = fb * tgt / speed, lr * tgt / speed
        th_t = 0.0 if look_xy is None else float(np.arctan2(look_xy[1] - y, look_xy[0] - x))
        head_err = float(np.degrees(_wv_wrap(th - th_t)))
        yaw = _wv_clamp(WV_K_YAW * head_err, WV_YAW_RC)
        if abs(head_err) <= PLANT_YAW_DB:
            yaw = 0
        elif abs(yaw) < PLANT_YAW_MIN_RC:
            yaw = PLANT_YAW_MIN_RC if head_err > 0 else -PLANT_YAW_MIN_RC
        if abs(head_err) > PLANT_TURN_FIRST:
            lr = fb = 0                                   # turn first, then translate
        return int(round(lr)), int(round(fb)), yaw, dist

    def _pl_finish(self, now, why):
        self._pl_end = True
        seen = ', '.join(f'ID{m}={"OK" if m in self._wv_seen else "NOT SEEN"}' for m in PLANT_IDS)
        self.ev(f'PLANT complete ({why}) -> landing  [{seen}]')
        self.cmd.request_land(f'plant test: {why}')
        return ZERO

    def _pl_stop_done(self, now, aligned):
        cfg = self.cfg
        kind, X, Y, mid, look, stand, tag = self._pl_cur
        cx, cy = PLANT_C
        x, y, th = (float(v) for v in self._wv.mu)
        self.ev(f'PLANT stop {tag} (ID{mid}) done{"" if aligned else " WITHOUT alignment"}: EKF=({x - cx:+.2f},{y - cy:+.2f}) '
                f'th={math.degrees(th):+.0f} sigma={self._wv.sigma_xy():.2f}')
        # No artificial EKF reset: only accepted landmark updates may correct pose.
        if cfg.plant_stop_after == '18' and tag == '18':
            # V35: ID18 done at ~0.65 m -> slide left ~0.5 m until ID19 is in view, then land
            self.ev('PLANT ID18 done: sliding left to see ID19')
            self._wv_goto('P_SLIDE', now)
            return ZERO
        if cfg.plant_stop_after == tag or tag == 'lap':
            return self._pl_finish(now, f'stage {tag}')
        if tag == 'align':
            # Initial ID13 alignment counts as the first visit; next waypoint is the first corner.
            self._pl_i = 1
        else:
            self._pl_i += 1
        self._wv_goto('P_WP', now)
        return ZERO

    def _plant_mission(self, now, obs):
        """P_STOP (line up on a marker with the camera + hover) and P_WP (EKF route to the next point). land / kill keys stay active (App.critical)."""
        cfg = self.cfg
        if self._wv is None:
            return self._abort('plant phase without plant state')
        if self._pl_end:
            return ZERO
        lvl = self._pl_level(obs)
        # Cache the actual target bearing; do not substitute a nominal EKF yaw
        # when the physical marker briefly leaves the image.
        for _mid, (_ex, _ey, _z) in lvl.items():
            if _mid in (13, 9, 11, 16, 18):
                self._pl_target_last[_mid] = (now, _ex)
        if lvl:
            self._pl_seen_t = now
        elif now - self._pl_seen_t > PLANT_LOST_S:
            return self._abort(f'plant: no marker in view for {PLANT_LOST_S:.0f} s')
        tof = valid_alt(self.tele.snapshot()[0].get('tof')) if self.tele else None
        cx, cy = PLANT_C
        x, y, th = (float(v) for v in self._wv.mu)
        sig = self._wv.sigma_xy()
        r_c = math.hypot(x - cx, y - cy)
        if sig > WV_SIGMA_LAND and not self._pl_vis18:
            return self._abort(f'plant: EKF lost (sigma={sig:.2f} m)')
        if r_c < PLANT_R_MIN and not self._pl_vis18:
            return self._abort(f'plant: EKF says {r_c:.2f} m from the plant centre (< PLANT_R_MIN {PLANT_R_MIN:.2f} m)')
        if x - cx > PLANT_X_MAX and not self._pl_vis18:   # V31: the 0.90 m limit was removed; standard 1.16 m limit now also applies to stage 18
            return self._abort(f'plant: EKF x={x - cx:+.2f} m, too close to the ID18/ID19 board (limit {PLANT_X_MAX:+.2f} m)')
        ph = self.phase
        lr = fb = ud = yaw_rc = 0
        if ph == 'P_WP':
            # No early capture. The drone must reach the EKF waypoint before visual stop.
            kind, X, Y, mid, look, stand, tag = self._pl_pts[self._pl_i]
            lr, fb, yaw_rc, dist = self._pl_fly_to(X, Y, look)
            ud = self._pl_ud(lvl, tof)
            # Compare camera horizontal range with marker-to-EKF range (same physical quantity).
            # Do not compare camera_z directly with distance to the *waypoint*.
            if kind == 'stop' and mid in lvl and tag not in ('align', '18'):
                ex_m, ey_m, z_m = lvl[mid]
                mx, my = plant_marker_map()[mid]
                observed_range = math.hypot(ex_m, z_m)
                predicted_range = math.hypot(mx-x, my-y)
                residual = abs(observed_range-predicted_range)
                if residual > PLANT_GEOM_RANGE_RESIDUAL_M:
                    return self._abort(
                        f'plant: MARKER GEOMETRY MISMATCH ID{mid}: '
                        f'camera_horizontal_range={observed_range:.2f}m, '
                        f'EKF_to_marker={predicted_range:.2f}m, '
                        f'residual={residual:.2f}m; check map and yaw')
                if observed_range < _plant_min_marker_range(mid):
                    return self._abort(
                        f'plant: ID{mid} marker too close: horizontal_range={observed_range:.2f}m')
                # Check straight-line waypoint segment for close passage by this marker.
                dx, dy = X-x, Y-y
                denom = dx*dx + dy*dy
                u = max(0.0, min(1.0, ((mx-x)*dx+(my-y)*dy)/denom)) if denom > 1e-9 else 0.0
                closest = math.hypot(x+u*dx-mx, y+u*dy-my)
                if closest < _plant_min_marker_range(mid):
                    return self._abort(
                        f'plant: PATH CLEARANCE UNVERIFIED ID{mid}: '
                        f'predicted_nearest_marker={closest:.2f}m '
                        f'(< {_plant_min_marker_range(mid):.2f}m); replan via waypoint')
            # ID9 leg: never translate unless the target marker is actually visible.
            # Keep the original EKF waypoint, but use the camera for heading.
            # A fresh bearing can trigger a limited yaw-only recovery.
            if kind == 'stop' and mid == 9:
                if mid not in lvl:
                    lr = fb = 0
                    last = self._pl_target_last.get(mid)
                    if last is not None and now - last[0] <= PLANT_STOP_RECOVERY_S:
                        yaw_rc = _plant_recovery_yaw(last[1])
                    else:
                        yaw_rc = 0
                    if now - self._pl_track_log_t >= PLANT_VISUAL_LOG_S:
                        self._pl_track_log_t = now
                        self.ev(f'PLANT ID9 NOT VISIBLE: translation blocked, yaw={yaw_rc}; '
                                f'last_age={now-last[0]:.2f}s' if last is not None else
                                'PLANT ID9 NOT VISIBLE: translation blocked, no last bearing')
                else:
                    ex9, _, z9 = lvl[mid]
                    bearing9 = math.degrees(math.atan2(ex9, max(z9, 0.05)))
                    # Camera is authoritative for alignment; do not move until
                    # the observed marker is close to the optical axis.
                    yaw_rc = 0 if abs(ex9) <= PLANT_STOP_TRACK_X_DB_M else _wv_clamp(
                        _wv_pcmd(ex9), PLANT_VISUAL_MAX_YAW)
                    if abs(bearing9) > 14.0:
                        lr = fb = 0
            # In the last stretch toward a face, keep its physical ArUco in view.
            # EKF still controls translation; camera x takes priority for yaw.
            if kind == 'stop' and mid != 9 and tag != 'align' and dist <= PLANT_STOP_TRACK_RADIUS_M:  # V31: also drives the ID18 approach
                if mid in lvl:
                    ex, _, z = lvl[mid]
                    if abs(ex) > PLANT_STOP_TRACK_X_DB_M:
                        yaw_rc = _wv_clamp(_wv_pcmd(ex), PLANT_VISUAL_MAX_YAW)
                        # Turn first if the marker is nearing the image edge.
                        if abs(math.degrees(math.atan2(ex, max(z, 0.05)))) > 14.0:
                            lr = fb = 0
                    else:
                        yaw_rc = 0
                    # Already in the acceptable visual distance band: do not
                    # move closer just to chase a nominal EKF waypoint.
                    # V15: only for stops whose nominal stand-off is the normal band (ID13/ID16).
                    # The ID11 stop (stand 0.70 m) lies inside this band, so the old rule froze the drone
                    # ~0.4 m before the waypoint and the stop could never be reached.
                    if (stand >= PLANT_VISUAL_STOP_M and z <= PLANT_VISUAL_STOP_M + PLANT_VISUAL_DIST_TOL
                            and dist <= PLANT_TRACK_FREEZE_MAX_DIST_BY_ID.get(mid, 1e9)):
                        lr = fb = 0
                else:
                    last = self._pl_target_last.get(mid)
                    if last is not None and now - last[0] <= PLANT_STOP_RECOVERY_S:
                        lr = fb = 0
                        yaw_rc = _plant_recovery_yaw(last[1])
                    else:
                        # No fresh bearing: do not continue translating toward
                        # an unseen target. Existing loss/timeout guards remain.
                        lr = fb = yaw_rc = 0
            is_stop = kind in ('stop', 'corner')
            if self._wv_debounce(dist < ((PLANT_STOP_TOL_BY_ID.get(mid, PLANT_STOP_TOL) if kind == 'stop' else PLANT_STOP_TOL) if is_stop else PLANT_VIA_TOL), 4 if is_stop else 1):
                if is_stop:
                    self._pl_cur, self._pl_hold_t, self._pl_stop_seen, self._pl_cap = self._pl_pts[self._pl_i], None, False, 0
                    self._pl_align_warned = False
                    self.ev(f'PLANT reached point {self._pl_i + 1}/{len(self._pl_pts)} ({tag}): EKF=({x - cx:+.2f},{y - cy:+.2f}) '
                            f'th={math.degrees(th):+.0f} sigma={sig:.2f}')
                    self._pl_corner_t = None
                    self._pl_corner_bad = 0
                    self._wv_goto('P_CORNER' if kind == 'corner' else 'P_STOP', now)
                    return ZERO
                self.ev(f'PLANT via {tag} passed: EKF=({x-cx:+.2f},{y-cy:+.2f}) dist={dist:.2f}m; next waypoint')
                self._pl_i += 1                                  # via point passed: next one, no stop
                self._wv_stable, self._phase_t = 0, now
        elif ph == 'P_CORNER':
            # Hover at the diagonal and look at plant centre. At this point yaw is ~45 deg
            # relative to the two adjacent faces, but simultaneous detection is NOT guaranteed.
            kind, X, Y, mids, look, stand, tag = self._pl_cur
            lr, fb, yaw_rc, dist = self._pl_fly_to(X, Y, look)
            ud = self._pl_ud(lvl, tof)
            # Fixed nominal diagonal heading: EKF position jitter must not move the yaw target.
            nominal_heading = math.atan2(cy-Y, cx-X)
            err = abs(math.degrees(_wv_wrap(th - nominal_heading)))
            visible = [mid for mid in mids if mid in lvl]
            fixed = [mid for mid in mids if
                     now - self._pl_fix_times.get(mid, -1e9) <= PLANT_CORNER_FIX_AGE_S]
            # Both markers must be visible, but only one accepted correction is required:
            # a systematic map error must not deadlock an otherwise valid waypoint.
            # V23: C11_16 is seen obliquely (~55 deg) from ID11; ID11 is gated out / leaves the FOV, so only ID16 is required here.
            if tag == 'C11_16':
                fixes_ok = 16 in visible and 16 in fixed
            elif tag == 'C16_13':   # V25: same oblique geometry; one fresh visible+accepted marker is enough
                fixes_ok = any(m in visible and m in fixed for m in mids)
            else:
                fixes_ok = len(visible) == len(mids) and (len(fixed) == 2 if PLANT_CORNER_NEED_BOTH else bool(fixed))
            self._pl_corner_t, self._pl_corner_bad, hold_xy, corner_confirmed = plant_corner_gate(
                now, self._pl_corner_t, self._pl_corner_bad, dist, err, sig, fixes_ok)
            # Once inside the 15 cm arrival radius, do not chase centimetre-scale
            # EKF jitter with minimum 15 RC commands. Allow hysteresis up to 20 cm.
            if hold_xy or dist <= PLANT_CORNER_ENTER_TOL:
                lr = fb = 0
            if corner_confirmed:
                self.ev(f'PLANT corner {tag} confirmed: accepted_fixes={fixed} yawerr={err:.1f} deg sigma={sig:.2f}m')
                self._pl_i += 1
                if cfg.plant_stop_after == '18' and tag == 'C16_13':
                    # V33: point 11 -> ID18 by camera only: turn to ID18, centre it, approach to PLANT_AUX_DIST
                    self._pl_cur, self._pl_hold_t, self._pl_stop_seen, self._pl_cap = self._pl_pts[self._pl_i], None, False, 0
                    self._pl_align_warned, self._pl_vis18 = False, True
                    self._wv_stable = 0
                    self.ev('PLANT ID18 leg: visual servo from point 11 (no EKF waypoint)')
                    self._wv_goto('P_STOP', now)
                    return ZERO
                self._wv_goto('P_WP', now)
                return ZERO
            if now - self._phase_t > PLANT_CORNER_TIMEOUT_S:
                return self._abort(f'plant: corner {tag} not confirmed; visible={visible}, accepted_fixes={fixed}, updates={getattr(self, "_pl_update_counts", {})}, dist={dist:.2f}m yawerr={err:.1f} deg, sigma={sig:.2f}m')
        elif ph == 'P_STOP':
            kind, X, Y, mid, look, stand, tag = self._pl_cur
            yaw_rc = 0  # visual-only yaw; no EKF yaw correction at the stop
            ud = self._pl_ud(lvl, tof)
            if mid in lvl:
                if not self._pl_stop_seen:
                    self._phase_t = now
                self._pl_stop_seen = True
                ex, ey, z = lvl[mid]
                target_z = PLANT_VISUAL_STOP_M if tag != 'align' else max(stand, PLANT_VISUAL_STOP_M)
                if tag == '18':
                    target_z = PLANT_AUX_DIST   # V31: 0.65 m from ID18
                if mid == 9 and tag != 'align':
                    mx, my = plant_marker_map()[9]
                    target_z = math.hypot(X-mx, Y-my)  # horizontal distance, approximate z only when centred
                # One-way approach: never retreat from the marker while confirming a stop.
                # Once in the stand-off band, zero forward/back even if the EKF drifts.
                range_ok = (PLANT_ALIGN_MIN_Z_M if tag == 'align' else PLANT_STOP_MIN_Z_M) <= z <= target_z + PLANT_VISUAL_DIST_TOL
                if tag == '18':
                    range_ok = PLANT_ID18_MIN_Z <= z <= PLANT_AUX_DIST + PLANT_ID18_MAX_OVER   # ID18 is only a waypoint for the slide to ID19: wide band (16:37 flight: stuck 30 s, z drifted 0.69 -> 0.36 with no retreat)
                x_ok = abs(ex) <= (max(PLANT_VISUAL_X_TOL, 0.07 * z) if tag == '18' else PLANT_VISUAL_X_TOL)   # V33: angular tolerance far from ID18
                # Visual centering alone is insufficient: waypoint, heading and
                # accepted EKF landmark correction must agree.
                pos_ok = math.hypot(X-x, Y-y) <= PLANT_STOP_WP_TOL_M
                desired_th = math.atan2(look[1]-y, look[0]-x)
                yaw_ok = abs(math.degrees(_wv_wrap(th-desired_th))) <= PLANT_STOP_YAW_TOL_DEG
                view_ok = self._pl_view_ok(obs[mid], mid) and (
                    math.degrees(math.acos(min(1.0, abs(float(cv2.Rodrigues(
                        np.asarray(obs[mid]['rvec'], float))[0][2,2]))))) <= PLANT_STOP_VIEW_TOL_DEG)
                fix_ok = now - self._pl_fix_times.get(mid, -1e9) <= PLANT_CORNER_FIX_AGE_S
                geometry_ok = (pos_ok and yaw_ok and view_ok and fix_ok and
                               sig <= PLANT_STOP_MAX_SIGMA_M)
                if self._pl_hold_t is None:
                    if (tag not in ('align', 'lap', '18') and mid != 9 and z <= PLANT_VISUAL_STOP_M + PLANT_VISUAL_DIST_TOL
                            and math.hypot(X-x, Y-y) > PLANT_RANGE_MAP_CONFLICT_M):
                        return self._abort(
                            f'plant: STOP MAP/RANGE CONFLICT ID{mid}: camera_z={z:.2f}m, '
                            f'EKF_waypoint_dist={math.hypot(X-x, Y-y):.2f}m')
                    if tag not in ('align', 'lap', '18') and math.hypot(X-x, Y-y) > PLANT_STOP_WP_TOL_M:  # V30: lap = visual-only like align
                        # Recover waypoint using EKF; do not require geometry_ok to move.
                        # If marker is already closer than the stand-off lower bound,
                        # do not command a blind forward approach.
                        lr, fb, yaw_rc, _ = self._pl_fly_to(X, Y, look)
                        if z < PLANT_STOP_MIN_Z_M:
                            # V26: block only the forward approach; retreating (fb<0) is allowed so a
                            # too-close stop (z<0.55, e.g. 15:51 flight at ID9 z=0.50) can recover.
                            fb = min(fb, 0)
                    else:
                        # Fine alignment: yaw on image x; approach only if farther
                        # than target, already centered and within EKF recovery region.
                        if tag == '18' and PLANT_ID18_LATERAL:
                            # centre ID18 by SLIDING sideways (heading kept): centring by yaw turned the drone ~15 deg (16:19 flight) and ID19 then had to be squared again
                            yaw_rc = 0
                            lr = 0 if x_ok else _wv_clamp(_wv_pcmd(ex), PLANT_ID18_MAX_LR)
                        else:
                            yaw_rc = 0 if x_ok else _wv_clamp(_wv_pcmd(ex), 14 if tag == '18' else PLANT_VISUAL_MAX_YAW)
                        fb = (min(15 if tag == '18' else PLANT_VISUAL_MAX_FB, max(0, _wv_pcmd(z - target_z)))
                              if x_ok and z > target_z + PLANT_VISUAL_DIST_TOL else 0)
                        if tag == '18':
                            if z < PLANT_ID18_MIN_Z:
                                fb = -10     # too close (overshoot/coast): back away into the band
                            elif fb:
                                fb = min(fb, 10 if z - target_z < 0.5 else 15)   # slow near the target: 15 rc coasted ~0.2 m past it
                        if not (tag == '18' and PLANT_ID18_LATERAL):
                            lr = 0
                        if tag == 'align' and z < PLANT_ALIGN_MIN_Z_M + PLANT_ALIGN_RETREAT_MARGIN:
                            # chained after the weave (it stops ~0.50-0.55 m from ID13): back away to the stand-off band, otherwise range_ok never holds (18:30 flight: 35 s stuck, z=0.50)
                            fb = -int(max(8, min(12, abs(_wv_pcmd(z - target_z)))))
                    # EKF geometry gates SUCCESS, not the correction commands.
                    stable = x_ok and range_ok and (tag in ('align', 'lap', '18') or geometry_ok)
                    self._wv_stable = self._wv_stable + 1 if stable else 0
                    if self._wv_stable >= PLANT_VISUAL_STABLE_FRAMES:
                        self._pl_hold_t = now
                        lr = fb = yaw_rc = 0
                        self.ev(f'PLANT stop {tag}: visually centred ID{mid} at z={z:.2f}m x={ex:+.3f}m '
                                f'(visual_y={ey:+.3f}m diagnostic only), tof={tof}cm; '
                                f'hovering {PLANT_HOVER_S:.1f} s')
                else:
                    lr = fb = yaw_rc = 0
                    if not (x_ok and range_ok and (tag in ('align', 'lap', '18') or geometry_ok)):
                        self._pl_hold_t = None
                        self._wv_stable = 0
                        self.ev(f'PLANT stop {tag}: visual hold reset (x={ex:+.3f}m z={z:.2f}m)')
                if now - self._pl_visual_log_t >= PLANT_VISUAL_LOG_S:
                    self._pl_visual_log_t = now
                    self.ev(f'PLANT ALIGN ID{mid}: x={ex:+.3f}m z={z:.2f}m target={target_z:.2f}m '
                            f'x_ok={x_ok} range_ok={range_ok} pos_ok={pos_ok} yaw_ok={yaw_ok} view_ok={view_ok} fix_ok={fix_ok} sigma={sig:.2f} y={ey:+.3f}m (not gated) '
                            f'tof={tof}cm stable={self._wv_stable}/{PLANT_VISUAL_STABLE_FRAMES} '
                            f'hold={self._pl_hold_t is not None} rc=({lr},{fb},{ud},{yaw_rc})')
            else:
                self._wv_stable = 0
                self._pl_hold_t = None
                lr = fb = yaw_rc = 0
                # Briefly recover the last observed bearing; rotate only, no
                # translation. Never search indefinitely on a stale direction.
                last = self._pl_target_last.get(mid)
                if last is not None and now - last[0] <= PLANT_STOP_RECOVERY_S:
                    yaw_rc = _plant_recovery_yaw(last[1])
                    if now - self._pl_track_log_t >= PLANT_VISUAL_LOG_S:
                        self._pl_track_log_t = now
                        self.ev(f'PLANT RECOVER ID{mid}: last_x={last[1]:+.3f}m age={now-last[0]:.2f}s yaw={yaw_rc} (no translation)')
                elif tag == '18' and self._pl_vis18:
                    # V34: ID18 not in view yet: search by turning slowly RIGHT (the marker sits right of the nose at the corner)
                    yaw_rc = PLANT_ID18_SEARCH_YAW
                    if now - self._pl_track_log_t >= PLANT_VISUAL_LOG_S:
                        self._pl_track_log_t = now
                        self.ev(f'PLANT SEARCH ID18: turning right yaw={yaw_rc} (no translation)')
                # Beyond the recovery window: hover and retain original guards.
            waited = now - self._phase_t
            if self._pl_hold_t is not None and mid in lvl and now - self._pl_hold_t >= PLANT_HOVER_S:
                return self._pl_stop_done(now, True)
            if self._pl_hold_t is None:
                if tag == 'align' and waited > PLANT_START_WAIT and not self._pl_align_warned:
                    self._pl_align_warned = True
                    self.ev(f'PLANT ALIGN ID{mid}: >{PLANT_START_WAIT:.0f}s; still aligning, no phase timeout')
                if tag != 'align' and waited > (45.0 if tag == '18' else 20.0):
                    return self._abort(f'plant: stop ID{mid} not confirmed within 20 s; landing')
                if False:
                    self._pl_align_warned = True
                    self.ev(f'PLANT ALIGN WAIT ID{mid}: {waited:.1f}s, continuing visual alignment; '
                            f'no landing on stop timer (global flight and safety limits still active)')
        elif ph == 'P_SLIDE':
            # V35: lateral slide to the left (rc lr < 0), no forward/back, no yaw; camera decides when to stop
            ud = self._pl_ud(lvl, tof)
            t_s = now - self._phase_t
            if 19 in lvl and abs(lvl[19][0]) <= PLANT_SLIDE19_X_TOL:
                self.ev(f'PLANT ID19 in view (x={lvl[19][0]:+.2f}m z={lvl[19][2]:.2f}m) after {t_s:.1f} s slide -> fine alignment')
                self._wv_stable, self._pl_19_lost = 0, now
                self._pl_19_settle = None
                self._pl_19_hist = None
                self._pl_19_devh = None
                self._wv_goto('P_ALIGN19', now)
                return ZERO
            if t_s > PLANT_SLIDE19_MAX_S:
                return self._pl_finish(now, f'ID18 aligned; slide timeout {t_s:.1f} s, ID19 not centred')
            lr = -PLANT_SLIDE19_RC
        elif ph == 'P_ALIGN19':
            # V36: centre on ID19 and stop PLANT_ID19_DIST in front of it, then land
            ud = self._pl_ud(lvl, tof)
            t_a = now - self._phase_t
            if 19 in lvl:
                self._pl_19_lost = now
                ex, ey, z = lvl[19]
                ex_raw, ex = ex, ex - PLANT_ID19_X_BIAS   # control on the biased lateral error
                x_ok = abs(ex) <= PLANT_ID19_X_TOL
                z_ok = -PLANT_ID19_Z_TOL_NEAR <= z - PLANT_ID19_DIST <= PLANT_ID19_Z_TOL
                # velocity from the last ~0.4 s -> predicted position after the coast; commands use the prediction, 'ok'/landing use the real values
                hist = getattr(self, '_pl_19_hist', None)
                if hist is None:
                    hist = self._pl_19_hist = []
                hist.append((now, ex, z))
                while hist and now - hist[0][0] > 0.4:
                    hist.pop(0)
                vx = vz = 0.0
                if len(hist) >= 3 and hist[-1][0] - hist[0][0] > 0.15:
                    dtv = hist[-1][0] - hist[0][0]
                    vx, vz = (hist[-1][1] - hist[0][1]) / dtv, (hist[-1][2] - hist[0][2]) / dtv
                ex_p, z_p = ex + vx * PLANT_ID19_LEAD_S, z + vz * PLANT_ID19_LEAD_S
                x_go = abs(ex_p) > PLANT_ID19_X_TOL and not x_ok
                z_go = not (-PLANT_ID19_Z_TOL_NEAR <= z_p - PLANT_ID19_DIST <= PLANT_ID19_Z_TOL) and not z_ok
                # rotation relative to the board from the marker pose: 0 = frontal; >0 = the drone is turned CLOCKWISE (15:45 flight: +16..19 deg after ID18 while the Tello yaw went +19)
                dev = None
                if obs.get(19) is not None:
                    try:
                        R19 = cv2.Rodrigues(np.asarray(obs[19]['rvec'], float))[0]
                        dev = math.degrees(math.atan2(R19[0, 2], -R19[2, 2]))
                    except Exception:
                        dev = None
                if dev is not None:
                    dh = getattr(self, '_pl_19_devh', None)
                    if dh is None:
                        dh = self._pl_19_devh = []
                    dh.append(dev)
                    del dh[:-PLANT_ID19_YAW_MED]
                    dev = float(np.median(dh))
                yaw_ok = dev is None or abs(dev) <= PLANT_ID19_YAW_TOL
                dev_txt = 'n/a' if dev is None else format(dev, '+.1f')
                yaw_rc = 0
                if not yaw_ok:
                    yaw_rc = int(max(-PLANT_ID19_MAX_YAW, min(PLANT_ID19_MAX_YAW, -PLANT_ID19_YAW_GAIN * dev)))
                    if abs(yaw_rc) < 8:
                        yaw_rc = -8 if dev > 0 else 8   # below ~8 the Tello does not rotate (16:57 flight: yaw stuck at -7..-11 deg with rc=6 for 9 s)
                cap_x = PLANT_ID19_NEAR_RC if abs(ex) < PLANT_ID19_NEAR_M else PLANT_ID19_MAX_RC
                cap_z = PLANT_ID19_NEAR_RC if abs(z - PLANT_ID19_DIST) < PLANT_ID19_NEAR_M else PLANT_ID19_MAX_RC
                lr = int(max(-cap_x, min(cap_x, _wv_pcmd(ex_p)))) if x_go else 0
                fb = int(max(-cap_z, min(cap_z, _wv_pcmd(z_p - PLANT_ID19_DIST)))) if z_go else 0
                if lr and abs(lr) < PLANT_ID19_MIN_RC:
                    lr = PLANT_ID19_MIN_RC if lr > 0 else -PLANT_ID19_MIN_RC
                if fb and abs(fb) < PLANT_ID19_MIN_RC:
                    fb = PLANT_ID19_MIN_RC if fb > 0 else -PLANT_ID19_MIN_RC
                all_ok = x_ok and z_ok and yaw_ok
                self._wv_stable = self._wv_stable + 1 if all_ok else 0
                settle = getattr(self, '_pl_19_settle', None)
                if settle is not None:
                    # settle hover: no translation or yaw; if the drone drifted out of the tolerances, go back to aligning
                    lr = fb = yaw_rc = 0
                    k = PLANT_ID19_EXIT_K
                    if not (abs(ex) <= PLANT_ID19_X_TOL * k and -PLANT_ID19_Z_TOL_NEAR * k <= z - PLANT_ID19_DIST <= PLANT_ID19_Z_TOL * k and (dev is None or abs(dev) <= PLANT_ID19_YAW_TOL * k)):
                        all_ok = False
                        self.ev(f'PLANT ID19 settle: drifted out (x={ex_raw:+.3f} z={z:.2f} yaw={dev_txt}) -> aligning again')
                        self._pl_19_settle = None
                        self._wv_stable = 0
                elif self._wv_stable >= PLANT_ID19_STABLE:
                    self._pl_19_settle = now
                    self.ev(f'PLANT ID19 inside the tolerances (x={ex_raw:+.3f} z={z:.2f} yaw={dev_txt}): hovering {PLANT_ID19_SETTLE_S:.1f} s to settle before landing')
                settling = getattr(self, '_pl_19_settle', None) is not None
                if now - self._pl_log_t > 0.6:
                    self.ev(f'PLANT ALIGN ID19: x={ex_raw:+.3f}m (err {ex:+.3f} after bias {PLANT_ID19_X_BIAS:+.2f}) z={z:.2f}m v=({vx:+.2f},{vz:+.2f}) target={PLANT_ID19_DIST:.2f}m x_ok={x_ok} z_ok={z_ok} yaw={dev_txt}deg yaw_ok={yaw_ok} stable={self._wv_stable}/{PLANT_ID19_STABLE} settle={settling} rc=({lr},{fb},{ud},{yaw_rc})')
                st = getattr(self, '_pl_19_settle', None)
                if st is not None and now - st >= PLANT_ID19_SETTLE_S:
                    return self._pl_finish(now, f'aligned with ID19 after settling (x={ex_raw:+.3f} m, z={z:.2f} m, yaw={0.0 if dev is None else dev:+.1f} deg)')
            elif now - self._pl_19_lost > PLANT_ID19_LOST_S:
                return self._pl_finish(now, 'ID19 lost during the final alignment')
            if t_a > PLANT_ID19_MAX_S:
                return self._pl_finish(now, f'ID19 final alignment timeout {t_a:.0f} s')
        else:
            return self._abort(f'unknown plant phase {ph}')
        # EKF confidence guard (as in the weave): too unsure -> stop translating, then land
        if sig > WV_SIGMA_HOVER and not self._pl_vis18:
            lr = fb = 0
            self._wv_unsure = self._wv_unsure or now
            if now - self._wv_unsure > WV_SIGMA_WAIT:
                return self._abort(f'plant: EKF unsure {WV_SIGMA_WAIT:.0f} s (sigma={sig:.2f} m)')
        else:
            self._wv_unsure = None
        if tof is not None and 20 < tof < WV_MIN_HEIGHT_CM:
            ud = max(ud, 0)
        if now - self._pl_log_t > 1.5:
            self._pl_log_t = now
            self.ev(f'PLANT {ph} pt={self._pl_i + 1}/{len(self._pl_pts)} EKF=({x - cx:+.2f},{y - cy:+.2f}) th={math.degrees(th):+.0f} '
                    f'r={r_c:.2f} sig={sig:.2f} seen={sorted(lvl)} rc=({lr},{fb},{ud},{yaw_rc})')
        return self._slew((lr, fb, ud, yaw_rc))

    # ------------------------------------------------------------------ hoop (T2.0, from t20) + HANDOFF to the slalom
    def _ek_align_state(self, now):
        """(x, y, psi_deg) from the EKF if fresh and confident, else None."""
        if self._ek is None or self._ek_x is None or now - self._ek_x_t > 0.5:
            return None
        if self._ek_sx is None or self._ek_sx > self.cfg.ekf_max_sx or self._ek_spsi is None or self._ek_spsi > 3.0:
            return None
        return self._ek_x, self._ek_y, self._ek_psi

    def _hoop_mission(self, now, flying, cstate, obs, fresh, rc, tage):
        """Phases ALIGN -> CROSS -> APPROACH -> DONE (hover) -> HANDOFF (-> ALIGN3). Logic copied from t20 _mission."""
        cfg, ph = self.cfg, self.phase
        if ph == 'ALIGN':
            if cstate == 'CROSSING':
                self._set_phase('CROSS', now)
                return ZERO
            if not flying:
                return rc
            if not self.auto:
                if now - self._last_want > 2.5:
                    self._last_want = now
                    self._want_auto = True
                return rc
            if now - self._phase_t > cfg.align_timeout:
                self.ev('MISSION: could not align/cross in time -> landing')
                self.cmd.request_land('alignment timeout')
                return ZERO
            probs = self.cross_problems(now, tage)
            if not probs:
                cm = int(round((float(self._cf[2]) + cfg.cross_beyond) * 100))
                if self.cmd.request_cross(cm):
                    self.auto = False
                    self.ev(f'MISSION: automatic CROSS, forward {cm} cm')
                    return ZERO
            if cfg.ekf_control:
                e = self._ek_align_state(now)
                if e is not None and abs(e[2]) > cfg.ekf_yaw_dead:
                    # psi is CCW-positive (nose left); rc_yaw>0 turns clockwise -> reduces psi
                    yaw_cmd = int(np.clip(cfg.ekf_kyaw * e[2], -15, 15))
                    rc = (rc[0], rc[1], rc[2], yaw_cmd)
                if any(q.startswith('EKF') for q in probs) and now - self._ek_gate_t > 2.0:
                    self._ek_gate_t = now
                    self.ev('CROSS held by EKF gate: ' + '; '.join(q for q in probs if q.startswith('EKF')))
            return rc
        if ph == 'CROSS':
            if cstate == 'FLYING' and self.cmd.cross_done_t > self._phase_t:
                xe = self._ek_x if (self._ek_x is not None and now - self._ek_x_t < 0.5) else None
                if xe is not None and xe < cfg.cross_min_x and not self._topup_done:
                    self._topup_done = True
                    cm = int(max(20, min(60, round((cfg.cross_min_x + 0.10 - xe) * 100))))
                    self.ev(f'CROSS short: EKF x={xe:+.2f} m < {cfg.cross_min_x:.2f} (still over the hoop rim) -> one top-up forward {cm} cm')
                    if self.cmd.request_cross(cm):
                        self._phase_t = now
                        return ZERO
                self._set_phase('APPROACH', now)
                self._ap, self._g_lost_t, self._ek_ref = None, None, None
                self.ev('MISSION: CROSS done -> APPROACH ID3')
            return ZERO
        if ph == 'APPROACH':
            return self._approach(now, obs, fresh) if flying else ZERO
        if ph == 'DONE':
            if not flying:
                return ZERO
            if cfg.stop_after == 'hoop':
                if now - self._phase_t > cfg.done_hover_s:
                    self.cmd.request_land('mission complete')
                return ZERO
            if now - self._phase_t > cfg.handoff_hover_s:
                self._handoff(now)
            return ZERO
        return rc

    def _handoff(self, now):
        """Hoop done, drone ~0.85 m in front of ID3: switch detection, EKF settings and the guide to the slalom (the EKF and the map
        are the SAME: one world frame), forget the hoop-start yaw reference, and start the slalom at ALIGN3."""
        cfg = self.cfg
        cfg.slalom, cfg.guide_id = True, None
        if self._ek is not None:
            a = self._ek.a
            a.use_tof, a.use_id3_full, a.fuse_exclude = False, True, ''
        self._yb, self._ye, self._ye_id, self._yrc, self._yc = {}, None, None, 0, None
        self._off, self._gap8 = {}, None
        e = self._ek_align_state(now)
        ek_txt = f' (EKF x={e[0]:+.2f} y={e[1]:+.2f} psi={e[2]:+.1f})' if e else ' (EKF not confident)'
        f3_txt = f', ID3 f={self._f3:.2f} m' if self._f3 is not None else ', ID3 not in view'
        self.ev('HANDOFF hoop -> slalom' + ek_txt + f3_txt)
        self._set_phase('ALIGN3', now)


    def _approach(self, now, obs, fresh):
        """After the hoop: creep forward looking only at ID3 until it is guide_stop metres ahead. Lost ID3 -> hover, then land."""
        cfg = self.cfg
        if now - self._phase_t > cfg.approach_timeout:
            self.ev('MISSION: approach timeout -> landing')
            self.cmd.request_land('approach timeout')
            return ZERO
        ek = None
        if cfg.ekf_control and self._ek is not None and self._ek_x is not None and now - self._ek_x_t < 0.5 \
                and self._ek_sx is not None and self._ek_sx < cfg.ekf_max_sx:
            ek = (self._ek_x, self._ek_y, self._ek_z)
            if self._ek_ref is None:
                self._ek_ref = (ek[1], ek[2])
                self.ev(f'EKF-control APPROACH: x={ek[0]:+.2f} y={ek[1]:+.2f} z={ek[2]:.2f}; stop at x>={cfg.ekf_x_stop:.2f}, hold y/z')
            if ek[0] >= cfg.ekf_x_stop:
                self._set_phase('DONE', now)
                self.ev(f'MISSION: EKF x={ek[0]:+.2f} m >= {cfg.ekf_x_stop:.2f} reached -> hover')
                return ZERO
            ez = ek[2] - self._ek_ref[1]
            ud_h = int(np.clip(-cfg.kp * ez, -12, 12)) if abs(ez) > 0.05 else 0
            ey = ek[1] - self._ek_ref[0]
            lr_h = int(np.clip(cfg.kp * ey, -12, 12)) if abs(ey) > 0.06 else 0
        g = obs.get(cfg.guide_id) if cfg.guide_id is not None else None
        if g is None or not fresh:
            if ek is not None:   # ID3 not visible: dead-reckon on the EKF instead of waiting to land
                self._g_lost_t = None
                if not self._ek_ctl_said:
                    self._ek_ctl_said = True
                    self.ev('ID3 not visible: continuing on EKF dead-reckoning')
                return (lr_h, cfg.approach_fb, ud_h, 0)
            self._g_lost_t = self._g_lost_t or now
            if now - self._g_lost_t > cfg.guide_lost_land:
                self.ev('MISSION: ID3 lost during approach -> landing')
                self.cmd.request_land('ID3 lost during approach')
            return ZERO
        self._g_lost_t = None
        th = math.radians(cfg.cam_pitch_deg)
        gx, gy, gz = map(float, g['xyz'])
        off = cfg.guide_cam_off or (0.0, 0.0, 0.0)
        cx, cy, cz = gx - off[0], gy - off[1], gz - off[2]
        meas = np.array([cx, cz * math.sin(th) + cy * math.cos(th), gz * math.cos(th) - gy * math.sin(th)])
        self._ap = meas if self._ap is None else self._ap + 0.4 * (meas - self._ap)
        x, yl, f3 = map(float, self._ap)
        if f3 < cfg.guide_min:
            self.ev(f'MISSION: ID3 too close ({f3:.2f} m) -> landing')
            self.cmd.request_land('too close to ID3')
            return ZERO
        if f3 <= cfg.guide_stop:
            self._set_phase('DONE', now)
            self.ev(f'MISSION: ID3 at {f3:.2f} m reached -> hover')
            return ZERO
        lr = int(np.clip(cfg.kp * x, -12, 12)) if abs(x) > 0.06 else 0
        ye = yl - cfg.y_offset
        ud = int(np.clip(-cfg.kp * ye, -12, 12)) if abs(ye) > 0.06 else 0
        fb = cfg.approach_fb if abs(x) < 0.25 else 0
        if ek is not None:
            ud = ud_h   # height held by the EKF (z fuses tof+vision+velocity)
        return (lr, fb, ud, 0)

    def _K_for(self, frame):
        h, w = frame.shape[:2]
        if (w, h) not in self._K_cache:
            Ks = self.K.copy()
            if self.calib_size:
                Ks[0, :] *= w / self.calib_size[0]
                Ks[1, :] *= h / self.calib_size[1]
            self._K_cache[(w, h)] = Ks
        return self._K_cache[(w, h)]

    # ---- one iteration
    def step(self, raw):
        now = time.monotonic()
        cfg = self.cfg
        if self.cmd:
            self.cmd.beat()
        if raw is not None and raw is not self._last_raw:
            self._last_raw, self._t_new = raw, now
            self._n_new += 1
        video_age = (now - self._t_new) if self._t_new is not None else 1e9
        self._video_age = video_age
        placeholder = raw is None or raw.shape[:2] == (300, 400)  # djitellopy starts with a black 400x300 frame
        live = (not placeholder) and self._n_new >= 3
        if placeholder:
            frame = np.zeros((720, 960, 3), np.uint8)
        else:
            frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if self.rgb else raw.copy()
        if self.cfg.record and live:
            self._rec_write('_cam.mp4', frame)           # raw frame, before any overlay: can be replayed with --video
        st, tage = self.tele.snapshot() if self.tele else ({}, 0.0)
        cstate = self.cmd.state if self.cmd else 'OFFLINE'

        while self._keys:
            self._handle(self._keys.popleft(), now, live, st, tage)

        if cstate == 'FLYING' and self._prev_state != 'FLYING':
            self.climb_t0, self.height_done, self.height_fail, self.ok_since, self.auto = now, False, False, None, False
            self.ev('FLYING: starting supervised height acquisition')
        if cstate != 'FLYING':
            self.auto = False
        self._prev_state = cstate

        # ---- vision
        obs, raw_ids, rejected = {}, [], []
        center, source, pair, pix, valid = None, 'NO_MARKERS', False, None, False
        Ks = self._K_for(frame)
        self._n_hoop, self._f3 = 0, None
        if live:
            extra = {cfg.guide_id: cfg.guide_size} if cfg.guide_id is not None else None
            lay = cfg.layout or make_layout(cfg.separation, 0.0)
            obs, raw_ids, rejected = detect(frame, self.det, Ks, self.dist, cfg.marker_size, extra, tuple(cfg.slalom_ids) if cfg.slalom else tuple(lay))
            center, source, pair = hoop_center(obs, cfg.separation, cfg.guide_id, cfg.guide_off, cfg.axis, lay, cfg.guide_cam_off)
            raw_ids = [i for i in raw_ids if (i in cfg.slalom_ids if cfg.slalom else (i in lay or i == cfg.guide_id))]
            self._n_hoop = sum(1 for i in lay if i in obs)
            _g = obs.get(cfg.guide_id) if cfg.guide_id is not None else None
            if _g is not None:
                _th = math.radians(cfg.cam_pitch_deg)
                self._f3 = float(_g['xyz'][2] * math.cos(_th) - _g['xyz'][1] * math.sin(_th))
            if cfg.guide_id in obs and pair and now - self._gd_t > 2.0:
                self._gd_t = now
                d = (obs[cfg.guide_id]['xyz'] - center) * 100
                self.ev(f'guide vs pair centre: dx={d[0]:+.1f} dy={d[1]:+.1f} dz={d[2]:+.1f} cm')
            pix = project(center, Ks, self.dist)
            valid = center is not None and .45 < center[2] < 4.0 and pix is not None
            if source != self._last_source:
                self._prev_c = None
            if valid and self._prev_c is not None and now - self._prev_t < 0.3 and \
                    float(np.linalg.norm(np.asarray(center) - self._prev_c)) > cfg.jump_max_m:
                self._jumps += 1
                self.n_jump_rejected += 1
                if self._jumps < 5:
                    valid = False  # isolated pose jump: ignore this frame
                    source += '+JUMP'
                else:
                    self.ev('pose re-anchored after 5 consistent jumped frames')
                    self.good = 0
            else:
                self._jumps = 0
            if valid:
                self._prev_c, self._prev_t = np.asarray(center, float), now
            if valid and cfg.cam_pitch_deg:
                # camera frame (X right, Y down, Z optical axis, pitched DOWN by th) -> drone-level frame:
                # y>0 = hoop centre is really BELOW the drone, z = horizontal distance to the hoop plane
                th = math.radians(cfg.cam_pitch_deg)
                cx_, cy_, cz_ = map(float, center)
                center = np.array([cx_, cz_ * math.sin(th) + cy_ * math.cos(th), cz_ * math.cos(th) - cy_ * math.sin(th)])
        # level-frame view of every slalom marker: (x right, y down, f forward), camera pitch removed
        self._lv, self._ang = {}, {}
        _th = math.radians(cfg.cam_pitch_deg)
        for _i, _o in obs.items():
            try:   # marker yaw relative to the camera (deg, sign NOT calibrated yet: logged only)
                _n = cv2.Rodrigues(np.asarray(_o['rvec'], float))[0][:, 2]
                self._ang[_i] = math.degrees(math.atan2(_n[0], abs(_n[2])))
            except Exception:
                pass
            _x, _y, _z = map(float, _o['xyz'])
            self._lv[_i] = (_x, _z * math.sin(_th) + _y * math.cos(_th), _z * math.cos(_th) - _y * math.sin(_th))
        self._filter_lv(now, st)
        self._learn_offsets()
        self._update_height(now, st)
        h, wid = frame.shape[:2]
        if valid:
            self.last_center = center
            self.good += 1   # consecutive frames with a valid centre (1 or 2 markers)
            zlo, zhi = (cfg.pair_zmin, cfg.pair_zmax) if pair else \
                (cfg.guide_zmin, cfg.guide_zmax) if source == 'GUIDE' else (cfg.single_zmin, cfg.single_zmax)
            self.z_ok = zlo < center[2] < zhi
            if self._cf is None or now - self._cf_t > 0.5:
                self._cf = np.asarray(center, float)
            else:
                self._cf = self._cf + cfg.center_alpha * (np.asarray(center, float) - self._cf)
            self._cf_t = now
            if self._n_new != self._vel_n:   # derivative only on NEW video frames
                if self._cf_prev is not None and 0 < now - self._vel_t < 0.3:
                    self._v = 0.8 * self._v + 0.2 * (self._cf - self._cf_prev) / (now - self._vel_t)
                else:
                    self._v = np.zeros(3)
                self._cf_prev, self._vel_t, self._vel_n = self._cf.copy(), now, self._n_new
            if source != self._last_source:
                self.ev(f'centre source: {self._last_source} -> {source}')
                self._last_source = source
            cv2.drawMarker(frame, pix, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.line(frame, (wid // 2, h // 2), pix, (255, 255, 0), 2)
        else:
            self.good = 0
            self.z_ok = False
        cv2.drawMarker(frame, (wid // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 25, 2)

        # ---- control (lateral/vertical + optional distance hold)
        rc = ZERO
        flying = cstate == 'FLYING'
        fresh = tage < 1.0 and video_age < cfg.video_stale_zero
        if flying and not self.height_done and not self.height_fail:
            if now - self.climb_t0 > 18.0:
                self.height_fail = True
                self.auto = False
                self.ev('HEIGHT TIMEOUT (18 s): landing')
                if self.cmd:
                    self.cmd.request_land('height not reached in 18 s')
            else:
                a = alt_of(st, cfg.height_source)
                if a is not None and 15 <= a <= 300 and tage < 1.0:
                    err = cfg.target_height - a
                    if abs(err) <= 12:
                        self.ok_since = self.ok_since or now
                        if now - self.ok_since >= 1.0:
                            self.height_done = True
                            self.ev(f'HEIGHT REACHED ({cfg.height_source}={a:.0f}cm). Press G to align.')
                    else:
                        self.ok_since = None
                        speed = int(np.clip(abs(err) * .35, 15, 20))
                        rc = (0, 0, speed if err > 0 else -speed, 0)
                else:
                    self.ok_since = None  # invalid/stale reading: hover and keep waiting (until the 12 s timeout)
        elif flying and self._want_auto:
            why = []
            if not self.height_done:
                why.append('height not reached')
            if not valid:
                why.append(f'no valid centre ({source}, raw={raw_ids}, rejected={rejected})')
            elif self.good < 5:
                why.append(f'only {self.good}/5 good frames')
            elif not self.z_ok:
                why.append(f'Z={center[2]:.2f}m outside the range for {source}')
            if not fresh:
                why.append(f'stale data (video {video_age:.1f}s, telemetry {tage:.1f}s)')
            if why:
                self.ev('AUTO REFUSED: ' + '; '.join(why))
            else:
                self.auto = True
                self.ev('AUTO ALIGN ON - lateral/vertical only')
        self._want_auto = False
        if flying and self.auto and self.phase == 'ALIGN':
            if valid and self.z_ok and self.good >= 5 and fresh and self._cf is not None:
                x, y, z = map(float, self._cf)
                vx, vy = float(self._v[0]), float(self._v[1])
                ye = y - cfg.y_offset
                # per-axis hysteresis (start beyond 5 cm; stop inside 3 cm AND slow) + PD: the derivative term brakes
                # before the target instead of overshooting
                self._act_x = abs(x) > .05 or (self._act_x and (abs(x) > .03 or abs(vx) > .06))
                self._act_y = abs(ye) > .05 or (self._act_y and (abs(ye) > .03 or abs(vy) > .06))
                A = cfg.rc_auto_max
                lr = int(np.clip(cfg.kp * x + cfg.kd * vx, -A, A)) if self._act_x else 0
                ud = int(np.clip(-(cfg.kp * ye + cfg.kd * vy), -A, A)) if self._act_y else 0
                if abs(lr) < 4:
                    lr = 0
                elif abs(lr) < cfg.rc_auto_min:
                    lr = cfg.rc_auto_min if lr > 0 else -cfg.rc_auto_min
                if abs(ud) < 4:
                    ud = 0
                elif abs(ud) < cfg.rc_auto_min:
                    ud = cfg.rc_auto_min if ud > 0 else -cfg.rc_auto_min
                # slew limit: commands may grow by at most rc_slew per frame; decreasing is immediate
                prev = self.last_rc
                lim = cfg.rc_slew
                lr = lr if abs(lr) <= abs(prev[0]) else int(np.clip(lr, prev[0] - lim, prev[0] + lim))
                ud = ud if abs(ud) <= abs(prev[2]) else int(np.clip(ud, prev[2] - lim, prev[2] + lim))
                fb = 0
                if cfg.hold_z > 0:
                    ez = z - cfg.hold_z          # + = too far from the hoop -> forward; - = too close -> backward
                    if abs(ez) > cfg.hold_z_tol:
                        fb = int(np.clip(cfg.kp * 0.4 * ez, -10, 10))
                        fb = max(6, abs(fb)) * (1 if ez > 0 else -1)
                    if fb and abs(fb) > abs(prev[1]):
                        fb = int(np.clip(fb, prev[1] - lim, prev[1] + lim))
                rc = (lr, fb, ud, 0)
                if abs(x) < .05 and abs(ye) < .05:
                    self._aligned_since = self._aligned_since or now
                    if now - self._aligned_since >= 1.0 and not self._aligned_said:
                        self._aligned_said = True
                        self.ev(f'ALIGNED within 5 cm for 1 s (x={x:+.3f} y={ye:+.3f})')
                else:
                    self._aligned_since = None
                    self._aligned_said = False
            else:
                rc = ZERO  # lost marker/stale data: never keep an old nonzero command
        if not (flying and self.auto):
            self._aligned_since, self._aligned_said = None, False
        if cfg.mission:
            rc = self._mission(now, flying, cstate, obs, fresh, rc, tage)
            self._yaw_learn()
            rc = self._yaw_add(now, flying, fresh, rc)
        if flying:
            if video_age > cfg.video_stale_land and self.cmd:
                self.cmd.request_land('video stale')
            if video_age > cfg.video_stale_zero or tage > cfg.tele_stale_zero:
                rc = ZERO
                self.auto = False
        self.last_rc = rc
        if self.cmd:
            self.cmd.set_setpoint(rc if flying else ZERO)

        if self.quit_req:
            if cstate in ('GROUNDED', 'LANDED', 'EMERGENCY', 'OFFLINE'):
                self.done = True
            elif now - self.quit_t > 40:
                self.done = True
                self.ev('quit timeout while airborne/uncertain: VERIFY THE DRONE PHYSICALLY')

        # ---- overlay + log
        bat = st.get('bat')
        confirm = now < self.confirm_until
        lines = [f'STATE={cstate} PHASE={self.phase} AUTO={self.auto} HEIGHT={"OK" if self.height_done else "FAIL" if self.height_fail else "..."} '
                 f'RC={rc}',
                 f'{source} ' + (f'X={center[0]:+.2f} Y={center[1]:+.2f} Z={center[2]:.2f}m' if valid else 'NO VALID CENTER')
                 + f' | raw={raw_ids} rej={rejected} H={"%.2f" % self._h_est if self._h_est is not None else "-"} paused={self.paused}',
                 f'h={st.get("h")} tof={st.get("tof")} bat={bat} | video_age={min(video_age, 99):.1f}s tele_age={tage:.1f}s']
        if confirm:
            lines.insert(0, '>>> CONFIRM ' + self.confirm_kind.upper() + ': press Y (N cancels) <<<')
        elif cstate == 'GROUNDED':
            pf = self.preflight_problems(now, live, st, tage)
            lines.insert(0, 'PREFLIGHT OK - T then Y to take off' if not pf else 'PREFLIGHT: ' + '; '.join(pf))
        if self._wv is not None and self._plant_on:
            _m = self._wv.mu
            lines.append(f'PLANT EKF x={_m[0] - PLANT_C[0]:+.2f} y={_m[1] - PLANT_C[1]:+.2f} th={math.degrees(_m[2]):+.0f} '
                         f'sig={self._wv.sigma_xy():.2f} pt={self._pl_i + 1}/{len(self._pl_pts)}')
        for i, text in enumerate(lines):
            cv2.putText(frame, text[:120], (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 0, 255) if cstate in ('EMERGENCY', 'UNCERTAIN') else (0, 255, 255), 2)
        cv2.putText(frame, 'E/K EMERGENCY | P/L land | SPACE pause (rc zero) | T,Y start | Q quit',
                    (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        loop_ms = (time.monotonic() - now) * 1000
        self.max_loop_ms = max(self.max_loop_ms, loop_ms)
        self.last_status = lines[0]
        if self._writer:
            xyz = [round(float(v), 4) for v in center] if valid else ['', '', '']
            self._writer.writerow([round(now - self.t_start, 3), cstate, int(self.height_done), int(self.height_fail),
                                   int(self.auto), round(min(video_age, 99), 2), round(tage, 2), st.get('h'),
                                   st.get('tof'), bat, ';'.join(map(str, sorted(obs))), ';'.join(map(str, raw_ids)),
                                   source, *xyz, *(pix if valid else ('', '')), *rc, round(loop_ms, 1),
                                   self.phase, '' if self._h_est is None else round(self._h_est, 3), self._h_ref,
                                   *self._csv_refs()])
            self._rows += 1
            if self._rows % 25 == 0:
                self._csv.flush()
        if self._ek is not None and live:
            self._ek_step(now, obs, st, tage, cstate, frame)
        if self._ek_f is not None and live:
            _t = round(now - self.t_start, 3)
            self._ek_n += 1
            self._ekw_f.writerow([_t, self._ek_n, len(obs), ';'.join(map(str, sorted(obs))), cstate, self.phase, round(tage, 2),
                                  st.get('pitch'), st.get('roll'), st.get('yaw'), st.get('vgx'), st.get('vgy'), st.get('vgz'),
                                  st.get('agx'), st.get('agy'), st.get('agz'), st.get('tof'), st.get('h'), st.get('baro'),
                                  st.get('bat'), *rc])
            for _i, _o in sorted(obs.items()):
                _tv, _rv = _o['xyz'], _o['rvec']
                self._ekw_m.writerow([_t, self._ek_n, _i, *[round(float(v), 5) for v in _tv], *[round(float(v), 5) for v in _rv],
                                      round(_o['rms'], 3), round(float(np.linalg.norm(_tv)), 4)])
            if self._ek_n % 25 == 0:
                self._ek_f.flush()
                self._ek_m.flush()
        if self.cfg.record and live:
            self._rec_write('_overlay.mp4', frame)
        return frame

    def _ek_step(self, now, obs, st, tage, cstate, frame):
        """EKF in shadow mode. ANY exception disables it; it never touches rc/commands."""
        try:
            t0 = time.perf_counter()
            from types import SimpleNamespace
            rows = [SimpleNamespace(id=int(i), tx=float(o['xyz'][0]), ty=float(o['xyz'][1]), tz=float(o['xyz'][2]),
                                    rx=float(o['rvec'][0]), ry=float(o['rvec'][1]), rz=float(o['rvec'][2]),
                                    rms_px=float(o['rms']), dist=float(np.linalg.norm(o['xyz']))) for i, o in obs.items() if i not in (6, 7, 13)]
            t = now - self.t_start
            out = self._ek.step(t, rows, st or {}, blind=False, stale=tage > 0.5)
            ms = (time.perf_counter() - t0) * 1000
            self._ek_ms += ms
            self._ek_cnt += 1
            if out is None:
                return
            self._ek_x, self._ek_y, self._ek_z, self._ek_x_t = float(out['x']), float(out['y']), float(out['z']), now
            self._ek_sx, self._ek_sz = float(out['sx']), float(out['sz'])
            self._ek_psi, self._ek_spsi = float(out['psi_deg']), float(out['spsi_deg'])
            if not self._ek_init_said:
                self._ek_init_said = True
                self.ev(f'EKF shadow initialised: x={out["x"]:+.2f} y={out["y"]:+.2f} z={out["z"]:.2f} psi={out["psi_deg"]:+.1f}deg')
            self._ek_w.writerow([round(t, 3), *[round(float(out[k]), 4) for k in ('x', 'y', 'z', 'psi_deg', 'sx', 'sy', 'sz', 'spsi_deg')],
                                 out['n_acc'], *[('' if out[k] != out[k] else round(float(out[k]), 4)) for k in ('vx', 'vy', 'vz', 'vpsi_deg')],
                                 cstate, self.phase, round(ms, 2)])
            if self._ek_cnt % 25 == 0:
                self._ek_file.flush()
            h = frame.shape[0]
            cv2.putText(frame, f'EKF(shadow) x={out["x"]:+.2f} y={out["y"]:+.2f} z={out["z"]:.2f} psi={out["psi_deg"]:+.0f}  +-{100 * out["sx"]:.0f}/{100 * out["sy"]:.0f}cm',
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 200, 0), 1)
        except Exception as e:
            self._ek = None
            self.ev(f'EKF shadow DISABLED after error: {e!r} (mission unaffected)')

    def _rec_write(self, suffix, img):
        try:
            if not hasattr(self, '_rec'):
                self._rec = {}
            w = self._rec.get(suffix)
            if w is None:
                base = str(getattr(self, '_csv_base', 'logs/rec'))
                h, wd = img.shape[:2]
                w = cv2.VideoWriter(base + suffix, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (wd, h))
                self._rec[suffix] = w
                self.ev(f'RECORD: {base + suffix} ({wd}x{h})')
            w.write(img)
        except Exception as e:
            self.cfg.record = False
            self.ev(f'RECORD failed ({e}); recording off')

    def close(self):
        for _w in getattr(self, '_rec', {}).values():
            _w.release()
        if self._ek_file:
            self._ek_file.flush()
            self._ek_file.close()
            self._ek_file = None
            if self._ek is not None and self._ek_cnt:
                self.ev(f'EKF shadow: {self._ek_cnt} steps, mean {self._ek_ms / self._ek_cnt:.2f} ms/step; NIS pos '
                        f'{(sum(self._ek.nis["pos"]) / len(self._ek.nis["pos"])) if self._ek.nis["pos"] else float("nan"):.2f}')
        if getattr(self, '_pl_diag_f', None):
            self._pl_diag_f.flush()
            self._pl_diag_f.close()
            self._pl_diag_f = None
        for _f in (self._ek_f, self._ek_m):
            if _f:
                _f.flush()
                _f.close()
        self._ek_f = self._ek_m = None
        if self._csv:
            self._csv.flush()
            self._csv.close()
            self._csv = None


class TerminalKeys(threading.Thread):
    """Second input path: works even if the OpenCV window is frozen (terminal must have focus)."""

    def __init__(self, app):
        super().__init__(daemon=True, name='terminal-keys')
        self.app = app
        self.stop_evt = threading.Event()
        self._old = None
        self._fd = None

    def run(self):
        if termios is None or not sys.stdin.isatty():
            return
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        atexit.register(self.restore)
        try:
            tty.setcbreak(self._fd)
            while not self.stop_evt.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    self.app.feed_char(sys.stdin.read(1).lower())
        finally:
            self.restore()

    def restore(self):
        if self._old is not None and self._fd is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None

    def stop(self):
        self.stop_evt.set()


# ----------------------------------------------------------------------------- entry points
def setup_logging(log_dir, stamp):
    logging.getLogger('djitellopy').setLevel(logging.WARNING)  # it logs one INFO line per rc: floods the terminal
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(message)s', '%H:%M:%S')
    for h in (logging.FileHandler(Path(log_dir) / f't21_{stamp}.events.log'), logging.StreamHandler()):
        h.setFormatter(fmt)
        LOG.addHandler(h)


def run_video(args, cfg, K, dist, size, stamp):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    app = App(cfg, K, dist, size, rgb=False, csv_path=Path(args.log_dir) / f't21_{stamp}.csv')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = app.step(frame)
            cv2.imshow('T1.8 offline', out)
            if cv2.waitKey(25) & 0xFF in (ord('q'), 27):
                break
    finally:
        app.close()
        cap.release()
        cv2.destroyAllWindows()


def shutdown_land(cmd, timeout=45.0):
    """Never abandon an airborne drone: land first (any exit path: window error, Ctrl+C, quit timeout)."""
    if cmd.state in AIRBORNE_STATES and not cmd.emergency.is_set():
        LOG.warning('Shutting down while %s: requesting landing first', cmd.state)
        cmd.request_land('shutdown')
        t0 = time.monotonic()
        while cmd.state in AIRBORNE_STATES and time.monotonic() - t0 < timeout and not cmd.emergency.is_set():
            time.sleep(0.2)


def run_live(args, cfg, K, dist, size, stamp):
    from djitellopy import Tello
    drone = Tello()
    # djitellopy prints one INFO line per command (rc at 20 Hz): keep only warnings/errors on the terminal
    for _lg in (getattr(Tello, 'LOGGER', None), logging.getLogger('djitellopy')):
        if _lg is not None:
            _lg.setLevel(logging.WARNING)
            for _h in list(_lg.handlers):
                _h.addFilter(lambda rec: rec.levelno >= logging.WARNING)
    drone.connect()
    bat = drone.get_battery()
    LOG.info(f'Battery: {bat}%')
    if bat < cfg.min_battery_takeoff:
        LOG.warning(f'Battery below {cfg.min_battery_takeoff}%: takeoff will be blocked')
    drone.streamoff()
    drone.streamon()
    reader = drone.get_frame_read()
    tele = Telemetry(drone)
    tele.poll()
    cmd = Commander(drone, tele, cfg, ev=lambda m: LOG.info(m))
    app = App(cfg, K, dist, size, cmd=cmd, tele=tele, rgb=True, csv_path=Path(args.log_dir) / f't21_{stamp}.csv')
    cmd.ev = app.ev
    cmd.start()
    keys = TerminalKeys(app)
    keys.start()
    LOG.info('T2.1 SLALOM part 1 ready. T,Y = start | SPACE = pause (rc zero, hover) | P/L = land | E/K = EMERGENCY (falls) | Q = quit')
    try:
        while not app.done:
            out = app.step(reader.frame)
            cv2.imshow('T1.9 CROSS', out)
            app.feed_key(cv2.waitKey(25) & 0xFF)
    except KeyboardInterrupt:
        LOG.info('Ctrl+C: landing')
        cmd.request_land('Ctrl+C')
        t0 = time.monotonic()
        while cmd.state in AIRBORNE_STATES and time.monotonic() - t0 < 40:
            time.sleep(0.2)
    finally:
        keys.stop()
        shutdown_land(cmd)
        final = cmd.state
        cmd.stop()
        cmd.join(timeout=2)
        app.close()
        if final in AIRBORNE_STATES:
            LOG.warning('!!! FINAL STATE %s: the drone may still be airborne. VERIFY PHYSICALLY / use the Tello app.', final)
        try:
            drone.send_command_without_return('streamoff')
            drone.stream_on = False
            drone.end()
        except Exception:
            pass
        cv2.destroyAllWindows()
        LOG.info(f'Final commander state: {final}. Logs in {args.log_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibration', default='tello_calibration.npz')
    ap.add_argument('--video', help='Offline video mode: no drone, no flight')
    ap.add_argument('--max-seconds', type=float, default=None, help='flight time limit (s); default 420 for the whole mission (the Tello battery is ~10 min), PLANT_MAX_S with --start-at plant')
    ap.add_argument('--target-height-cm', type=int, default=133)
    ap.add_argument('--height-source', choices=('tof', 'h'), default='tof',
                    help='tof = distance to floor below; h = Tello height. Both are logged.')
    ap.add_argument('--marker-size', type=float, default=0.115)
    ap.add_argument('--separation', type=float, default=0.55)
    ap.add_argument('--log-dir', default='logs')
    ap.add_argument('--hsep', type=float, default=0.50, help='distance between the centres of the LEFT and RIGHT markers (m); 0 = not used')
    ap.add_argument('--hids', default='10,12', help='ids of the LEFT and RIGHT markers seen from the FRONT (drone side)')
    ap.add_argument('--axis', choices=('v', 'h'), default='v',
                    help='v = ID0 below / ID1 above the hoop centre (default); h = ID0 LEFT / ID1 RIGHT (use when you are close: the camera sees more sideways than up/down)')
    ap.add_argument('--y-offset', type=float, default=0.0,
                    help='hold the hoop centre this far BELOW the image centre (m at hoop distance); negative = above')
    ap.add_argument('--hoop-height-cm', type=int, default=133, help='hoop centre height as read by tof; F is blocked outside +-10 cm of it')
    ap.add_argument('--hold-z', type=float, default=1.0, help='keep this distance (m) to the hoop plane with small forward/back moves; 0 = off')
    ap.add_argument('--cam-pitch-deg', type=float, default=10.0, help='camera downward tilt (deg); 0 = off')
    ap.add_argument('--cross-beyond', type=float, default=0.45, help='metres past the hoop plane for the blind forward move (then the ID3 approach takes over)')
    ap.add_argument('--no-mission', action='store_true', help='manual mode (G, F keys as in T1.9)')
    ap.add_argument('--guide-stop', type=float, default=0.85, help='stop when ID3 is this far ahead (m)')
    ap.add_argument('--guide-id', type=int, default=3, help='ID of the guide marker behind the hoop (-1 = off). Only used while the hoop runs; the slalom turns it off')
    ap.add_argument('--guide-cam-off', default='-0.265,-0.288,1.285', help='dx,dy,dz in metres from the "guide vs pair centre" log, e.g. -0.265,-0.288,1.285')
    ap.add_argument('--guide-size', type=float, default=0.115)
    ap.add_argument('--over-height-cm', type=int, default=215, help='height for flying over the top bar')
    ap.add_argument('--gap-height-cm', type=int, default=150, help='height of the gap between the top and middle bars')
    ap.add_argument('--pole-gap', type=float, default=1.20, help='m between the frame and the vertical poles')
    ap.add_argument('--back-dist', type=float, default=0.80, help='fallback: stop the back pass this far in front of the frame if ID3 was not seen')
    ap.add_argument('--low-height-cm', type=float, default=95., help='height for crossing the low gap (between ID4 and ID3)')
    ap.add_argument('--final-dist', type=float, default=0.70, help='m from ID8 where it stops and lands')
    ap.add_argument('--low-dist', type=float, default=0.95, help='m in front of the frame while going down to the low gap')
    ap.add_argument('--clear2', type=float, default=0.60, help='m beyond the frame plane before the drone lines up with ID8')
    ap.add_argument('--cross2-fb', type=int, default=12, help='rc forward speed through the low gap')
    ap.add_argument('--no-cross', action='store_true', help='stop after the back pass (land when ID3 is seen)')
    ap.add_argument('--no-yaw-hold', action='store_true', help='disable the heading hold (rc yaw)')
    ap.add_argument('--no-back', action='store_true', help='land after the descent instead of flying back')
    ap.add_argument('--turn', action='store_true', help='(not used by default) yaw 180 deg after the back pass')
    ap.add_argument('--pole-min', type=float, default=0.40, help='abort (land) if closer than this to the poles while descending (m)')
    ap.add_argument('--past', type=float, default=0.45, help='m past the frame before descending')
    ap.add_argument('--start-dist', type=float, default=0.85, help='m from the frame at the start')
    ap.add_argument('--mk-h', default='4:0.70,3:1.30,2:1.85,5:1.50,17:1.82', help='marker centre heights id:m')
    ap.add_argument('--guide-depth', type=float, default=0.0,
                    help='how far BEHIND the hoop plane the guide is (m); the hoop centre is taken this far towards the camera')
    ap.add_argument('--over-boost', type=int, default=14, help='rc forward while crossing the top bar in OVER (14 = no boost)')
    ap.add_argument('--ekf-control', action='store_true', help='EKF holds x/y/z in DESCEND and keeps going if ID5/ID17 drop out')
    ap.add_argument('--no-ekf-shadow', action='store_true', help='disable the EKF shadow mode (it only logs anyway)')
    ap.add_argument('--ekf-map', default='aruco_map_circulo_calibrado.yaml', help='ONE map for hoop + slalom (same world frame)')
    ap.add_argument('--start-at', choices=('hoop', 'slalom', 'weave', 'plant', 'id18'), default='hoop',
                    help='hoop = whole mission (hoop, slalom, weave+plant); slalom = start at ALIGN3 in front of the frame (old t21); '
                         'weave = only the weave between the poles + plant (part_2), start 0.75 m in front of the purple pole; '
                         'plant = only the plant lap (take off PLANT_START_DIST from the plant centre, nose to ID13; all tunables are the PLANT_* constants)')
    ap.add_argument('--record', action='store_true', help='save the camera video (<log>_cam.mp4 raw, <log>_overlay.mp4 with the drawings) next to the logs')
    ap.add_argument('--no-plant', action='store_true', help='whole mission: land at the plant after the weave (old behaviour) instead of flying the plant lap + ID18 + ID19')
    ap.add_argument('--plant-stop-after', choices=('align', '09', '11', '16', 'lap', '18'), default='18',
                    help='--start-at plant: land once this stage is done. align = lined up on ID13 at the start; 13/09/11/16 = that marker stop; '
                         'lap = back at ID13 after the full lap (default); 18, 19 = the final leg (19 = land aligned with ID19)')
    ap.add_argument('--stop-after', choices=('hoop', 'slalom'), default=None, help='hoop = land after the ID3 approach (old t20); slalom = land after the vertical slalom (no weave)')
    ap.add_argument('--ekf-x-stop', type=float, default=0.55, help='hoop APPROACH ends when EKF x >= this (m)')
    ap.add_argument('--ekf-vg-gain', type=float, default=1.2)
    args = ap.parse_args()
    id18_only = args.start_at == 'id18'
    if id18_only:   # only the last leg: reuse the plant machinery, stopping at ID18 -> slide -> ID19
        args.start_at, args.plant_stop_after = 'plant', '18'
        if args.max_seconds is None:
            args.max_seconds = 90.0
    if args.max_seconds is None:
        args.max_seconds = (PLANT_MAX_S_LAP if args.plant_stop_after in ('lap', '18') else PLANT_MAX_S_FINAL if args.plant_stop_after in ('19', '16') else PLANT_MAX_S) if args.start_at == 'plant' else (420.0 if args.no_plant or args.stop_after else FULL_MAX_S)   # V19: 16 needs ~66-81 s to ID11 + the 11->16 leg
    if args.max_seconds > (180 if args.plant_stop_after in ('lap', '18') else 150 if args.plant_stop_after == '16' else 90) and args.start_at == 'plant':
        ap.error('--start-at plant: max-seconds cannot exceed 90 (150 with --plant-stop-after 16, 180 with lap/18)')
    if not 5 <= args.max_seconds <= 600:
        ap.error('--max-seconds must be 5..600')
    if not 80 <= args.target_height_cm <= 160:
        ap.error('--target-height-cm must be 80..160')
    if args.ekf_control and not args.no_ekf_shadow and args.start_at != 'plant':
        # fail BEFORE touching the drone: --ekf-control without the EKF files would hold the hoop CROSS forever
        _missing = [f for f in ('ekf_tello.py', 'pose_global.py', args.ekf_map) if not (Path(__file__).resolve().parent / f).exists() and not Path(f).exists()]
        if _ekf_mod is None or _missing:
            ap.error(f'--ekf-control needs these next to the script / in the current folder: {_missing or "ekf_tello.py + pose_global.py"} '
                     f'(import error: {_ekf_import_err if _ekf_mod is None else "none"})')
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_logging(args.log_dir, stamp)
    cfg = Cfg(max_seconds=args.max_seconds, target_height=args.target_height_cm, height_source=args.height_source,
              marker_size=args.marker_size, separation=args.separation,
              guide_id=None if args.guide_id < 0 else args.guide_id, guide_size=args.guide_size,
              guide_off=(0.0, 0.0, args.guide_depth),
              ekf_shadow=not args.no_ekf_shadow, over_boost=args.over_boost, ekf_control=args.ekf_control and not args.no_ekf_shadow, ekf_map=args.ekf_map, ekf_vg_gain=args.ekf_vg_gain,
              mission=not args.no_mission, guide_stop=args.guide_stop,
              over_height=args.over_height_cm / 100, gap_height=args.gap_height_cm / 100, pole_gap=args.pole_gap, past=args.past, pole_min=args.pole_min, back=not args.no_back, back_dist=args.back_dist, turn=args.turn, yaw_hold=not args.no_yaw_hold, cross2=not args.no_cross, low_height=args.low_height_cm / 100, final_dist=args.final_dist, clear2=args.clear2, low_dist=args.low_dist, cross2_fb=args.cross2_fb,
              start_dist=args.start_dist, mk_h={int(k): float(v) for k, v in (kv.split(':') for kv in args.mk_h.split(','))},
              guide_cam_off=tuple(float(v) for v in args.guide_cam_off.split(',')) if args.guide_cam_off else None, y_offset=args.y_offset, cam_pitch_deg=args.cam_pitch_deg, hoop_height_cm=args.hoop_height_cm, hold_z=args.hold_z, cross_beyond=args.cross_beyond, axis=args.axis,
              layout=make_layout(args.separation, args.hsep, tuple(int(v) for v in args.hids.split(','))))
    cfg.circle = args.start_at == 'hoop'
    cfg.slalom = not cfg.circle
    cfg.stop_after = args.stop_after
    cfg.start_weave = args.start_at == 'weave'
    cfg.weave = args.stop_after is None
    cfg.start_plant = args.start_at == 'plant'
    cfg.start_id18 = id18_only
    cfg.plant_chain = (not args.no_plant) and args.stop_after is None and not cfg.start_plant
    cfg.plant_stop_after = args.plant_stop_after
    cfg.record = args.record
    if cfg.start_plant:
        cfg.slalom_ids = tuple(PLANT_IDS)          # detect only the plant markers (the floor mat uses the same dictionary)
        cfg.target_height = PLANT_TAKEOFF_CM       # take off and climb to the plant marker height (--target-height-cm is ignored here)
        cfg.ekf_shadow = cfg.ekf_control = False   # the global EKF + YAML belong to the hoop/slalom; the plant uses its own WvEKF + the PLANT_* map
        cfg.weave = False
        if cfg.start_id18:
            LOG.info(f'ID18 TEST: place the drone at plant point 11 (C16_13: {PLANT_CORNER_D - 0.09:.2f} m before ID13 and {PLANT_CORNER_D:.2f} m to the RIGHT of the plant axis), nose parallel to the plant axis (towards ID18/ID19, +X). '
                     f'It turns right if ID18 is not in view. Limit {cfg.max_seconds:.0f} s. SPACE = pause | P/L = land | E/K = EMERGENCY | Q = land and quit')
        LOG.info(f'PLANT TEST: place the drone {PLANT_START_DIST:.2f} m from the plant centre, nose towards ID13. stop_R={PLANT_R:.2f}m corner_xy=±{PLANT_CORNER_D:.2f}m, hover {PLANT_HOVER_S:.1f} s, '
                 f'climb to tof {PLANT_TAKEOFF_CM} cm, stop after "{cfg.plant_stop_after}", limit {cfg.max_seconds:.0f} s. '
                 f'SPACE = pause | P/L = land | E/K = EMERGENCY motor cut | Q = land and quit')
    if cfg.start_weave:
        cfg.slalom_ids = tuple(sorted(set(cfg.slalom_ids) | {6, 7, 13}))
    cfg.ekf_x_stop = args.ekf_x_stop
    if not cfg.circle:
        cfg.guide_id = None            # old t21 start: ID3 is a frame marker, not the hoop guide
    elif cfg.guide_id is None:
        ap.error('--start-at hoop needs --guide-id 3 (the hoop approach and the hand-over use ID3)')
    K, dist, size = load_calibration(args.calibration)
    (run_video if args.video else run_live)(args, cfg, K, dist, size, stamp)


if __name__ == '__main__':
    main()