"""Tello I/O: one owner for ordinary flight commands, no automatic takeoff retry.

IMPORTANT: SDK 'ok' means command acknowledged, not physical touchdown.
Do not use Tello.takeoff()/land() here (some versions retry).
"""
import csv
import time
from enum import Enum


class FlightState(Enum):
    GROUNDED = 'GROUNDED'
    TAKING_OFF = 'TAKING_OFF'
    FLYING = 'FLYING'
    LANDING = 'LANDING'
    UNKNOWN = 'UNKNOWN'
    EMERGENCY = 'EMERGENCY'


class TelloIO:
    def __init__(self, log_path=None, tello=None):
        if tello is None:
            from djitellopy import Tello
            tello = Tello()
        self.t = tello
        self.state = FlightState.GROUNDED  # preflight assumption only
        self.takeoff_attempted = False
        self.land_latched = False
        self.land_acknowledged = False
        self.frame_reader = None
        self.stream_started = False
        self._t0 = time.monotonic()
        self._log = open(log_path, 'w', newline='') if log_path else None
        self._writer = csv.writer(self._log) if self._log else None
        if self._writer:
            self._writer.writerow(['t', 'state', 'bat', 'h_cm', 'tof_cm', 'yaw',
                                   'vgx', 'vgy', 'vgz', 'lr', 'fb', 'ud', 'yaw_cmd'])

    def connect(self, video=True):
        self.t.connect()
        bat = self.t.get_battery()
        print(f'[TELLO] bateria: {bat}%')
        if video:
            self.t.streamon()
            self.stream_started = True
            self.frame_reader = self.t.get_frame_read()
        return self.t

    def frame(self):
        return self.frame_reader.frame if self.frame_reader else None

    def _one_shot(self, cmd, timeout=8):
        # This is the raw djitellopy API. Verify behavior for your installed version.
        # Never substitute send_control_command (may retry).
        return self.t.send_command_with_return(cmd, timeout=timeout)

    def takeoff_once(self):
        if self.takeoff_attempted or self.land_latched or self.state != FlightState.GROUNDED:
            raise RuntimeError('Takeoff bloqueado: já tentado, pouso solicitado ou estado incompatível')
        self.takeoff_attempted = True
        self.state = FlightState.TAKING_OFF
        print('[FLIGHT] takeoff único (sem retry no nosso código)', flush=True)
        try:
            response = self._one_shot('takeoff', timeout=8)
            if str(response).strip().lower() != 'ok':
                raise RuntimeError(f'takeoff sem confirmação: {response!r}')
            self.state = FlightState.FLYING
        except BaseException:
            self.state = FlightState.UNKNOWN
            raise

    def rc(self, lr=0, fb=0, ud=0, yaw=0):
        if self.land_latched or self.state != FlightState.FLYING:
            return
        vals = [max(-100, min(100, int(v))) for v in (lr, fb, ud, yaw)]
        self.t.send_rc_control(*vals)
        self._log_row(vals)

    def hover(self):
        self.rc(0, 0, 0, 0)

    def land_once(self):
        if self.land_latched or self.state in (FlightState.GROUNDED, FlightState.EMERGENCY):
            return
        self.land_latched = True
        self.land_acknowledged = False
        self.state = FlightState.LANDING
        print('[FLIGHT] land único (sem retry no nosso código)', flush=True)
        # No RC packet immediately before land: avoid an extra concurrent UDP
        # command at the transition. A land command itself is the priority.
        try:
            response = self._one_shot('land', timeout=8)
            if str(response).strip().lower() == 'ok':
                self.land_acknowledged = True
                # 'ok' does not prove physical touchdown.
                self.state = FlightState.UNKNOWN
                print('[FLIGHT] land aceito; toque no solo NÃO verificado', flush=True)
            else:
                self.state = FlightState.UNKNOWN
                print(f'[FLIGHT] pouso NÃO confirmado: {response!r}', flush=True)
        except BaseException as exc:
            self.state = FlightState.UNKNOWN
            print(f'[FLIGHT] pouso incerto: {type(exc).__name__}: {exc}', flush=True)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise

    def mark_emergency(self):
        self.land_latched = True
        self.state = FlightState.EMERGENCY

    def _log_row(self, cmd):
        if not self._writer:
            return
        s = self.t.get_current_state() or {}
        self._writer.writerow([round(time.monotonic() - self._t0, 3), self.state.value,
                               *[s.get(k) for k in ('bat', 'h', 'tof', 'yaw', 'vgx', 'vgy', 'vgz')], *cmd])

    def close(self):
        # Cleanup is NOT a landing or flight-state confirmation.
        if self.state in (FlightState.UNKNOWN, FlightState.LANDING, FlightState.TAKING_OFF, FlightState.FLYING):
            print('[ALERTA] Estado físico NÃO confirmado. Encerrar conexão não garante pouso.', flush=True)
        try:
            if self.stream_started:
                self.t.streamoff()
        except Exception as exc:
            print('[TELLO] streamoff:', exc)
        finally:
            try:
                self.t.end()
            except Exception as exc:
                print('[TELLO] end:', exc)
            if self._log:
                self._log.close()
