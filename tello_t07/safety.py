"""Safety requests. Only main thread sends ordinary flight commands.

K sends motor-cut emergency and can cause a fall. Never use it as a normal landing.
Watchdog requests cannot interrupt a blocked SDK call.
"""
import select
import sys
import termios
import threading
import time
import tty


class Safety:
    def __init__(self, tello, max_height_cm=200, min_battery=20,
                 stall_stop_s=1.0, stall_land_s=3.0, max_mission_s=30):
        self.t = tello
        self.max_height = max_height_cm
        self.min_battery = min_battery
        self.stall_stop_s = stall_stop_s
        self.stall_land_s = stall_land_s
        self.max_mission_s = max_mission_s
        self.abort = threading.Event()
        self.land_requested = threading.Event()
        self.hover_requested = threading.Event()
        self.emergency_requested = threading.Event()
        self._stop = threading.Event()
        self._armed = threading.Event()
        self._lock = threading.Lock()
        self._beat = time.monotonic()
        self._t0 = None
        self._threads = []
        self.land_reason = ''
        self._terminal_fd = None
        self._terminal_original = None

    def request_land(self, why=''):
        with self._lock:
            if self.emergency_requested.is_set() or self.land_requested.is_set():
                return
            self.land_reason = why
            self.land_requested.set()
            self.abort.set()
        print(f'\n[SAFETY] POUSO SOLICITADO {why}', flush=True)

    def request_hover(self):
        if not self.land_requested.is_set() and not self.emergency_requested.is_set():
            self.hover_requested.set()
            print('\n[SAFETY] HOVER SOLICITADO (SPACE)', flush=True)

    def emergency(self):
        with self._lock:
            if self.emergency_requested.is_set():
                return
            self.emergency_requested.set()
            self.abort.set()
        print('\n[EMERGENCY] CORTE DE MOTORES SOLICITADO! Pode causar queda.', flush=True)
        # Emergency is the ONLY concurrent direct command; cannot guarantee delivery.
        for _ in range(3):
            try:
                self.t.send_command_without_return('emergency')
            except Exception as exc:
                print('[EMERGENCY] falha no envio:', exc, flush=True)
            time.sleep(0.05)

    def handle_key(self, ch):
        ch = (ch or '').lower()
        if ch == ' ':
            self.request_hover()
        elif ch in ('l', 'q'):
            self.request_land('(tecla L/Q)')
        elif ch == 'k':
            self.emergency()

    def should_stop(self):
        return self.abort.is_set() or self.land_requested.is_set() or self.emergency_requested.is_set()

    def consume_hover_request(self):
        if self.hover_requested.is_set():
            self.hover_requested.clear()
            return True
        return False

    def arm_watchdog(self):
        self._beat = time.monotonic()
        self._t0 = self._beat
        self._armed.set()

    def heartbeat(self):
        self._beat = time.monotonic()

    def _keys(self):
        if self._terminal_fd is None:
            return
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._terminal_fd], [], [], 0.1)
                if ready:
                    self.handle_key(sys.stdin.read(1))
            except (OSError, ValueError) as exc:
                print('[SAFETY] teclado indisponivel:', exc, flush=True)
                return

    def _watchdog(self):
        last_check = 0
        while not self._stop.wait(0.1):
            if not self._armed.is_set() or self.abort.is_set():
                continue
            now = time.monotonic()
            stall = now - self._beat
            if stall > self.stall_land_s:
                self.request_land(f'(watchdog {stall:.1f}s; aguardando loop principal)')
            elif stall > self.stall_stop_s:
                self.hover_requested.set()
            if self._t0 is not None and now - self._t0 > self.max_mission_s:
                self.request_land('(tempo máximo)')
            if now - last_check > 2:
                last_check = now
                try:
                    state = self.t.get_current_state() or {}
                    bat, h = state.get('bat'), state.get('h')
                    if isinstance(bat, (int, float)) and bat < self.min_battery:
                        self.request_land(f'(bateria {bat}%)')
                    if isinstance(h, (int, float)) and h > self.max_height:
                        self.request_land(f'(altura {h}cm)')
                except Exception as exc:
                    print('[SAFETY] telemetria indisponível:', exc)

    def start(self):
        print('[SAFETY] SPACE=HOVER | L/Q=POUSAR | K=EMERGENCIA (corte de motores)')
        self._stop.clear()
        if sys.stdin.isatty():
            self._terminal_fd = sys.stdin.fileno()
            self._terminal_original = termios.tcgetattr(self._terminal_fd)
            try:
                tty.setcbreak(self._terminal_fd)
            except BaseException:
                self._restore_terminal()
                raise
        try:
            for fn in (self._keys, self._watchdog):
                th = threading.Thread(target=fn, daemon=True)
                th.start()
                self._threads.append(th)
        except BaseException:
            self.stop()
            raise

    def _restore_terminal(self):
        if self._terminal_fd is not None and self._terminal_original is not None:
            try:
                termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._terminal_original)
            except Exception as exc:
                print('[SAFETY] falha restaurando terminal:', exc, flush=True)
        self._terminal_fd = None
        self._terminal_original = None

    def stop(self):
        self._stop.set()
        self._restore_terminal()  # main thread restores even if keyboard daemon is interrupted
        for th in self._threads:
            if th is not threading.current_thread():
                th.join(timeout=1)
        self._threads.clear()
