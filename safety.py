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
    def __init__(self, tello, max_height_cm=200, min_battery=20, stall_stop_s=1.0, stall_land_s=3.0, max_mission_s=30):
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

    def arm_watchdog(self):
        self._beat = time.monotonic()
        self._t0 = self._beat
        self._armed.set()

    def heartbeat(self):
        self._beat = time.monotonic()

    def _keys(self):
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    self.handle_key(sys.stdin.read(1))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

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
        print('[SAFETY] Terminal ou janela: SPACE=HOVER | L/Q=POUSAR | K=EMERGÊNCIA (cai)')
        for fn in (self._keys, self._watchdog):
            th = threading.Thread(target=fn, daemon=True)
            th.start()
            self._threads.append(th)

    def stop(self):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=1)
