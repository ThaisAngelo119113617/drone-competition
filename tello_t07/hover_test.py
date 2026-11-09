"""T0.7: one short physical hover, then observe landing telemetry.
No retry of takeoff/land. Flight remains unverified without visual inspection.
"""
import csv
import time
from datetime import datetime
from pathlib import Path
from safety import Safety
from tello_io import TelloIO, FlightState

HOVER_SECONDS = 5
OBSERVE_SECONDS = 20
MIN_BATTERY = 40


def sample(io, writer, phase):
    now = time.monotonic()
    try:
        s = io.t.get_current_state() or {}
    except Exception as exc:
        print('[TELEMETRY] indisponivel:', exc, flush=True)
        s = {}
    row = [round(now, 3), phase, io.state.value,
           *[s.get(k, '') for k in ('bat', 'h', 'tof', 'vgz', 'time', 'yaw')]]
    writer.writerow(row)
    print('[STATE]', phase, 'h=', s.get('h'), 'tof=', s.get('tof'),
          'vgz=', s.get('vgz'), 'bat=', s.get('bat'), flush=True)
    return s


def main():
    log = Path('t07_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')
    io = TelloIO(log_path=None)
    safety = Safety(io.t, max_height_cm=200, min_battery=20,
                    stall_stop_s=1, stall_land_s=3, max_mission_s=15)
    started = False
    airborne_possible = False
    io.emergency_event = safety.emergency_requested
    with log.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['monotonic_s', 'phase', 'software_state', 'bat', 'h_cm',
                         'tof_cm', 'vgz', 'flight_time_s', 'yaw'])
        try:
            io.connect(video=False)
            battery = io.t.get_battery()
            preflight = sample(io, writer, 'PREFLIGHT')
            # The SDK state is cached: plausibility checks are NOT freshness proof.
            h, tof = preflight.get('h'), preflight.get('tof')
            if not isinstance(h, (int, float)) or not (-10 <= h <= 20):
                print('[ABORT] altura de preflight ausente/incompativel:', h)
                return
            if not isinstance(tof, (int, float)) or not (0 <= tof <= 300):
                print('[ABORT] ToF de preflight ausente/incompativel:', tof)
                return
            if battery < MIN_BATTERY:
                print('[ABORT] bateria insuficiente')
                return
            if input('Area livre, protecoes e drone no solo? Digite VOAR: ').strip() != 'VOAR':
                return
            safety.start()
            started = True
            if safety.land_requested.is_set() or safety.emergency_requested.is_set():
                return
            airborne_possible = True  # once a takeoff is attempted, physical state may be uncertain
            try:
                io.takeoff_once()
            except Exception as exc:
                print('[TAKEOFF INCERTO]', exc, 'Nao repetir takeoff.', flush=True)
            sample(io, writer, 'AFTER_TAKEOFF')
            if io.state == FlightState.FLYING and not safety.land_requested.is_set() and not safety.emergency_requested.is_set():
                safety.arm_watchdog()
                end = time.monotonic() + HOVER_SECONDS
                next_sample = 0
                while time.monotonic() < end and not safety.land_requested.is_set() and not safety.emergency_requested.is_set():
                    safety.heartbeat()
                    if safety.consume_hover_request():
                        print('[FLIGHT] SPACE recebido: hover zero')
                    io.hover()
                    if time.monotonic() >= next_sample:
                        sample(io, writer, 'HOVER')
                        next_sample = time.monotonic() + 0.5
                    time.sleep(0.05)
                safety.request_land('(fim do hover)')
            elif not safety.emergency_requested.is_set():
                safety.request_land('(takeoff incerto ou pouso solicitado)')
        except KeyboardInterrupt:
            safety.request_land('(Ctrl+C)')
        except Exception as exc:
            print('[ERRO]', repr(exc), flush=True)
            safety.request_land('(erro)')
        finally:
            # Keep keyboard listener active during blocking land for emergency requests.
            try:
                if safety.emergency_requested.is_set():
                    io.mark_emergency()
                elif airborne_possible and not io.land_latched:
                    io.land_once()  # ONE command, never retry automatically
                if airborne_possible:
                    print('[OBSERVACAO] Telemetria passiva. Verifique fisicamente o drone.', flush=True)
                    end = time.monotonic() + OBSERVE_SECONDS
                    while time.monotonic() < end:
                        if safety.emergency_requested.is_set():
                            io.mark_emergency()
                        sample(io, writer, 'POST_LAND_MONITOR')
                        f.flush()
                        time.sleep(0.5)
            except KeyboardInterrupt:
                print('[INTERRUPCAO] Monitoramento interrompido; estado fisico incerto.', flush=True)
            except Exception as exc:
                print('[ERRO FINALIZACAO]', repr(exc), flush=True)
            finally:
                if started:
                    safety.stop()  # main thread restores terminal, including after Ctrl+C
                try:
                    if safety.emergency_requested.is_set():
                        io.mark_emergency()
                    io.close()
                except Exception as exc:
                    print('[ERRO CLOSE]', repr(exc), flush=True)
                print('[FINAL] log=', log, 'state=', io.state.value,
                      'land_ack=', io.land_acknowledged, flush=True)
                print('[ATENCAO] Verifique visualmente o pouso antes de encerrar a supervisao.', flush=True)


if __name__ == '__main__':
    main()
