#!/usr/bin/env python3
"""T0.9: Ground-only Tello SDK UDP diagnostic. NEVER sends takeoff/land/rc/emergency."""
import argparse
import csv
import datetime as dt
import json
import select
import socket
import time
from pathlib import Path

DRONE = ('192.168.10.1', 8889)
QUERIES = ('battery?', 'speed?', 'time?')

def now():
    return dt.datetime.now().astimezone().isoformat(timespec='milliseconds')

def log(writer, fp, phase, seq, event, cmd='', payload='', elapsed_ms='', note=''):
    writer.writerow([now(), f'{time.monotonic():.6f}', phase, seq, event, cmd, payload, elapsed_ms, note])
    fp.flush()

def drain(sock, writer, fp, phase, seq, limit_s=0.05):
    count = 0
    until = time.monotonic() + limit_s
    while time.monotonic() < until:
        ready, _, _ = select.select([sock], [], [], max(0, until-time.monotonic()))
        if not ready: break
        data, peer = sock.recvfrom(4096)
        count += 1
        log(writer, fp, phase, seq, 'UNSOLICITED_RESPONSE', payload=data.decode('utf-8','replace'), note=str(peer))
    return count

def main():
    p = argparse.ArgumentParser(description='Ground-only UDP diagnostic; no flight commands')
    p.add_argument('--per-phase', type=int, default=30, help='queries per phase (default 30)')
    p.add_argument('--timeout', type=float, default=2.5)
    p.add_argument('--output', default='t09_ground_events.csv')
    args = p.parse_args()
    if args.per_phase < 1 or args.timeout <= 0: p.error('Invalid count or timeout')
    print('T0.9 GROUND ONLY — sends command, battery?, speed?, time?; NO flight commands.')
    print('Drone MUST be on the ground. Close the Tello app and other scripts first.')
    if input('Type SOLO to continue: ').strip() != 'SOLO':
        print('Cancelled.'); return
    out = Path(args.output)
    with out.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        writer.writerow(['wall_time','monotonic_s','phase','seq','event','command','payload','rtt_ms','note'])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', 8889))
            sock.setblocking(False)
            def exchange(phase, seq, cmd):
                old = drain(sock, writer, fp, phase, seq)
                t0 = time.monotonic()
                sock.sendto(cmd.encode('ascii'), DRONE)
                log(writer, fp, phase, seq, 'SEND', cmd=cmd, note=f'drained={old}')
                until = t0 + args.timeout
                while time.monotonic() < until:
                    ready, _, _ = select.select([sock], [], [], max(0,until-time.monotonic()))
                    if ready:
                        data, peer = sock.recvfrom(4096)
                        reply = data.decode('utf-8','replace').strip()
                        rtt = round((time.monotonic()-t0)*1000, 2)
                        if peer[0] != DRONE[0]:
                            log(writer, fp, phase, seq, 'FOREIGN_RESPONSE', cmd, reply, rtt, str(peer)); continue
                        log(writer, fp, phase, seq, 'RESPONSE', cmd, reply, rtt)
                        print(f'[{phase} #{seq}] {cmd} -> {reply!r} ({rtt} ms)')
                        return reply
                log(writer, fp, phase, seq, 'TIMEOUT', cmd)
                print(f'[{phase} #{seq}] {cmd} -> TIMEOUT')
                return None
            if exchange('init',0,'command') != 'ok':
                print('SDK initialization failed. No more commands sent.'); return
            phases = [('paced_500ms', .5), ('paced_200ms', .2), ('paced_50ms', .05)]
            for phase, delay in phases:
                print(f'\n{phase}: {args.per_phase} sequential queries; pause {delay}s')
                for i in range(args.per_phase):
                    cmd = QUERIES[i % len(QUERIES)]
                    exchange(phase, i+1, cmd)
                    time.sleep(delay)
                drain(sock, writer, fp, phase, args.per_phase+1, .3)
    print(f'\nSaved {out.resolve()}')
    print('WARNING: UDP has no command IDs. A response may be stale despite queue draining.')

if __name__ == '__main__':
    try: main()
    except KeyboardInterrupt: print('\nInterrupted safely; no flight commands were sent.')
    except OSError as exc: print(f'Network/socket error: {exc}; ensure no other Tello script is running.')
