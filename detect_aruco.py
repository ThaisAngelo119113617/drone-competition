#!/usr/bin/env python3
"""T0.8: Tello camera ArUco detection (ground-only, no flight commands)."""
import argparse
import logging
import time
from collections import Counter

import cv2
from djitellopy import Tello


def args_parse():
    p = argparse.ArgumentParser(description='Detecta ArUcos usando a camera do Tello, sem voar')
    p.add_argument('--dict', default='DICT_4X4_50', help='Ex.: DICT_4X4_50, DICT_5X5_100')
    p.add_argument('--log-interval', type=float, default=1.0, help='Segundos entre mensagens de deteccao')
    p.add_argument('--no-flip', action='store_true', help='Mantem orientacao original (padrao tambem nao espelha)')
    return p.parse_args()


def create_detector(name):
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('cv2.aruco nao encontrado. Instale opencv-contrib-python no ambiente ativo.')
    if not hasattr(cv2.aruco, name):
        names = [x for x in dir(cv2.aruco) if x.startswith('DICT_')]
        raise ValueError(f'Dicionario {name!r} desconhecido. Exemplos: {names[:12]}')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    if hasattr(cv2.aruco, 'ArucoDetector'):
        return lambda img: cv2.aruco.ArucoDetector(dictionary).detectMarkers(img)
    return lambda img: cv2.aruco.detectMarkers(img, dictionary)


def main():
    opts = args_parse()
    detect = create_detector(opts.dict)
    logging.getLogger('djitellopy').setLevel(logging.WARNING)
    tello = Tello()
    connected = False
    stream_on = False
    last_log = 0.0
    last_frame_time = time.monotonic()
    prev = None
    fps = 0.0
    seen = Counter()
    print(f'[CONFIG] Dicionario: {opts.dict}')
    print('[SEGURANCA] Apenas camera: nenhum comando de voo sera enviado.')
    print('[TECLAS] Q ou ESC: sair | S: salvar frame')
    try:
        tello.connect()
        connected = True
        print(f'[TELLO] Bateria: {tello.get_battery()}%')
        # Reiniciar stream evita herdar estado de video de execucoes anteriores.
        try:
            tello.streamoff()
        except Exception as exc:
            print(f'[AVISO] streamoff inicial: {exc}')
        tello.streamon()
        stream_on = True
        reader = tello.get_frame_read()
        time.sleep(1.0)
        while True:
            frame = reader.frame
            if frame is None or frame.size == 0:
                if time.monotonic() - last_frame_time > 8:
                    print('[ERRO] Sem frames validos por 8 s; encerrando.')
                    break
                cv2.waitKey(30)
                continue
            last_frame_time = time.monotonic()
            frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect(gray)
            now = time.monotonic()
            if prev is not None:
                instant = 1 / max(now - prev, 1e-6)
                fps = instant if fps == 0 else 0.85 * fps + 0.15 * instant
            prev = now
            details = []
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for marker_id, c in zip(ids.flatten(), corners):
                    pts = c.reshape(4, 2)
                    cx, cy = pts.mean(axis=0)
                    x, y = int(round(cx)), int(round(cy))
                    marker_id = int(marker_id)
                    seen[marker_id] += 1
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f'ID {marker_id}', (max(0, x - 30), max(20, y - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    details.append(f'ID={marker_id} centro=({x},{y})')
            if now - last_log >= opts.log_interval:
                print(f'[ARUCO] {len(details)} marcador(es): ' + (', '.join(details) if details else 'nenhum'), flush=True)
                last_log = now
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (20, 20, 20), -1)
            cv2.putText(frame, f'{opts.dict} | ArUcos: {len(details)} | FPS: {fps:.1f} | Q sair / S salvar',
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Tello T0.8 - ArUco (sem voo)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('s'), ord('S')):
                filename = f'aruco_frame_{time.strftime("%Y%m%d_%H%M%S")}.png'
                if cv2.imwrite(filename, frame):
                    print(f'[SALVO] {filename}')
        print('[RESUMO] IDs vistos (frames com deteccao):', dict(sorted(seen.items())))
    except KeyboardInterrupt:
        print('\n[PARADA] Ctrl+C')
    finally:
        cv2.destroyAllWindows()
        if stream_on:
            try:
                tello.streamoff()
            except Exception as exc:
                print(f'[AVISO] Falha ao encerrar video: {exc}')
        if connected:
            try:
                tello.end()
            except Exception as exc:
                print(f'[AVISO] Falha ao encerrar conexao: {exc}')
        print('[FIM] Teste encerrado.')


if __name__ == '__main__':
    main()
