#!/usr/bin/env python3
"""ekf_tello.py - EKF OFFLINE para localizar o Tello no mapa do circulo (TEL310: EKF com correspondencias conhecidas).

NAO voa, NAO mexe em t20/t21. Le os CSVs gravados por pose_global.py (frames + markers) e o mapa YAML.

Estado  x = [x, y, z, psi, b]   (mundo W; psi = yaw da camera, + anti-horario; b = deslocamento entre o yaw do Tello e o mundo)
   telemetria de yaw (horario, graus inteiros) vira MEDIDA:  -tele_yaw = psi - b   (nao derivamos taxa: graus inteiros a 10 Hz dariam ruido enorme)
Predicao (modelo de movimento, controle u = velocidades do Tello):
   v_W = Rz(b) [0.1*vgx, 0.1*vgy*sinal_y]  (vg em dm/s, referencial do Tello = yaw de ligacao)   vz = -0.1*vgz (vgz<0 = subindo; confirmado nos dados)
   x_k = x_{k-1} + v_W dt ;  psi, b = passeio aleatorio ;  G = df/dx, V = df/du  ->  P = G P G' + V S_u V' + Q
Medidas (cada marcador do mapa = correspondencia conhecida, atualizacao sequencial, gating de Mahalanobis):
   (1) posicao do marcador no referencial da camera:  z = tvec,   h(x) = R_wc(psi)' (p_marcador - p_camera)    [3 gl, gate chi2 99% = 11.34]
   (2) yaw implicado pela orientacao do marcador:      z = psi_i,   h(x) = psi                                   [1 gl, gate 6.63]
   (3) telemetria de yaw: h = psi - b ;  (4) altura pelo ToF (opcional): h = z
Saidas: ekf_<run>.csv, resumo no terminal (NIS = consistencia: esperado ~ gl), grafico PNG (--plot).

Uso:
  python3 ekf_tello.py --map aruco_map_circulo_calibrado.yaml --frames logs/pose_X_frames.csv --markers logs/pose_X_markers.csv --plot ekf.png
  ... --blackout 26:36          # remove TODAS as medidas de marcador entre 26 e 36 s e compara com a visao completa
  ... --fit-vg                  # com voo/carregada em movimento: testa sinal e referencial de vgx,vgy
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pose_global import camera_pose_world, load_map, yaw_pitch_deg

CHI2_3, CHI2_1 = 11.34, 6.63


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def R_wc(psi, pitch, roll=0.0):
    """Colunas = eixos da camera (direita, baixo, frente) em W. pitch (+ = olhando para cima); roll = tele_roll (+ = lado direito para baixo)."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    f = np.array([cp * math.cos(psi), cp * math.sin(psi), sp])
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    c, s = math.cos(roll), math.sin(roll)
    return np.column_stack([right * c + down * s, -right * s + down * c, f])


class EKF:
    def __init__(self, x0, P0, a):
        self.x, self.P, self.a = x0.copy(), P0.copy(), a
        self.tilt = (0.0, 0.0)                                 # (pitch, roll) do IMU em graus

    # ---- predicao ----
    def predict(self, u, dt, extra=0.0):
        """u = (vgx, vgy, vgz) do Tello (dm/s)."""
        a, x = self.a, self.x
        g = 0.1 * a.vg_gain
        vx, vy, vz = g * u[0], a.vy_sign * g * u[1], -g * u[2]
        k = 4 if a.vg_frame == 'ground' else 3
        c, s = math.cos(x[k]), math.sin(x[k])
        vwx, vwy = c * vx - s * vy, s * vx + c * vy
        x[0] += vwx * dt
        x[1] += vwy * dt
        x[2] += vz * dt
        G = np.eye(5)
        G[0, k] = (-s * vx - c * vy) * dt
        G[1, k] = (c * vx - s * vy) * dt
        sv = a.q_v0 + a.q_v1 * math.hypot(vx, vy) + extra  # ruido de velocidade (m/s)
        svz = a.q_v0 + a.q_v1 * abs(vz) + extra
        V = np.zeros((5, 3))                                 # dx/du (ruido nas 3 velocidades), u em m/s
        V[0, 0], V[0, 1], V[1, 0], V[1, 1], V[2, 2] = c * dt, -s * dt, s * dt, c * dt, dt
        Su = np.diag([sv ** 2, sv ** 2, svz ** 2])
        Q = np.diag([0, 0, 0, (math.radians(a.q_psi) ** 2) * dt, (math.radians(a.q_b) ** 2) * dt])
        self.P = G @ self.P @ G.T + V @ Su @ V.T + Q + np.diag([a.q_floor ** 2 * dt] * 3 + [0, 0])
        x[3] = wrap(x[3])

    def inflate(self, f):
        """Recuperacao: se so ha medidas rejeitadas, a predicao (vg) provavelmente esta errada -> aumenta P (so posicao)."""
        self.P[:3, :3] *= f
        d = np.sqrt(np.diag(self.P)[:3])
        cap = self.a.p_max
        if np.any(d > cap):                                   # limite: sem isso, rejeicoes repetidas fazem P explodir
            sc = np.minimum(1.0, cap / d)
            self.P[:3, :3] *= np.outer(sc, sc)

    # ---- atualizacao generica (forma de Joseph) ----
    def _update(self, nu, H, R, gate, angle_idx=None):
        S = H @ self.P @ H.T + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if d2 > gate:
            return d2, False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[3] = wrap(self.x[3])
        I = np.eye(5) - K @ H
        self.P = I @ self.P @ I.T + K @ R @ K.T
        return d2, True

    def update_marker_pos(self, p_m, tvec, dist):
        a = self.a
        pitch = math.radians(a.cam_pitch + (self.tilt[0] if a.use_imu_tilt else 0.0))
        roll = math.radians(self.tilt[1]) if a.use_imu_tilt else 0.0
        R = R_wc(self.x[3], pitch, roll)
        pc = self.x[:3]
        h = R.T @ (p_m - pc)
        eps = 1e-5
        dR = (R_wc(self.x[3] + eps, pitch, roll).T - R_wc(self.x[3] - eps, pitch, roll).T) / (2 * eps)
        H = np.zeros((3, 5))
        H[:, :3] = -R.T
        H[:, 3] = dR @ (p_m - pc)
        sl = math.hypot(0.008 + 0.012 * dist, a.map_sigma)
        sd = math.hypot(0.015 + 0.03 * dist, a.map_sigma)
        Rm = np.diag([sl ** 2, sl ** 2, sd ** 2]) * a.r_scale ** 2
        return self._update(tvec - h, H, Rm, CHI2_3)

    def update_marker_yaw(self, psi_i, dist):
        a = self.a
        H = np.zeros((1, 5))
        H[0, 3] = 1.0
        sig = math.radians(a.yaw_sigma0 + a.yaw_sigma1 * dist) * a.r_scale
        return self._update(np.array([wrap(psi_i - self.x[3])]), H, np.array([[sig ** 2]]), CHI2_1)

    def update_tele_yaw(self, tele_yaw_deg):
        H = np.zeros((1, 5))
        H[0, 3], H[0, 4] = 1.0, -1.0
        z = -math.radians(tele_yaw_deg)
        nu = wrap(z - (self.x[3] - self.x[4]))
        return self._update(np.array([nu]), H, np.array([[math.radians(self.a.tele_sigma) ** 2]]), CHI2_1)

    def update_tof(self, tof_m):
        H = np.zeros((1, 5))
        H[0, 2] = 1.0
        return self._update(np.array([tof_m + self.a.tof_offset - self.x[2]]), H, np.array([[self.a.tof_sigma ** 2]]), CHI2_1)


def _idset(v):
    """ids as a set from a tuple/list or a '1,2,3' string ('' = none)."""
    if isinstance(v, str):
        return {int(t) for t in v.replace(' ', '').split(',') if t}
    return {int(t) for t in v}


def marker_obs(row, mp, a):
    """Retorna (p_marcador, tvec, psi_implicito, dist) ou None."""
    i = int(row.id)
    if i not in mp or row.dist > a.max_dist or row.rms_px > a.max_rms:
        return None
    tvec = np.array([row.tx, row.ty, row.tz])
    rvec = np.array([row.rx, row.ry, row.rz])
    _, Rwc = camera_pose_world(mp[i], rvec, tvec)
    return mp[i]['p'], tvec, math.radians(yaw_pitch_deg(Rwc)[0]), float(row.dist), i, Rwc, rvec


def vision_fused(rows, mp, a):
    """Referencia so-visao (media 1/d^2 das poses implicadas), usada como pseudo-verdade."""
    P, Y, W = [], [], []
    for r in rows:
        if r.id not in mp or r.id in _idset(a.fuse_exclude) or r.dist > a.max_dist:
            continue
        p, Rwc = camera_pose_world(mp[int(r.id)], np.array([r.rx, r.ry, r.rz]), np.array([r.tx, r.ty, r.tz]))
        P.append(p)
        Y.append(math.radians(yaw_pitch_deg(Rwc)[0]))
        W.append(1.0 / max(r.dist, 0.2) ** 2)
    if len(P) < a.init_min:
        return None
    W = np.array(W) / np.sum(W)
    return np.sum(np.array(P) * W[:, None], axis=0), math.atan2(np.sum(W * np.sin(Y)), np.sum(W * np.cos(Y)))


class OnlineEKF:
    """O MESMO filtro para offline (CSV) e online (modo sombra no voo): step() por quadro, nada de E/S.
    rows: objetos com id, tx,ty,tz, rx,ry,rz, rms_px, dist.  tele: objeto/dict com yaw, vgx, vgy, vgz, pitch, roll, tof."""

    def __init__(self, mp, a):
        self.mp, self.a = mp, a
        self.ekf, self.tlast, self.tele_last, self.meas_last, self.bad_since = None, None, -9.0, -9.0, None
        self.nis = {'pos': [], 'yaw': [], 'tele': [], 'tof': []}
        self.acc = {'pos': 0, 'yaw': 0, 'tele': 0, 'tof': 0}
        self.rej = {'pos': 0, 'yaw': 0, 'tele': 0, 'tof': 0}

    def step(self, t, rows, tele, blind=False, stale=False):
        a, mp = self.a, self.mp
        g = (lambda k: float(tele.get(k) or 0.0)) if isinstance(tele, dict) else (lambda k: float(getattr(tele, k) or 0.0))
        ref = vision_fused(rows, mp, a)
        if self.ekf is None:
            hoop = [r for r in rows if r.id in _idset(a.init_ids)]
            if len(hoop) < a.init_min or ref is None:
                return None
            p0, psi0 = ref
            b0 = psi0 + math.radians(g('yaw'))                  # -tele = psi - b  ->  b = psi + tele
            s_xy, s_z, s_psi = a.p0_xy, a.p0_z, math.radians(a.p0_psi)
            P0 = np.diag([s_xy ** 2, s_xy ** 2, s_z ** 2, s_psi ** 2, s_psi ** 2])
            P0[3, 4] = P0[4, 3] = s_psi ** 2                      # b = psi + tele_yaw: perfeitamente correlacionados
            self.ekf = EKF(np.array([p0[0], p0[1], p0[2], psi0, b0]), P0, a)
            self.tlast = t
            return None
        ekf = self.ekf
        dt = t - self.tlast
        self.tlast = t
        if 0 < dt < 1.0:   #PREDICTION
            if stale:                                            # telemetria velha: vg desconhecido -> sem velocidade, mais ruido
                ekf.predict((0.0, 0.0, 0.0), dt, extra=0.3)
            else:
                ekf.predict((g('vgx'), g('vgy'), g('vgz')), dt)
        ekf.tilt = (g('pitch'), g('roll'))
        n_acc = 0
        do_meas = (t - self.meas_last) >= 1.0 / a.meas_hz
        if do_meas and rows:
            self.meas_last = t
        if not blind and do_meas: # CORRECTION
            for r in rows:
                o = marker_obs(r, mp, a)
                if o is None:
                    continue
                p_m, tvec, psi_i, dist, i, _, _ = o
                d2, ok = ekf.update_marker_pos(p_m, tvec, dist)
                (self.acc if ok else self.rej)['pos'] += 1
                if ok:
                    self.nis['pos'].append(d2)
                    n_acc += 1
                if not (i == 3 and not a.use_id3_full):          # ID3 (longe): so a posicao, sem yaw
                    d2, ok = ekf.update_marker_yaw(psi_i, dist)
                    (self.acc if ok else self.rej)['yaw'] += 1
                    if ok:
                        self.nis['yaw'].append(d2)
        if do_meas and rows and not blind:
            if n_acc > 0:
                self.bad_since = None
            else:
                self.bad_since = t if self.bad_since is None else self.bad_since
                if t - self.bad_since > a.recover_s and any(marker_obs(r, mp, a) for r in rows):
                    ekf.inflate(a.recover_gain)                  # ha marcadores validos, mas todos rejeitados: o filtro se perdeu
                    self.bad_since = t
        if not stale and t - self.tele_last >= 0.1:
            self.tele_last = t
            d2, ok = ekf.update_tele_yaw(g('yaw'))
            (self.acc if ok else self.rej)['tele'] += 1
            if ok:
                self.nis['tele'].append(d2)
        tof = g('tof')
        if a.use_tof and not blind and not stale and a.tof_min < tof < a.tof_max:
            d2, ok = ekf.update_tof(tof / 100.0)
            (self.acc if ok else self.rej)['tof'] += 1
            if ok:
                self.nis['tof'].append(d2)
        sd = np.sqrt(np.diag(ekf.P))
        return {'t': t, 'x': ekf.x[0], 'y': ekf.x[1], 'z': ekf.x[2], 'psi_deg': math.degrees(ekf.x[3]),
                'sx': sd[0], 'sy': sd[1], 'sz': sd[2], 'spsi_deg': math.degrees(sd[3]), 'n_acc': n_acc,
                'blind': int(blind), 'vx': ref[0][0] if ref else np.nan, 'vy': ref[0][1] if ref else np.nan,
                'vz': ref[0][2] if ref else np.nan, 'vpsi_deg': math.degrees(ref[1]) if ref else np.nan}


def run(a):
    mp = load_map(a.map)
    F = pd.read_csv(a.frames)
    M = pd.read_csv(a.markers)
    groups = {k: list(g.itertuples(index=False)) for k, g in M.groupby('frame')}
    bo = None
    if a.blackout:
        t0, t1 = (float(v) for v in a.blackout.split(':'))
        bo = (t0, t1)
    oe = OnlineEKF(mp, a)
    log = []
    for fr in F.itertuples(index=False):
        tele = {'yaw': fr.tele_yaw, 'vgx': fr.tele_vgx, 'vgy': fr.tele_vgy, 'vgz': fr.tele_vgz,
                'pitch': fr.tele_pitch, 'roll': fr.tele_roll, 'tof': fr.tele_tof}
        blind = bo is not None and bo[0] <= fr.t < bo[1]
        out = oe.step(fr.t, groups.get(fr.frame, []), tele, blind=blind)
        if out is None:
            continue
        out.update({'tele_yaw': fr.tele_yaw, 'tele_vgx': fr.tele_vgx, 'tele_vgy': fr.tele_vgy, 'tele_vgz': fr.tele_vgz})
        log.append(out)
    return pd.DataFrame(log), oe.nis, oe.acc, oe.rej


def report(L, nis, acc, rej, a):
    print(f'\nquadros processados: {len(L)}   t = {L.t.min():.1f} .. {L.t.max():.1f} s')
    print('\nConsistencia (NIS medio; esperado ~ graus de liberdade se R e Q estiverem bem ajustados):')
    for k, g in (('pos', 3), ('yaw', 1), ('tele', 1), ('tof', 1)):
        if nis[k]:
            tot = acc[k] + rej[k]
            m = float(np.mean(nis[k]))
            adv = 'ok' if 0.6 * g < m < 1.6 * g else ('R muito PEQUENO (aumente --r-scale)' if m > g else 'R muito GRANDE (diminua --r-scale)')
            print(f'  {k:5s}: NIS medio {m:5.2f} (esperado {g})   aceitos {acc[k]}/{tot} ({100 * acc[k] / max(tot, 1):.0f}%)   -> {adv}')
    V = L.dropna(subset=['vx'])
    # janela estatica de comparacao
    for name, w in (('todo o periodo com visao', V),):
        e = np.array([V.x - V.x, V.y - V.y])
    ev = V[(V.blind == 0)]
    if len(ev):
        dx, dy, dz = ev.x - ev.vx, ev.y - ev.vy, ev.z - ev.vz
        dp = np.degrees(np.array([wrap(math.radians(u - v)) for u, v in zip(ev.psi_deg, ev.vpsi_deg)]))
        print('\nEKF - visao pura (sem blackout): RMS dx %.1f cm  dy %.1f cm  dz %.1f cm  dpsi %.1f deg' % (
            100 * np.sqrt(np.mean(dx ** 2)), 100 * np.sqrt(np.mean(dy ** 2)), 100 * np.sqrt(np.mean(dz ** 2)), np.sqrt(np.mean(dp ** 2))))
        print('Ruido (desvio-padrao em torno da media local de 0.5 s):')
        for nm, e, v in (('x', ev.x, ev.vx), ('y', ev.y, ev.vy), ('z', ev.z, ev.vz)):
            re = e - e.rolling(25, center=True, min_periods=5).mean()
            rv = v - v.rolling(25, center=True, min_periods=5).mean()
            print(f'   {nm}: visao pura {100 * rv.std():.2f} cm   EKF {100 * re.std():.2f} cm')
    B = V[V.blind == 1]
    if len(B):
        dx, dy, dz = B.x - B.vx, B.y - B.vy, B.z - B.vz
        dp = np.degrees(np.array([wrap(math.radians(u - v)) for u, v in zip(B.psi_deg, B.vpsi_deg)]))
        print(f'\nBLACKOUT {a.blackout} s (sem marcadores; so predicao + telemetria):')
        print('   erro vs visao completa: max |dx| %.1f cm  max |dy| %.1f cm  max |dz| %.1f cm  max |dpsi| %.1f deg' % (
            100 * abs(dx).max(), 100 * abs(dy).max(), 100 * abs(dz).max(), abs(dp).max()))
        b0, b1 = L[L.blind == 1].iloc[0], L[L.blind == 1].iloc[-1]
        print('   incerteza (1 sigma) no inicio -> fim:  x %.1f -> %.1f cm  y %.1f -> %.1f cm  z %.1f -> %.1f cm  psi %.1f -> %.1f deg' % (
            100 * b0.sx, 100 * b1.sx, 100 * b0.sy, 100 * b1.sy, 100 * b0.sz, 100 * b1.sz, b0.spsi_deg, b1.spsi_deg))
    return


def fit_vg(L):
    """Testa referencial (ground/body) e sinal de vgy comparando a velocidade da visao com a do Tello."""
    V = L.dropna(subset=['vx']).copy()
    V = V[V.blind == 0].reset_index(drop=True)
    step = 25                                              # ~0.5 s a 50 Hz
    i0 = np.arange(0, len(V) - step, 5)
    dt = V.t.values[i0 + step] - V.t.values[i0]
    vis = np.column_stack([V.vx.values[i0 + step] - V.vx.values[i0], V.vy.values[i0 + step] - V.vy.values[i0]]) / dt[:, None]
    tele = np.array([V[['tele_vgx', 'tele_vgy']].values[i0 + k].mean(axis=0) for k in range(0, step, 5)]).mean(axis=0) if False else \
        np.mean([V[['tele_vgx', 'tele_vgy']].values[i0 + k] for k in range(0, step, 4)], axis=0) * 0.1
    psi = np.radians(V.vpsi_deg.values[i0])
    b = psi + np.radians(V.tele_yaw.values[i0])             # -tele = psi - b  ->  b = psi + tele  (tele em graus, horario +)
    mov = np.hypot(vis[:, 0], vis[:, 1]) > 0.08
    print(f'\nvelocidade da visao: RMS {np.sqrt(np.mean(vis ** 2)):.3f} m/s ; amostras em movimento (>0.08 m/s): {int(mov.sum())}')
    if mov.sum() < 10:
        print('POUCO MOVIMENTO: teste inconclusivo. Grave uma corrida movendo o drone ~0.3 m/s.')
    res = []
    for frame in ('ground', 'body'):
        for sy in (1, -1):
            ang = b if frame == 'ground' else psi
            vwx = np.cos(ang) * tele[:, 0] - np.sin(ang) * sy * tele[:, 1]
            vwy = np.sin(ang) * tele[:, 0] + np.cos(ang) * sy * tele[:, 1]
            rm = np.sqrt(np.mean(((vwx - vis[:, 0]) ** 2 + (vwy - vis[:, 1]) ** 2)[mov]))
            num = np.sum((vwx * vis[:, 0] + vwy * vis[:, 1])[mov])
            gain = num / max(np.sum((vwx ** 2 + vwy ** 2)[mov]), 1e-9)
            res.append((rm, frame, sy, gain))
    for rm, fm, sy, g in sorted(res):
        print(f'   vg_frame={fm:6s} vy_sign={sy:+d}: erro RMS {rm:.3f} m/s   ganho otimo {g:.2f}')
    rm, fm, sy, g = sorted(res)[0]
    print(f'==> melhor: --vg-frame {fm} --vy-sign {sy}   (ganho {g:.2f}; ~1 = dm/s correto)')


def plot(L, a, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    names = [('x', 'vx', 'sx', 'x (m)', 1), ('y', 'vy', 'sy', 'y (m)', 1), ('z', 'vz', 'sz', 'z (m)', 1),
             ('psi_deg', 'vpsi_deg', 'spsi_deg', 'yaw (graus)', 1)]
    for k, (e, v, s, lab, _) in enumerate(names):
        ax[k].plot(L.t, L[v], '.', ms=2.5, color='#bbbbbb', label='visão pura (média dos marcadores)')
        ax[k].plot(L.t, L[e], '-', lw=1.4, color='#1f5fbf', label='EKF')
        ax[k].fill_between(L.t, L[e] - 2 * L[s], L[e] + 2 * L[s], color='#1f5fbf', alpha=0.18, label='EKF ±2σ')
        ax[k].set_ylabel(lab)
        ax[k].grid(alpha=0.25)
        if a.blackout:
            t0, t1 = (float(u) for u in a.blackout.split(':'))
            ax[k].axvspan(t0, t1, color='#f4a261', alpha=0.2)
    ax[0].legend(loc='upper right', ncol=3, fontsize=8)
    ax[4].plot(L.t, L.n_acc, drawstyle='steps-post', color='#2a9d8f')
    ax[4].set_ylabel('medidas aceitas\npor quadro')
    ax[4].set_xlabel('t (s)')
    ax[4].grid(alpha=0.25)
    if a.blackout:
        ax[0].set_title(f'EKF do Tello — blackout dos marcadores em {a.blackout} s (faixa laranja)')
    else:
        ax[0].set_title('EKF do Tello — posição global no mapa do círculo')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f'grafico: {path}')


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map', default='aruco_map_circulo_calibrado.yaml')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--markers', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--plot', default=None)
    ap.add_argument('--blackout', default=None, help='t0:t1 em segundos: ignora marcadores (e ToF) nesse intervalo')
    ap.add_argument('--fit-vg', action='store_true')
    ap.add_argument('--cam-pitch', dest='cam_pitch', type=float, default=-10.5, help='pitch da camera (graus, - = olhando para baixo)')
    ap.add_argument('--vg-frame', dest='vg_frame', choices=['ground', 'body'], default='ground')  # confirmado nos dados
    ap.add_argument('--vy-sign', dest='vy_sign', type=int, choices=[1, -1], default=-1)  # vgy positivo = DIREITA (confirmado)
    ap.add_argument('--vg-gain', dest='vg_gain', type=float, default=1.2, help='ganho de vg: em voo real o Tello le ~20%% a menos que o movimento medido pelos marcadores (2 voos)')
    ap.add_argument('--r-scale', dest='r_scale', type=float, default=0.6)
    ap.add_argument('--map-sigma', dest='map_sigma', type=float, default=0.012, help='incerteza do mapa (m), soma ao ruido da medida')
    ap.add_argument('--yaw-sigma0', dest='yaw_sigma0', type=float, default=1.0)
    ap.add_argument('--yaw-sigma1', dest='yaw_sigma1', type=float, default=1.0, help='graus por metro de distancia')
    ap.add_argument('--tele-sigma', dest='tele_sigma', type=float, default=0.35)
    ap.add_argument('--q-v0', dest='q_v0', type=float, default=0.04)
    ap.add_argument('--q-v1', dest='q_v1', type=float, default=0.15)
    ap.add_argument('--q-psi', dest='q_psi', type=float, default=2.0, help='graus/sqrt(s)')
    ap.add_argument('--q-b', dest='q_b', type=float, default=0.2, help='graus/sqrt(s)')
    ap.add_argument('--q-floor', dest='q_floor', type=float, default=0.01, help='m/sqrt(s)')
    ap.add_argument('--p0-xy', dest='p0_xy', type=float, default=0.30)
    ap.add_argument('--p0-z', dest='p0_z', type=float, default=0.20)
    ap.add_argument('--p0-psi', dest='p0_psi', type=float, default=10.0)
    ap.add_argument('--max-dist', dest='max_dist', type=float, default=3.0)
    ap.add_argument('--max-rms', dest='max_rms', type=float, default=2.5)
    ap.add_argument('--use-id3-full', dest='use_id3_full', action='store_true', help='usa tambem o yaw do ID 3')
    ap.add_argument('--meas-hz', dest='meas_hz', type=float, default=10.0, help='taxa maxima de atualizacao por marcadores (quadros a 50 Hz sao muito correlacionados)')
    ap.add_argument('--recover-s', dest='recover_s', type=float, default=0.4, help='s com so rejeicoes antes de inflar P')
    ap.add_argument('--recover-gain', dest='recover_gain', type=float, default=6.0)
    ap.add_argument('--p-max', dest='p_max', type=float, default=1.0, help='desvio maximo (m) de posicao apos inflar P')
    ap.add_argument('--no-imu-tilt', dest='use_imu_tilt', action='store_false', help='ignora pitch/roll do IMU no modelo da camera')
    ap.add_argument('--no-tof', dest='use_tof', action='store_false')
    ap.add_argument('--tof-sigma', dest='tof_sigma', type=float, default=0.04)
    ap.add_argument('--tof-offset', dest='tof_offset', type=float, default=0.0)
    ap.add_argument('--tof-min', dest='tof_min', type=float, default=25.0)
    ap.add_argument('--tof-max', dest='tof_max', type=float, default=300.0)
    ap.add_argument('--init-ids', dest='init_ids', default='0,1,10,12', help='ids que podem inicializar o filtro (aro: 0,1,10,12; slalom: 2,3,4,5,17)')
    ap.add_argument('--init-min', dest='init_min', type=int, default=3, help='minimo de marcadores para inicializar e para a referencia so-visao')
    ap.add_argument('--fuse-exclude', dest='fuse_exclude', default='3', help='ids fora da referencia so-visao (aro: 3 por ser longe; slalom: vazio)')
    return ap


def default_args(**kw):
    a = build_parser().parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def main():
    ap = build_parser()
    a = ap.parse_args()
    if not a.frames or not a.markers:
        ap.error('--frames e --markers sao obrigatorios')
    L, nis, acc, rej = run(a)
    report(L, nis, acc, rej, a)
    out = a.out or ('ekf_' + Path(a.frames).stem.replace('_frames', '') + '.csv')
    L.to_csv(out, index=False)
    print(f'csv: {out}')
    if a.fit_vg:
        fit_vg(L)
    if a.plot:
        plot(L, a, a.plot)


if __name__ == '__main__':
    main()