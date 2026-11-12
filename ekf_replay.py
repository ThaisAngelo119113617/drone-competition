"""Run the EKF (Lab 3, Table 7.2: EKF localization with known correspondences)
OFFLINE over a hand-carry log from ekf_logger.py and plot the result.



Output:
    - error at every checkpoint (EKF estimate vs. taped position) in the terminal
    - <log>_ekf.png  : map, checkpoints, dead-reckoning path, EKF path
    - <log>_ekf.csv  : the EKF estimate (t, x, y, theta, sigma_x, sigma_y, sigma_theta)
"""
import argparse
from pathlib import Path

import numpy as np


# Taped checkpoints, LOCAL frame, metres: X forward, Y left(+) / right(-)
CHECKPOINTS = {
    1: (0.00,  0.00),
    2: (0.05,  0.00),
    3: (0.45, -0.40),
    4: (1.15, -0.40),
    7: (1.55,  0.00),
    5: (1.95,  0.40),
    6: (2.70,  0.40),
    8: (2.86,  0.40),
}

# ---------------------------------------------------------------- EKF tuning
Q_XY = 0.08           # process noise, m/sqrt(s): how much we distrust the Tello speeds
Q_TH = np.radians(3)   # process noise for heading, rad/sqrt(s)
SIGMA_R0 = 0.03        # range noise = SIGMA_R0 + SIGMA_R1 * r   (measured: ~1 cm at 0.6 m)
SIGMA_R1 = 0.03
SIGMA_PHI = np.radians(1.5)
GATE = 9.21            # chi-square, 2 dof, 99 %  (Lab 4 idea: reject outliers)
MAX_RANGE_M = 3.0      # ignore detections further away than this (far-away look-alikes)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    """State mu = (x, y, theta): position on the floor and heading (CCW, rad)."""

    def __init__(self, mu0, Sigma0):
        self.mu = np.array(mu0, float)
        self.Sigma = np.array(Sigma0, float)

    def predict(self, v_fwd, v_left, w, dt):
        """Lab 3 lines 2-7, holonomic version (the drone can also fly sideways).
            x' = x + (v_fwd cos th - v_left sin th) dt
            y' = y + (v_fwd sin th + v_left cos th) dt
            th' = th + w dt
        """
        x, y, th = self.mu
        c, s = np.cos(th), np.sin(th)
        self.mu = np.array([x + (v_fwd * c - v_left * s) * dt,
                            y + (v_fwd * s + v_left * c) * dt,
                            wrap(th + w * dt)])
        G = np.array([[1, 0, -(v_fwd * s + v_left * c) * dt],
                      [0, 1,  (v_fwd * c - v_left * s) * dt],
                      [0, 0, 1]])
        R = np.diag([Q_XY**2 * dt, Q_XY**2 * dt, Q_TH**2 * dt])   # random-walk noise
        self.Sigma = G @ self.Sigma @ G.T + R

    def update(self, cam_x, cam_z, landmark):
        """Lab 3 lines 9-17. Measurement z = (range, bearing) from the PnP result:
            r = sqrt(x^2 + z^2),  phi = atan2(-x, z)   (phi > 0 = marker to the LEFT)
        Returns True if used, False if rejected by the gate."""
        r_meas = np.hypot(cam_x, cam_z)
        z = np.array([r_meas, np.arctan2(-cam_x, cam_z)])
        mx, my = landmark
        dx, dy = mx - self.mu[0], my - self.mu[1]
        q = dx * dx + dy * dy
        z_hat = np.array([np.sqrt(q), wrap(np.arctan2(dy, dx) - self.mu[2])])
        H = np.array([[-dx / np.sqrt(q), -dy / np.sqrt(q), 0],
                      [dy / q, -dx / q, -1]])
        Q = np.diag([(SIGMA_R0 + SIGMA_R1 * r_meas)**2, SIGMA_PHI**2])
        S = H @ self.Sigma @ H.T + Q
        nu = z - z_hat
        nu[1] = wrap(nu[1])
        if nu @ np.linalg.solve(S, nu) > GATE:      # Mahalanobis gate
            return False
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ nu
        self.mu[2] = wrap(self.mu[2])
        self.Sigma = (np.eye(3) - K @ H) @ self.Sigma
        return True
    
    def sigma_xy(self):
        """1-sigma position uncertainty in metres."""
        return float(np.sqrt(self.Sigma[0, 0] + self.Sigma[1, 1]))

def body_speeds(vgx, vgy, yaw_deg):
    """Tello vgx/vgy are dm/s in the Tello's own fixed frame, turned by yaw
    (yaw positive CLOCKWISE). Convert to the drone's forward / left speed in m/s."""
    yw = np.radians(yaw_deg)
    fwd = 0.1 * (vgx * np.cos(yw) + vgy * np.sin(yw))
    right = 0.1 * (-vgx * np.sin(yw) + vgy * np.cos(yw))
    return fwd, -right


def main():
    import pandas as pd  
    from part_2 import MAP
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--no-plot', action='store_true')
    opts = ap.parse_args()
    log = Path(opts.log)

    df = pd.read_csv(log)
    for c in ('vgx', 'vgy', 'yaw', 'cam_x', 'cam_z', 'tof'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values('t', kind='stable')

    cps = df[df.kind == 'checkpoint']
    t_start = cps[cps.id == 1].t.iloc[0] if (cps.id == 1).any() else df.t.iloc[0]
    st = df[df.kind == 'state']
    tof_med = st.tof.median()
    if tof_med < 20:
        print(f'[AVISO] tof mediano = {tof_med:.0f} cm: o sensor de baixo estava TAPADO (mao embaixo?). '
              'As velocidades deste log nao sao confiaveis.')

    ekf = EKF([0.0, 0.0, 0.0], np.diag([0.05**2, 0.05**2, np.radians(5)**2]))
    dr = np.zeros(3)                       # dead reckoning only (no markers), for comparison
    last_t, last_yaw = None, None
    last_meas = {}
    out, n_used, n_rej = [], 0, 0

    for row in df[df.t >= t_start].itertuples():
        if row.kind == 'state':
            if np.isnan(row.yaw) or np.isnan(row.vgx):
                continue
            if last_t is not None:
                dt = row.t - last_t
                if dt > 0:
                    v_f, v_l = body_speeds(row.vgx, row.vgy, row.yaw)
                    w = -wrap(np.radians(row.yaw - last_yaw)) / dt   # yaw is clockwise -> CCW rate
                    ekf.predict(v_f, v_l, w, dt)
                    c, s = np.cos(dr[2]), np.sin(dr[2])
                    dr = dr + np.array([(v_f * c - v_l * s) * dt, (v_f * s + v_l * c) * dt, w * dt])
            last_t, last_yaw = row.t, row.yaw
            out.append((row.t, *ekf.mu, *np.sqrt(np.diag(ekf.Sigma)), *dr[:2]))
        elif row.kind == 'marker':
            mid = int(row.id)
            if mid not in MAP or np.hypot(row.cam_x, row.cam_z) > MAX_RANGE_M:
                continue
            key = (row.cam_x, row.cam_z)
            if last_meas.get(mid) == key:          # same video frame logged twice -> use once
                continue
            last_meas[mid] = key
            if ekf.update(row.cam_x, row.cam_z, MAP[mid]):
                n_used += 1
            else:
                n_rej += 1

    est = pd.DataFrame(out, columns=['t', 'x', 'y', 'theta', 'sx', 'sy', 'sth', 'dr_x', 'dr_y'])
    print(f'[EKF] {log.name}: marker updates used={n_used}, rejected={n_rej}')
    print(f'{"CP":>3} {"true (x, y)":>15} {"EKF (x, y)":>15} {"err":>6} {"+-1sigma":>9} {"dead-reck err":>13}')
    for r in cps.itertuples():
        k = int(r.id)
        if k not in CHECKPOINTS:
            continue
        e = est.iloc[(est.t - r.t).abs().argmin()]
        tx, ty = CHECKPOINTS[k]
        err = np.hypot(e.x - tx, e.y - ty)
        derr = np.hypot(e.dr_x - tx, e.dr_y - ty)
        print(f'{k:>3} ({tx:+.2f}, {ty:+.2f})  ({e.x:+.2f}, {e.y:+.2f}) {err:6.2f} {np.hypot(e.sx, e.sy):9.2f} {derr:13.2f}')

    est.to_csv(log.with_name(log.stem + '_ekf.csv'), index=False)
    if opts.no_plot:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    # plot with X forward to the right and Y left pointing UP (seen from above)
    ax.plot(est.dr_x, est.dr_y, color='0.6', lw=1, ls='--', label='speeds only (dead reckoning)')
    ax.plot(est.x, est.y, color='C0', lw=2, label='EKF estimate')
    for mid, (mx, my) in MAP.items():
        ax.plot(mx, my, 'ks', ms=9)
        ax.annotate(f'ID {mid}', (mx, my), textcoords='offset points', xytext=(6, 6), fontsize=8)
    for k, (cx, cy) in CHECKPOINTS.items():
        ax.plot(cx, cy, 'x', color='C3', ms=9, mew=2)
        ax.annotate(f'CP{k}', (cx, cy), textcoords='offset points', xytext=(6, -12), fontsize=8, color='C3')
    for r in cps.itertuples():
        e = est.iloc[(est.t - r.t).abs().argmin()]
        ax.plot(e.x, e.y, 'o', color='C0', ms=5)
    ax.set_xlabel('X forward (m)')
    ax.set_ylabel('Y left (m)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(log.name)
    fig.tight_layout()
    png = log.with_name(log.stem + '_ekf.png')
    fig.savefig(png, dpi=130)
    print('[PLOT]', png)


if __name__ == '__main__':
    main()
    