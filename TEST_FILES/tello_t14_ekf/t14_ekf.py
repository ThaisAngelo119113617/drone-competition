#!/usr/bin/env python3
"""T1.4 ground-only EKF: relative hoop-center in camera coordinates.
Approximate constant-velocity model; NOT a full drone navigation estimator.
"""
import argparse, csv, time
import cv2
import numpy as np
from djitellopy import Tello

MARKER_SIZE=.115
HALF_SPACING=.275
# Measured in T1.3: ID3 approximately 1.266m behind hoop, with small offsets.
# Approximate camera-aligned frame, valid only for nearly frontal views.
ID3_MINUS_HOOP=np.array([.013, -.013, 1.266]) # camera X right, Y down, Z forward
RMS_LIMIT=2.0

def marker_pose(c,K,D):
    s=MARKER_SIZE/2
    obj=np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]],np.float32)
    im=np.asarray(c,dtype=np.float32).reshape(4,2)
    ok,rv,tv=cv2.solvePnP(obj,im,K,D,flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:return None
    projected,_=cv2.projectPoints(obj,rv,tv,K,D)
    rms=float(np.sqrt(np.mean(np.sum((projected.reshape(4,2)-im)**2,axis=1))))
    return tv.reshape(3),rms

class EKF:
    def __init__(self):
        self.x=None;self.P=None
    def predict(self,dt):
        if self.x is None:return
        dt=min(max(dt,.001),.5)
        F=np.eye(6);F[:3,3:]=np.eye(3)*dt
        G=np.vstack([np.eye(3)*(.5*dt**2),np.eye(3)*dt])
        self.x=F@self.x
        self.P=F@self.P@F.T+G@(np.eye(3)*.7**2)@G.T+np.eye(6)*1e-7
    def update(self,z,std,gate=16.27):
        z=np.asarray(z,float)
        if not np.all(np.isfinite(z)) or z[2]<=0:return 'BAD'
        if self.x is None:
            self.x=np.r_[z,np.zeros(3)]
            self.P=np.diag([.08**2]*3+[.4**2]*3)
            return 'INIT'
        H=np.c_[np.eye(3),np.zeros((3,3))]
        R=np.eye(3)*std**2
        innov=z-H@self.x
        S=H@self.P@H.T+R
        d2=float(innov@np.linalg.solve(S,innov))
        if d2>gate:return f'GATED({d2:.1f})'
        K=self.P@H.T@np.linalg.inv(S)
        self.x+=K@innov
        I=np.eye(6);A=I-K@H
        self.P=A@self.P@A.T+K@R@K.T
        return 'OK'

def center_from_marker(i,t):
    # Local frontal approximation: OpenCV camera Y points downward.
    # Bottom ID0: center above; top ID1: center below.
    if i==0:return t+np.array([0,-HALF_SPACING,0])
    if i==1:return t+np.array([0,+HALF_SPACING,0])
    if i==3:return t-ID3_MINUS_HOOP
    raise ValueError(i)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--calibration',default='tello_calibration.npz')
    ap.add_argument('--csv',default='t14_ekf.csv')
    args=ap.parse_args()
    cal=np.load(args.calibration)
    K=cal['camera_matrix'];D=cal['dist_coeffs'];size=tuple(int(a) for a in cal['image_size'])
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),cv2.aruco.DetectorParameters())
    print('GROUND ONLY. NO FLIGHT COMMANDS. Q quit; S screenshot.')
    if input('Type SOLO to confirm motors OFF: ').strip().upper()!='SOLO':return
    drone=Tello();connected=False;ekf=EKF();last=time.monotonic()
    with open(args.csv,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['timestamp','visible_ids','accepted_ids','updates','center_x_m','center_y_m','center_z_m','sigma_x_m','sigma_y_m','sigma_z_m','suggestion','mode'])
        try:
            drone.connect();connected=True
            try:drone.streamoff()
            except Exception:pass
            drone.streamon();reader=drone.get_frame_read()
            next_log=0
            while True:
                frame=reader.frame
                if frame is None:time.sleep(.03);continue
                frame=frame.copy();h,ww=frame.shape[:2]
                if (ww,h)!=size:
                    cv2.putText(frame,'CALIBRATION SIZE MISMATCH - no EKF',(15,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
                    cv2.imshow('T1.4 EKF - GROUND ONLY',frame)
                    if cv2.waitKey(1)&255==ord('q'):break
                    continue
                now=time.monotonic();ekf.predict(now-last);last=now
                corners,ids,_=detector.detectMarkers(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
                visible={}; accepted=[];updates=[]
                if ids is not None:
                    for c,i in zip(corners,ids.flatten()):
                        i=int(i)
                        if i not in (0,1,3):continue
                        p=marker_pose(c,K,D)
                        if p is None or p[1]>RMS_LIMIT or p[0][2]<=0:continue
                        visible[i]=p
                        cv2.aruco.drawDetectedMarkers(frame,[c],np.array([[i]],dtype=np.int32))
                if 0 in visible and 1 in visible:
                    a,b=visible[0][0],visible[1][0]
                    spacing=np.linalg.norm(b-a)
                    if abs(spacing-.55)<.045:
                        z=(a+b)/2
                        res=ekf.update(z,.025)
                        updates.append('01:'+res)
                        if res in ('OK','INIT'):accepted.extend([0,1])
                    else:updates.append(f'SPACING_REJECT({spacing:.2f})')
                else:
                    for i in (0,1):
                        if i in visible:
                            res=ekf.update(center_from_marker(i,visible[i][0]),.075)
                            updates.append(f'{i}:{res}')
                            if res in ('OK','INIT'):accepted.append(i)
                # ID03 is weaker: mapping is approximate and rotates with camera.
                # Only use it near frontal alignment; disable if heading not verified.
                if 3 in visible and ekf.x is not None and (0 in accepted or 1 in accepted):
                    # Compare, don't fuse ID03 to avoid double counting correlated camera data.
                    d=np.linalg.norm(center_from_marker(3,visible[3][0])-ekf.x[:3])
                    updates.append(f'3:CHECK({d:.2f}m)')
                elif 3 in visible:updates.append('3:VISIBLE_NOT_FUSED')
                suggestion='WAIT'
                if ekf.x is not None:
                    x,y,z=ekf.x[:3]
                    sx,sy,sz=np.sqrt(np.diag(ekf.P)[:3])
                    if max(sx,sy,sz)>.12:suggestion='UNCERTAIN_HOVER'
                    elif y<-.06:suggestion='CENTER_BELOW_CAMERA (DESCEND TO ALIGN)'
                    elif y>.06:suggestion='CENTER_ABOVE_CAMERA (ASCEND TO ALIGN)'
                    elif x<-.06:suggestion='CENTER_LEFT (MOVE LEFT)'
                    elif x>.06:suggestion='CENTER_RIGHT (MOVE RIGHT)'
                    else:suggestion='ALIGNED_ESTIMATE'
                    cv2.putText(frame,f'Hoop center X={x:+.2f} Y={y:+.2f} Z={z:.2f}m',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
                    cv2.putText(frame,f'Uncertainty std X={sx:.2f} Y={sy:.2f} Z={sz:.2f}m',(12,55),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),2)
                else:x=y=z=sx=sy=sz=float('nan')
                cv2.putText(frame,f'IDs visible: {sorted(visible)} | accepted: {accepted}',(12,85),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,0),2)
                cv2.putText(frame,suggestion[:57],(12,113),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),2)
                cv2.putText(frame,'GROUND ONLY - NOT A FLIGHT CONTROLLER | Q quit S screenshot',(12,h-20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
                if now>=next_log:
                    w.writerow([f'{time.time():.3f}',';'.join(map(str,sorted(visible))),';'.join(map(str,accepted)),';'.join(updates),x,y,z,sx,sy,sz,suggestion,'GROUND_ONLY'])
                    f.flush();next_log=now+.2
                cv2.imshow('T1.4 EKF - GROUND ONLY',frame)
                key=cv2.waitKey(1)&255
                if key==ord('q'):break
                if key==ord('s'):
                    name=f't14_{time.strftime("%Y%m%d_%H%M%S")}.png';cv2.imwrite(name,frame);print('Saved',name)
        except KeyboardInterrupt:print('Interrupted')
        finally:
            if connected:
                try:drone.streamoff()
                except Exception:pass
            cv2.destroyAllWindows()
            print('CSV saved:',args.csv)

if __name__=='__main__':main()
