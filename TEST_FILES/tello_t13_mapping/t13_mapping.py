#!/usr/bin/env python3
"""T1.3: ground-only relative map of circle markers 0/1 and rear marker 3."""
import argparse, csv, math, time
from pathlib import Path
import cv2
import numpy as np
from djitellopy import Tello

SIZE=0.115
SEP=0.55
TARGET_Z=0.75

def pose(corners, K, D):
    s=SIZE/2
    obj=np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]],np.float32)
    img=np.asarray(corners,np.float32).reshape(4,2)
    ok,rvec,tvec=cv2.solvePnP(obj,img,K,D,flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok: return None
    proj,_=cv2.projectPoints(obj,rvec,tvec,K,D)
    rms=float(np.sqrt(np.mean(np.sum((proj.reshape(4,2)-img)**2,axis=1))))
    R,_=cv2.Rodrigues(rvec)
    return np.asarray(tvec).reshape(3),R,rms

def circle_frame(p0,p1):
    # Local +Y points from bottom ID0 to top ID1; +Z points away from camera.
    center=(p0[0]+p1[0])/2
    up=p1[0]-p0[0]
    up/=np.linalg.norm(up)
    normal=-(p0[1][:,2]+p1[1][:,2])
    normal-=np.dot(normal,up)*up
    if np.linalg.norm(normal)<0.2: return None
    depth=normal/np.linalg.norm(normal)
    if depth[2]<0: depth=-depth
    right=np.cross(depth,up)
    right/=np.linalg.norm(right)
    depth=np.cross(up,right)
    axes=np.column_stack([right,up,depth])
    return center,axes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--calibration',default='tello_calibration.npz')
    ap.add_argument('--csv',default='t13_map.csv')
    args=ap.parse_args()
    cal=np.load(args.calibration)
    K=cal['camera_matrix'];D=cal['dist_coeffs'];sz=tuple(int(v) for v in cal['image_size'])
    print('GROUND ONLY: no takeoff, landing, or RC commands. Q quit; S screenshot.')
    if input('Drone on ground? Type SOLO: ').strip().upper()!='SOLO':return
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params=cv2.aruco.DetectorParameters()
    detector=cv2.aruco.ArucoDetector(dictionary,params)
    tello=Tello(); connected=False
    f=open(args.csv,'w',newline='')
    writer=csv.writer(f)
    writer.writerow(['timestamp','id0_z_m','id1_z_m','id3_z_m','circle_x_m','circle_y_m','circle_z_m','sep_m','id3_circle_x_m','id3_circle_y_m','id3_circle_depth_m','id3_range_m','id3_stop_error_m','id0_rms_px','id1_rms_px','id3_rms_px','status'])
    try:
        tello.connect();connected=True
        try:tello.streamoff()
        except Exception:pass
        tello.streamon();reader=tello.get_frame_read()
        last_frame=None; last_print=0; n=0
        while True:
            frame=reader.frame
            if frame is None:
                time.sleep(.04);continue
            frame=frame.copy()
            h,w=frame.shape[:2]
            if (w,h)!=sz:
                cv2.putText(frame,f'CALIBRATION SIZE MISMATCH: {(w,h)} != {sz}',(15,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
                cv2.imshow('T1.3 mapping - ground only',frame)
                if cv2.waitKey(1)&255==ord('q'):break
                continue
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            corners,ids,_=detector.detectMarkers(gray)
            poses={}; pixels={}
            if ids is not None:
                for c,idarr in zip(corners,ids.flatten()):
                    i=int(idarr)
                    if i not in (0,1,3):continue
                    pp=pose(c,K,D)
                    if pp is None or pp[2]>2 or pp[0][2]<=0:continue
                    poses[i]=pp;pixels[i]=np.asarray(c).reshape(4,2).mean(axis=0)
                    cv2.aruco.drawDetectedMarkers(frame,[c],np.array([[i]],np.int32))
                    cv2.drawFrameAxes(frame,K,D,cv2.Rodrigues(pp[1])[0],pp[0].reshape(3,1),.06)
                    cv2.putText(frame,f'ID {i}: Z={pp[0][2]:.2f}m rms={pp[2]:.1f}px',tuple(pixels[i].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),2)
            cxyz=np.array([np.nan]*3);sep=np.nan;rel=np.array([np.nan]*3)
            status='CIRCLE_NOT_VISIBLE'
            if 0 in poses and 1 in poses:
                sep=float(np.linalg.norm(poses[1][0]-poses[0][0]))
                cf=circle_frame(poses[0],poses[1])
                if cf is not None:
                    cxyz,axes=cf
                    # Midpoint is only the opening center if markers are symmetric.
                    if 3 in poses:rel=axes.T@(poses[3][0]-cxyz)
                    cp=(pixels[0]+pixels[1])/2
                    cv2.circle(frame,tuple(cp.astype(int)),8,(0,255,0),-1)
                    cv2.putText(frame,f'CIRCLE X={cxyz[0]:+.2f} Y={cxyz[1]:+.2f} Z={cxyz[2]:.2f}m', (12,27),cv2.FONT_HERSHEY_SIMPLEX,.59,(255,255,0),2)
                    cv2.putText(frame,f'ID0-ID1: {sep:.3f}m expected {SEP:.3f}m',(12,52),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,0),2)
                    status='MAP_READY' if 3 in poses else 'CIRCLE_ONLY'
            if 3 in poses:
                z3=float(poses[3][0][2]); err=z3-TARGET_Z
                cv2.putText(frame,f'ID03 Z={z3:.2f}m; stop target={TARGET_Z:.2f}m; delta={err:+.2f}m',(12,78),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,0),2)
                if np.isfinite(rel).all():
                    cv2.putText(frame,f'ID03 wrt circle: x={rel[0]:+.2f} y={rel[1]:+.2f} depth={rel[2]:+.2f}m',(12,103),cv2.FONT_HERSHEY_SIMPLEX,.51,(255,255,0),2)
            cv2.putText(frame,f'GROUND ONLY | {status} | Q quit S screenshot',(12,h-17),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),2)
            t=time.time()
            if t-last_print>.3:
                def getz(i):return poses[i][0][2] if i in poses else np.nan
                def getr(i):return poses[i][2] if i in poses else np.nan
                z3=getz(3)
                writer.writerow([f'{t:.3f}',getz(0),getz(1),z3,*cxyz,sep,*rel,float(np.linalg.norm(poses[3][0])) if 3 in poses else np.nan,z3-TARGET_Z if 3 in poses else np.nan,getr(0),getr(1),getr(3),status])
                f.flush();last_print=t
                if n%10==0:print(f'{status}: circle Z={cxyz[2]:.2f}m, ID03 Z={z3:.2f}m, ID03 relative={rel.round(2)}')
                n+=1
            cv2.imshow('T1.3 mapping - ground only',frame)
            key=cv2.waitKey(1)&255
            if key==ord('q'):break
            if key==ord('s'):
                name=f't13_{time.strftime("%Y%m%d_%H%M%S")}.png';cv2.imwrite(name,frame);print('Saved',name)
    except KeyboardInterrupt:print('\nInterrupted; closing stream.')
    finally:
        try:
            if connected:tello.streamoff()
        except Exception:pass
        cv2.destroyAllWindows();f.close();print('CSV:',args.csv)

if __name__=='__main__':main()
