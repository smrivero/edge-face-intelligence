#!/usr/bin/env python3
"""Edge Face Intelligence: experimental RGB+Depth facial relief probe (NOT a biometric security system)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

BASE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 640, 480
PORT = 8766
FACE_MIN_CONF = 0.65
# Experimental indicator ONLY, NOT a calibrated anti-spoofing classifier.
RELIEF_MM = 12.0
MIN_COVERAGE = 0.60

model = str(BASE / 'models/yunet.onnx')
detector = cv2.FaceDetectorYN_create(model, '', (WIDTH, HEIGHT), FACE_MIN_CONF, 0.3, 5000)
cond = threading.Condition()
state = {'jpeg': None, 'seq': 0, 'metrics': {'status': 'waiting'}}

PAGE = b'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Edge Face Intelligence - 3D Depth Probe</title><style>body{font:16px Arial;background:#151515;color:#eee;text-align:center;margin:16px}img{max-width:1280px;width:100%}pre{background:#252525;white-space:pre-wrap;display:inline-block;text-align:left;padding:14px;border-radius:8px}</style></head><body><h2>Edge Face Intelligence | Relieve facial RGB + Depth</h2><p>Diagnostico experimental: NO es una prueba de vida confiable</p><img src="/stream.mjpg"><div><pre id="metrics">Cargando...</pre></div><script>async function poll(){try{const r=await fetch('/metrics',{cache:'no-store'});document.getElementById('metrics').textContent=JSON.stringify(await r.json(),null,2)}catch(e){document.getElementById('metrics').textContent=e.toString()}}poll();setInterval(poll,500)</script></body></html>'''


def patch(depth_mm, xy, radius):
    x, y = [int(round(v)) for v in xy]
    x0, x1 = max(0, x-radius), min(WIDTH, x+radius+1)
    y0, y1 = max(0, y-radius), min(HEIGHT, y+radius+1)
    arr = depth_mm[y0:y1, x0:x1]
    good = arr[(arr >= 300) & (arr <= 2000) & np.isfinite(arr)]
    return (float(np.median(good)) if len(good) >= 5 else None,
            float(len(good) / max(1, arr.size)))


def get_metrics(depth_mm, face):
    x,y,w,h = [int(v) for v in face[:4]]
    landmarks = face[4:14].reshape(5,2)
    # Landmarks returned by YuNet: eyes (0,1), nose (2), mouth corners (3,4).
    eyes = sorted(landmarks[:2], key=lambda p:p[0])
    nose = landmarks[2]
    # Use cheek-region patches between eyes and mouth to measure lateral face surface.
    mouth = sorted(landmarks[3:], key=lambda p:p[0])
    cheek_l = eyes[0]*0.40 + mouth[0]*0.60
    cheek_r = eyes[1]*0.40 + mouth[1]*0.60
    radius = max(3, int(min(w,h)*0.055))
    points = {'nose': nose, 'left_cheek':cheek_l, 'right_cheek':cheek_r,
              'left_eye': eyes[0], 'right_eye':eyes[1]}
    readings = {key: patch(depth_mm, pt, radius) for key,pt in points.items()}
    valid_depths = [v for v,coverage in readings.values() if v is not None]
    x0,x1 = max(0,x+int(.15*w)), min(WIDTH,x+int(.85*w))
    y0,y1 = max(0,y+int(.20*h)), min(HEIGHT,y+int(.85*h))
    roi = depth_mm[y0:y1,x0:x1]
    roi_valid = np.isfinite(roi) & (roi>=300) & (roi<=2000)
    coverage = float(np.mean(roi_valid)) if roi.size else 0.0
    nose_d = readings['nose'][0]
    cheek_ds = [readings[k][0] for k in ('left_cheek','right_cheek')]
    relief = (float(np.mean(cheek_ds)-nose_d)
              if nose_d is not None and all(v is not None for v in cheek_ds) else None)
    if len(valid_depths)<3 or coverage<MIN_COVERAGE or relief is None:
        status='INCONCLUSIVE'
    elif relief>=RELIEF_MM:
        status='3D_RELIEF_DETECTED'
    else:
        status='LOW_RELIEF'
    metrics={'status':status,'relief_mm':round(relief,1) if relief is not None else None,
             'nose_mm':round(nose_d,1) if nose_d is not None else None,
             'left_cheek_mm':round(cheek_ds[0],1) if cheek_ds[0] is not None else None,
             'right_cheek_mm':round(cheek_ds[1],1) if cheek_ds[1] is not None else None,
             'face_depth_coverage_percent':round(coverage*100,1),
             'threshold_mm_experimental':RELIEF_MM}
    return metrics,points


class Handler(BaseHTTPRequestHandler):
    def respond(self, content, ctype):
        self.send_response(200);self.send_header('Content-Type',ctype)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(content)))
        self.end_headers();self.wfile.write(content)
    def do_GET(self):
        if self.path=='/':return self.respond(PAGE,'text/html; charset=utf-8')
        if self.path=='/metrics':
            with cond:data=json.dumps(state['metrics']).encode()
            return self.respond(data,'application/json')
        if self.path!='/stream.mjpg':
            self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control','no-store');self.end_headers()
        seq=-1
        try:
            while True:
                with cond:
                    cond.wait_for(lambda:state['seq'] != seq,timeout=4)
                    jpeg=state['jpeg'];seq=state['seq']
                if jpeg:self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n')
        except (BrokenPipeError,ConnectionResetError,OSError):pass
    def log_message(self,*args):pass


def main():
    pipe=rs.pipeline(); cfg=rs.config()
    cfg.enable_stream(rs.stream.color,WIDTH,HEIGHT,rs.format.bgr8,30)
    cfg.enable_stream(rs.stream.depth,WIDTH,HEIGHT,rs.format.z16,30)
    align=rs.align(rs.stream.color)
    pipe.start(cfg)
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f'Abra http://localhost:{PORT} desde la Mac por tunel SSH',flush=True)
    last=time.perf_counter();fps=0
    try:
        while True:
            frames=align.process(pipe.wait_for_frames(5000))
            color,depth=frames.get_color_frame(),frames.get_depth_frame()
            if not color or not depth:continue
            rgb=np.asarray(color.get_data()).copy()
            raw=np.asarray(depth.get_data())
            depth_mm=raw.astype(np.float32)*depth.get_units()*1000
            _,faces=detector.detect(rgb)
            metrics={'status':'NO_FACE' if faces is None or len(faces)==0 else 'MULTIPLE_FACES'}
            if faces is not None and len(faces)==1:
                face=faces[0];metrics,pts=get_metrics(depth_mm,face)
                x,y,w,h=face[:4].astype(int)
                cv2.rectangle(rgb,(x,y),(x+w,y+h),(0,255,0),2)
                for key,pt in pts.items():
                    p=tuple(np.round(pt).astype(int));cv2.circle(rgb,p,4,(0,255,255),-1)
                cv2.putText(rgb,f"{metrics['status']} relief={metrics['relief_mm']}mm",
                            (8,60),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
            now=time.perf_counter();fps=.9*fps+.1/max(.001,now-last);last=now
            metrics['fps']=round(fps,1)
            cv2.putText(rgb,f'FPS {fps:.1f} | Experimental depth probe',
                        (8,27),cv2.FONT_HERSHEY_SIMPLEX,.60,(0,255,0),2)
            vis=cv2.applyColorMap(cv2.convertScaleAbs(depth_mm,alpha=255/2500),cv2.COLORMAP_JET)
            vis[raw==0]=0
            ok,jpg=cv2.imencode('.jpg',np.hstack((rgb,vis)),[cv2.IMWRITE_JPEG_QUALITY,80])
            if ok:
                with cond:
                    state['jpeg']=jpg.tobytes();state['metrics']=metrics
                    state['seq']+=1;cond.notify_all()
    except KeyboardInterrupt:pass
    finally:server.shutdown();server.server_close();pipe.stop()

if __name__=='__main__':main()
