#!/usr/bin/env python3
"""InsightFace RGB liveness camera stream. Experimental, not secure authentication."""
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pyrealsense2 as rs
from insightface.app import FaceAnalysis

W, H, PORT = 640, 480, 8767
THRESHOLD = 0.8
condition = threading.Condition()
state = {"jpeg": None, "seq": 0, "metrics": {"state": "starting"}}

PAGE = '''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Edge Face Intelligence | RGB Liveness</title><style>
body{background:#151515;color:#eee;font:16px Arial,sans-serif;text-align:center;margin:16px}img{width:100%;max-width:960px;border-radius:6px}main{max-width:960px;margin:auto}#result{font-size:26px;font-weight:bold;margin:12px}#details{background:#252525;padding:15px;border-radius:8px;text-align:left;white-space:pre-wrap;overflow:auto}small{color:#bbb} .metrics{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}.metric{background:#272727;border-radius:6px;padding:14px;min-width:160px}.metric b{display:block;margin-top:8px;font-size:20px}</style></head><body><main><h2>Jetson | InsightFace 2.1 RGB Liveness</h2><small>Solo RGB · CPU · Modelo experimental · No es prueba de vida certificada</small><div id="result">Iniciando...</div><div class="metrics"><div class="metric">Live score<b id="score">--</b></div><div class="metric">FPS (pipeline)<b id="fps">--</b></div><div class="metric">Inferencia<b id="ms">--</b></div></div><p><img src="/stream.mjpg" alt="Camara RGB en vivo"></p><details><summary>Datos tecnicos</summary><pre id="details"></pre></details><p><small>Probá con tu cara y después con una fotografía. No utilizamos profundidad para esta prueba.</small></p></main><script>
async function poll(){try{const r=await fetch('/status',{cache:'no-store'});const x=await r.json();document.getElementById('result').textContent=x.label||'Esperando cámara';document.getElementById('result').style.color=x.label==='REAL (modelo)'?'#8fd44b':x.label==='SPOOF (modelo)'?'#ff8d8d':'#eee';document.getElementById('score').textContent=x.score==null?'--':Number(x.score).toFixed(3);document.getElementById('fps').textContent=x.fps==null?'--':Number(x.fps).toFixed(1);document.getElementById('ms').textContent=x.inference_ms==null?'--':Number(x.inference_ms).toFixed(1)+' ms';document.getElementById('details').textContent=JSON.stringify(x,null,2)}catch(e){document.getElementById('result').textContent='Sin conexion con Jetson'}}setInterval(poll,500);poll();</script></body></html>'''.encode()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return
        if self.path == '/status':
            with condition:
                obj = dict(state['metrics'])
            payload = json.dumps(obj, default=str).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path != '/stream.mjpg':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        seq = -1
        try:
            while True:
                with condition:
                    condition.wait_for(lambda: state['seq'] != seq, timeout=5)
                    jpeg, seq = state['jpeg'], state['seq']
                if jpeg is None:
                    continue
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
        except (ConnectionResetError, BrokenPipeError, OSError):
            return


def main():
    print('Cargando InsightFace CPU...', flush=True)
    app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'],
                       addons=['liveness'], liveness_mode='observe',
                       liveness_threshold=THRESHOLD,
                       providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 480))
    print('Modelos:', list(app.models.keys()), 'addons:', list(app.addons.keys()), flush=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    server = None
    pipe.start(cfg)
    try:
        for _ in range(15):
            pipe.wait_for_frames(5000)
        server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f'Abri http://localhost:{PORT} desde tu Mac (con tunel SSH). Ctrl+C para salir.', flush=True)
        prev = time.perf_counter()
        fps = 0.0
        while True:
            frames = pipe.wait_for_frames(5000)
            frame = frames.get_color_frame()
            if not frame:
                continue
            img = np.asanyarray(frame.get_data()).copy()
            t0 = time.perf_counter()
            faces = app.get(img)
            inference_ms = (time.perf_counter() - t0) * 1000.0
            now = time.perf_counter()
            instant = 1.0 / max(now - prev, 1e-6)
            fps = instant if fps == 0 else fps * 0.85 + instant * 0.15
            prev = now

            metrics = dict(label='NO FACE', score=None, status=None,
                           is_live=None, faces=len(faces), fps=round(fps, 2),
                           inference_ms=round(inference_ms, 2), threshold=THRESHOLD,
                           provider='CPUExecutionProvider')
            if faces:
                face = max(faces, key=lambda f: float((f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])))
                live = getattr(face, 'liveness', None)
                status = getattr(live, 'status', None)
                is_live = getattr(live, 'is_live', None)
                score = getattr(live, 'live_score', None)
                # Do not silently treat missing or rejected inputs as spoof.
                if live is None or score is None or status == 'input_rejected':
                    label = 'INCONCLUSIVE'
                elif is_live is True or is_live == np.bool_(True):
                    label = 'REAL (modelo)'
                elif is_live is False or is_live == np.bool_(False):
                    label = 'SPOOF (modelo)'
                else:
                    label = 'INCONCLUSIVE'
                metrics.update(label=label, status=str(status),
                               is_live=None if is_live is None else bool(is_live),
                               score=None if score is None else float(score))
                x1,y1,x2,y2 = map(int,face.bbox)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,200,255), 2)
                cv2.putText(img, label, (max(0,x1),max(25,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (0,220,255),2)
            cv2.putText(img, f'FPS {fps:.1f} | {inference_ms:.1f} ms', (12,29),
                        cv2.FONT_HERSHEY_SIMPLEX,.7,(50,240,80),2)
            ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with condition:
                    state['jpeg'] = encoded.tobytes()
                    state['seq'] += 1
                    state['metrics'] = metrics
                    condition.notify_all()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        pipe.stop()

if __name__ == '__main__':
    main()
