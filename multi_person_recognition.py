#!/usr/bin/env python3
"""Edge Face Intelligence: two-person, consent-based local face recognition demo."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / 'models'
DATA_DIR = BASE / 'data' / 'faces'
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.chmod(0o700)
PERSONS = {'sebastian': 'Sebastian', 'benjamin': 'Benjamin'}
W, H, PORT = 640, 480, 8769
SAMPLES = 8
THRESHOLD = 0.45  # Experimental. Must calibrate for actual use.
MIN_MARGIN = 0.06  # Reject ambiguous matches.
TARGET = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)

def normalized(v):
    v = np.asarray(v, dtype=np.float32).flatten()
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError('Embedding invalid')
    return v / n

# Import earlier, single-person enrollment as Sebastian only if no separate template exists.
legacy = BASE / 'data' / 'face_template.npy'
if legacy.exists() and not (DATA_DIR / 'sebastian.npy').exists():
    np.save(DATA_DIR / 'sebastian.npy', normalized(np.load(legacy, allow_pickle=False)))
    (DATA_DIR / 'sebastian.npy').chmod(0o600)
    print('Imported existing face_template.npy as Sebastian; original preserved', flush=True)

templates = {}
for key in PERSONS:
    p = DATA_DIR / f'{key}.npy'
    if p.exists():
        templates[key] = normalized(np.load(p, allow_pickle=False))

net = cv2.dnn.readNetFromONNX(str(MODEL_DIR / 'arcface.onnx'))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
detector = cv2.FaceDetectorYN_create(
    str(MODEL_DIR / 'yunet.onnx'), '', (W, H), 0.65, 0.3, 5000
)

mutex = threading.Lock()
cond = threading.Condition()
state = {'person': None, 'vectors': [], 'last_sample': 0.0,
         'message': 'Ready', 'jpeg': None, 'seq': 0, 'fps': 0., 'faces': 0}

def embed(img, face):
    p = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    eyes = sorted(p[:2], key=lambda x: x[0])
    mouth = sorted(p[3:], key=lambda x: x[0])
    landmarks = np.array([eyes[0], eyes[1], p[2], mouth[0], mouth[1]], np.float32)
    mat, _ = cv2.estimateAffinePartial2D(landmarks, TARGET, method=cv2.LMEDS)
    if mat is None:
        return None
    aligned = cv2.warpAffine(img, mat, (112, 112), borderValue=0)
    blob = cv2.dnn.blobFromImage(
        aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    return normalized(net.forward())

def match(vec, gallery):
    if not gallery:
        return 'UNKNOWN', None
    scores = sorted(((float(np.dot(vec, ref)), key) for key, ref in gallery.items()), reverse=True)
    best, key = scores[0]
    if best < THRESHOLD:
        return 'UNKNOWN', best
    if len(scores) > 1 and best - scores[1][0] < MIN_MARGIN:
        return 'AMBIGUOUS', best
    return PERSONS[key], best

PAGE = '''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Edge Face Intelligence | Reconocimiento de 2 personas</title>
<style>body{margin:20px auto;padding:0 16px;max-width:1240px;background:#13161a;color:#f5f5f5;font:16px Arial,sans-serif}h1{font-size:24px}button{padding:11px 16px;margin:5px 8px 5px 0;cursor:pointer;border:0;border-radius:7px;font-weight:600;background:#76b900;color:#111}button.secondary{background:#444;color:white}button:disabled{opacity:.4;cursor:default}.panel{padding:14px;background:#232830;border-radius:10px;margin-bottom:12px}img{display:block;width:100%;height:auto;border-radius:8px}progress{width:100%;height:17px;accent-color:#76b900}.muted{color:#aaa;font-size:13px}</style></head>
<body><h1>Edge Face Intelligence | Sebastian + Benjamin</h1><div class="panel"><button onclick="enroll('sebastian')">Registrar Sebastian</button><button onclick="enroll('benjamin')">Registrar Benjamin</button><button class="secondary" onclick="erase('sebastian')">Borrar Sebastian</button><button class="secondary" onclick="erase('benjamin')">Borrar Benjamin</button><p id="status">Cargando estado...</p><progress id="progress" value="0" max="8" hidden></progress><p id="known"></p><p class="muted">Registro: colocá solo a la persona elegida frente a la cámara. Las 8 muestras son automáticas. Al terminar, ambas personas pueden aparecer juntas.</p></div>
<img src="/stream.mjpg" alt="Cámara en vivo"><p class="muted">Demostración local, sin verificación de vida. Los nombres dependen de las plantillas registradas y un umbral experimental.</p>
<script>
async function call(url){const r=await fetch(url,{method:'POST'});const body=await r.json();if(!r.ok)throw Error(body.error||r.status);return body}
async function enroll(p){if(!confirm('Registrar la cara de '+p+'? Debe aparecer una sola persona y participar con su consentimiento.'))return;try{await call('/enroll/'+p);update()}catch(e){alert(e.message)}}
async function erase(p){if(!confirm('Borrar plantilla de '+p+'?'))return;try{await call('/delete/'+p);update()}catch(e){alert(e.message)}}
async function update(){try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();document.getElementById('status').textContent=(s.enrolling?'Registrando '+s.enrolling+': '+s.samples+'/8. ':'')+s.message+' | FPS: '+s.fps.toFixed(1)+' | Caras: '+s.faces;document.getElementById('known').textContent='Registrados: '+(s.registered.length?s.registered.join(', '):'ninguno');const prog=document.getElementById('progress');prog.hidden=!s.enrolling;prog.value=s.samples;document.querySelectorAll('button').forEach(b=>b.disabled=!!s.enrolling)}catch(e){document.getElementById('status').textContent='Servidor no disponible'}}
update();setInterval(update,500);
</script></body></html>'''.encode('utf-8')

class Handler(BaseHTTPRequestHandler):
    def json_reply(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parts = self.path.strip('/').split('/')
        if len(parts) != 2 or parts[1] not in PERSONS or parts[0] not in ('enroll', 'delete'):
            return self.json_reply(404, {'error': 'Not found'})
        action, key = parts
        with mutex:
            if state['person'] is not None:
                return self.json_reply(409, {'error': 'Enrollment already running'})
            if action == 'enroll':
                state['person'] = key
                state['vectors'] = []
                state['last_sample'] = 0.0
                state['message'] = 'Waiting for exactly one face'
            else:
                templates.pop(key, None)
                (DATA_DIR / f'{key}.npy').unlink(missing_ok=True)
                state['message'] = f'{PERSONS[key]} deleted'
        self.json_reply(200, {'ok': True})

    def do_GET(self):
        if self.path == '/status':
            with mutex:
                result = {'registered': [PERSONS[k] for k in templates],
                          'enrolling': PERSONS.get(state['person']),
                          'samples': len(state['vectors']),
                          'message': state['message'],
                          'fps': state['fps'], 'faces': state['faces']}
            return self.json_reply(200, result)
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(PAGE)))
            self.end_headers()
            return self.wfile.write(PAGE)
        if self.path != '/stream.mjpg':
            return self.json_reply(404, {'error': 'Not found'})
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        seq = -1
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: state['seq'] != seq, timeout=5)
                    jpeg, seq = state['jpeg'], state['seq']
                if jpeg:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, *_):
        pass

camera = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
profile = camera.start(config)
server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f'Listening at http://127.0.0.1:{PORT}', flush=True)
last = time.perf_counter()
fps = 0.0
try:
    while True:
        frames = camera.wait_for_frames(5000)
        cf = frames.get_color_frame()
        if not cf:
            continue
        img = np.asanyarray(cf.get_data()).copy()
        _, faces = detector.detect(img)
        faces = [] if faces is None else faces
        with mutex:
            enrolling = state['person']
            gallery = {k: v.copy() for k, v in templates.items()}
        boxes = []
        if enrolling and len(faces) != 1:
            with mutex:
                state['message'] = 'Registration requires exactly one visible face'
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            vec = embed(img, face)
            label, score = 'UNKNOWN', None
            if vec is not None:
                if enrolling and len(faces) == 1:
                    with mutex:
                        if state['person'] == enrolling and time.monotonic() - state['last_sample'] >= 0.35:
                            state['vectors'].append(vec)
                            state['last_sample'] = time.monotonic()
                            count = len(state['vectors'])
                            state['message'] = f'Capturing {count}/{SAMPLES}'
                            if count >= SAMPLES:
                                avg = normalized(np.mean(state['vectors'], axis=0))
                                out = DATA_DIR / f'{enrolling}.npy'
                                np.save(out, avg)
                                out.chmod(0o600)
                                templates[enrolling] = avg
                                state['person'] = None
                                state['message'] = f'{PERSONS[enrolling]} registered successfully'
                    label = f'REGISTERING {PERSONS[enrolling]}'
                elif not enrolling:
                    label, score = match(vec, gallery)
            text = f'{label} {score:.3f}' if score is not None else label
            color = (0, 220, 0) if label in PERSONS.values() else (0, 180, 255)
            boxes.append((x, y, w, h, text, color))
        for x, y, w, h, text, color in boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, text, (max(x, 0), max(24, y - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 0.001)
        last = now
        cv2.putText(img, f'Faces: {len(faces)}  FPS: {fps:.1f}',
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 83])
        with mutex:
            state['fps'] = fps
            state['faces'] = len(faces)
        if ok:
            with cond:
                state['jpeg'] = buf.tobytes()
                state['seq'] += 1
                cond.notify_all()
except KeyboardInterrupt:
    print('Stopping', flush=True)
finally:
    server.shutdown()
    server.server_close()
    camera.stop()
