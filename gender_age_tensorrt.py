#!/usr/bin/env python3
"""Edge Face Intelligence: YuNet CPU + InsightFace genderage TensorRT GPU + RealSense RGB MJPEG."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'models/genderage-fp16.engine'
YUNET = ROOT / 'models/yunet.onnx'
PORT = 8768
W, H = 640, 480


def check(result, operation):
    if int(result[0]) != 0:
        raise RuntimeError(f'{operation} failed: {result[0]}')
    return result[1] if len(result) > 1 else None


class TRTGenderAge:
    def __init__(self):
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(ENGINE.read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Cannot load {ENGINE}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Cannot create TensorRT execution context')
        self.input_name = 'data'
        self.output_name = 'fc1'
        assert tuple(self.engine.get_tensor_shape(self.input_name)) == (1, 3, 96, 96)
        assert tuple(self.engine.get_tensor_shape(self.output_name)) == (1, 3)
        self.input = np.empty((1, 3, 96, 96), dtype=np.float32)
        self.output = np.empty((1, 3), dtype=np.float32)
        self.device_input = check(cudart.cudaMalloc(self.input.nbytes), 'cudaMalloc input')
        self.device_output = check(cudart.cudaMalloc(self.output.nbytes), 'cudaMalloc output')
        self.context.set_tensor_address(self.input_name, int(self.device_input))
        self.context.set_tensor_address(self.output_name, int(self.device_output))

    def infer(self, face_bgr):
        blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (96, 96), (0, 0, 0), swapRB=True)
        self.input[:] = blob
        kind = cudart.cudaMemcpyKind
        check(cudart.cudaMemcpy(int(self.device_input), self.input.ctypes.data,
                                self.input.nbytes, kind.cudaMemcpyHostToDevice), 'copy to GPU')
        if not self.context.execute_async_v3(stream_handle=0):
            raise RuntimeError('TensorRT inference failed')
        check(cudart.cudaMemcpy(self.output.ctypes.data, int(self.device_output),
                                self.output.nbytes, kind.cudaMemcpyDeviceToHost), 'copy from GPU')
        values = self.output[0].copy()
        # Official InsightFace Attribute model: 0=Female, 1=Male
        gender_index = int(np.argmax(values[:2]))
        gender = 'Female' if gender_index == 0 else 'Male'
        age = int(np.round(float(values[2]) * 100))
        return gender, age, values

    def close(self):
        if self.device_input is not None:
            cudart.cudaFree(self.device_input)
            self.device_input = None
        if self.device_output is not None:
            cudart.cudaFree(self.device_output)
            self.device_output = None


def crop_face(image, x, y, w, h):
    # Same bbox-centered geometric transform as InsightFace Attribute.get():
    # scale = 96 / (max(width,height)*1.5), zero rotation.
    center = (float(x + w / 2), float(y + h / 2))
    scale = 96.0 / (max(float(w), float(h), 1.0) * 1.5)
    mat = cv2.getRotationMatrix2D(center, 0.0, scale)
    mat[0, 2] += 48.0 - center[0]
    mat[1, 2] += 48.0 - center[1]
    return cv2.warpAffine(image, mat, (96, 96), flags=cv2.INTER_LINEAR)


shared = {'frame': None, 'seq': 0, 'metrics': {}}
condition = threading.Condition()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            page = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge Face Intelligence | Age &amp; Gender TRT</title><style>
body{background:#151515;color:#eee;text-align:center;font:16px Arial,sans-serif;margin:20px}
img{max-width:100%;width:960px;border-radius:9px} .muted{color:#aaa}
#metrics{background:#222;padding:12px;border-radius:9px;margin:12px auto;max-width:600px;font-family:monospace;white-space:pre-wrap}
</style></head><body><h2>Jetson | Age &amp; Gender</h2><p class="muted">YuNet CPU + InsightFace GenderAge TensorRT GPU</p>
<div id="metrics">Starting...</div><img src="/stream.mjpg">
<p class="muted">Experimental predictions. Age is approximate; gender is a model appearance classification.</p>
<script>async function poll(){try{let r=await fetch('/metrics',{cache:'no-store'});document.getElementById('metrics').textContent=JSON.stringify(await r.json(),null,2)}catch(e){}}setInterval(poll,750);poll();</script>
</body></html>'''.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path == '/metrics':
            with condition:
                data = json.dumps(shared['metrics']).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
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
                    condition.wait_for(lambda: shared['seq'] != seq, timeout=5)
                    payload, seq = shared['frame'], shared['seq']
                if payload is None:
                    continue
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                 str(len(payload)).encode() + b'\r\n\r\n' + payload + b'\r\n')
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, *args):
        return


def main():
    if not ENGINE.is_file() or not YUNET.is_file():
        raise FileNotFoundError(f'Required: {ENGINE} and {YUNET}')
    print('Loading TensorRT engine and YuNet...', flush=True)
    trt_model = TRTGenderAge()
    detector = cv2.FaceDetectorYN_create(str(YUNET), '', (W, H), 0.65, 0.3, 5000)
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    server = None
    try:
        pipe.start(config)
        server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f'Running: http://127.0.0.1:{PORT} (SSH tunnel from Mac)', flush=True)
        smoothed_fps, last = 0.0, time.perf_counter()
        while True:
            frames = pipe.wait_for_frames(5000)
            frame = frames.get_color_frame()
            if not frame:
                continue
            rgb = np.asanyarray(frame.get_data()).copy()
            _, faces = detector.detect(rgb)
            predictions = []
            trt_ms = 0.0
            if faces is not None:
                for f in faces:
                    x, y, w, h = map(int, f[:4])
                    crop = crop_face(rgb, x, y, w, h)
                    t0 = time.perf_counter()
                    gender, age, values = trt_model.infer(crop)
                    trt_ms += (time.perf_counter() - t0) * 1000.0
                    predictions.append({'age_estimate': age, 'gender_model': gender})
                    cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 220, 90), 2)
                    cv2.putText(rgb, f'{gender} | Age ~{age}', (max(0, x), max(22, y-9)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 90), 2)
            now = time.perf_counter()
            fps = 1.0 / max(0.001, now - last)
            smoothed_fps = fps if smoothed_fps == 0 else .9 * smoothed_fps + .1 * fps
            last = now
            cv2.putText(rgb, f'FPS: {smoothed_fps:.1f} | TensorRT: {trt_ms:.1f} ms',
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
            ok, jpeg = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 83])
            if ok:
                with condition:
                    shared['frame'] = jpeg.tobytes()
                    shared['seq'] += 1
                    shared['metrics'] = {'fps_pipeline': round(smoothed_fps, 1),
                                         'tensorrt_ms_faces': round(trt_ms, 2),
                                         'faces': len(predictions), 'predictions': predictions}
                    condition.notify_all()
    finally:
        if server:
            server.shutdown()
            server.server_close()
        pipe.stop()
        trt_model.close()


if __name__ == '__main__':
    main()
