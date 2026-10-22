import cv2
import time
import threading
import numpy as np
import pyrealsense2 as rs

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8765
WIDTH = 640
HEIGHT = 480

MODEL = str(Path("models/yunet.onnx"))

detector = cv2.FaceDetectorYN_create(
    MODEL, "", (WIDTH, HEIGHT),
    0.65, 0.3, 5000
)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, WIDTH, HEIGHT,
    rs.format.bgr8, 30
)
config.enable_stream(
    rs.stream.depth, WIDTH, HEIGHT,
    rs.format.z16, 30
)

align = rs.align(rs.stream.color)

condition = threading.Condition()
state = {"jpeg": None, "seq": 0}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            html = b"""
            <!DOCTYPE html>
            <html>
            <head>
              <meta name="viewport"
                    content="width=device-width,initial-scale=1">
              <title>Edge Face Intelligence - Face Detection</title>
              <style>
                body {background:#151515;color:white;
                      font-family:Arial;text-align:center}
                img {width:100%;max-width:1280px}
              </style>
            </head>
            <body>
              <h2>Jetson - Face Detection + Depth</h2>
              <p>YuNet | RealSense D435 | RGB + Depth</p>
              <img src="/stream.mjpg">
            </body>
            </html>
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        seq = -1

        try:
            while True:
                with condition:
                    condition.wait_for(
                        lambda: state["seq"] != seq,
                        timeout=5
                    )
                    jpeg = state["jpeg"]
                    seq = state["seq"]

                if jpeg is None:
                    continue

                self.wfile.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n"
                      .encode()
                    + b"\r\n" + jpeg + b"\r\n"
                )

        except (BrokenPipeError, ConnectionResetError,
                OSError):
            pass

    def log_message(self, *args):
        pass

print("Iniciando RealSense...")
pipeline.start(config)

server = ThreadingHTTPServer(
    ("0.0.0.0", PORT), Handler
)
threading.Thread(
    target=server.serve_forever, daemon=True
).start()

print(f"Servidor HTTP en puerto {PORT}")
print("Presiona Ctrl+C para detener")

last = time.perf_counter()
fps = 0

try:
    while True:
        frames = pipeline.wait_for_frames(5000)
        frames = align.process(frames)

        color = frames.get_color_frame()
        depth = frames.get_depth_frame()

        if not color or not depth:
            continue

        rgb = np.asanyarray(
            color.get_data()
        ).copy()

        raw_depth = np.asanyarray(
            depth.get_data()
        )

        depth_mm = (
            raw_depth.astype(np.float32)
            * depth.get_units() * 1000
        )

        _, faces = detector.detect(rgb)

        count = 0 if faces is None else len(faces)

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                score = float(face[-1])

                cv2.rectangle(
                    rgb, (x, y),
                    (x+w, y+h),
                    (0, 255, 0), 2
                )

                for i in range(5):
                    px = int(face[4 + i*2])
                    py = int(face[5 + i*2])
                    cv2.circle(
                        rgb, (px, py), 2,
                        (0, 255, 255), -1
                    )

                # Profundidad del centro de la cara
                cx = max(0, min(WIDTH-1, x+w//2))
                cy = max(0, min(HEIGHT-1, y+h//2))

                roi = depth_mm[
                    max(0,cy-12):min(HEIGHT,cy+12),
                    max(0,cx-12):min(WIDTH,cx+12)
                ]

                valid = roi[
                    (roi > 250) & (roi < 3000)
                ]

                distance = (
                    f"{np.median(valid):.0f} mm"
                    if len(valid) else "N/A"
                )

                cv2.putText(
                    rgb,
                    f"Face {score:.2f} | {distance}",
                    (max(0,x), max(20,y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0,255,0), 2
                )

        depth_vis = cv2.convertScaleAbs(
            depth_mm, alpha=255.0/2500
        )
        depth_vis = cv2.applyColorMap(
            depth_vis, cv2.COLORMAP_JET
        )
        depth_vis[raw_depth == 0] = 0

        now = time.perf_counter()
        instant_fps = 1.0 / max(now-last, 0.001)
        fps = 0.9*fps + 0.1*instant_fps
        last = now

        cv2.putText(
            rgb,
            f"Faces: {count} | FPS: {fps:.1f}",
            (15,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0,255,0), 2
        )

        combined = np.hstack(
            (rgb, depth_vis)
        )

        ok, buffer = cv2.imencode(
            ".jpg", combined,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if ok:
            with condition:
                state["jpeg"] = buffer.tobytes()
                state["seq"] += 1
                condition.notify_all()

except KeyboardInterrupt:
    print("\nDeteniendo...")
finally:
    server.shutdown()
    server.server_close()
    pipeline.stop()
