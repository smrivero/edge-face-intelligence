import cv2
import time
import json
import statistics
import pyrealsense2 as rs
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_sc",
    allowed_modules=["detection"],
    addons=["liveness"],
    liveness_mode="observe",
    liveness_threshold=0.8,
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1, det_size=(640, 480))

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(
    rs.stream.color, 640, 480,
    rs.format.bgr8, 30
)

print("\nIniciando camara...", flush=True)
pipeline.start(config)

times = []
scores = []
live_count = 0
fake_count = 0
rejected_count = 0
no_face_count = 0

try:
    for i in range(30):
        frame = pipeline.wait_for_frames(5000)
        color = frame.get_color_frame()

        if not color:
            continue

        import numpy as np
        image = np.asanyarray(color.get_data())

        t0 = time.perf_counter()
        faces = app.get(image)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

        if not faces:
            no_face_count += 1
            print(f"Frame {i+1}: NO_FACE")
            continue

        face = max(
            faces,
            key=lambda f:
                (f.bbox[2]-f.bbox[0]) *
                (f.bbox[3]-f.bbox[1])
        )

        result = face.liveness

        if result is None:
            raise RuntimeError("Liveness no ejecutado")

        if result.status == "input_rejected":
            rejected_count += 1
        elif result.is_live:
            live_count += 1
        else:
            fake_count += 1

        score = result.live_score

        if score is not None:
            scores.append(float(score))

        print(
            f"Frame {i+1:02d} | "
            f"status={result.status} | "
            f"is_live={result.is_live} | "
            f"score={score} | "
            f"latency={elapsed:.1f}ms",
            flush=True
        )

    print("\n===== RESUMEN =====")

    summary = {
        "frames": len(times),
        "live": live_count,
        "fake": fake_count,
        "input_rejected": rejected_count,
        "no_face": no_face_count,
        "avg_score": (
            round(statistics.mean(scores), 4)
            if scores else None
        ),
        "avg_latency_ms": (
            round(statistics.mean(times), 2)
            if times else None
        ),
        "fps_inference": (
            round(1000/statistics.mean(times), 2)
            if times else None
        )
    }

    print(json.dumps(summary, indent=2))

finally:
    pipeline.stop()
