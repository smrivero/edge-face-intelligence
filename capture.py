import cv2
import json
import time
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

OUT = Path("output")
OUT.mkdir(exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth, 640, 480, rs.format.z16, 30
)
config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)

align = rs.align(rs.stream.color)

print("Iniciando RealSense D435...")
profile = pipeline.start(config)

try:
    # Estabilizar autoexposicion
    for i in range(30):
        frames = pipeline.wait_for_frames(5000)

    frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        raise RuntimeError("RGB o Depth no disponible")

    rgb = np.asanyarray(color_frame.get_data())
    raw_depth = np.asanyarray(depth_frame.get_data())

    depth_scale = depth_frame.get_units()
    depth_mm = raw_depth.astype(np.float32) * depth_scale * 1000

    # Visualizacion de 0 a 2500 mm
    normalized = cv2.convertScaleAbs(
        depth_mm, alpha=255.0 / 2500
    )
    depth_vis = cv2.applyColorMap(
        normalized, cv2.COLORMAP_JET
    )
    depth_vis[raw_depth == 0] = (0, 0, 0)

    combined = np.hstack((rgb, depth_vis))

    cv2.imwrite(str(OUT / "rgb.jpg"), rgb)
    cv2.imwrite(str(OUT / "depth.jpg"), depth_vis)
    cv2.imwrite(str(OUT / "combined.jpg"), combined)

    # Estadisticas de la region central
    h, w = raw_depth.shape
    roi = depth_mm[
        h//2-50:h//2+50,
        w//2-50:w//2+50
    ]

    valid = roi[(roi > 250) & (roi < 3000)]

    result = {
        "resolution": f"{w}x{h}",
        "fps_configured": 30,
        "depth_scale_m": depth_scale,
        "valid_depth_percent": round(
            100 * len(valid) / roi.size, 2
        ),
        "central_depth_mm": (
            round(float(np.median(valid)), 1)
            if len(valid) else None
        )
    }

    (OUT / "metrics.json").write_text(
        json.dumps(result, indent=2)
    )

    print("\nRESULTADOS")
    print(json.dumps(result, indent=2))
    print("\nArchivos en:", OUT.resolve())

finally:
    pipeline.stop()
