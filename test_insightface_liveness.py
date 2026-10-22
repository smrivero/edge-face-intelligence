import time
import onnxruntime as ort
from pathlib import Path
from insightface.app import FaceAnalysis

print("===== INSIGHTFACE RGB LIVENESS =====", flush=True)

print("ONNX Runtime:", ort.__version__)
print("Providers:", ort.get_available_providers())

start = time.perf_counter()

app = FaceAnalysis(
    name="buffalo_sc",
    allowed_modules=["detection"],
    addons=["liveness"],
    liveness_mode="observe",
    liveness_threshold=0.8,
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

print(
    "Modelos cargados:",
    list(app.models.keys())
)

print(
    "Addons cargados:",
    list(app.addons.keys())
)

app.prepare(
    ctx_id=0,
    det_size=(640, 480)
)

elapsed = time.perf_counter() - start

print(f"Tiempo de carga: {elapsed:.2f} s")

path = Path.home() / ".insightface/addons/liveness.onnx"

print("Modelo existe:", path.exists())

if path.exists():
    print(
        "Tamaño:",
        round(path.stat().st_size / 1024**2, 2),
        "MB"
    )

print("===== RESULTADO =====")
print("InsightFace RGB Liveness inicializado")
