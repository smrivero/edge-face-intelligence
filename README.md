# Edge Face Intelligence: Multi-Model Facial Analysis

Local facial recognition and multimodel computer vision on NVIDIA Jetson Orin Nano Super.

**Face detection, identity embeddings, multi-person recognition and TensorRT-accelerated facial-attribute analysis.**

This project implements several local inference workflows using YuNet, ArcFace, InsightFace and TensorRT. Camera-based applications support identity enrollment, embedding comparison and browser-based visualization.

Facial attributes and experimental liveness analysis are implemented as separate components.

## Overview

Edge Face Intelligence is a collection of GPU-accelerated computer vision components for local face analysis.

The implementation combines OpenCV, ArcFace, InsightFace, TensorRT and Intel RealSense camera processing across several application entry points.

## Capabilities

- Face detection using YuNet.
- Face embeddings using ArcFace.
- Enrollment and comparison of registered identities.
- Multi-person recognition.
- Real-time browser-based camera previews.
- Experimental RGB and depth-based liveness analysis.
- Separate age and gender appearance classification with TensorRT.

The capabilities are implemented across different scripts and are not presented as one fully unified production service.

## Architecture

Camera frames are processed locally. Facial detections are used for alignment and feature extraction.

For identity recognition, embeddings are compared with locally enrolled reference embeddings.

Facial-attribute estimation uses a separate TensorRT engine.

Browser interfaces expose video streams and application status.

## Components

| File | Purpose |
|---|---|
| `face_recognition.py` | Identity enrollment and recognition |
| `multi_person_recognition.py` | Multi-identity enrollment and matching |
| `face_pipeline.py` | Combined face analysis workflow |
| `gender_age_tensorrt.py` | GPU-accelerated facial attributes |
| `depth_liveness.py` | Experimental depth-based analysis |
| `rgb_liveness.py` | Experimental RGB liveness |
| `capture.py` | Camera capture |
| `live_face.py` | Live face detection |

## Engineering Highlights

- Multi-model computer vision pipelines.
- Face embeddings and similarity-based identity matching.
- Multiple registered identities and unknown-face handling.
- TensorRT acceleration for facial attributes.
- Real-time camera processing and browser-based interfaces.
- Local biometric data handling and experimental liveness analysis.

## Requirements

- NVIDIA Jetson Orin Nano Super 8 GB
- Intel RealSense D435
- Python
- OpenCV with appropriate DNN support
- NumPy
- Intel RealSense Python SDK
- ONNX Runtime
- InsightFace for relevant entry points
- NVIDIA TensorRT for GPU attribute inference

The local model directory must contain the required models for the selected entry point.

Examples include:

    models/yunet.onnx
    models/arcface.onnx
    models/genderage-fp16.engine

Additional InsightFace assets may be needed for the liveness components.

Model assets and private identity templates are excluded from Git.

## Execution

Run multi-person recognition:

    python3 multi_person_recognition.py

Run facial-attribute inference:

    python3 gender_age_tensorrt.py

Run the face recognition application:

    python3 face_recognition.py

Camera access, model files and compatible runtime dependencies must be configured before execution.

## Privacy

All inference is designed to execute locally.

Enrolled biometric templates, photographs, identity databases and camera recordings are excluded from this repository.

Registration should only be performed with participant consent.

## Limitations

Identity thresholds require calibration for the target environment.

Depth and RGB liveness implementations are experimental and are not validated anti-spoofing security systems.

Age estimation is approximate. Gender outputs are model-based appearance classifications and do not establish personal identity.

No reproducible recognition accuracy or false-match benchmark is claimed.
