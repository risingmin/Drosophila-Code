# AMCAM Embryo Detection System - AI Coding Instructions

## Architecture Overview

This is a real-time computer vision system for automated embryo detection and classification using AmScope/ToupTek cameras. The system combines:

- **Hardware Integration**: AmScope camera SDK (`amcam.py`) for live video capture
- **Computer Vision Pipeline**: Multi-stage detection using background subtraction, edge detection, and morphological operations
- **ML Inference**: ResNet18-based classification running in separate processes
- **Dual Processing Modes**: Live camera feed or video file analysis

## Core Components

### `main.py` - Main Application

- **EmbryoDetector class**: Central orchestrator managing camera, detection pipeline, and multiprocessing
- **Detection Pipeline**: ROI-based processing with dual background subtractors (main + sensitive)
- **Temporal NMS**: Sophisticated non-maximum suppression across frames with region cooldowns
- **Edge Priority System**: Special handling for objects near frame edges using tripwires at 10%/90% width

### `amcam.py` - Camera SDK Wrapper

- **Low-level camera interface**: ctypes-based wrapper for AmScope/ToupTek native libraries
- **Callback-driven**: Uses `_event_callback` → `_camera_callback` for frame processing
- **RGB24 format**: Frames come as RGB, convert to BGR for OpenCV consistency in video mode

## Critical Detection Parameters

```python
# AMCAM Embryo Detection System — AI coding guide

This repo runs real-time embryo detection on AmScope/ToupTek cameras, with optional ML classification. Two entry points exist with different trade‑offs; pick the one that matches your task.

Architecture and processes
- Core: `EmbryoDetector` in `main.py` and `main_3.py` drives camera/video input, detection, and multiprocessing.
- Workers: separate processes for inference and (optionally) disk I/O. Queues: `crop_queue` (main→inference), `results_queue` (inference→main), `save_queue` (main→save).
- Camera: `amcam.py` wraps the native SDK via ctypes. Camera frames arrive as RGB24; always convert to BGR before OpenCV ops.

Two run modes and their differences
- `main.py` (balanced): temporal NMS across frames, region cooldowns, shape gates, and Arduino trigger only when inference label == 'Correct'.
- `main_3.py` (maximum sensitivity/low‑latency): permissive gates, no temporal NMS (same‑frame dedup only), `INFERENCE_BATCH_SIZE=1`, optional `--show-window`, Arduino trigger on detection stage (not gated by inference result).

How to run (macOS; SDK required for camera)
- Camera: `python3 main_3.py` (or `python3 main.py`). Optional: `--show-window` (may fail headless; default is headless saving to disk).
- Video file: `--video path/to/video.mp4` on either entry point.
- Model: `resnet18_model.pth` in repo root is loaded by the inference worker; classes: ['Incorrect','Correct'] with 224x224 + ImageNet normalization.

Detection pipeline (both variants)
- ROI is the horizontal channel: approximately 35%–65% of frame height (`_initialize_common`), saved overlay on first frame.
- Enhancement: CLAHE on L channel (LAB) → mild blur → dual MOG2 subtractors (stable + sensitive) with different learning rates (camera warmup vs steady state).
- Optional color sensitivity after camera warmup: LAB background accumulation → ΔE mask gated by gradient.
- Morphology: median blur → light erosion → open/close (isotropic + horizontal) → dilate.
- Shape/contrast gates: intensity delta, gradient, solidity, circularity, aspect ratio; side‑margin suppression near vertical borders.
- Dedupe: `main.py` uses temporal NMS window; `main_3.py` only same‑frame IoU/proximity dedup plus per‑frame cap.

Integration notes and gotchas
- AmScope SDK: `libamcam.dylib` must be present/compatible; `amcam.py` allocates a RGB24 buffer and pulls frames via `PullImageV3`. Keep heavy work out of the camera callback.
- Device: selects MPS (Apple Silicon) > CUDA > CPU automatically. Torch/torchvision versions must match (`requirements.txt`).
- Disk I/O: set `ENABLE_DISK_SAVING=True` to save overlays/crops under `live_process/frames` and `live_process/cropped`; disable for max performance.
- Arduino: update `SERIAL_PORT` in the chosen entry point. `main.py` writes only on 'Correct'; `main_3.py` writes on any detection.

Useful code anchors
- ROI setup and scaling: `EmbryoDetector._initialize_common`.
- Camera callback and RGB→BGR fix: `EmbryoDetector._event_callback` in both mains.
- Crop production for ML: end of `process_frame` (mask→boundingRect→crop→queue).

When editing
- Preserve multiprocessing and queue contracts; never block in the camera callback.
- Keep ROI coordinates consistent (ROI→full‑frame y offset when promoting contours).
- If you add debugging, guard it behind `ENABLE_DISK_SAVING` or `--show-window` and avoid extra per‑frame allocations in hot paths.

Questions or gaps? If you need clarification on which entry point to standardize on, Arduino gating behavior, or SDK setup expectations, call it out and we’ll refine these rules.
# Communication via Queues:
```
