# Drosophila Embryo Detection and Classification System

**Real-time computer vision and ML pipeline for automated embryo detection, classification, and hardware-triggered sorting.**

---

## 1. Introduction and Purpose

This repository contains the software for an **automated Drosophila embryo detection and classification system** used in a research or sorting setup. The pipeline acquires live video from an AmScope/ToupTek camera (or from a saved video file), detects moving embryos within a defined channel region, classifies each detection as **Correct** or **Incorrect** using a deep learning model, and can send delayed serial commands to an Arduino—for example to trigger a valve or sorter—when a “correct” embryo enters a designated trigger zone. The design supports **dual-zone operation**: one zone for full ML-based keep/discard decisions, and a second zone for low-latency hardware triggering based on those decisions.


---

## 2. High-Level Architecture

The system is built around a central **EmbryoDetector** class (in `main_3.py`) that:

1. **Captures frames** from the camera (via the AmScope SDK) or from a video file.
2. **Preprocesses and detects** embryos in a configurable region of interest (ROI) using background subtraction, morphology, and shape/contrast gates.
3. **Crops** each detection and sends it to a **separate inference process** that runs a ResNet18-based binary classifier.
4. **Consumes classification results** and, in dual-zone mode, records “Keep”/“Discard” decisions for Zone 1 and uses Zone 2 detections to schedule **Arduino signals** with optional delay (latency compensation).
5. **Optionally** saves overlay frames and cropped images to disk and computes **per-object speed** (pixels per frame) for display in the GUI.

Process and thread layout:

- **Main process**: camera/video loop, detection, ROI logic, result handling, GUI (if running via `gui.py`).
- **Inference worker** (separate process): loads the ResNet18 model, reads from a crop queue, runs inference, pushes results to a results queue.
- **Save worker** (optional, separate process): writes overlay and crop images to disk when disk saving is enabled.
- **Arduino sender thread**: waits until scheduled time then writes a byte (e.g. `b'C'`) to the serial port so that triggers are delayed as desired (e.g. to align with physical arrival at a valve).

Data flows between these via **queues**: `crop_queue` (main → inference), `results_queue` (inference → main), `save_queue` (main → save worker). The camera callback must stay non-blocking; heavy work is done in the main loop or in workers.

---

## 3. Repository Structure and Entry Points

| File or folder        | Role |
|-----------------------|------|
| **`gui.py`**          | **Recommended entry point.** Tkinter GUI that instantiates `EmbryoDetector` from `main_3.py`, starts/stops camera or video, and exposes ROI, Zone 2, exposure/gain, and classification log. Run with `python gui.py`. |
| **`main_3.py`**       | Detection/classification script. Maximum-sensitivity detection, dual-zone logic, speed calculation, Arduino trigger in Zone 2 (when Zone 1 has a recent “Keep”). Can run headless or with `--show-window`; supports `--video <path>` for file input. |
| **`amcam.py`**        | ctypes-based wrapper for the AmScope/ToupTek SDK (`libamcam.dylib` on macOS). Handles enumeration, open/close, start/stop, and callback-driven frame delivery (e.g. RGB24). Frames are converted to BGR for OpenCV. |
| **`Valve_Control.py`**| Optional: PicoScope (e.g. 2204A) + Arduino integration for voltage-based triggering and valve control (separate from the camera pipeline). |
| **`GiveVoltage 1.py`**| Simple script to send a byte (e.g. `b'C'`) to the Arduino on keypress; useful for testing serial connectivity. |
| **`ps2000_block_read_min.py`**, **`open_ps2000_diag.py`** | PicoScope 2000 series diagnostics / block read examples; used in conjunction with oscilloscope-style triggering if needed. |
| **`cameraonly.py`**   | Minimal camera test script (no full detection/ML pipeline). |
| **`SPEED_DETECTION_GUIDE.md`** | Describes the speed (px/frame) feature and where it is implemented. |
| **`ARDUINO_DEBUG_GUIDE.md`**  | Debugging guide for the Arduino delayed-trigger path (queue format, logging, serial checks). |
| **`.github/copilot-instructions.md`** | Concise architecture and coding notes for AI assistants. |

There is no committed **model file** in the repo. The inference worker expects a file named **`resnet18_model.pth`** in the **project root**. That model should be a ResNet18 with the final fully connected layer replaced to output 2 classes: `['Incorrect', 'Correct']`. Input: 224×224 RGB, ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`). You must train or obtain this model separately and place it in the repo root before running classification.

---

## 4. Detection Pipeline (Conceptual)

The pipeline below is implemented in `main_3.py`.

1. **ROI**  
   Only a horizontal “channel” strip of the frame is processed. By default this is set to about 35%–65% of frame height (configurable by fraction or by pixel x, y, width, height). The ROI is intended to match the physical channel where embryos flow.

2. **Preprocessing**  
   - CLAHE on the L channel (LAB) for local contrast.  
   - Light blur.  
   - Two MOG2 background subtractors (one more stable, one more sensitive) with different learning rates; the sensitive one is used more after camera warmup.  
   - Optional color-based sensitivity: LAB background accumulation and ΔE mask gated by gradient.  
   - Morphology: median blur, light erosion, open/close (isotropic and horizontal kernels), dilation.  

3. **Contours and gates**  
   Contours are found on the processed ROI. Each contour is filtered by:
   - Area (min/max bandpass, sometimes scaled by resolution).  
   - Intensity contrast and gradient (center and edge).  
   - Shape: solidity, circularity, aspect ratio.  
   - Optional “wall strip” logic for objects near vertical edges.  
   - Motion: minimum displacement over a few frames to reject stationary debris (e.g. dust).  

4. **Deduplication and caps**  
   Same-frame IoU/proximity deduplication and a per-frame cap on accepted detections.  

5. **Crops and inference**  
   For each accepted detection, a crop is taken (in ROI/full-frame coordinates as appropriate), resized to 224×224, normalized, and sent to the inference worker. Results (label, timing, optional speed and object_id) are consumed in the main thread and reflected in the GUI and in Arduino scheduling when applicable.

6. **Speed**  
   Detections are matched across frames (e.g. within 100 px); each object gets an ID and a short position history. Speed is computed as distance / frame difference (pixels per frame), then smoothed (e.g. average of last N measurements) and shown in the GUI classification log.

---

## 5. Dual-Zone System (main_3.py)

- **Zone 1**  
  The main ROI used for detection and ML. Every classification is recorded as a “Keep” (Correct) or “Discard” (Incorrect) with a timestamp. A sliding window (e.g. 2 s) of these decisions is kept; if there is at least one recent “Keep,” Zone 1 is considered “active” for triggering.

- **Zone 2**  
  A separate region (by default the rightmost ~35% of the frame width, same vertical span as the main ROI) with its own background subtractor and simpler motion+size criteria. When an object is detected in Zone 2 **and** Zone 1 has a recent “Keep,” the system schedules an Arduino signal with configurable delay (latency compensation).  
  So: Zone 1 decides *whether* an embryo is correct; Zone 2 decides *when* it has reached the trigger point, and the delay aligns the serial command with the physical arrival at the valve/sorter.

Zone 2 can be enabled/disabled and its rectangle (x, y, width, height) and the Zone 1 decision window are adjustable in the GUI.

---

## 6. Arduino and Serial

- **Role**  
  The pipeline can send a single byte (e.g. `b'C'`, defined as `CORRECT_EMBRYO_SIGNAL`) over a serial port to an Arduino to trigger an actuator (e.g. valve).

- **Scheduling**  
  To compensate for detection and processing latency, the signal is **delayed** by `max(0, T_target - T_detect)` ms. So if the desired trigger time is 1000 ms after “decision” and detection took 15 ms, the byte is sent 985 ms after the decision. A dedicated thread waits until the due time then writes to the serial port so the main loop is not blocked.

- **Cooldown**  
  A cooldown (e.g. 500 ms) limits how often triggers can occur.

- **Configuration**  
  In `main_3.py`, set `SERIAL_PORT` to your Arduino’s port (e.g. `'COM3'` on Windows, `/dev/cu.usbmodem*` on macOS) and `BAUD_RATE` (e.g. 9600). If `pyserial` is not installed, serial is disabled and no Arduino code path runs.

- **Debugging**  
  See **`ARDUINO_DEBUG_GUIDE.md`** for queue format, thread startup, and serial checks. The guide also explains the fix for the earlier queue unpacking bug (2-tuple vs 3-tuple with sequence number).

---

## 7. Dependencies and Environment

- **Python**  
  Use a recent 3.x (3.9+ recommended).

- **Pip packages** (see **`requirements.txt`**):
  - `numpy`, `opencv-python`, `pillow`
  - `torch`, `torchvision` (versions must match each other)
  - `pyserial` for Arduino

- **AmScope/ToupTek camera**  
  The `amcam` module is not installable via pip. You must:
  - Install the vendor’s SDK/drivers for your OS.
  - Ensure `libamcam.dylib` (macOS) is on the library path and that the Python process can load it.
  - The repo’s `amcam.py` is the wrapper; the manual (e.g. `doc/en.html`) applies to the C API that this wrapper calls.

- **Model file**  
  Place **`resnet18_model.pth`** in the project root. The inference worker loads it with `torch.load(..., map_location=device)` and uses the same device as the rest of the app (MPS → CUDA → CPU).

---

## 8. How to Run

1. **Install dependencies**  
   `pip install -r requirements.txt`  
   Install the AmScope SDK and ensure the camera is connected and visible to the OS.

2. **Add the model**  
   Put `resnet18_model.pth` in the repo root (2-class ResNet18 as described above).

3. **GUI (camera or video)**  
   ```bash
   python gui.py
   ```  
   Start the camera, then enable “Start Detection.” Or use “Load Video File” to run on a video; the same pipeline (including speed and dual-zone) applies.

4. **Headless / video from command line**  
   ```bash
   python main_3.py --video /path/to/video.mp4
   ```  
   Optional: `--show-window` to show the feed (may not work in headless environments).

5. **Arduino**  
   Set `SERIAL_PORT` (and optionally `BAUD_RATE`, `arduino_target_delay_ms`, `trigger_cooldown_ms`) in `main_3.py`. Connect the Arduino before starting so the sender thread can open the port and log correctly.

6. **Disk saving**  
   Set `ENABLE_DISK_SAVING = True` in `main_3.py` to write overlays and crops to `live_process/frames/` and `live_process/cropped/`. Disable for maximum performance.

---

## 9. Configuration Highlights

- **ROI**  
  In the GUI: ROI panel (pixel x, y, width, height or legacy fraction-based). In code: `_initialize_common` (fractions), `update_roi()`.

- **Zone 2**  
  GUI: “Zone 2 - Trigger Zone” (enable, x, y, width, height, decision window ms). In code: `update_zone2_roi()`, `update_zone1_decision_window()`.

- **Detection sensitivity**  
  In `main_3.py`: MOG2 parameters, area bandpass, contrast/gradient/solidity/circularity/aspect ratio, motion filter (`min_motion_px`, `motion_check_frames`), `MAX_ACCEPTS_PER_FRAME`.

- **Speed**  
  See **`SPEED_DETECTION_GUIDE.md`**. Key: match radius (e.g. 100 px), position history length, and number of samples for averaging. Speed is in pixels per frame; multiply by frame rate for pixels per second.

- **Inference**  
  `INFERENCE_BATCH_SIZE` (e.g. 1 for lowest latency), device selection (MPS/CUDA/CPU is automatic).

---

## 10. Notes and Caveats

- **Platform**  
  Camera path is written for macOS (AmScope SDK + `libamcam.dylib`). Other platforms would need the corresponding SDK and library name (e.g. `libamcam.so`, DLL).

- **Port names**  
  `SERIAL_PORT` is often set to `'COM3'` in the repo; change it to match your Arduino (e.g. `/dev/cu.usbmodem*` on macOS).

- **Accuracy**  
  Detection and classification accuracy depend on lighting, optics, ROI alignment, and model training. Tune ROI and Zone 2 to your rig; use saved crops and overlays (with disk saving) to inspect false positives/negatives.

- **Performance**  
  Keep heavy work out of the camera callback. Avoid enabling disk saving or debug branches in hot paths if you need maximum frame rate.

---

## 11. References and Further Reading

- **In-repo**: `SPEED_DETECTION_GUIDE.md`, `ARDUINO_DEBUG_GUIDE.md`, `.github/copilot-instructions.md`.
- **AmScope/ToupTek**: vendor SDK documentation and `doc/en.html` (or equivalent) for the C API that `amcam.py` wraps.
- **PyTorch / ResNet**: standard ImageNet preprocessing and `torchvision.models.resnet18` for the backbone and custom classifier head.

---

*This README was written to reflect the codebase as of the current revision. If you extend the system (e.g. new hardware or another classifier), updating this document and the copilot instructions will help future researchers and assistants.*
