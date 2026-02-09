# main.py
# Combines the AmScope camera SDK with the embryo detection and sorting logic.
# This version keeps ALL structure from the provided main.py and swaps in the
# "detect_emryo"-style single-frame detection pipeline (edge-assist + gating + tNMS)
# from detect_embryo.py. Serial communication remains disabled.
 
# --- Core Python & Computer Vision Libs ---
import cv2
import numpy as np
import time
import os
import sys
from collections import deque
 
# --- AmScope Camera SDK ---
import amcam
import ctypes
 
# --- PyTorch for Inference ---
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
 
# --- Multiprocessing & Arduino Communication ---
from multiprocessing import Process, Queue, Event
from queue import Empty
import torch.multiprocessing as mp
# import serial # --- DISABLED ---
 
# ===================================================================
# === CONFIGURATION (UNCHANGED FROM main.py EXCEPT WHERE NEEDED) ===
# ===================================================================
 
# --- Detection & Morphology (relaxed for better sensitivity) ---
MIN_AREA = 6000              # Reduced from 8000 to catch smaller objects
MOG2_HISTORY = 100
MOG2_VAR_THRESHOLD = 40      # Reduced from 60 for more sensitive background subtraction
MOG2_DETECT_SHADOWS = False
DILATION_ITERATIONS = 3
 
# --- Performance & Feature Toggles ---
ENABLE_DISK_SAVING = False  # Set to False for max performance, True to save files
INFERENCE_BATCH_SIZE = 8    # Number of images to batch for the model
 
# --- Worker & I/O Paths ---
if ENABLE_DISK_SAVING:
    CROPPED_OUTPUT_DIR = "live_process/cropped/"
    FRAME_OUTPUT_DIR = "live_process/frames/"
    os.makedirs(CROPPED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
 
# ===================================================================
# === WORKER PROCESSES (UNCHANGED) ===
# ===================================================================
 
def save_worker(save_queue, stop_event):
    print("Save worker started.")
    while not stop_event.is_set():
        try:
            save_type, path, image = save_queue.get(timeout=1)
            cv2.imwrite(path, image)
        except Empty:
            continue
        except Exception as e:
            print(f"Save worker error: {e}")
 
def inference_worker(model_path, crop_queue, results_queue, stop_event, device):
    class_names = ['Incorrect', 'Correct']
    infer_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Inference worker started on device: {device}")
 
    while not stop_event.is_set():
        try:
            batch_data = crop_queue.get(timeout=1)
            images, frame_indices = zip(*batch_data)
            tensors = [infer_transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) for img in images]
            input_batch = torch.stack(tensors).to(device)
 
            with torch.no_grad():
                outputs = model(input_batch)
                _, preds = torch.max(outputs, 1)
 
            for i in range(len(images)):
                results_queue.put({
                    'frame_index': frame_indices[i],
                    'inference_label': class_names[preds[i].item()]
                })
        except Empty:
            continue
        except Exception as e:
            print(f"Inference worker error: {e}")
 
# ===================================================================
# === MAIN APPLICATION CLASS ===
# ===================================================================
 
class EmbryoDetector:
    def __init__(self):
        # --- Handles and State ---
        self.hcam = None
        # self.ser = None # --- DISABLED ---
        self.frame_count = 0
        self.detection_count = 0
        self.crop_batch = []
 
        # --- Multiprocessing ---
        self.stop_event = Event()
        self.crop_queue = Queue(maxsize=50)
        self.results_queue = Queue()
        self.save_queue = Queue(maxsize=100)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
 
        # --- OpenCV objects (base subtractor kept) ---
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=MOG2_DETECT_SHADOWS)
 
        # === detect_emryo pipeline additions (from detect_embryo.py) ===
        # Secondary, more sensitive subtractor (made even more sensitive)
        self.bg_subtractor_sens = cv2.createBackgroundSubtractorMOG2(history=15, varThreshold=35, detectShadows=False)
 
        # ROI setup is finalized in _initialize_camera()
        self.roi_start_y = 0
        self.roi_end_y = 0
        self.height = 0
        self.width = 0
 
        # Morphology
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel_iso = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.close_kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 5))
 
        # Edge assist & gating parameters (relaxed for better detection)
        self.AREA_MIN_BANDPASS = 8000  # Reduced from 12000 to catch smaller embryos
        self.AREA_MAX_BANDPASS = 50000  # Increased from 40000 to catch larger embryos
        self.CONTRAST_MIN_CENTER = 2  # Reduced from 3 for lower contrast embryos
        self.CONTRAST_MIN_EDGE = 5    # Reduced from 8 for edge detection
        self.GRAD_MIN_CENTER = 4      # Reduced from 6 for subtle gradients
        self.GRAD_MIN_EDGE = 6        # Reduced from 9 for edge gradients
        self.SOLIDITY_MIN = 0.70          # Reduced from 0.80 for irregular shapes
        self.CIRCULARITY_RANGE = (0.25, 0.95)  # Widened from (0.35, 0.92)
        self.ASPECT_RATIO_MIN = 0.25      # Reduced from 0.35 for elongated embryos
        self.EDGE_SOLIDITY_MIN = 0.75     # Reduced from 0.82
        self.EDGE_CIRC_MIN = 0.35         # Reduced from 0.42
        self.EDGE_BAND_FRACTION = 0.15
        self.TNMS_WINDOW_FRAMES = 12
        self.TNMS_IOU_THRESH = 0.55
        self.TNMS_DX = 80
        self.TNMS_DY = 50
        self.TNMS_DX_EDGE = 120
        self.SINGLE_SHOT_COOLDOWN_FRAMES = 3
        self.WALL_STRIP_PX = 12
        self.WALL_STRIP_OVERLAP_MAX = 0.40     # Increased from 0.30 to be less restrictive
        self.WALL_STRIP_STRICT_DELTAI = 8      # Reduced from 12 to be more lenient
        self.MAX_ACCEPTS_PER_FRAME = 3
 
        self.prev_roi_gray = None
        self.recent_accepts = deque(maxlen=self.TNMS_WINDOW_FRAMES)
        self.region_cooldowns = deque()  # (cx, cy, expire_frame)
 
        # Tripwires set in camera init once width known
        self.left_tripwire_x = 0
        self.right_tripwire_x = 0
 
    # ------------------------ camera init -------------------------
    def _initialize_camera(self):
        print("Searching for camera...")
        devices = amcam.Amcam.EnumV2()
        if len(devices) <= 0:
            print("ERROR: No camera found. Please check connection.")
            return False
 
        selected_camera = devices[0]
        print(f"Found camera: {selected_camera.displayname}")
 
        self.hcam = amcam.Amcam.Open(selected_camera.id)
        if not self.hcam:
            print(f"ERROR: Failed to open camera: {selected_camera.displayname}")
            return False
 
        width, height = self.hcam.get_Size()
        self.hcam.put_eSize(0)
        self.width, self.height = width, height
 
        # Narrow horizontal ROI band (0.15 H) centered vertically, as in detect script
        roi_height_fraction = 0.15
        roi_half = int(height * roi_height_fraction / 2)
        cy = height // 2
        self.roi_start_y, self.roi_end_y = cy - roi_half, cy + roi_half
 
        # Tripwires for edge-priority
        self.left_tripwire_x = int(self.width * 0.10)
        self.right_tripwire_x = int(self.width * 0.90)
 
        print(f"Camera opened successfully at {width}x{height}.")
        print(f"ROI: y={self.roi_start_y}..{self.roi_end_y} (h={self.roi_end_y - self.roi_start_y})")
        return True
 
    # --------------------- helpers (from detect) -------------------
    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2-x1)*(y2-y1)
        union = aw*ah + bw*bh - inter
        return inter/union if union > 0 else 0.0
 
    @staticmethod
    def _contour_shape_metrics(cnt):
        A = cv2.contourArea(cnt)
        if A <= 0:
            return 0, 0, 0
        hull = cv2.convexHull(cnt)
        hullA = cv2.contourArea(hull)
        P = cv2.arcLength(cnt, True)
        circularity = (4*np.pi*A)/(P*P + 1e-6)
        solidity = A / (hullA + 1e-6)
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            aratio = 0
        else:
            short, long_ = (min(w, h), max(w, h))
            aratio = short/long_
        return solidity, circularity, aratio
 
    @staticmethod
    def _mean_intensity_delta(gray, mask):
        dil = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
        ring = cv2.subtract(dil, mask)
        inner_vals = gray[mask > 0]
        outer_vals = gray[ring > 0]
        if inner_vals.size == 0 or outer_vals.size == 0:
            return 0.0
        return float(np.mean(outer_vals) - np.mean(inner_vals))
 
    @staticmethod
    def _mean_gradient_inside(gray, mask):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        vals = mag[mask > 0]
        if vals.size == 0:
            return 0.0
        return float(np.mean(vals))
 
    def _in_region_cooldown(self, cx, cy, frame_idx):
        while self.region_cooldowns and self.region_cooldowns[0][2] <= frame_idx:
            self.region_cooldowns.popleft()
        for rx, ry, exp in self.region_cooldowns:
            if abs(rx - cx) <= 50 and abs(ry - cy) <= 50:
                return True
        return False
 
    def _add_region_cooldown(self, cx, cy, frame_idx):
        self.region_cooldowns.append((cx, cy, frame_idx + 24))
 
    # --------------------- callback (detection) --------------------
    @staticmethod
    def _event_callback(nEvent, pContext):
        detector = pContext
        if not detector or not detector.hcam:
            return
 
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            try:
                # Frame info struct
                info = amcam.AmcamFrameInfoV3()
 
                # Get size
                width, height = detector.hcam.get_FinalSize()
 
                # Allocate raw buffer (RGB24 → 3 bytes per pixel)
                nbytes = width * height * 3
                cbuf = ctypes.create_string_buffer(nbytes)
 
                # Call PullImageV3 with proper types
                detector.hcam.PullImageV3(
                    cbuf,  # c_char_p-compatible buffer
                    0,  # bStill
                    24,  # bits (RGB24)
                    0,  # rowPitch
                    info  # Frame info struct
                )
 
                # Convert to NumPy array
                frame = np.frombuffer(cbuf, dtype=np.uint8).reshape((height, width, 3))
 
                # Forward to your pipeline
                EmbryoDetector._camera_callback(frame, info, detector)
 
            except Exception as e:
                print(f"Error pulling image: {e}")
 
    @staticmethod
    def _camera_callback(pData, pInfo, pContext):
        detector = pContext
        if not detector:
            return
        if pInfo.width <= 0 or pInfo.height <= 0:
            return
 
        frame_data = np.ctypeslib.as_array(pData, shape=(pInfo.height, pInfo.width, 3))
        frame = frame_data.copy()
        detector.frame_count += 1
        fc = detector.frame_count
 
        # --- ROI extraction ---
        roi_frame = frame[detector.roi_start_y:detector.roi_end_y, :]
 
        # Local contrast boost (CLAHE on L-channel)
        lab = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
        roi_enhanced = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
 
        # Background subtraction (two streams)
        fg_main = detector.bg_subtractor.apply(roi_enhanced, learningRate=0.02)
        fg_sens = detector.bg_subtractor_sens.apply(roi_enhanced, learningRate=0.04)
        fg_mask = cv2.bitwise_or(fg_main, fg_sens)
 
        # Edge differencing assist (made more sensitive)
        roi_gray = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2GRAY)
        assist = None
        if detector.prev_roi_gray is not None:
            diff = cv2.absdiff(roi_gray, detector.prev_roi_gray)
            _, diff_thresh = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)  # Reduced from 18 to 12
            band_w = int(roi_frame.shape[1] * detector.EDGE_BAND_FRACTION)
            assist = np.zeros_like(diff_thresh)
            assist[:, :band_w] = diff_thresh[:, :band_w]
            assist[:, -band_w:] = diff_thresh[:, -band_w:]
            fg_mask = cv2.bitwise_or(fg_mask, assist)
        detector.prev_roi_gray = roi_gray
 
        # Noise reduction & smoothing
        fg_mask = cv2.medianBlur(fg_mask, 5)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, detector.open_kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, detector.close_kernel_iso, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, detector.close_kernel_h, iterations=1)
        fg_mask = cv2.dilate(fg_mask, detector.dilation_kernel, iterations=DILATION_ITERATIONS)
 
        # Find contours in ROI coords
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        # Build valid polygons in FULL-FRAME coords
        valid_polys = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            approx_full = approx.copy()
            approx_full[:, 0, 1] += detector.roi_start_y
            valid_polys.append((approx_full, area))
 
        if not valid_polys:
            return
 
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # Precompute wall strips (full frame)
        wall_top_y1 = max(detector.roi_start_y - detector.WALL_STRIP_PX, 0)
        wall_top_y2 = detector.roi_start_y + detector.WALL_STRIP_PX
        wall_bot_y1 = detector.roi_end_y - detector.WALL_STRIP_PX
        wall_bot_y2 = min(detector.roi_end_y + detector.WALL_STRIP_PX, detector.height)
 
        # Candidate list (single-frame primary)
        candidates = []
        width = detector.width
 
        for poly, area in valid_polys:
            if not (detector.AREA_MIN_BANDPASS <= area <= detector.AREA_MAX_BANDPASS):
                continue
 
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
 
            deltaI = detector._mean_intensity_delta(gray_full, mask)
            deltaI_abs = abs(deltaI)
            grad_mean = detector._mean_gradient_inside(gray_full, mask)
 
            M = cv2.moments(poly)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
 
            in_edge_band = (cx <= int(width * detector.EDGE_BAND_FRACTION)) or (cx >= int(width * (1.0 - detector.EDGE_BAND_FRACTION)))
 
            # Edge-specific: require assist evidence (relaxed threshold)
            if in_edge_band and assist is not None:
                roi_mask = mask[detector.roi_start_y:detector.roi_end_y, :]
                if roi_mask.dtype != np.uint8:
                    roi_mask = roi_mask.astype(np.uint8)
                roi_mask = (roi_mask > 0).astype(np.uint8) * 255
                assist_overlap = cv2.bitwise_and(assist, roi_mask)
                if int(np.count_nonzero(assist_overlap)) < 80:  # Reduced from 120 to 80
                    continue
 
            # Minimal abs-contrast + gradient gates (more permissive)
            if in_edge_band:
                if deltaI_abs < detector.CONTRAST_MIN_EDGE and grad_mean < detector.GRAD_MIN_EDGE:  # Changed OR to AND
                    continue
            else:
                if deltaI_abs < detector.CONTRAST_MIN_CENTER and grad_mean < detector.GRAD_MIN_CENTER:
                    continue
 
            # Wall strip overlap veto
            mask_pixels = np.count_nonzero(mask)
            if mask_pixels > 0:
                top_overlap = np.count_nonzero(mask[wall_top_y1:wall_top_y2, :])
                bot_overlap = np.count_nonzero(mask[wall_bot_y1:wall_bot_y2, :])
                overlap_frac = (top_overlap + bot_overlap) / float(mask_pixels)
                if overlap_frac > detector.WALL_STRIP_OVERLAP_MAX and deltaI_abs < detector.WALL_STRIP_STRICT_DELTAI:
                    continue
 
            # Shape gates
            cnt = poly.reshape(-1, 1, 2)
            solidity, circularity, aratio = detector._contour_shape_metrics(cnt)
            if not in_edge_band:
                if solidity < detector.SOLIDITY_MIN:
                    continue
                if not (detector.CIRCULARITY_RANGE[0] <= circularity <= detector.CIRCULARITY_RANGE[1]):
                    continue
                if aratio < detector.ASPECT_RATIO_MIN:
                    continue
            else:
                if not (solidity >= detector.EDGE_SOLIDITY_MIN or circularity >= detector.EDGE_CIRC_MIN):
                    continue
 
            x, y, w, h = cv2.boundingRect(mask)
            candidates.append({
                "poly": poly,
                "area": area,
                "bbox": (x, y, w, h),
                "centroid": (cx, cy),
                "edge_priority": min(abs(cx - detector.left_tripwire_x), abs(cx - detector.right_tripwire_x)),
                "in_edge_band": in_edge_band
            })
 
        if not candidates:
            return
 
        # Sort by edge-priority (closer to tripwires first)
        candidates.sort(key=lambda c: c["edge_priority"])
 
        # Temporal NMS + region cooldown + per-frame cap
        accepts = []
        for cand in candidates:
            if len(accepts) >= detector.MAX_ACCEPTS_PER_FRAME:
                break
            cx, cy = cand["centroid"]
            bbox = cand["bbox"]
            in_edge_band = cand["in_edge_band"]
 
            if detector._in_region_cooldown(cx, cy, fc):
                continue
 
            # Against recent memory
            duplicate = False
            for item in list(detector.recent_accepts):
                if fc - item["frame"] > detector.TNMS_WINDOW_FRAMES:
                    continue
                dx_allow = detector.TNMS_DX_EDGE if in_edge_band else detector.TNMS_DX
                if EmbryoDetector._iou(bbox, item["bbox"]) > detector.TNMS_IOU_THRESH:
                    duplicate = True; break
                icx, icy = item["centroid"]
                if abs(icx - cx) < dx_allow and abs(icy - cy) < detector.TNMS_DY:
                    duplicate = True; break
            if duplicate:
                continue
 
            # Against same-frame accepts
            for item in accepts:
                dx_allow = detector.TNMS_DX_EDGE if (in_edge_band or item.get("in_edge_band")) else detector.TNMS_DX
                if EmbryoDetector._iou(bbox, item["bbox"]) > detector.TNMS_IOU_THRESH:
                    duplicate = True; break
                icx, icy = item["centroid"]
                if abs(icx - cx) < dx_allow and abs(icy - cy) < detector.TNMS_DY:
                    duplicate = True; break
            if duplicate:
                continue
 
            accepts.append(cand)
            detector.recent_accepts.append({"frame": fc, "bbox": bbox, "centroid": (cx, cy)})
            detector._add_region_cooldown(cx, cy, fc)
 
        if not accepts:
            return
 
        detector.detection_count += len(accepts)
 
        # Optional debug saving
        if ENABLE_DISK_SAVING:
            dbg = frame.copy()
            for acc in accepts:
                hull = cv2.convexHull(acc["poly"])
                cv2.polylines(dbg, [hull], True, (0, 255, 0), 2)
                cx, cy = acc["centroid"]
                cv2.circle(dbg, (cx, cy), 4, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(FRAME_OUTPUT_DIR, f"embryo_{fc:06d}.png"), dbg)
 
        # Enqueue crops for inference (batching unchanged)
        for i, acc in enumerate(accepts):
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [acc["poly"]], 255)
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            x, y, w, h = cv2.boundingRect(mask)
            crop = masked[y:y+h, x:x+w]
 
            if ENABLE_DISK_SAVING:
                cv2.imwrite(os.path.join(CROPPED_OUTPUT_DIR, f"embryo_{fc:06d}_{i}.png"), crop)
 
            detector.crop_batch.append((crop.copy(), fc))
 
        if len(detector.crop_batch) >= INFERENCE_BATCH_SIZE and not detector.crop_queue.full():
            detector.crop_queue.put(detector.crop_batch)
            detector.crop_batch = []
 
    # -------------------------- run loop --------------------------
    def run(self):
        if not self._initialize_camera():
            return
 
        print(f"Main process using device: {self.device}")
        inference_process = Process(target=inference_worker, args=("resnet18_model.pth", self.crop_queue, self.results_queue, self.stop_event, self.device))
        inference_process.start()
 
        save_process = None
        if ENABLE_DISK_SAVING:
            save_process = Process(target=save_worker, args=(self.save_queue, self.stop_event))
            save_process.start()
 
        try:
            self.hcam.StartPullModeWithCallback(EmbryoDetector._event_callback, self)
            print("\nCamera stream started. Detecting embryos... Press Ctrl+C to stop.")
 
            while not self.stop_event.is_set():
                try:
                    result = self.results_queue.get(timeout=1)
                    print(f"Inference Result (Frame {result['frame_index']}): {result['inference_label']}")
                except Empty:
                    continue
        except KeyboardInterrupt:
            print("\nShutdown signal received.")
        finally:
            print("Cleaning up resources...")
            self.stop_event.set()
            if self.hcam:
                self.hcam.Close()
                print("Camera closed.")
            inference_process.join(timeout=5)
            if save_process:
                save_process.join(timeout=5)
            if inference_process.is_alive(): inference_process.terminate()
            if save_process and save_process.is_alive(): save_process.terminate()
            print("Program finished.")
 
 
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    app = EmbryoDetector()
    app.run()
 
 