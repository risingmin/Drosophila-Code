# main.py
# --- Core Python & Computer Vision Libs ---
import cv2
import numpy as np
import time
import threading
import queue
import os
import sys
import argparse
from typing import Any, Optional

# Ensure the project directory is in the Python path for amcam import
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

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
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    serial = None  # Placeholder to avoid NameError

# ===================================================================
# === CONFIGURATION (UNCHANGED FROM main.py EXCEPT WHERE NEEDED) ===
# ===================================================================

# --- Detection & Morphology (EMBRYO-FOCUSED SETTINGS) ---
MIN_AREA = 2500                        # Lower to allow smaller embryo contours (was 5000)
MOG2_HISTORY = 30                      # Lower = faster adaptation & higher sensitivity (was 50)
MOG2_VAR_THRESHSOLD = 15               # Much lower threshold for aggressive motion detection (was 30)
MOG2_DETECT_SHADOWS = False
DILATION_ITERATIONS = 4                # Increased to expand detected regions (was 3)

# --- Performance & Feature Toggles ---
ENABLE_DISK_SAVING = True  # Set to False for max performance, True to save files
INFERENCE_BATCH_SIZE = 1    # ULTRA LOW LATENCY: Process immediately, no batching (was 8)

# --- Worker & I/O Paths ---
CROPPED_OUTPUT_DIR = "live_process/cropped/"
FRAME_OUTPUT_DIR = "live_process/frames/"

if ENABLE_DISK_SAVING:
    os.makedirs(CROPPED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

SERIAL_PORT = 'COM3' # CHANGE TO CORRECT ARDUNIO PORT
BAUD_RATE = 9600
CORRECT_EMBRYO_SIGNAL = b'C'

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
    
    # Load model with error handling
    try:
        print(f"[INFERENCE] Loading model from: {model_path}")
        if not os.path.exists(model_path):
            print(f"[INFERENCE] ERROR: Model file not found at {model_path}")
            return
        
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"[INFERENCE] Worker started on device: {device}, model loaded successfully")
    except Exception as e:
        print(f"[INFERENCE] ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    inference_count = 0

    while not stop_event.is_set():
        try:
            batch_data = crop_queue.get(timeout=1)
            # Handle multiple formats: (img, fc, i) or (img, fc, i, speed, obj_id)
            if len(batch_data[0]) >= 5:
                # New format with speed and object_id
                images, frame_indices, detection_indices, speeds, obj_ids = zip(*batch_data)
            elif len(batch_data[0]) == 4:
                # Format with speed but no obj_id
                images, frame_indices, detection_indices, speeds = zip(*batch_data)
                obj_ids = [None] * len(images)
            elif len(batch_data[0]) == 3:
                # Format without speed
                images, frame_indices, detection_indices = zip(*batch_data)
                speeds = [None] * len(images)
                obj_ids = [None] * len(images)
            else:
                images, frame_indices = zip(*batch_data)
                detection_indices = [0] * len(images)
                speeds = [None] * len(images)
                obj_ids = [None] * len(images)
            
            inference_start = time.time()
            tensor_list: list[torch.Tensor] = []
            for img in images:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                tensor = infer_transform(pil_img)
                if isinstance(tensor, torch.Tensor):
                    tensor_list.append(tensor)
            input_batch = torch.stack(tensor_list).to(device)

            with torch.no_grad():
                outputs = model(input_batch)
                _, preds = torch.max(outputs, 1)
            
            inference_time = time.time() - inference_start

            for i in range(len(images)):
                pred_idx = int(preds[i].item())
                result = {
                    'frame_index': frame_indices[i],
                    'detection_index': detection_indices[i] if i < len(detection_indices) else 0,
                    'inference_label': class_names[pred_idx],
                    'inference_time': inference_time / len(images),  # Average time per image
                    'speed': speeds[i] if i < len(speeds) else None,
                    'object_id': obj_ids[i] if i < len(obj_ids) else None
                }
                results_queue.put(result)
                inference_count += 1
                
            # Log progress periodically
            if inference_count % 10 == 0:
                print(f"[INFERENCE] Processed {inference_count} classifications")
                
        except Empty:
            continue
        except Exception as e:
            print(f"[INFERENCE] Worker error: {e}")
            import traceback
            traceback.print_exc()

# ===================================================================
# === MAIN APPLICATION CLASS ===
# ===================================================================

class EmbryoDetector:
    def __init__(self):
        # --- Arduino delayed trigger (latency compensation) ---
        self.arduino_target_delay_ms = 1  # The delay time you want, just as needed
        self._arduino_send_q = queue.PriorityQueue()  # (due_time, payload)
        self._arduino_thread_stop = threading.Event()
        self._arduino_sender_thread = None
        
        # --- Trigger cooldown (only one trigger per window) ---
        self.trigger_cooldown_ms = 500  # only one trigger every 500ms
        self._next_trigger_allowed_time = 0.0  # epoch seconds

        # --- DUAL-ZONE DETECTION SYSTEM ---
        # Zone 1 (Left): Full ML classification - determines if embryo is "Keep" or "Discard"
        # Zone 2 (Right): Fast motion+size detection - triggers Arduino if recent Zone 1 "Keep"
        
        # Zone 1 decision tracking: list of (timestamp, decision) for recent classifications
        self.zone1_decisions = []  # [(timestamp, 'Keep'/'Discard', object_id), ...]
        self.zone1_decision_window_ms = 2000  # How long a Zone 1 decision is valid (ms)
        self.zone1_keep_active = False  # Quick flag: is there an active "Keep" decision?
        
        # Zone 2 ROI configuration (separate from Zone 1)
        self.zone2_enabled = True  # Enable/disable Zone 2
        self.zone2_x = 0
        self.zone2_y = 0
        self.zone2_width = 0
        self.zone2_height = 0
        
        # Zone 2 detection parameters (simpler than Zone 1 - just motion + size)
        self.zone2_min_area = 800   # Lower than Zone 1 for more sensitivity (was 1500)
        self.zone2_max_area = 250000  # Maximum contour area
        self.zone2_var_threshold = 16  # Background subtractor sensitivity (lower = more sensitive)
        self.zone2_bg_subtractor = None  # Separate background subtractor for Zone 2
        
        # Zone 2 statistics
        self.zone2_detection_count = 0
        self.zone2_trigger_count = 0  # Actual Arduino triggers from Zone 2

        # --- Handles and State ---
        self.hcam: Optional[Any] = None
        self.ser = None
        self.frame_count = 0
        self.detection_count = 0
        self.crop_batch = []
        
        # --- Detection toggle (can be paused while camera runs) ---
        self.detection_enabled = False  # Starts OFF - user must click 'Start Detection'
        
        # --- Motion requirement (filter out stationary dust) ---
        self.min_motion_px = 15  # Minimum movement in pixels to be considered moving
        self.motion_check_frames = 3  # Check motion over this many frames
        self.candidate_history = {}  # {(approx_x, approx_y): [(frame, cx, cy), ...]} for motion tracking

        # --- Zone 1 temporal tracking (reduce dust FP + duplicate detections) ---
        self.zone1_tracks = {}  # {track_id: {'first_frame','last_frame','centroids','bbox','hits','emitted','max_displacement'}}
        self.zone1_next_track_id = 1
        self.zone1_track_match_px = 85
        self.zone1_track_stale_frames = 20
        self.zone1_min_confirm_frames = 2
        self.zone1_min_path_px = 12
        self.zone1_min_speed_px_per_frame = 0.8
        
        # --- Classification tracking ---
        self.classification_history = []  # List of {frame_index, label, timestamp, latency, speed}
        self.classification_stats = {'Correct': 0, 'Incorrect': 0}
        
        # --- Object tracking for speed calculation ---
        self.object_positions = {}  # {object_id: [(frame_count, cx, cy, timestamp), ...]} - keeps last 5 detections
        self.object_counter = 0  # Auto-incrementing ID for new objects
        self.position_history_max = 10  # Keep last N detections for each object (increased for better speed accuracy)
        self.speed_px_per_frame = {}  # {object_id: [speed1, speed2, ...]} for averaging
        
        # --- Performance metrics (limited size to prevent memory growth) ---
        self._metrics_max_size = 100  # Keep only last 100 measurements
        self.performance_metrics = {
            'frame_capture_time': [],
            'detection_time': [],
            'preprocessing_time': [],
            'inference_time': [],
            'total_latency': []
        }

        # --- Camera vs Video Mode ---
        self.is_camera_mode = False
        self.camera_warmup_frames = 0
        self.show_window = False

        # --- Multiprocessing ---
        self.stop_event = Event()
        self.crop_queue = Queue(maxsize=50)
        self.results_queue = Queue()
        self.save_queue = Queue(maxsize=100)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

        # --- OpenCV objects (placeholders; re-created in camera/video init if needed) ---
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=16, detectShadows=False)

        # === detect_emryo pipeline additions (MAXIMUM SENSITIVITY) ===
        self.bg_subtractor_sens = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=12, detectShadows=False)

        # LAB background model accumulation for color deltaE sensitivity
        self._lab_bg_accum = None  # running float accumulation
        self._lab_bg_count = 0
        self._deltaE_enabled = True

        # ROI setup is finalized in _initialize_camera() or _initialize_video()
        # Full frame dimensions
        self.height = 0
        self.width = 0
        
        # ROI configuration - using x, y, width, height for precise control
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = 0
        self.roi_height = 0
        
        # Camera exposure and gain settings (can be updated dynamically)
        # Note: Gain uses camera SDK units where 100 = 1x (baseline), 5000 = 50x
        self.exposure_time_us = 264  # microseconds
        self.gain = 5000  # camera-specific gain units (5000 = 50x baseline, minimum is 100)
        
        # Legacy ROI parameters (kept for backward compatibility during transition)
        self.roi_start_y = 0
        self.roi_end_y = 0
        self.roi_top_fraction = 0.35
        self.roi_bottom_fraction = 0.65
        self.roi_buffer_px = 5
        
        # Zone 2 legacy ROI (vertical bounds, same as Zone 1 by default)
        self.zone2_roi_start_y = 0
        self.zone2_roi_end_y = 0

        # Morphology
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel_iso = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.close_kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 5))

        # Edge assist & gating parameters (MAXIMUM SENSITIVITY FOR ALL EMBRYOS)
        # Area bandpass (will be dynamically scaled). More permissive range.
        self.AREA_MIN_BANDPASS = 1500        # Even lower min area to catch small embryos
        self.AREA_MAX_BANDPASS = 250000       # Higher max area for large embryos
        self.CONTRAST_MIN_CENTER = 0.15       # Very low contrast requirement (was 0.3)
        self.CONTRAST_MIN_EDGE = 0.15         # Very low edge contrast (was 0.3)
        self.GRAD_MIN_CENTER = 0.15           # Very low gradient requirement (was 0.3)
        self.GRAD_MIN_EDGE = 0.25             # Lower edge gradient (was 0.5)
        self.SOLIDITY_MIN = 0.4               # Much more permissive shape filtering
        self.CIRCULARITY_RANGE = (0.2, 0.98) # More permissive circular shapes
        self.ASPECT_RATIO_MIN = 0.2           # More permissive aspect ratios
        self.EDGE_SOLIDITY_MIN = 0.3          # More permissive edge solidity
        self.EDGE_CIRC_MIN = 0.15             # More permissive edge circularity
        self.EDGE_BAND_FRACTION = 0.05        # Narrow edge band; less noisy side emphasis

        # === WALL STRIP PARAMETERS (MAXIMUM SENSITIVITY) ===
        self.WALL_STRIP_PX = 20               # Pixel distance for wall overlap detection
        self.WALL_STRIP_OVERLAP_MAX = 0.8     # Allow more wall overlap (was 0.3)
        self.WALL_STRIP_STRICT_DELTAI = 2     # Lower intensity requirement (was 5)

        # Keep only per-frame limit for performance - increased for multiple embryos
        self.MAX_ACCEPTS_PER_FRAME = 20      # Higher limit to capture multiple embryos (was 10)

        self.prev_roi_gray = None

        # Tripwires set in camera init once width known (COMMENTED OUT - NO EDGE PRIORITY)
        # self.left_tripwire_x = 0
        # self.right_tripwire_x = 0

    # ------------------------ QX added on 12/26/2025 -------------------------
    def _arduino_sender_loop(self):
        """Send queued Arduino signals at their due_time without blocking the camera pipeline."""
        print(f"[ARDUINO] Sender thread started. Queue: {self._arduino_send_q}")
        sent_count = 0
        while not self._arduino_thread_stop.is_set() and not self.stop_event.is_set():
            try:
                due_time, seq_num, payload = self._arduino_send_q.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ARDUINO] Queue get error: {e}")
                continue

            # Wait until due (but remain stoppable)
            while True:
                remaining = due_time - time.time()
                if remaining <= 0:
                    break
                if self._arduino_thread_stop.is_set() or self.stop_event.is_set():
                    print(f"[ARDUINO] Sender thread stopped (seq={seq_num})")
                    return
                time.sleep(min(remaining, 0.005))  # 5ms granularity

            # Send (double-check stop before writing)
            if self._arduino_thread_stop.is_set() or self.stop_event.is_set():
                print(f"[ARDUINO] Send cancelled due to stop signal (seq={seq_num})")
                return
            if self.ser and getattr(self.ser, "is_open", False):
                try:
                    self.ser.write(payload)
                    sent_count += 1
                    print(f"[ARDUINO] SENT (seq={seq_num}, total={sent_count}) payload={payload}")
                except Exception as e:
                    print(f"[ARDUINO] Send failed (seq={seq_num}): {e}")
            else:
                print(f"[ARDUINO] Serial port not available (seq={seq_num}), dropping payload")

    
    def schedule_arduino_signal(self, payload: bytes, detect_elapsed_ms: float):
        """Schedule Arduino signal with latency compensation. T_wait = max(0, T_target - T_detect)"""
        # Get sequence number for tracking
        seq_num = self.frame_count
        
        target = float(self.arduino_target_delay_ms)
        wait_ms = max(0.0, target - float(detect_elapsed_ms))
        due_time = time.time() + wait_ms / 1000.0
        
        try:
            self._arduino_send_q.put_nowait((due_time, seq_num, payload))
            print(f"[ARDUINO] Scheduled (seq={seq_num}): detect={detect_elapsed_ms:.2f}ms, "
                  f"target={target:.0f}ms, wait={wait_ms:.2f}ms, due_time={due_time:.4f}")
        except Exception as e:
            # Queue full or other issue; drop to avoid blocking real-time pipeline
            print(f"[ARDUINO] Failed to schedule (seq={seq_num}): {e}")

    # ------------------------ camera init -------------------------
    def _initialize_camera(self):
        # Close camera if already open (cleanup from previous run)
        if self.hcam is not None:
            try:
                print("Closing previously opened camera...")
                # Stop camera first if it's running
                try:
                    self.hcam.Stop()
                    time.sleep(0.2)
                except Exception:
                    pass  # May not be running, that's okay
                
                self.hcam.Close()
                time.sleep(0.5)  # Give camera time to fully close
            except Exception as e:
                print(f"Warning: Error closing previous camera: {e}")
            finally:
                self.hcam = None
        
        print("Searching for camera...")
        try:
            devices = amcam.Amcam.EnumV2()
        except Exception as e:
            print(f"ERROR: Failed to enumerate cameras: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        if len(devices) <= 0:
            print("ERROR: No camera found. Please check connection.")
            return False

        selected_camera = devices[0]
        print(f"Found camera: {selected_camera.displayname}")

        # Try to open camera with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Retry attempt {attempt + 1}/{max_retries} to open camera...")
                    time.sleep(1.0)  # Wait before retry
                
                self.hcam = amcam.Amcam.Open(selected_camera.id)
                if self.hcam:
                    print(f"Camera opened successfully on attempt {attempt + 1}")
                    break
                else:
                    print(f"Attempt {attempt + 1}: Failed to open camera (returned None)")
                    if attempt == max_retries - 1:
                        print(f"ERROR: Failed to open camera after {max_retries} attempts")
                        return False
            except Exception as e:
                print(f"Attempt {attempt + 1}: Exception opening camera: {e}")
                if attempt == max_retries - 1:
                    import traceback
                    traceback.print_exc()
                    return False
                time.sleep(1.0)
        
        if not self.hcam:
            print(f"ERROR: Failed to open camera: {selected_camera.displayname}")
            return False

        # ---- Exposure & Gain Lock -------------------------------------------------
        try:
            # Disable auto exposure (0 = manual) if supported
            try:
                self.hcam.put_AutoExpoEnable(0)
                print("[EXPOSURE] Auto exposure disabled (manual mode)")
            except Exception as e:
                print(f"[EXPOSURE] Could not disable auto exposure: {e}")

            # Set initial exposure time and gain from instance variables
            try:
                self.hcam.put_ExpoTime(self.exposure_time_us)
                actual_expo = self.hcam.get_ExpoTime()
                print(f"[EXPOSURE] Requested {self.exposure_time_us}us, camera set to {actual_expo}us")
            except Exception as e:
                print(f"[EXPOSURE] Failed to set exposure time: {e}")

            try:
                self.hcam.put_ExpoAGain(self.gain)
                print(f"[GAIN] Set analog gain to {self.gain}")
            except Exception as e:
                print(f"[GAIN] Failed to set gain: {e}")
                
            # Set frame rate to maximum (0 = no limit)
            try:
                self.hcam.put_Option(amcam.AMCAM_OPTION_FRAMERATE, 0)
                print("[FRAMERATE] Set to maximum (no limit)")
            except Exception as e:
                print(f"[FRAMERATE] Failed to set frame rate: {e}")
                
            # Try to set precise frame rate to maximum if supported
            try:
                max_fps = self.hcam.get_Option(amcam.AMCAM_OPTION_MAX_PRECISE_FRAMERATE)
                if max_fps > 0:
                    self.hcam.put_Option(amcam.AMCAM_OPTION_PRECISE_FRAMERATE, max_fps)
                    print(f"[FRAMERATE] Set precise frame rate to {max_fps/10.0:.1f} fps")
            except Exception as e:
                print(f"[FRAMERATE] Precise frame rate not available: {e}")
                
        except Exception as e:
            print(f"[EXPOSURE] Unexpected error configuring exposure/gain: {e}")

        # Set camera resolution to 1824 x 1216
        try:
            print("Setting camera resolution to 1824 x 1216...")
            self.hcam.put_Size(1824, 1216)
            
            # Apply binning if specified (must be done before starting camera)
            if hasattr(self, 'binning') and self.binning != 0x01:
                try:
                    self.hcam.put_Option(amcam.AMCAM_OPTION_BINNING, self.binning)
                    binning_names = {0x01: "None", 0x82: "2x2", 0x83: "3x3", 0x84: "4x4"}
                    print(f"Binning set to {binning_names.get(self.binning, 'Unknown')}")
                except Exception as e:
                    print(f"Warning: Could not set binning: {e}")
            
            # Get actual camera resolution (may differ if camera doesn't support exact size or after binning)
            width, height = self.hcam.get_Size()
            self.width, self.height = width, height
            print(f"Camera resolution set: {self.width}x{self.height}")
            if self.width != 1824 or self.height != 1216:
                print(f"Note: Resolution is {self.width}x{self.height} (may be due to binning or camera limitations)")
        except Exception as e:
            print(f"ERROR: Failed to get camera size: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Critical: Allocate image buffer for RGB24 (3 bytes per pixel)
        # amcam.Amcam_PullImageV3 expects a (char *) buffer (ctypes.c_char_p), so allocate as c_char array
        try:
            buffer_size = self.width * self.height * 3
            print(f"Allocating camera buffer: {buffer_size} bytes ({self.width}x{self.height} RGB24)...")
            self.camera_buf = (ctypes.c_char * buffer_size)()
            print(f"Camera buffer allocated successfully: {buffer_size} bytes")
            # Basic sanity check
            if len(self.camera_buf) != buffer_size:
                print(f"ERROR: Camera buffer size mismatch! Expected {buffer_size}, got {len(self.camera_buf)}")
                return False
        except Exception as e:
            print(f"ERROR: Failed to allocate camera buffer: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Initialize ROI based on actual camera dimensions
        try:
            print("Initializing ROI and background subtractors...")
            self._initialize_common(self.width, self.height, 
                                     self.roi_top_fraction, 
                                     self.roi_bottom_fraction, 
                                     self.roi_buffer_px)
            
            # Set initial ROI to full frame (can be adjusted via GUI)
            self.roi_x = 0
            self.roi_y = 0
            self.roi_width = self.width
            self.roi_height = self.height
            
            # === CAMERA-SPECIFIC OPTIMIZATIONS ===
            # Reset background subtractors for live camera feed
            self.is_camera_mode = True
            self.camera_warmup_frames = 60  # number of frames used for background & LAB model accumulation
            
            # More aggressive background learning for camera
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=160,           # longer history for stability at high resolution
                varThreshold=24,
                detectShadows=False
            )
            self.bg_subtractor_sens = cv2.createBackgroundSubtractorMOG2(
                history=60,            # secondary faster adapting model
                varThreshold=16,
                detectShadows=False
            )
            
            print("Camera-specific background subtractors initialized")
        except Exception as e:
            print(f"ERROR: Failed to initialize ROI/background subtractors: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Debug: Print ROI bounds
        roi_height = self.roi_end_y - self.roi_start_y
        print(f"ROI configured: y={self.roi_start_y}..{self.roi_end_y} (height={roi_height})")
        
        if roi_height <= 0:
            print(f"ERROR: Invalid ROI height: {roi_height}")
            return False
        
        # Initialize Zone 2 (trigger zone) - default to right side of frame, same vertical bounds
        # Zone 2 is positioned to the RIGHT of Zone 1 (embryos flow left to right)
        self.zone2_x = int(self.width * 0.6)  # Start at 60% of frame width
        self.zone2_y = self.roi_y
        self.zone2_width = int(self.width * 0.35)  # 35% of frame width
        self.zone2_height = self.roi_height
        self.zone2_roi_start_y = self.roi_start_y
        self.zone2_roi_end_y = self.roi_end_y
        self.zone2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=20, detectShadows=False
        )
        print(f"Zone 2 (trigger) initialized: x={self.zone2_x}, y={self.zone2_y}, w={self.zone2_width}, h={self.zone2_height}")
    
        print(f"Camera opened successfully at {self.width}x{self.height}.")
        print(f"ROI: y={self.roi_start_y}..{self.roi_end_y} (h={self.roi_end_y - self.roi_start_y})")
        return True

    def _initialize_video(self, video_path):
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._initialize_common(self.width, self.height,
                                 self.roi_top_fraction,
                                 self.roi_bottom_fraction,
                                 self.roi_buffer_px)
        
        # Set initial ROI to full frame (can be adjusted via GUI)
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = self.width
        self.roi_height = self.height
        
        # Initialize Zone 2 (trigger zone) - default to right side of frame, same vertical bounds
        self.zone2_x = int(self.width * 0.6)
        self.zone2_y = self.roi_y
        self.zone2_width = int(self.width * 0.35)
        self.zone2_height = self.roi_height
        self.zone2_roi_start_y = self.roi_start_y
        self.zone2_roi_end_y = self.roi_end_y
        self.zone2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=20, detectShadows=False
        )
        print(f"Zone 2 (trigger) initialized: x={self.zone2_x}, y={self.zone2_y}, w={self.zone2_width}, h={self.zone2_height}")
        
        print(f"Video opened successfully at {self.width}x{self.height}.")
        print(f"ROI: y={self.roi_start_y}..{self.roi_end_y} (h={self.roi_end_y - self.roi_start_y})")
        return cap

    def _initialize_common(self, width, height, roi_top_fraction=None, roi_bottom_fraction=None, roi_buffer_px=None):
        # Set ROI to channel area between the two horizontal lines
        # EXPANDED ROI: Increased bounds by ~25% for maximum sensitivity
        # Allow these to be overridden if provided
        if roi_top_fraction is None:
            roi_top_fraction = 0.35      # Expanded up from 40% to 35% (5% increase)
        if roi_bottom_fraction is None:
            roi_bottom_fraction = 0.65   # Expanded down from 60% to 65% (5% increase)
        if roi_buffer_px is None:
            roi_buffer_px = 5            # Reduced buffer for more detection area
        
        # Store ROI fractions for potential updates
        self.roi_top_fraction = roi_top_fraction
        self.roi_bottom_fraction = roi_bottom_fraction
        self.roi_buffer_px = roi_buffer_px
        
        # Calculate ROI boundaries
        channel_top_y = int(height * roi_top_fraction)
        channel_bottom_y = int(height * roi_bottom_fraction)
        
        # Add buffer inside the channel to avoid detecting the lines themselves
        self.roi_start_y = max(0, channel_top_y + roi_buffer_px)
        self.roi_end_y = min(height, channel_bottom_y - roi_buffer_px)
        
        # Store dimensions for reference
        self.height = height
        self.width = width
        
        # --- Dynamic area scaling relative to reference video resolution ---
        ref_w, ref_h = 1824, 1216  # reference dimensions used for original tuning
        self._area_scale = (self.width * self.height) / (ref_w * ref_h)
        # Scale existing bandpass thresholds if not already scaled (heuristic: only scale if large resolution)
        if self.width > ref_w or self.height > ref_h:
            # Use much more permissive scaling to catch all embryos
            pre_min = self.AREA_MIN_BANDPASS
            self.AREA_MIN_BANDPASS = max(1000, int(self.AREA_MIN_BANDPASS * self._area_scale * 0.2))  # Even more permissive
            self.AREA_MAX_BANDPASS = int(self.AREA_MAX_BANDPASS * self._area_scale * 1.5)  # Higher max
            print(f"[INFO] Scaled area thresholds (factor {self._area_scale:.2f}, very permissive): pre_min={pre_min} -> {self.AREA_MIN_BANDPASS}, max={self.AREA_MAX_BANDPASS}")

        print(f"Channel ROI set: y={self.roi_start_y}..{self.roi_end_y} (height={self.roi_end_y - self.roi_start_y}px)")

        # Tripwires for edge-priority (COMMENTED OUT - NO EDGE PRIORITY)
        # self.left_tripwire_x = int(width * 0.10)
        # self.right_tripwire_x = int(width * 0.90)
    
    def update_roi(self, top_fraction=None, bottom_fraction=None, buffer_px=None, 
                  x=None, y=None, width=None, height=None):
        """Update ROI parameters dynamically (only if camera/video is initialized)
        
        Can use either legacy fraction-based (top_fraction, bottom_fraction, buffer_px)
        or new pixel-based (x, y, width, height) parameters.
        """
        if self.width == 0 or self.height == 0:
            print("Warning: Cannot update ROI - camera/video not initialized")
            return False
        
        # New pixel-based ROI (takes precedence if provided)
        if x is not None or y is not None or width is not None or height is not None:
            if x is not None:
                self.roi_x = max(0, min(self.width, int(x)))
            if y is not None:
                self.roi_y = max(0, min(self.height, int(y)))
            if width is not None:
                self.roi_width = max(1, min(self.width - self.roi_x, int(width)))
            if height is not None:
                self.roi_height = max(1, min(self.height - self.roi_y, int(height)))
            
            # Ensure ROI stays within frame bounds
            if self.roi_x + self.roi_width > self.width:
                self.roi_width = self.width - self.roi_x
            if self.roi_y + self.roi_height > self.height:
                self.roi_height = self.height - self.roi_y
            
            # Update legacy values for compatibility
            self.roi_start_y = self.roi_y
            self.roi_end_y = self.roi_y + self.roi_height
            
            print(f"ROI updated: x={self.roi_x}, y={self.roi_y}, w={self.roi_width}, h={self.roi_height}")
        else:
            # Legacy fraction-based ROI
            if top_fraction is not None:
                self.roi_top_fraction = max(0.0, min(1.0, top_fraction))
            if bottom_fraction is not None:
                self.roi_bottom_fraction = max(0.0, min(1.0, bottom_fraction))
            if buffer_px is not None:
                self.roi_buffer_px = max(0, buffer_px)
            
            # Recalculate ROI boundaries
            channel_top_y = int(self.height * self.roi_top_fraction)
            channel_bottom_y = int(self.height * self.roi_bottom_fraction)
            
            self.roi_start_y = max(0, channel_top_y + self.roi_buffer_px)
            self.roi_end_y = min(self.height, channel_bottom_y - self.roi_buffer_px)
            
            # Update new values from legacy
            self.roi_x = 0
            self.roi_y = self.roi_start_y
            self.roi_width = self.width
            self.roi_height = self.roi_end_y - self.roi_start_y
            
            print(f"ROI updated: y={self.roi_start_y}..{self.roi_end_y} (height={self.roi_end_y - self.roi_start_y}px)")
        
        # Reset background models when ROI changes
        if self.is_camera_mode:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=160, varThreshold=24, detectShadows=False
            )
            self.bg_subtractor_sens = cv2.createBackgroundSubtractorMOG2(
                history=60, varThreshold=16, detectShadows=False
            )
            self._lab_bg_accum = None
            self._lab_bg_count = 0
            self.frame_count = 0  # Reset to trigger warmup again
        
        return True
    
    def update_zone2_roi(self, x=None, y=None, width=None, height=None, enabled=None):
        """Update Zone 2 (trigger zone) ROI parameters dynamically.
        
        Zone 2 is the downstream trigger zone - it uses simple motion detection
        and only triggers Arduino if Zone 1 recently classified an embryo as 'Keep'.
        """
        if self.width == 0 or self.height == 0:
            print("Warning: Cannot update Zone 2 ROI - camera/video not initialized")
            return False
        
        if enabled is not None:
            self.zone2_enabled = enabled
            print(f"Zone 2 {'enabled' if enabled else 'disabled'}")
        
        if x is not None:
            self.zone2_x = max(0, min(self.width, int(x)))
        if y is not None:
            self.zone2_y = max(0, min(self.height, int(y)))
        if width is not None:
            self.zone2_width = max(1, min(self.width - self.zone2_x, int(width)))
        if height is not None:
            self.zone2_height = max(1, min(self.height - self.zone2_y, int(height)))
        
        # Ensure Zone 2 ROI stays within frame bounds
        if self.zone2_x + self.zone2_width > self.width:
            self.zone2_width = self.width - self.zone2_x
        if self.zone2_y + self.zone2_height > self.height:
            self.zone2_height = self.height - self.zone2_y
        
        # Update legacy values for compatibility
        self.zone2_roi_start_y = self.zone2_y
        self.zone2_roi_end_y = self.zone2_y + self.zone2_height
        
        # Reset Zone 2 background subtractor
        if self.is_camera_mode:
            self.zone2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=30, varThreshold=20, detectShadows=False
            )
        
        print(f"Zone 2 ROI updated: x={self.zone2_x}, y={self.zone2_y}, w={self.zone2_width}, h={self.zone2_height}")
        return True
    
    def update_zone1_decision_window(self, window_ms):
        """Update the time window for Zone 1 decisions to remain valid."""
        self.zone1_decision_window_ms = max(100, int(window_ms))
        print(f"Zone 1 decision window updated to {self.zone1_decision_window_ms}ms")
    
    def _record_zone1_decision(self, decision, object_id=None):
        """Record a Zone 1 classification decision with timestamp.
        
        Args:
            decision: 'Keep' or 'Discard'
            object_id: Optional object tracking ID
        """
        now = time.time()
        self.zone1_decisions.append((now, decision, object_id))
        
        # Update quick flag
        if decision == 'Keep':
            self.zone1_keep_active = True
        
        # Clean up old decisions outside the time window
        cutoff = now - (self.zone1_decision_window_ms / 1000.0)
        self.zone1_decisions = [(t, d, oid) for t, d, oid in self.zone1_decisions if t > cutoff]
        
        # Update quick flag based on remaining decisions
        self.zone1_keep_active = any(d == 'Keep' for _, d, _ in self.zone1_decisions)
        
        print(f"[ZONE1] Recorded decision: {decision} (object_id={object_id}), "
              f"active_keeps={sum(1 for _, d, _ in self.zone1_decisions if d == 'Keep')}")
    
    def _check_zone1_keep_active(self):
        """Check if there's an active 'Keep' decision from Zone 1 within the time window.
        
        Returns:
            bool: True if Arduino should trigger on Zone 2 detection
        """
        now = time.time()
        cutoff = now - (self.zone1_decision_window_ms / 1000.0)
        
        # Clean up expired decisions
        self.zone1_decisions = [(t, d, oid) for t, d, oid in self.zone1_decisions if t > cutoff]
        
        # Check for any active 'Keep' decisions
        self.zone1_keep_active = any(d == 'Keep' for _, d, _ in self.zone1_decisions)
        
        return self.zone1_keep_active
    
    def _consume_zone1_keep(self):
        """Consume (remove) the oldest 'Keep' decision after triggering.
        
        This prevents the same 'Keep' decision from triggering multiple times
        if multiple objects pass through Zone 2.
        """
        for i, (t, d, oid) in enumerate(self.zone1_decisions):
            if d == 'Keep':
                self.zone1_decisions.pop(i)
                print(f"[ZONE1] Consumed 'Keep' decision (object_id={oid})")
                break
        
        # Update quick flag
        self.zone1_keep_active = any(d == 'Keep' for _, d, _ in self.zone1_decisions)

    def update_exposure_gain(self, exposure_time_us=None, gain=None):
        """Update camera exposure time and/or gain dynamically"""
        if not self.hcam:
            print("Warning: Cannot update exposure/gain - camera not initialized")
            return False
        
        try:
            if exposure_time_us is not None:
                # Ensure exposure is at least 1 microsecond
                self.exposure_time_us = max(1, int(exposure_time_us))
                try:
                    self.hcam.put_ExpoTime(self.exposure_time_us)
                    actual = self.hcam.get_ExpoTime()
                    print(f"[EXPOSURE] Updated to {actual}us (requested {self.exposure_time_us}us)")
                except Exception as e:
                    print(f"[EXPOSURE] Failed to update: {e}")
                    return False
            
            if gain is not None:
                # Camera SDK minimum gain is 100 (1x), so ensure we don't go below that
                self.gain = max(100, int(gain))
                try:
                    self.hcam.put_ExpoAGain(self.gain)
                    print(f"[GAIN] Updated to {self.gain} (camera SDK units, 100=1x)")
                except Exception as e:
                    print(f"[GAIN] Failed to update: {e}")
                    return False
            
            return True
        except Exception as e:
            print(f"[EXPOSURE/GAIN] Error updating: {e}")
            return False
    
    def _calculate_object_speed(self, centroid, max_distance_px=200):
        """Calculate speed of an object by matching it to recent detections.
        Uses temporal distance (time-based) for more accurate speed calculation.
        Returns (speed_px_per_frame, object_id) or (None, None) if no match found.
        """
        cx, cy = centroid
        current_time = time.time()
        best_match_id = None
        best_match_distance = max_distance_px
        
        # Clean up stale objects (older than 5 seconds) to prevent memory growth
        stale_threshold = current_time - 5.0
        stale_ids = []
        for obj_id, positions in list(self.object_positions.items()):
            if positions and positions[-1][3] < stale_threshold:
                stale_ids.append(obj_id)
        for obj_id in stale_ids:
            del self.object_positions[obj_id]
            if obj_id in self.speed_px_per_frame:
                del self.speed_px_per_frame[obj_id]
        
        # Try to match with existing objects (find closest one)
        # Check all recent positions, not just the last one, to handle fast-moving embryos
        for obj_id, positions in list(self.object_positions.items()):
            if positions:
                last_cx, last_cy = positions[-1][1], positions[-1][2]
                distance = ((cx - last_cx)**2 + (cy - last_cy)**2)**0.5
                if distance < best_match_distance:
                    best_match_distance = distance
                    best_match_id = obj_id
        
        # If no existing object matched, create new one
        if best_match_id is None:
            best_match_id = self.object_counter
            self.object_counter += 1
            self.object_positions[best_match_id] = []
            self.speed_px_per_frame[best_match_id] = []
        
        # Get position history
        positions = self.object_positions[best_match_id]
        positions.append((self.frame_count, cx, cy, current_time))
        
        # Keep last N positions (increased for better speed stability)
        if len(positions) > self.position_history_max:
            positions.pop(0)
        
        # Calculate speed using MOST DISTANT positions for accuracy
        # (not just consecutive detections, which might miss fast motion)
        speed = None
        if len(positions) >= 2:
            # Use first and last positions for max accuracy with fast-moving objects
            frame_count1, cx1, cy1, time1 = positions[0]
            frame_count2, cx2, cy2, time2 = positions[-1]
            
            time_diff = time2 - time1  # seconds
            
            # Only calculate if meaningful time has passed (avoid division by near-zero)
            if time_diff > 0.001:  # At least 1ms between detections
                distance = ((cx2 - cx1)**2 + (cy2 - cy1)**2)**0.5
                frame_count_diff = frame_count2 - frame_count1
                
                # Calculate speed in pixels per frame (more stable metric)
                if frame_count_diff > 0:
                    speed = distance / frame_count_diff  # pixels per frame
                    self.speed_px_per_frame[best_match_id].append(speed)
                    
                    # Keep only last 10 speed measurements
                    if len(self.speed_px_per_frame[best_match_id]) > 10:
                        self.speed_px_per_frame[best_match_id].pop(0)
        
        return speed, best_match_id
    
    def _process_classification_result(self, result):
        """Process ML classification result: log it, update stats, and record Zone 1 decision"""
        frame_idx = result['frame_index']
        label = result['inference_label']
        inference_time = result.get('inference_time', 0.0)
        detection_idx = result.get('detection_index', 0)
        speed = result.get('speed', None)
        object_id = result.get('object_id', None)
        
        # Record classification with timestamp
        classification_entry = {
            'frame_index': frame_idx,
            'detection_index': detection_idx,
            'label': label,
            'timestamp': time.time(),
            'inference_time': inference_time,
            'speed': speed,
            'object_id': object_id
        }
        self.classification_history.append(classification_entry)
        
        # Update statistics
        if label in self.classification_stats:
            self.classification_stats[label] += 1
        
        # Track inference time (with size limit)
        self.performance_metrics['inference_time'].append(inference_time)
        if len(self.performance_metrics['inference_time']) > self._metrics_max_size:
            self.performance_metrics['inference_time'] = self.performance_metrics['inference_time'][-self._metrics_max_size:]
        
        # === ZONE 1 DECISION RECORDING ===
        # Map 'Correct' -> 'Keep', 'Incorrect' -> 'Discard' for Zone 1 decision tracking
        zone1_decision = 'Keep' if label == 'Correct' else 'Discard'
        self._record_zone1_decision(zone1_decision, object_id)
        
        # Log classification with speed information
        speed_str = f", speed: {speed:.2f}px/fr" if speed is not None else ""
        print(f"[CLASSIFICATION/ZONE1] Frame {frame_idx}, Detection {detection_idx}: {label} -> {zone1_decision} (inference: {inference_time*1000:.2f}ms{speed_str})")
        
        # Keep only last 1000 classifications to prevent memory issues
        if len(self.classification_history) > 1000:
            self.classification_history = self.classification_history[-1000:]
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        stats = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                stats[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                stats[metric_name] = {'mean': 0, 'min': 0, 'max': 0, 'count': 0}
        return stats

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

    def _cleanup_zone1_tracks(self, current_frame):
        """Remove stale Zone 1 tracks to keep memory bounded."""
        stale_ids = [
            tid for tid, tr in self.zone1_tracks.items()
            if (current_frame - tr["last_frame"]) > self.zone1_track_stale_frames
        ]
        for tid in stale_ids:
            del self.zone1_tracks[tid]

    def _update_zone1_track(self, centroid, bbox, current_frame):
        """Match a candidate to an existing track or create a new one."""
        cx, cy = centroid
        matched_id = None
        best_dist = float("inf")

        for tid, tr in self.zone1_tracks.items():
            if (current_frame - tr["last_frame"]) > self.zone1_track_stale_frames:
                continue
            tcx, tcy = tr["centroids"][-1]
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            if dist < self.zone1_track_match_px and dist < best_dist:
                matched_id = tid
                best_dist = dist

        if matched_id is None:
            track_id = self.zone1_next_track_id
            self.zone1_next_track_id += 1
            self.zone1_tracks[track_id] = {
                "first_frame": current_frame,
                "last_frame": current_frame,
                "centroids": [centroid],
                "bbox": bbox,
                "hits": 1,
                "emitted": False,
                "max_displacement": 0.0,
            }
            return track_id, self.zone1_tracks[track_id]

        tr = self.zone1_tracks[matched_id]
        tr["last_frame"] = current_frame
        tr["bbox"] = bbox
        tr["hits"] += 1
        tr["centroids"].append(centroid)
        if len(tr["centroids"]) > 12:
            tr["centroids"] = tr["centroids"][-12:]

        fx, fy = tr["centroids"][0]
        tr["max_displacement"] = max(
            float(tr.get("max_displacement", 0.0)),
            ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5,
        )
        return matched_id, tr

    def _should_emit_zone1_track(self, tr):
        """Decide if a track is reliable enough to send to ML exactly once."""
        if tr["emitted"]:
            return False
        if tr["hits"] < self.zone1_min_confirm_frames:
            return False
        if tr["max_displacement"] < self.zone1_min_path_px:
            return False

        frame_span = max(1, tr["last_frame"] - tr["first_frame"])
        avg_speed = tr["max_displacement"] / frame_span
        if avg_speed < self.zone1_min_speed_px_per_frame:
            return False
        return True



    # --------------------- callback (detection) --------------------
    @staticmethod
    def _event_callback(nEvent, pContext):
        detector = pContext
        if not detector or not detector.hcam:
            return

        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            try:
                # Ensure buffer exists (should be allocated during _initialize_camera)
                if not hasattr(detector, 'camera_buf'):
                    # Fallback (should not normally happen)
                    width, height = detector.hcam.get_Size()
                    buffer_size = width * height * 3
                    detector.camera_buf = (ctypes.c_char * buffer_size)()
                    print(f"[WARN] Allocated fallback camera buffer in callback: {buffer_size} bytes")

                # Pull RGB24 image from camera (bStill=0, bits=24, rowPitch=0)
                # NOTE: amcam sets errcheck on PullImageV3, so it returns None on success or raises on failure.
                detector.hcam.PullImageV3(detector.camera_buf, 0, 24, 0, None)

                # Convert buffer to numpy array (RGB format from camera)
                expected_size = detector.width * detector.height * 3
                # Safety: ensure underlying buffer length matches expectations
                if len(detector.camera_buf) != expected_size:
                    print(f"[ERROR] Camera buffer length mismatch: have {len(detector.camera_buf)}, expected {expected_size}")
                    return
                frame_rgb = np.frombuffer(detector.camera_buf, dtype=np.uint8)
                if frame_rgb.size != expected_size:
                    print(f"[WARN] Incomplete frame: got {frame_rgb.size} bytes, expected {expected_size}")
                    return
                frame_rgb = frame_rgb.reshape((detector.height, detector.width, 3))
                
                # CRITICAL: Convert RGB to BGR for OpenCV compatibility
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Debug: Check frame on first few captures
                if detector.frame_count < 3:
                    print(f"Camera frame {detector.frame_count}: RGB→BGR conversion")
                    print(f"  Frame shape: {frame_bgr.shape}, dtype: {frame_bgr.dtype}")
                    print(f"  Value range: {frame_bgr.min()}-{frame_bgr.max()}")
                
                # Process the BGR frame (same as video processing)
                detector.process_frame(frame_bgr)
                
            except Exception as e:
                print(f"Camera callback error: {e}")
                import traceback
                traceback.print_exc()

    def process_frame(self, frame):
        frame_start_time = time.time()
        self.frame_count += 1
        fc = self.frame_count
        
        # Skip detection if disabled (but still update background model for when it's re-enabled)
        if not self.detection_enabled:
            # PRE-TRAIN background models while detection is OFF
            # This way Zone 1 and Zone 2 are ready immediately when detection is enabled
            if self.is_camera_mode and fc > 5:
                frame_h, frame_w = frame.shape[:2]
                
                # Pre-train Zone 1 background model
                if self.roi_width > 0 and self.roi_height > 0:
                    roi_x = max(0, min(self.roi_x, frame_w - 1))
                    roi_y = max(0, min(self.roi_y, frame_h - 1))
                    roi_w = max(1, min(self.roi_width, frame_w - roi_x))
                    roi_h = max(1, min(self.roi_height, frame_h - roi_y))
                    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    roi_pre = cv2.GaussianBlur(roi_frame, (5, 5), 0)
                    self.bg_subtractor.apply(roi_pre, learningRate=0.05)
                    self.bg_subtractor_sens.apply(roi_pre, learningRate=0.08)
                
                # Pre-train Zone 2 background model
                if self.zone2_enabled and self.zone2_width > 0 and self.zone2_height > 0:
                    # Initialize Zone 2 bg subtractor if needed
                    if self.zone2_bg_subtractor is None:
                        self.zone2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                            history=50, varThreshold=self.zone2_var_threshold, detectShadows=False
                        )
                    
                    z2_x = max(0, min(self.zone2_x, frame_w - 1))
                    z2_y = max(0, min(self.zone2_y, frame_h - 1))
                    z2_w = max(1, min(self.zone2_width, frame_w - z2_x))
                    z2_h = max(1, min(self.zone2_height, frame_h - z2_y))
                    zone2_frame = frame[z2_y:z2_y+z2_h, z2_x:z2_x+z2_w]
                    self.zone2_bg_subtractor.apply(zone2_frame, learningRate=0.1)
                    
                    # Track pre-training progress
                    if not hasattr(self, '_zone2_pretrain_frames'):
                        self._zone2_pretrain_frames = 0
                    self._zone2_pretrain_frames += 1
                    
                    if self._zone2_pretrain_frames % 30 == 0:
                        print(f"[PRETRAIN] Zone 2 background model: {self._zone2_pretrain_frames} frames processed")
                
            return  # Skip rest of detection
        
        # Track frame capture time (for camera, this is already done; for video, minimal)
        frame_capture_time = time.time() - frame_start_time
        self.performance_metrics['frame_capture_time'].append(frame_capture_time)
        # Limit metrics list size to prevent memory growth
        if len(self.performance_metrics['frame_capture_time']) > self._metrics_max_size:
            self.performance_metrics['frame_capture_time'] = self.performance_metrics['frame_capture_time'][-self._metrics_max_size:]

        # Save first frame with ROI overlay to see where camera is looking
        if fc == 1 and ENABLE_DISK_SAVING:
            first_frame = frame.copy()
            # Draw ROI rectangle (detection area)
            cv2.rectangle(first_frame, 
                         (0, self.roi_start_y), 
                         (self.width, self.roi_end_y), 
                         (0, 255, 0), 8)  # Thick green rectangle
            
            # Draw channel boundary lines for reference
            channel_top_y = int(self.height * 0.35)  # Updated to match new fractions
            channel_bottom_y = int(self.height * 0.65)
            cv2.line(first_frame, (0, channel_top_y), (self.width, channel_top_y), (255, 0, 0), 4)  # Blue top line
            cv2.line(first_frame, (0, channel_bottom_y), (self.width, channel_bottom_y), (255, 0, 0), 4)  # Blue bottom line
            
            # Draw tripwires (edge detection zones) - COMMENTED OUT
            # cv2.line(first_frame, (self.left_tripwire_x, 0), (self.left_tripwire_x, self.height), (0, 255, 255), 3)  # Yellow left
            # cv2.line(first_frame, (self.right_tripwire_x, 0), (self.right_tripwire_x, self.height), (0, 255, 255), 3)  # Yellow right
            
            # Add text labels
            cv2.putText(first_frame, 'GREEN: ROI Detection Zone', (50, self.roi_start_y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
            cv2.putText(first_frame, 'BLUE: Channel Boundaries', (50, channel_top_y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
            # cv2.putText(first_frame, 'YELLOW: Edge Tripwires', (50, 100),   # COMMENTED OUT
            #            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
            cv2.putText(first_frame, f'Camera: {self.width}x{self.height}', (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
            cv2.putText(first_frame, f'ROI: y={self.roi_start_y} to {self.roi_end_y}', (50, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
            
            # Save the annotated first frame (async via save_queue to avoid blocking)
            if hasattr(self, 'save_queue') and self.save_queue is not None:
                try:
                    self.save_queue.put_nowait(('frame', os.path.join(FRAME_OUTPUT_DIR, "FIRST_FRAME_WITH_ROI.png"), first_frame))
                except Exception:
                    # Queue full, skip to avoid blocking
                    pass
            else:
                # Fallback: save synchronously if queue not available
                cv2.imwrite(os.path.join(FRAME_OUTPUT_DIR, "FIRST_FRAME_WITH_ROI.png"), first_frame)
            print(f"*** SAVED FIRST FRAME WITH ROI OVERLAY: {FRAME_OUTPUT_DIR}/FIRST_FRAME_WITH_ROI.png ***")

        # --- ROI extraction ---
        preprocessing_start = time.time()
        # Use new x, y, width, height ROI if available, otherwise fall back to legacy
        if self.roi_width > 0 and self.roi_height > 0:
            # Ensure ROI is within frame bounds
            frame_h, frame_w = frame.shape[:2]
            roi_x = max(0, min(self.roi_x, frame_w - 1))
            roi_y = max(0, min(self.roi_y, frame_h - 1))
            roi_w = max(1, min(self.roi_width, frame_w - roi_x))
            roi_h = max(1, min(self.roi_height, frame_h - roi_y))
            roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        else:
            # Legacy ROI (full width, vertical only)
            frame_h = frame.shape[0]
            roi_start = max(0, min(self.roi_start_y, frame_h - 1))
            roi_end = max(roi_start + 1, min(self.roi_end_y, frame_h))
            roi_frame = frame[roi_start:roi_end, :]

        # Local contrast boost (CLAHE on L-channel)
        # Reuse CLAHE object to avoid recreating it every frame (performance optimization)
        if not hasattr(self, '_clahe'):
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = self._clahe.apply(L)
        roi_enhanced = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

        # Background subtraction with camera-specific adaptation
        if self.is_camera_mode:
            # Warmup: build stable background & LAB mean image
            if self.frame_count <= self.camera_warmup_frames:
                # moderate learning (not too high) to avoid oscillation
                lr_main = 0.35
                lr_sens = 0.55
                if self.frame_count % 10 == 0 or self.frame_count == 1:
                    print(f"Camera warmup frame {self.frame_count}/{self.camera_warmup_frames}")
            else:
                # Post-warmup: very low adaptation for stable channel
                lr_main = 0.08
                lr_sens = 0.12
        else:
            # Video file processing (original rates)
            lr_main = 0.25
            lr_sens = 0.40
        
        # Mild blur to reduce sensor noise before subtraction
        roi_pre = cv2.GaussianBlur(roi_enhanced, (5, 5), 0)
        fg_main = self.bg_subtractor.apply(roi_pre, learningRate=lr_main)
        fg_sens = self.bg_subtractor_sens.apply(roi_pre, learningRate=lr_sens)
        fg_mask = cv2.bitwise_or(fg_main, fg_sens)

        # Store individual contributions for debug ratios
        contrib_deltaE = None
        contrib_assist = None

        # Accumulate LAB background during warmup for deltaE color sensitivity
        if self.is_camera_mode and self.frame_count <= self.camera_warmup_frames:
            lab_frame = cv2.cvtColor(roi_pre, cv2.COLOR_BGR2LAB).astype(np.float32)
            if self._lab_bg_accum is None:
                self._lab_bg_accum = lab_frame
            else:
                self._lab_bg_accum += lab_frame
            self._lab_bg_count += 1
        # After warmup, compute color difference mask (deltaE) to catch subtle embryo color contrast
        if self._deltaE_enabled and self.is_camera_mode and self.frame_count == self.camera_warmup_frames + 1 and self._lab_bg_count > 0 and self._lab_bg_accum is not None:
            self._lab_bg_mean = (self._lab_bg_accum / self._lab_bg_count).astype(np.float32)
            print("[INFO] LAB background model finalized for color sensitivity.")
        if self._deltaE_enabled and hasattr(self, '_lab_bg_mean') and self.frame_count > self.camera_warmup_frames:
            # Only compute deltaE every N frames to reduce computational load
            # This is a performance optimization - deltaE is expensive
            if self.frame_count % 2 == 0:  # Compute every other frame
                lab_cur = cv2.cvtColor(roi_pre, cv2.COLOR_BGR2LAB).astype(np.float32)
                # CIE76 deltaE approximation
                dE = np.linalg.norm(lab_cur - self._lab_bg_mean, axis=2)
                # Increase threshold to reduce noise; apply gradient gating
                # Compute gradient magnitude on L channel for gating
                lab_L = lab_cur[...,0]
                gx = cv2.Sobel(lab_L, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(lab_L, cv2.CV_32F, 0, 1, ksize=3)
                grad_mag = cv2.magnitude(gx, gy)
                dE_mask_raw = (dE > 12.0).astype(np.uint8)  # raised from 6.0
                grad_gate = (grad_mag > 6.0).astype(np.uint8)
                # Gate out low-gradient color noise
                dE_mask = (dE_mask_raw & grad_gate) * 255
                # Morphological open to clean speckles
                dE_mask = cv2.morphologyEx(dE_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
                contrib_deltaE = dE_mask.copy()
                fg_mask = cv2.bitwise_or(fg_mask, dE_mask)

        # Foreground flood guard: if mask is mostly filled, salvage
        if self.frame_count > self.camera_warmup_frames:
            fg_ratio = float(np.count_nonzero(fg_mask)) / fg_mask.size
            if fg_ratio > 0.70:
                # Salvage: keep largest components instead of skipping
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
                if num_labels > 1:
                    # Skip label 0 (background)
                    component_areas = []
                    salv_mask = np.zeros_like(fg_mask)
                    # Sort by area descending, keep top K
                    K = 12
                    for lbl in range(1, num_labels):
                        area = stats[lbl, cv2.CC_STAT_AREA]
                        component_areas.append((area, lbl))
                    component_areas.sort(reverse=True)
                    kept = 0
                    for area, lbl in component_areas:
                        salv_mask[labels == lbl] = 255
                        kept += 1
                        if kept >= K:
                            break
                    if self.frame_count % 30 == 0:
                        print(f"[INFO] Frame {self.frame_count}: FG flood {fg_ratio:.2f}. Salvaged {kept}/{num_labels-1} largest components.")
                    fg_mask = salv_mask
                else:
                    if self.frame_count % 30 == 0:
                        print(f"[INFO] Frame {self.frame_count}: FG flood {fg_ratio:.2f} (no salvage components).")

        # Edge differencing assist - MAXIMUM SENSITIVITY
        roi_gray = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2GRAY)
        assist = None  # Edge differencing ASSIST disabled to reduce side noise
        self.prev_roi_gray = roi_gray

        # Noise reduction & smoothing
        fg_mask = cv2.medianBlur(fg_mask, 5)
        # Light erosion to separate merged large blobs before opening (helps at high res)
        fg_mask = cv2.erode(fg_mask, np.ones((3,3), np.uint8), iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.open_kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.close_kernel_iso, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.close_kernel_h, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self.dilation_kernel, iterations=DILATION_ITERATIONS)

        # Debug: mask contribution ratios every 30 frames (after salvage, before morphology post-processing)
        if self.frame_count % 30 == 0:
            total_fg = max(1, np.count_nonzero(fg_mask))
            main_pct = (np.count_nonzero(fg_main) / fg_main.size) * 100.0
            sens_pct = (np.count_nonzero(fg_sens) / fg_sens.size) * 100.0
            dE_pct = (np.count_nonzero(contrib_deltaE) / contrib_deltaE.size * 100.0) if contrib_deltaE is not None else 0.0
            assist_pct = (np.count_nonzero(contrib_assist) / contrib_assist.size * 100.0) if contrib_assist is not None else 0.0
            print(f"[MASK RATIOS] Frame {self.frame_count}: main={main_pct:.1f}%, sens={sens_pct:.1f}%, dE={dE_pct:.1f}%, assist={assist_pct:.1f}% (fg_ratio={total_fg/fg_mask.size*100:.1f}%)")

        # Track preprocessing time (with size limit)
        preprocessing_time = time.time() - preprocessing_start
        self.performance_metrics['preprocessing_time'].append(preprocessing_time)
        if len(self.performance_metrics['preprocessing_time']) > self._metrics_max_size:
            self.performance_metrics['preprocessing_time'] = self.performance_metrics['preprocessing_time'][-self._metrics_max_size:]
        
        # Find contours in ROI coords
        detection_start = time.time()
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Build valid polygons in FULL-FRAME coords
        valid_polys = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            approx_full = approx.copy()
            approx_full[:, 0, 1] += self.roi_start_y
            valid_polys.append((approx_full, area))

        if not valid_polys:
            return

        # === DEBUG OUTPUT FOR AREA ANALYSIS ===
        if self.frame_count % 30 == 0:  # Every 30 frames
            areas = [area for _, area in valid_polys]
            print(f"\nFrame {self.frame_count} - Contour Areas Found:")
            print(f"  Total contours: {len(areas)}")
            print(f"  Area range: {min(areas):.0f} - {max(areas):.0f} pixels")
            print(f"  All areas: {sorted([int(a) for a in areas])}")
            
            # Show what passes current filter
            filtered = [a for a in areas if self.AREA_MIN_BANDPASS <= a <= self.AREA_MAX_BANDPASS]
            print(f"  After area filter ({self.AREA_MIN_BANDPASS}-{self.AREA_MAX_BANDPASS}): {len(filtered)} objects")
            if filtered:
                print(f"    Filtered areas: {sorted([int(a) for a in filtered])}")

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Precompute wall strips (full frame)
        wall_top_y1 = max(self.roi_start_y - self.WALL_STRIP_PX, 0)
        wall_top_y2 = self.roi_start_y + self.WALL_STRIP_PX
        wall_bot_y1 = self.roi_end_y - self.WALL_STRIP_PX
        wall_bot_y2 = min(self.roi_end_y + self.WALL_STRIP_PX, self.height)

        # Candidate list (single-frame primary)
        candidates = []
        width = self.width

        for poly, area in valid_polys:
            # === DEBUG: Log area rejections ===
            if area < self.AREA_MIN_BANDPASS and self.frame_count % 60 == 0:
                print(f"  REJECTED (too small): {area:.0f} < {self.AREA_MIN_BANDPASS}")
            elif area > self.AREA_MAX_BANDPASS and self.frame_count % 60 == 0:
                print(f"  REJECTED (too large): {area:.0f} > {self.AREA_MAX_BANDPASS}")
            
            if not (self.AREA_MIN_BANDPASS <= area <= self.AREA_MAX_BANDPASS):
                continue

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)

            deltaI = self._mean_intensity_delta(gray_full, mask)
            deltaI_abs = abs(deltaI)
            grad_mean = self._mean_gradient_inside(gray_full, mask)

            M = cv2.moments(poly)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

            in_edge_band = False  # Treat all regions uniformly now

            # Unified minimal contrast/gradient gate
            if deltaI_abs < self.CONTRAST_MIN_CENTER or grad_mean < self.GRAD_MIN_CENTER:
                continue

            # Wall strip overlap veto
            mask_pixels = np.count_nonzero(mask)
            if mask_pixels > 0:
                top_overlap = np.count_nonzero(mask[wall_top_y1:wall_top_y2, :])
                bot_overlap = np.count_nonzero(mask[wall_bot_y1:wall_bot_y2, :])
                overlap_frac = (top_overlap + bot_overlap) / float(mask_pixels)
                if overlap_frac > self.WALL_STRIP_OVERLAP_MAX and deltaI_abs < self.WALL_STRIP_STRICT_DELTAI:
                    continue

            # Shape gates
            cnt = poly.reshape(-1, 1, 2)
            solidity, circularity, aratio = self._contour_shape_metrics(cnt)
            # Unified shape gates
            if solidity < self.SOLIDITY_MIN:
                continue
            if not (self.CIRCULARITY_RANGE[0] <= circularity <= self.CIRCULARITY_RANGE[1]):
                continue
            if aratio < self.ASPECT_RATIO_MIN:
                continue

            x, y, w, h = cv2.boundingRect(mask)
            # Side margin suppression: reject boxes hugging vertical borders (likely noise)
            side_margin = int(self.width * 0.02) + 15  # dynamic + constant
            if x < side_margin or (x + w) > (self.width - side_margin):
                # Allow if area very large (likely real) else skip
                if area < (self.AREA_MIN_BANDPASS * 1.5):
                    continue

            candidates.append({
                "poly": poly,
                "area": area,
                "bbox": (x, y, w, h),
                "centroid": (cx, cy),
                "in_edge_band": in_edge_band
            })

        self._cleanup_zone1_tracks(fc)

        if not candidates:
            if self.frame_count % 60 == 0:  # Log when no candidates found
                print(f"  Frame {self.frame_count}: No candidates after filtering")
            return

        # === DEBUG: Log candidate processing ===
        if self.frame_count % 30 == 0:
            candidate_areas = [cand["area"] for cand in candidates]
            print(f"  Candidates after all filters: {len(candidates)}")
            if candidate_areas:
                print(f"    Candidate areas: {sorted([int(a) for a in candidate_areas])}")

        # No sorting - process candidates in order found (TRIPWIRE PRIORITY DISABLED)
        # candidates.sort(key=lambda c: c["edge_priority"])  # COMMENTED OUT - was sorting by tripwire distance

        # Accept candidates only after short temporal confirmation:
        # this suppresses static dust and prevents re-detecting one embryo every frame.
        accepts = []
        for cand in candidates:
            if len(accepts) >= self.MAX_ACCEPTS_PER_FRAME:
                break
            cx, cy = cand["centroid"]
            bbox = cand["bbox"]
            track_id, track = self._update_zone1_track((cx, cy), bbox, fc)
            if not self._should_emit_zone1_track(track):
                continue

            # Same-frame dedup only (to avoid immediate duplicates within the same frame)
            duplicate = False
            for item in accepts:
                # Use basic IoU overlap check with a high threshold to avoid only exact duplicates
                if EmbryoDetector._iou(bbox, item["bbox"]) > 0.8:
                    duplicate = True
                    break
                # Check spatial proximity with small distance to avoid exact duplicates
                icx, icy = item["centroid"]
                if abs(icx - cx) < 20 and abs(icy - cy) < 20:
                    duplicate = True
                    break
            if duplicate:
                continue

            cand["track_id"] = track_id
            accepts.append(cand)
            track["emitted"] = True

        if not accepts:
            if self.frame_count % 60 == 0:  # Log when no accepts
                print(f"  Frame {self.frame_count}: No Zone 1 embryos detected")
            # Still process Zone 2 even if Zone 1 has no detections (to maintain background model)
            self._process_zone2(frame, frame_start_time)
            return
        
        # --- Zone 1: Detection count (no Arduino trigger here - that's Zone 2's job) ---
        self.detection_count += len(accepts)

        # NOTE: Arduino triggering has been REMOVED from Zone 1
        # Zone 1 only does ML classification and records 'Keep'/'Discard' decisions
        # Zone 2 handles the actual Arduino triggering based on Zone 1 decisions

        # === DEBUG: Log Zone 1 detections ===
        if len(accepts) > 0:
            accept_areas = [acc["area"] for acc in accepts]
            print(f"[ZONE1] Frame {self.frame_count}: DETECTED {len(accepts)} embryos (sent to ML)")
            print(f"    Areas: {[int(a) for a in accept_areas]}")
        elif self.frame_count % 120 == 0:  # Reduced frequency 
            print(f"Frame {self.frame_count}: No embryos detected in Zone 1")

        # Optional debug saving and window display
        if ENABLE_DISK_SAVING or self.show_window:
            dbg = frame.copy()
            for acc in accepts:
                hull = cv2.convexHull(acc["poly"])
                cv2.polylines(dbg, [hull], True, (0, 255, 0), 2)
                cx, cy = acc["centroid"]
                cv2.circle(dbg, (cx, cy), 4, (0, 0, 255), -1)
            
            if ENABLE_DISK_SAVING and hasattr(self, 'save_queue') and self.save_queue is not None:
                # Use async save queue to avoid blocking camera callback
                try:
                    self.save_queue.put_nowait(('frame', os.path.join(FRAME_OUTPUT_DIR, f"embryo_{fc:06d}.png"), dbg))
                except Exception:
                    # Queue full, skip this save to avoid blocking
                    pass
            
            if self.show_window:
                cv2.imshow('Embryo Detection', dbg)
                cv2.waitKey(1)  # Non-blocking wait

        # Process crops: save rectangular crops from bounding boxes and enqueue for inference
        for i, acc in enumerate(accepts):
            # Build a mask for the polygon and compute its bounding rectangle in full-frame coords
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [acc["poly"].astype(np.int32)], 255)
            x, y, w, h = cv2.boundingRect(mask)
            if w <= 0 or h <= 0:
                continue
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            crop = masked[y:y+h, x:x+w]
            
            # Calculate speed for this object
            centroid = acc["centroid"]
            speed, obj_id = self._calculate_object_speed(centroid)

            # Save crop to disk (async via save_queue to avoid blocking camera callback)
            if ENABLE_DISK_SAVING and hasattr(self, 'save_queue') and self.save_queue is not None:
                try:
                    self.save_queue.put_nowait(('crop', os.path.join(CROPPED_OUTPUT_DIR, f"embryo_{fc:06d}_{i}.png"), crop))
                except Exception:
                    # Queue full, skip this save to avoid blocking camera callback
                    pass

            # Enqueue crop immediately for inference (no batching delay)
            # Include detection index and speed to track which embryo this is and its speed
            try:
                self.crop_queue.put([(crop.copy(), fc, i, speed, obj_id)])  # Add speed and object_id
            except Exception:
                print(f"Warning: Inference queue full, skipping crop {fc}_{i}")
        
        # Track detection time for Zone 1 (with size limit)
        detection_time = time.time() - detection_start
        self.performance_metrics['detection_time'].append(detection_time)
        if len(self.performance_metrics['detection_time']) > self._metrics_max_size:
            self.performance_metrics['detection_time'] = self.performance_metrics['detection_time'][-self._metrics_max_size:]
        
        # === ZONE 2 PROCESSING ===
        # Process Zone 2 (trigger zone) - uses simple motion detection
        # Arduino only triggers if Zone 1 has an active 'Keep' decision
        self._process_zone2(frame, frame_start_time)
    
    def _process_zone2(self, frame, frame_start_time):
        """Process Zone 2 (trigger zone) with simple motion + size detection.
        
        Zone 2 uses a lightweight detection method (background subtraction + size filter)
        to detect any embryo-sized moving object. If detected AND there's an active
        'Keep' decision from Zone 1, it triggers the Arduino.
        
        This separation allows Zone 1 to do thorough ML classification while Zone 2
        provides fast trigger response.
        """
        if not self.zone2_enabled:
            return
        
        if self.zone2_width <= 0 or self.zone2_height <= 0:
            return
        
        # Ensure Zone 2 background subtractor exists
        if self.zone2_bg_subtractor is None:
            self.zone2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=50, varThreshold=self.zone2_var_threshold, detectShadows=False
            )
        
        fc = self.frame_count
        
        # Extract Zone 2 ROI
        frame_h, frame_w = frame.shape[:2]
        z2_x = max(0, min(self.zone2_x, frame_w - 1))
        z2_y = max(0, min(self.zone2_y, frame_h - 1))
        z2_w = max(1, min(self.zone2_width, frame_w - z2_x))
        z2_h = max(1, min(self.zone2_height, frame_h - z2_y))
        
        zone2_frame = frame[z2_y:z2_y+z2_h, z2_x:z2_x+z2_w]
        
        # Simple background subtraction for Zone 2 (fast, no CLAHE or LAB)
        # Check if pre-training happened (skip warmup if so)
        pretrain_frames = getattr(self, '_zone2_pretrain_frames', 0)
        needs_warmup = (pretrain_frames < 15) and (self.frame_count <= 15)
        
        if needs_warmup:
            # Still in warmup phase - build background model
            fg_mask = self.zone2_bg_subtractor.apply(zone2_frame, learningRate=0.2)
            if self.frame_count % 5 == 0:
                print(f"[ZONE2] Warmup frame {self.frame_count}/15 (pretrain={pretrain_frames})")
            return  # Don't trigger during warmup
        else:
            # Ready for detection - use normal learning rate
            fg_mask = self.zone2_bg_subtractor.apply(zone2_frame, learningRate=0.05)
        
        # Light morphology cleanup (more aggressive dilation for better detection)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # More closing
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)  # More dilation (was 1)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size (embryo-sized objects only)
        zone2_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.zone2_min_area <= area <= self.zone2_max_area:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + z2_x  # Convert to full-frame coords
                    cy = int(M["m01"] / M["m00"]) + z2_y
                    zone2_detections.append({
                        'centroid': (cx, cy),
                        'area': area,
                        'contour': contour
                    })
        
        # Debug: Log Zone 2 detection stats periodically
        if fc % 60 == 0:
            fg_pct = np.count_nonzero(fg_mask) / fg_mask.size * 100
            print(f"[ZONE2 DEBUG] Frame {fc}: {len(contours)} contours found, "
                  f"fg_mask={fg_pct:.1f}%, ROI=({z2_x},{z2_y},{z2_w}x{z2_h})")
            if contours:
                areas = [cv2.contourArea(c) for c in contours]
                print(f"  Contour areas: {sorted([int(a) for a in areas if a > 100])}")
                print(f"  After size filter ({self.zone2_min_area}-{self.zone2_max_area}): {len(zone2_detections)} detections")
        
        if not zone2_detections:
            return
        
        self.zone2_detection_count += len(zone2_detections)
        
        # Log Zone 2 detections
        if len(zone2_detections) > 0:
            det_areas = [d['area'] for d in zone2_detections]
            print(f"[ZONE2] Frame {fc}: Detected {len(zone2_detections)} objects, areas={[int(a) for a in det_areas]}")
        
        # === TRIGGER DECISION ===
        # Only trigger if there's an active 'Keep' decision from Zone 1
        if self._check_zone1_keep_active():
            # Trigger gating: prevent rapid-fire triggers
            now = time.time()
            if now >= self._next_trigger_allowed_time:
                self._next_trigger_allowed_time = now + self.trigger_cooldown_ms / 1000.0
                
                # Calculate detection latency
                detect_elapsed_ms = (now - frame_start_time) * 1000.0
                
                # Trigger Arduino
                if SERIAL_AVAILABLE and self.ser and getattr(self.ser, 'is_open', False):
                    self.schedule_arduino_signal(CORRECT_EMBRYO_SIGNAL, detect_elapsed_ms)
                    self.zone2_trigger_count += 1
                    
                    # Consume the 'Keep' decision to prevent double-triggering
                    self._consume_zone1_keep()
                    
                    print(f"[ZONE2 TRIGGER] Frame {fc}: Embryo in trigger zone! "
                          f"Zone1='Keep' active, Arduino scheduled. "
                          f"(detect={detect_elapsed_ms:.2f}ms, trigger_count={self.zone2_trigger_count})")
                else:
                    print(f"[ZONE2] Frame {fc}: Would trigger but Arduino not available")
            else:
                print(f"[ZONE2] Frame {fc}: Detection but in cooldown period")
        else:
            if fc % 60 == 0:  # Log periodically
                print(f"[ZONE2] Frame {fc}: Detected {len(zone2_detections)} objects but no active 'Keep' from Zone 1")

    # -------------------------- run loop --------------------------
    def run(self, video_path: Optional[str] = None, show_window: bool = False):
        self.show_window = show_window
        cap: Optional[cv2.VideoCapture] = None
        if video_path:
            cap = self._initialize_video(video_path)
            if not cap:
                return
        else:
            if not self._initialize_camera():
                print("ERROR: Camera initialization failed. Exiting.")
                return
        
        # CRITICAL: Clear stop_event at the start of each run
        # This ensures a fresh start even if the detector was stopped before
        if self.stop_event.is_set():
            print("WARNING: stop_event was already set. Resetting it.")
        self.stop_event.clear()
        print(f"DEBUG: stop_event cleared. Current state: {self.stop_event.is_set()}")
            
        if not SERIAL_AVAILABLE:
            print("Warning: pyserial not installed. Arduino communication disabled.")
            print("Install with: pip install pyserial")
            self.ser = None
        else:
            try:
                self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                time.sleep(2)  # Give the connection time to establish
                print(f"Successfully connected to Arduino on {SERIAL_PORT}")
                # Clear any pending signals from a previous run
                while not self._arduino_send_q.empty():
                    try:
                        self._arduino_send_q.get_nowait()
                    except queue.Empty:
                        break
                # >>> START ARDUINO DELAYED SENDER THREAD <<<
                if self.ser and self.ser.is_open:
                    print(f"[ARDUINO] Starting sender thread (ser={self.ser}, is_open={self.ser.is_open})")
                    self._arduino_thread_stop.clear()
                    self._arduino_sender_thread = threading.Thread(
                        target=self._arduino_sender_loop,
                        daemon=False  # Not daemon - we want to explicitly stop it
                    )
                    self._arduino_sender_thread.start()
                    print(f"[ARDUINO] Sender thread started (thread_id={self._arduino_sender_thread.ident})")
                else:
                    print(f"[ARDUINO] Cannot start sender thread: ser={self.ser}, "
                          f"is_open={self.ser.is_open if self.ser else 'N/A'}")
            except serial.SerialException as e:
                print(f"Warning: Could not connect to Arduino. {e}")
                self.ser = None

        print(f"Main process using device: {self.device}")
        # Use absolute path for model to ensure it works when GUI changes working directory
        model_path = os.path.join(_project_dir, "resnet18_model.pth")
        if not os.path.exists(model_path):
            print(f"WARNING: Model file not found at {model_path}")
        inference_process = Process(target=inference_worker, args=(model_path, self.crop_queue, self.results_queue, self.stop_event, self.device))
        inference_process.start()

        save_process = None
        if ENABLE_DISK_SAVING:
            save_process = Process(target=save_worker, args=(self.save_queue, self.stop_event))
            save_process.start()

        try:
            if video_path:
                # Video file loop
                while not self.stop_event.is_set():
                    if cap is not None:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        self.process_frame(frame)
                    
                    # Handle results from inference worker
                    try:
                        result = self.results_queue.get_nowait()
                        # Process classification for GUI logging (no Arduino interaction)
                        self._process_classification_result(result)
                    except Empty:
                        pass
                print("Video processing finished.")

            else:
                # Camera loop
                if self.hcam is not None:
                    # Double-check stop_event before starting callback
                    if self.stop_event.is_set():
                        print("ERROR: stop_event was set before starting camera callback. Clearing it.")
                        self.stop_event.clear()
                    
                    # Ensure camera is still valid before starting
                    if self.hcam is None:
                        print("ERROR: Camera handle became None before starting callback")
                        return
                    
                    # Stop camera if it's already running (important to avoid "resource in use" error)
                    try:
                        self.hcam.Stop()
                        time.sleep(0.2)  # Brief pause to ensure stop is complete
                    except Exception as e:
                        print(f"Warning: Could not stop camera (may not be running): {e}")
                    
                    try:
                        self.hcam.StartPullModeWithCallback(EmbryoDetector._event_callback, self)
                    except Exception as e:
                        print(f"ERROR: Failed to start camera pull mode: {e}")
                        import traceback
                        traceback.print_exc()
                        # Close camera if it's in a bad state
                        try:
                            self.hcam.Close()
                        except:
                            pass
                        self.hcam = None
                        raise
                    print("\nCamera stream started. Detecting embryos... Press Ctrl+C to stop.")
                    print(f"Waiting for frames... (stop_event is {'SET' if self.stop_event.is_set() else 'NOT SET'})")
                    
                    # Keep track of iterations to detect if loop exits immediately
                    loop_iterations = 0
                    try:
                        while not self.stop_event.is_set():
                            loop_iterations += 1
                            if loop_iterations == 1:
                                print("DEBUG: Entered camera loop for the first time")
                            if loop_iterations % 60 == 0:
                                print(f"DEBUG: Camera loop still running (iteration {loop_iterations})")
                            try:
                                result = self.results_queue.get(timeout=1)
                                # Process classification for GUI logging (no Arduino interaction)
                                self._process_classification_result(result)
                            except Empty:
                                # No result yet, continue waiting
                                continue
                        print("Camera loop exited (stop_event was set)")
                    except Exception as e:
                        print(f"Error in camera loop: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                else:
                    print("ERROR: Camera handle is None, cannot start stream")
        
        except KeyboardInterrupt:
            print("\nShutdown signal received.")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up resources...")
            self.stop_event.set()
            if self.show_window:
                cv2.destroyAllWindows()
                print("Display windows closed.")
            if self.hcam:
                try:
                    # Stop camera before closing (important for proper cleanup)
                    try:
                        self.hcam.Stop()
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Warning: Could not stop camera before close: {e}")
                    
                    self.hcam.Close()
                    print("Camera closed.")
                except Exception as e:
                    print(f"Warning: Error closing camera: {e}")
                finally:
                    self.hcam = None
            if cap is not None and cap.isOpened():
                cap.release()
                print("Video file closed.")

            # >>> ADD HERE: stop Arduino delayed sender thread <<<
            self._arduino_thread_stop.set()
            if self._arduino_sender_thread and self._arduino_sender_thread.is_alive():
                self._arduino_sender_thread.join(timeout=1.0)
            
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("Arduino serial port closed.")

            inference_process.join(timeout=5)
            if save_process:
                save_process.join(timeout=5)
            if inference_process.is_alive(): inference_process.terminate()
            if save_process and save_process.is_alive(): save_process.terminate()
            print("Program finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embryo Detection and Sorting')
    parser.add_argument('--video', type=str, default=None, help='Path to a video file to process instead of using the camera.')
    parser.add_argument('--show-window', action='store_true', help='Display the detection results in a window.')
    args = parser.parse_args()

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    app = EmbryoDetector()
    app.run(video_path=args.video, show_window=args.show_window)