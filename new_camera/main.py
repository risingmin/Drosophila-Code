# main.py
# Combines the AmScope camera SDK with the embryo detection and sorting logic.
# This version is modified to IGNORE Arduino communication for camera-only testing.

# --- Core Python & Computer Vision Libs ---
import cv2
import numpy as np
import time
import os
import sys

# --- AmScope Camera SDK ---
import amcam 

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
# === CONFIGURATION (ADJUST THESE SETTINGS) ===
# ===================================================================

# --- Detection & Morphology ---
MIN_AREA = 8000
MOG2_HISTORY = 100
MOG2_VAR_THRESHOLD = 60
MOG2_DETECT_SHADOWS = False
DILATION_ITERATIONS = 3

# --- Performance & Feature Toggles ---
ENABLE_DISK_SAVING = False # Set to False for max performance, True to save files
INFERENCE_BATCH_SIZE = 8 # Number of images to batch for the model

# --- Worker & I/O Paths ---
if ENABLE_DISK_SAVING:
    CROPPED_OUTPUT_DIR = "live_process/cropped/"
    FRAME_OUTPUT_DIR = "live_process/frames/"
    os.makedirs(CROPPED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

# --- Arduino Serial Configuration (DISABLED) ---
# SERIAL_PORT = 'COM3' # IMPORTANT: Change to your Arduino's port!
# BAUD_RATE = 9600
# CORRECT_EMBRYO_SIGNAL = b'C'

# ===================================================================
# === WORKER PROCESSES (No changes needed here) ===
# ===================================================================

def save_worker(save_queue, stop_event):
    """A worker process that handles saving images to disk."""
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
    """A worker process that performs model inference on BATCHES of cropped images."""
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
        
        # --- OpenCV objects ---
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=MOG2_DETECT_SHADOWS)
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.roi_start_y = 0
        self.roi_end_y = 0

    def _initialize_camera(self):
        """Finds and opens the AmScope camera."""
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
            
        width, height = self.hcam.get_Size(0)
        self.hcam.put_eSize(0) # Use the first available resolution
        self.roi_start_y, self.roi_end_y = int(height * 0.25), int(height * 0.75)
        print(f"Camera opened successfully at {width}x{height}.")
        return True

    # --- _initialize_serial METHOD DISABLED ---
    # def _initialize_serial(self):
    #     """Initializes the connection to the Arduino."""
    #     if SERIAL_PORT:
    #         try:
    #             self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    #             time.sleep(2) # Wait for connection to establish
    #             print(f"Successfully connected to Arduino on {SERIAL_PORT}")
    #         except serial.SerialException as e:
    #             print(f"WARNING: Could not open serial port {SERIAL_PORT}. {e}")

    @staticmethod
    def _camera_callback(pData, pInfo, pContext):
        """
        This function is called by the SDK on a separate thread for each new frame.
        'pContext' is a pointer to the EmbryoDetector instance (self).
        """
        detector_instance = pContext
        if not detector_instance:
            return

        # --- Convert raw frame data to an OpenCV image ---
        if pInfo.width > 0 and pInfo.height > 0:
            frame_data = np.ctypeslib.as_array(pData, shape=(pInfo.height, pInfo.width, 3))
            frame = frame_data.copy() # Make a writable copy
        else:
            return
            
        detector_instance.frame_count += 1
        
        # --- Apply the detection logic from your original script ---
        roi_frame = frame[detector_instance.roi_start_y:detector_instance.roi_end_y, :]
        fg_mask = detector_instance.bg_subtractor.apply(roi_frame)
        fg_mask_blur = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        fg_mask_eroded = cv2.erode(fg_mask_blur, detector_instance.dilation_kernel, iterations=2)
        fg_mask_dilated = cv2.dilate(fg_mask_eroded, detector_instance.dilation_kernel, iterations=DILATION_ITERATIONS)
        contours, _ = cv2.findContours(fg_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA:
                continue
            
            detector_instance.detection_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            
            y_full = y + detector_instance.roi_start_y
            cropped = frame[y_full:y_full+h, x:x+w]

            detector_instance.crop_batch.append((cropped.copy(), detector_instance.frame_count))

            if len(detector_instance.crop_batch) >= INFERENCE_BATCH_SIZE:
                if not detector_instance.crop_queue.full():
                    detector_instance.crop_queue.put(detector_instance.crop_batch)
                detector_instance.crop_batch = []

    def run(self):
        """Main application entry point."""
        if not self._initialize_camera():
            return
        # self._initialize_serial() # --- DISABLED ---

        # --- Start worker processes ---
        print(f"Main process using device: {self.device}")
        inference_process = Process(target=inference_worker, args=("resnet18_model.pth", self.crop_queue, self.results_queue, self.stop_event, self.device))
        inference_process.start()

        save_process = None
        if ENABLE_DISK_SAVING:
            save_process = Process(target=save_worker, args=(self.save_queue, self.stop_event))
            save_process.start()

        try:
            # --- Start the camera stream ---
            # We pass 'self' as the context, so the callback can access our class instance
            self.hcam.StartPullModeWithCallback(self._camera_callback, self)
            print("\nCamera stream started. Detecting embryos... Press Ctrl+C to stop.")

            while not self.stop_event.is_set():
                try:
                    # --- Process results from the inference worker ---
                    result = self.results_queue.get(timeout=1)
                    print(f"Inference Result (Frame {result['frame_index']}): {result['inference_label']}")
                    
                    # --- ARDUINO SIGNALING DISABLED ---
                    # if result['inference_label'] == 'Correct' and self.ser and self.ser.is_open:
                    #     print("--> Correct embryo detected. Sending signal to Arduino.")
                    #     self.ser.write(CORRECT_EMBRYO_SIGNAL)
                except Empty:
                    continue # No results yet, just keep waiting
        
        except KeyboardInterrupt:
            print("\nShutdown signal received.")
        finally:
            # --- Cleanup ---
            print("Cleaning up resources...")
            self.stop_event.set()
            
            if self.hcam:
                self.hcam.Close()
                print("Camera closed.")
            
            # --- SERIAL PORT CLEANUP DISABLED ---
            # if self.ser and self.ser.is_open:
            #     self.ser.close()
            #     print("Serial port closed.")
            
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
