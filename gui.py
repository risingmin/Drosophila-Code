import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import queue
import time
import os
import sys


# COMMENT
# Ensure the project directory is in the Python path for imports
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from main_3 import EmbryoDetector

class EmbryoDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Embryo Detection System")
        self.root.geometry("1400x900")
        self.root.minsize(900, 600)  # Minimum window size
        
        # Configure styles for better aesthetics
        self.setup_styles()
        
        # State variables
        self.detector = None
        self.is_running = False
        self.detection_enabled = False  # Detection can be toggled while camera runs
        self.current_frame = None
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue - we'll always show latest frame
        self.log_queue = queue.Queue()  # Thread-safe queue for log messages
        self._lock = threading.Lock()  # Lock for thread-safe state access
        
        # Flip options (disabled by default)
        self.flip_horizontal = False
        self.flip_vertical = False
        
        # Camera exposure and gain controls (for display adjustment)
        self.display_brightness = 1.0  # Legacy display brightness
        self.exposure_time_us = 2000
        self.gain = 5000  # Default to 5000 (50x), will be updated from detector
        
        # Debouncing for automatic camera settings updates
        self._exposure_update_job = None
        self._gain_update_job = None
        self._wb_update_job = None
        
        # FPS tracking
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # GUI display FPS tracking (separate from detector FPS)
        self.gui_last_fps_time = time.time()
        self.gui_fps_frame_count = 0
        self.gui_current_fps = 0.0
        
        # Stats update throttling
        self.last_stats_update = 0
        self.stats_update_interval = 0.1  # Update stats every 100ms (10 times per second)
        
        # Create GUI layout
        self.setup_ui()
        
        # Start processing log messages from queue
        self.process_log_queue()
        
        # ROI editing mode
        self.roi_edit_mode = False
        self.roi_drag_start = None
        self.roi_drag_mode = None  # 'move', 'resize_nw', 'resize_ne', 'resize_sw', 'resize_se', etc.
        self.active_zone = None  # 'zone1' or 'zone2'
        self.display_scale = 1.0  # Scale factor from display to actual frame coords
        self.display_width = 640  # Initial display width (will be updated dynamically)
        self.display_height = 480  # Initial display height
        self.handle_size = 8  # Size of resize handles in display pixels
        
        # Bind mouse events to video label after UI setup
        self.root.after(100, self._bind_mouse_events)
        
        # Bind window resize event
        self.root.bind('<Configure>', self._on_window_resize)
        self._resize_scheduled = False
        
    def setup_styles(self):
        """Configure ttk styles for better aesthetics"""
        style = ttk.Style()
        
        # Use clam theme as base (cleaner than default)
        style.theme_use('clam')
        
        # Configure colors
        bg_color = '#f5f5f5'
        accent_color = '#2196F3'  # Blue accent
        success_color = '#4CAF50'  # Green
        warning_color = '#FF9800'  # Orange
        error_color = '#f44336'  # Red
        
        # Frame styles
        style.configure('TFrame', background=bg_color)
        style.configure('TLabelframe', background=bg_color)
        style.configure('TLabelframe.Label', background=bg_color, font=('Segoe UI', 9, 'bold'))
        
        # Label styles
        style.configure('TLabel', background=bg_color, font=('Segoe UI', 9))
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 9), padding=5)
        
        # Stat labels with colors
        style.configure('Success.TLabel', foreground=success_color, font=('Segoe UI', 9, 'bold'))
        style.configure('Warning.TLabel', foreground=warning_color, font=('Segoe UI', 9, 'bold'))
        style.configure('Error.TLabel', foreground=error_color, font=('Segoe UI', 9, 'bold'))
        style.configure('Info.TLabel', foreground=accent_color, font=('Segoe UI', 9, 'bold'))
        
        # Button styles
        style.configure('TButton', font=('Segoe UI', 9), padding=6)
        style.configure('Accent.TButton', font=('Segoe UI', 9, 'bold'))
        style.map('Accent.TButton',
                  background=[('active', accent_color), ('!active', '#1976D2')])
        
        # Treeview style
        style.configure('Treeview', font=('Segoe UI', 8), rowheight=22)
        style.configure('Treeview.Heading', font=('Segoe UI', 8, 'bold'))
        
        # Scale style
        style.configure('TScale', background=bg_color)
        
    def setup_ui(self):
        # Main container with THREE columns: video, scrollable controls, fixed log
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left side - Video display and stats (fixed, no scrolling)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Video panel with dark background for contrast
        video_frame = ttk.LabelFrame(left_frame, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        self.video_frame = video_frame  # Store reference for resize calculations
        
        # Video container frame with black background
        self.video_container = tk.Frame(video_frame, bg='#1a1a1a', relief=tk.SUNKEN, bd=2)
        self.video_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2, pady=2)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(self.video_container, background='#1a1a1a')
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Status bar under video with better styling
        self.status_label = ttk.Label(video_frame, text="● Ready", 
                                      style='Status.TLabel', anchor=tk.W)
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Fixed Stats section below video (always visible)
        stats_container = ttk.Frame(left_frame)
        stats_container.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Statistics display - left side of stats container
        stats_frame = ttk.LabelFrame(stats_container, text="Statistics", padding="10")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 3))
        
        self.frames_label = ttk.Label(stats_frame, text="📊 Frames: 0")
        self.frames_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.detections_label = ttk.Label(stats_frame, text="🎯 Detections: 0")
        self.detections_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.fps_label = ttk.Label(stats_frame, text="⚡ FPS: 0")
        self.fps_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Separator
        ttk.Separator(stats_frame, orient=tk.VERTICAL).grid(row=0, column=1, rowspan=3, sticky=(tk.N, tk.S), padx=10)
        
        self.correct_label = ttk.Label(stats_frame, text="✓ Correct: 0", style='Success.TLabel')
        self.correct_label.grid(row=0, column=2, sticky=tk.W, pady=2)
        
        self.incorrect_label = ttk.Label(stats_frame, text="✗ Incorrect: 0", style='Error.TLabel')
        self.incorrect_label.grid(row=1, column=2, sticky=tk.W, pady=2)
        
        # Performance metrics - right side of stats container
        perf_frame = ttk.LabelFrame(stats_container, text="Performance", padding="10")
        perf_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(3, 0))
        
        self.preprocessing_time_label = ttk.Label(perf_frame, text="Preprocess: -- ms")
        self.preprocessing_time_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.detection_time_label = ttk.Label(perf_frame, text="Detection: -- ms")
        self.detection_time_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.inference_time_label = ttk.Label(perf_frame, text="ML Infer: -- ms")
        self.inference_time_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Zone 2 stats - below stats and performance
        zone2_stats_frame = ttk.LabelFrame(left_frame, text="Zone 2 Triggers", padding="10")
        zone2_stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.zone2_detections_label = ttk.Label(zone2_stats_frame, text="🔍 Detections: 0", style='Info.TLabel')
        self.zone2_detections_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=2)
        
        self.zone2_triggers_label = ttk.Label(zone2_stats_frame, text="⚡ Triggers: 0", style='Warning.TLabel')
        self.zone2_triggers_label.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        stats_container.columnconfigure(0, weight=1)
        stats_container.columnconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)  # Video frame gets extra space
        
        # Middle - Scrollable controls and stats
        middle_frame = ttk.Frame(main_frame, padding="10")
        middle_frame.grid(row=0, column=1, sticky=(tk.N, tk.S), padx=5)
        
        # Create scrollable frame for middle side controls
        middle_canvas = tk.Canvas(middle_frame, highlightthickness=0)
        middle_scrollbar = ttk.Scrollbar(middle_frame, orient=tk.VERTICAL, command=middle_canvas.yview)
        middle_scrollable = ttk.Frame(middle_canvas)
        
        middle_scrollable.bind(
            "<Configure>",
            lambda e: middle_canvas.configure(scrollregion=middle_canvas.bbox("all"))
        )
        
        middle_canvas.create_window((0, 0), window=middle_scrollable, anchor="nw")
        middle_canvas.configure(yscrollcommand=middle_scrollbar.set)
        
        middle_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        middle_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        middle_frame.columnconfigure(0, weight=1)
        middle_frame.rowconfigure(0, weight=1)
        
        # Control buttons
        control_frame = ttk.LabelFrame(middle_scrollable, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.start_camera_btn = ttk.Button(control_frame, text="Start Camera", 
                                           command=self.start_camera, width=20)
        self.start_camera_btn.grid(row=0, column=0, pady=5)
        
        self.load_video_btn = ttk.Button(control_frame, text="Load Video File", 
                                         command=self.load_video, width=20)
        self.load_video_btn.grid(row=1, column=0, pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", 
                                   command=self.stop_detection, 
                                   state=tk.DISABLED, width=20)
        self.stop_btn.grid(row=2, column=0, pady=5)
        
        # Detection toggle button (separate from camera)
        self.detection_btn = ttk.Button(control_frame, text="Start Detection", 
                                        command=self.toggle_detection, 
                                        state=tk.DISABLED, width=20)
        self.detection_btn.grid(row=3, column=0, pady=5)
        
        # Detection status indicator
        self.detection_status_label = ttk.Label(control_frame, text="Detection: OFF", 
                                                foreground="gray")
        self.detection_status_label.grid(row=4, column=0, pady=2)
        
        # Flip options
        flip_frame = ttk.LabelFrame(control_frame, text="Display Options", padding="10")
        flip_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.flip_horizontal_var = tk.BooleanVar(value=False)
        self.flip_horizontal_cb = ttk.Checkbutton(flip_frame, text="Flip Horizontal", 
                                                 variable=self.flip_horizontal_var,
                                                 command=self.update_flip_settings)
        self.flip_horizontal_cb.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.flip_vertical_var = tk.BooleanVar(value=False)
        self.flip_vertical_cb = ttk.Checkbutton(flip_frame, text="Flip Vertical", 
                                                variable=self.flip_vertical_var,
                                                command=self.update_flip_settings)
        self.flip_vertical_cb.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Exposure Time slider (in milliseconds for user convenience)
        ttk.Label(flip_frame, text="Exposure (ms):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.exposure_var = tk.DoubleVar(value=2.000)  # Default 264 μs = 0.264 ms
        self.exposure_scale = ttk.Scale(flip_frame, from_=0.0, to=10.0, 
                                        variable=self.exposure_var, orient=tk.HORIZONTAL, length=150)
        self.exposure_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.exposure_label = ttk.Label(flip_frame, text="2.000")
        self.exposure_label.grid(row=2, column=2, padx=5, pady=2)
        self.exposure_scale.configure(command=lambda v: self.update_exposure_display(float(v)))
        
        # Gain slider
        # Note: Camera SDK uses units where 100 = 1x gain, 5000 = 50x gain
        # Minimum is 100, so slider range is 100-5000 (mapped to 0-500 for user convenience)
        ttk.Label(flip_frame, text="Gain (0-500 → 100-5000):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.gain_var = tk.IntVar(value=500)
        self.gain_scale = ttk.Scale(flip_frame, from_=0, to=500, 
                                    variable=self.gain_var, orient=tk.HORIZONTAL, length=150)
        self.gain_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.gain_label = ttk.Label(flip_frame, text="5000")
        self.gain_label.grid(row=3, column=2, padx=5, pady=2)
        self.gain_scale.configure(command=lambda v: self.update_gain_display(int(float(v))))

        flip_frame.columnconfigure(1, weight=1)
        
        # Zone Editing Mode toggle
        zone_edit_frame = ttk.LabelFrame(control_frame, text="Zone Editing", padding="10")
        zone_edit_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.zone_edit_var = tk.BooleanVar(value=False)
        self.zone_edit_cb = ttk.Checkbutton(zone_edit_frame, text="Enable Zone Editing", 
                                            variable=self.zone_edit_var,
                                            command=self.toggle_zone_edit_mode)
        self.zone_edit_cb.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # Instructions label
        self.zone_edit_instructions = ttk.Label(zone_edit_frame, 
            text="• Click & drag corner to resize\n• Click & drag center to move\n• Green=Zone1, Orange=Zone2",
            font=('TkDefaultFont', 8), foreground='gray')
        self.zone_edit_instructions.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Active zone indicator
        self.active_zone_label = ttk.Label(zone_edit_frame, text="Active: None", foreground='blue')
        self.active_zone_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # ROI Configuration panel - Pixel-based controls
        roi_frame = ttk.LabelFrame(middle_scrollable, text="Zone 1 - Classification Region", padding="10")
        roi_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # ROI X position
        ttk.Label(roi_frame, text="X (pixels):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.roi_x_var = tk.IntVar(value=0)
        self.roi_x_scale = ttk.Scale(roi_frame, from_=0, to=1920, 
                                      variable=self.roi_x_var, orient=tk.HORIZONTAL, length=150)
        self.roi_x_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.roi_x_label = ttk.Label(roi_frame, text="0")
        self.roi_x_label.grid(row=0, column=2, padx=5, pady=2)
        self.roi_x_scale.configure(command=lambda v: self.roi_x_label.config(text=f"{int(float(v))}"))
        
        # ROI Y position
        ttk.Label(roi_frame, text="Y (pixels):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.roi_y_var = tk.IntVar(value=276)
        self.roi_y_scale = ttk.Scale(roi_frame, from_=0, to=1080, 
                                      variable=self.roi_y_var, orient=tk.HORIZONTAL, length=150)
        self.roi_y_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.roi_y_label = ttk.Label(roi_frame, text="276")
        self.roi_y_label.grid(row=1, column=2, padx=5, pady=2)
        self.roi_y_scale.configure(command=lambda v: self.roi_y_label.config(text=f"{int(float(v))}"))
        
        # ROI Width
        ttk.Label(roi_frame, text="Width (pixels):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.roi_width_var = tk.IntVar(value=618)
        self.roi_width_scale = ttk.Scale(roi_frame, from_=1, to=1920, 
                                          variable=self.roi_width_var, orient=tk.HORIZONTAL, length=150)
        self.roi_width_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.roi_width_label = ttk.Label(roi_frame, text="618")
        self.roi_width_label.grid(row=2, column=2, padx=5, pady=2)
        self.roi_width_scale.configure(command=lambda v: self.roi_width_label.config(text=f"{int(float(v))}"))
        
        # ROI Height
        ttk.Label(roi_frame, text="Height (pixels):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.roi_height_var = tk.IntVar(value=618)
        self.roi_height_scale = ttk.Scale(roi_frame, from_=1, to=1080, 
                                           variable=self.roi_height_var, orient=tk.HORIZONTAL, length=150)
        self.roi_height_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.roi_height_label = ttk.Label(roi_frame, text="618")
        self.roi_height_label.grid(row=3, column=2, padx=5, pady=2)
        self.roi_height_scale.configure(command=lambda v: self.roi_height_label.config(text=f"{int(float(v))}"))
        
        # Apply ROI button
        self.apply_roi_btn = ttk.Button(roi_frame, text="Apply Zone 1 Settings", 
                                        command=self.apply_roi_settings, width=20)
        self.apply_roi_btn.grid(row=4, column=0, columnspan=3, pady=5)
        
        # Current ROI display
        self.roi_info_label = ttk.Label(roi_frame, text="ROI: Not initialized", 
                                        relief=tk.SUNKEN, anchor=tk.W)
        self.roi_info_label.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2)
        
        roi_frame.columnconfigure(1, weight=1)
        
        # === ZONE 2 (Trigger Zone) Configuration ===
        zone2_frame = ttk.LabelFrame(middle_scrollable, text="Zone 2 - Trigger Zone", padding="10")
        zone2_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Zone 2 Enable checkbox
        self.zone2_enabled_var = tk.BooleanVar(value=True)
        self.zone2_enabled_cb = ttk.Checkbutton(zone2_frame, text="Enable Zone 2 (Trigger)", 
                                                 variable=self.zone2_enabled_var,
                                                 command=self.apply_zone2_settings)
        self.zone2_enabled_cb.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # Zone 2 X position
        ttk.Label(zone2_frame, text="X (pixels):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.zone2_x_var = tk.IntVar(value=1140)
        self.zone2_x_scale = ttk.Scale(zone2_frame, from_=0, to=1920, 
                                        variable=self.zone2_x_var, orient=tk.HORIZONTAL, length=150)
        self.zone2_x_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.zone2_x_label = ttk.Label(zone2_frame, text="1140")
        self.zone2_x_label.grid(row=1, column=2, padx=5, pady=2)
        self.zone2_x_scale.configure(command=lambda v: self.zone2_x_label.config(text=f"{int(float(v))}"))
        
        # Zone 2 Y position
        ttk.Label(zone2_frame, text="Y (pixels):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.zone2_y_var = tk.IntVar(value=276)
        self.zone2_y_scale = ttk.Scale(zone2_frame, from_=0, to=1080, 
                                        variable=self.zone2_y_var, orient=tk.HORIZONTAL, length=150)
        self.zone2_y_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.zone2_y_label = ttk.Label(zone2_frame, text="276")
        self.zone2_y_label.grid(row=2, column=2, padx=5, pady=2)
        self.zone2_y_scale.configure(command=lambda v: self.zone2_y_label.config(text=f"{int(float(v))}"))
        
        # Zone 2 Width
        ttk.Label(zone2_frame, text="Width (pixels):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.zone2_width_var = tk.IntVar(value=618)
        self.zone2_width_scale = ttk.Scale(zone2_frame, from_=1, to=1920, 
                                            variable=self.zone2_width_var, orient=tk.HORIZONTAL, length=150)
        self.zone2_width_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.zone2_width_label = ttk.Label(zone2_frame, text="618")
        self.zone2_width_label.grid(row=3, column=2, padx=5, pady=2)
        self.zone2_width_scale.configure(command=lambda v: self.zone2_width_label.config(text=f"{int(float(v))}"))
        
        # Zone 2 Height
        ttk.Label(zone2_frame, text="Height (pixels):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.zone2_height_var = tk.IntVar(value=618)
        self.zone2_height_scale = ttk.Scale(zone2_frame, from_=1, to=1080, 
                                             variable=self.zone2_height_var, orient=tk.HORIZONTAL, length=150)
        self.zone2_height_scale.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.zone2_height_label = ttk.Label(zone2_frame, text="618")
        self.zone2_height_label.grid(row=4, column=2, padx=5, pady=2)
        self.zone2_height_scale.configure(command=lambda v: self.zone2_height_label.config(text=f"{int(float(v))}"))
        
        # Zone 1 Decision Window (how long a 'Keep' is valid)
        ttk.Label(zone2_frame, text="Decision Window (ms):").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.zone1_window_var = tk.IntVar(value=2000)
        self.zone1_window_scale = ttk.Scale(zone2_frame, from_=100, to=5000, 
                                             variable=self.zone1_window_var, orient=tk.HORIZONTAL, length=150)
        self.zone1_window_scale.grid(row=5, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.zone1_window_label = ttk.Label(zone2_frame, text="2000")
        self.zone1_window_label.grid(row=5, column=2, padx=5, pady=2)
        self.zone1_window_scale.configure(command=lambda v: self.zone1_window_label.config(text=f"{int(float(v))}"))
        
        # Apply Zone 2 button
        self.apply_zone2_btn = ttk.Button(zone2_frame, text="Apply Zone 2 Settings", 
                                          command=self.apply_zone2_settings, width=20)
        self.apply_zone2_btn.grid(row=6, column=0, columnspan=3, pady=5)
        
        # Zone 2 status display
        self.zone2_info_label = ttk.Label(zone2_frame, text="Zone 2: Not initialized", 
                                          relief=tk.SUNKEN, anchor=tk.W)
        self.zone2_info_label.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2)
        
        zone2_frame.columnconfigure(1, weight=1)
        
        # Right side - FIXED panels (no scrolling)
        right_frame = ttk.Frame(main_frame, padding="10")
        right_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Classification log - top right
        classification_frame = ttk.LabelFrame(right_frame, text="Classification Log", padding="10")
        classification_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create treeview for classification log
        columns = ('Frame', 'Detection', 'Label', 'Speed (px/fr)', 'Time (ms)')
        self.classification_tree = ttk.Treeview(classification_frame, columns=columns, show='headings', height=10)
        for col in columns:
            self.classification_tree.heading(col, text=col)
            if col == 'Speed (px/fr)':
                self.classification_tree.column(col, width=70)
            else:
                self.classification_tree.column(col, width=55)
        
        # Scrollbar for classification log
        class_scrollbar = ttk.Scrollbar(classification_frame, orient=tk.VERTICAL, command=self.classification_tree.yview)
        self.classification_tree.configure(yscrollcommand=class_scrollbar.set)
        
        self.classification_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        class_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.clear_class_log_btn = ttk.Button(classification_frame, text="Clear Log", command=self.clear_classification_log)
        self.clear_class_log_btn.grid(row=1, column=0, sticky=(tk.E), pady=(5, 0))
        
        classification_frame.columnconfigure(0, weight=1)
        classification_frame.rowconfigure(0, weight=1)
        
        # System log - bottom right
        log_frame = ttk.LabelFrame(right_frame, text="System Log", padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=35, height=25, 
                                                   state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        middle_scrollable.columnconfigure(0, weight=1)
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)  # Video gets more space
        main_frame.columnconfigure(1, weight=1)  # Controls
        main_frame.columnconfigure(2, weight=1)  # Right panels (log + classification)
        main_frame.rowconfigure(0, weight=1)
        
        # Store reference to main_frame for resize calculations
        self.main_frame = main_frame
        
    def _on_window_resize(self, event):
        """Handle window resize to update video display size"""
        # Only process resize events from the root window
        if event.widget != self.root:
            return
        
        # Debounce resize events
        if self._resize_scheduled:
            return
        self._resize_scheduled = True
        self.root.after(100, self._update_display_size)
    
    def _update_display_size(self):
        """Calculate and update the display size based on available space"""
        self._resize_scheduled = False
        
        try:
            # Get the available width for the video frame
            # Account for padding, other columns, and scrollbar
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            
            # Estimate space for other elements (controls ~280px, right panel ~350px, padding ~60px)
            available_width = max(320, window_width - 690)
            
            # Calculate height based on video frame available space
            # Stats take ~140px, padding ~50px
            available_height = max(240, window_height - 250)
            
            # Maintain aspect ratio (assume 4:3 camera)
            aspect_ratio = 4 / 3
            
            # Fit within available space
            if available_width / aspect_ratio <= available_height:
                self.display_width = available_width
                self.display_height = int(available_width / aspect_ratio)
            else:
                self.display_height = available_height
                self.display_width = int(available_height * aspect_ratio)
            
            # Minimum sizes
            self.display_width = max(320, min(self.display_width, 1280))
            self.display_height = max(240, min(self.display_height, 960))
            
        except Exception:
            # Fallback to default size
            self.display_width = 640
            self.display_height = 480
        
    def log_message(self, message):
        """Add message to log window (thread-safe)"""
        # Put message in queue - can be called from any thread
        try:
            self.log_queue.put_nowait(message)
        except queue.Full:
            pass  # Drop message if queue is full
    
    def process_log_queue(self):
        """Process log messages from queue (called from main thread)"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)  # Auto-scroll to bottom
                self.log_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        # Schedule next check
        self.root.after(100, self.process_log_queue)
    
    def clear_classification_log(self):
        """Clear the classification log treeview"""
        for item in self.classification_tree.get_children():
            self.classification_tree.delete(item)
        
    def update_stats(self):
        """Update statistics display (thread-safe)"""
        with self._lock:
            detector = self.detector
        
        if detector:
            try:
                # Access detector attributes safely (read-only)
                frame_count = detector.frame_count
                detection_count = detector.detection_count
                
                self.frames_label.config(text=f"📊 Frames: {frame_count}")
                self.detections_label.config(text=f"🎯 Detections: {detection_count}")
                
                # Update FPS from detector (actual processing rate)
                if hasattr(detector, 'current_fps'):
                    self.fps_label.config(text=f"⚡ FPS: {detector.current_fps:.1f}")
                else:
                    self.fps_label.config(text=f"⚡ FPS: --")
            except (AttributeError, RuntimeError):
                # Detector may be None or in inconsistent state
                pass
            
            # Update ROI info display
            try:
                with self._lock:
                    detector = self.detector
                if detector and detector.width > 0:
                    if hasattr(detector, 'roi_width') and detector.roi_width > 0:
                        self.roi_info_label.config(
                            text=f"ROI: x={detector.roi_x}, y={detector.roi_y}, "
                                 f"w={detector.roi_width}, h={detector.roi_height} "
                                 f"(Frame: {detector.width}x{detector.height})"
                        )
                    else:
                        roi_height = detector.roi_end_y - detector.roi_start_y
                        self.roi_info_label.config(
                            text=f"ROI: y={detector.roi_start_y}-{detector.roi_end_y} "
                                 f"(h={roi_height}px, {detector.width}x{detector.height})"
                        )
                    
                    if hasattr(detector, 'classification_stats'):
                        stats = detector.classification_stats
                        self.correct_label.config(text=f"✓ Correct: {stats.get('Correct', 0)}")
                        self.incorrect_label.config(text=f"✗ Incorrect: {stats.get('Incorrect', 0)}")
                    
                    if hasattr(detector, 'zone2_detection_count'):
                        self.zone2_detections_label.config(text=f"🔍 Detections: {detector.zone2_detection_count}")
                    if hasattr(detector, 'zone2_trigger_count'):
                        self.zone2_triggers_label.config(text=f"⚡ Triggers: {detector.zone2_trigger_count}")
                    
                    # Update Zone 2 info display
                    if hasattr(detector, 'zone2_width') and detector.zone2_width > 0:
                        status = "enabled" if detector.zone2_enabled else "disabled"
                        self.zone2_info_label.config(
                            text=f"Zone 2: x={detector.zone2_x}, y={detector.zone2_y}, "
                                 f"w={detector.zone2_width}, h={detector.zone2_height} ({status})"
                        )
                    
                    # Update performance metrics
                    if hasattr(detector, 'get_performance_stats'):
                        perf_stats = detector.get_performance_stats()
                        
                        prep_ms = perf_stats.get('preprocessing_time', {}).get('mean', 0) * 1000
                        self.preprocessing_time_label.config(text=f"Preprocess: {prep_ms:.1f} ms")
                        
                        det_ms = perf_stats.get('detection_time', {}).get('mean', 0) * 1000
                        self.detection_time_label.config(text=f"Detection: {det_ms:.1f} ms")
                        
                        inf_ms = perf_stats.get('inference_time', {}).get('mean', 0) * 1000
                        self.inference_time_label.config(text=f"ML Infer: {inf_ms:.1f} ms")
                    
                    # Update classification log (only if history changed to avoid expensive rebuilds)
                    if hasattr(detector, 'classification_history'):
                        history = detector.classification_history
                        current_count = len(history)
                        # Only rebuild if we have new entries (check if count changed)
                        if not hasattr(self, '_last_history_count') or self._last_history_count != current_count:
                            # Clear existing items
                            for item in self.classification_tree.get_children():
                                self.classification_tree.delete(item)
                            
                            # Add recent classifications (last 50)
                            for entry in history[-50:]:
                                frame_idx = entry.get('frame_index', 'N/A')
                                det_idx = entry.get('detection_index', 0)
                                label = entry.get('label', 'Unknown')
                                inf_time = entry.get('inference_time', 0) * 1000
                                speed = entry.get('speed', None)
                                
                                # Format speed display
                                if speed is not None:
                                    speed_str = f"{speed:.2f}"
                                else:
                                    speed_str = "--"
                                
                                # Color code: green for Correct, red for Incorrect
                                item_id = self.classification_tree.insert('', 'end', values=(
                                    frame_idx, det_idx, label, speed_str, f"{inf_time:.2f}"
                                ))
                                if label == 'Correct':
                                    self.classification_tree.set(item_id, 'Label', '✓ Correct')
                                elif label == 'Incorrect':
                                    self.classification_tree.set(item_id, 'Label', '✗ Incorrect')
                            
                            # Auto-scroll to bottom
                            children = self.classification_tree.get_children()
                            if children:
                                self.classification_tree.see(children[-1])
                            
                            self._last_history_count = current_count
            except (AttributeError, RuntimeError):
                pass
    
    def update_flip_settings(self):
        """Update flip settings from checkboxes"""
        with self._lock:
            self.flip_horizontal = self.flip_horizontal_var.get()
            self.flip_vertical = self.flip_vertical_var.get()
    
    def toggle_detection(self):
        """Toggle detection ON/OFF while camera keeps running"""
        with self._lock:
            detector = self.detector
            
        if not detector:
            self.log_message("Cannot toggle detection: Camera not running")
            return
        
        # Toggle detection state
        self.detection_enabled = not self.detection_enabled
        detector.detection_enabled = self.detection_enabled
        
        if self.detection_enabled:
            self.detection_btn.config(text="Stop Detection")
            self.detection_status_label.config(text="Detection: ON", foreground="green")
            self.log_message("Detection ENABLED - now detecting embryos")
        else:
            self.detection_btn.config(text="Start Detection")
            self.detection_status_label.config(text="Detection: OFF", foreground="gray")
            self.log_message("Detection DISABLED - camera still running")
    
    def update_wb_temp_display(self, value):
        """Update white balance temperature label and apply to camera"""
        # Value is in Kelvin (2000-8000K)
        self.wb_temp_label.config(text=f"{value}K")
        with self._lock:
            self.wb_temp_k = value
        
        # Cancel any pending update
        if self._wb_update_job is not None:
            self.root.after_cancel(self._wb_update_job)
        
        # Schedule update after 300ms of no changes (debouncing)
        self._wb_update_job = self.root.after(300, self._apply_wb_immediately)
    
    def update_wb_tint_display(self, value):
        """Update white balance tint label and apply to camera"""
        # Value ranges -50 (Magenta) to +50 (Green)
        if value < 0:
            tint_text = f"Magenta {abs(value)}"
        elif value > 0:
            tint_text = f"Green {value}"
        else:
            tint_text = "Neutral"
        
        self.wb_tint_label.config(text=tint_text)
        with self._lock:
            self.wb_tint = value
        
        # Cancel any pending update
        if self._wb_update_job is not None:
            self.root.after_cancel(self._wb_update_job)
        
        # Schedule update after 300ms of no changes (debouncing)
        self._wb_update_job = self.root.after(300, self._apply_wb_immediately)
    
    def _apply_wb_immediately(self):
        """Apply white balance (temperature + tint) setting to camera immediately"""
        with self._lock:
            detector = self.detector
            wb_temp_k = self.wb_temp_k
            wb_tint = self.wb_tint
        
        if detector and hasattr(detector, 'hcam') and detector.hcam is not None:
            try:
                # Map temperature: 2000K (warm/yellow) to 8000K (cool/blue)
                # Normalize to 0-255 range for camera SDK
                wb_temp_value = int((wb_temp_k - 2000) / (8000 - 2000) * 255)
                
                # Map tint: -50 (Magenta/Red) to +50 (Green/Cyan)
                # Normalize to 0-255 range
                wb_tint_value = int((wb_tint + 50) / 100 * 255)
                
                # Try to apply white balance via camera SDK
                # Different SDKs may have different methods
                import amcam
                
                # Method 1: Separate temperature and tint options
                if hasattr(amcam, 'AMCAM_OPTION_WHITEBALANCE_TEMP') and hasattr(amcam, 'AMCAM_OPTION_WHITEBALANCE_TINT'):
                    try:
                        detector.hcam.put_Option(amcam.AMCAM_OPTION_WHITEBALANCE_TEMP, wb_temp_value)
                        detector.hcam.put_Option(amcam.AMCAM_OPTION_WHITEBALANCE_TINT, wb_tint_value)
                        self.log_message(f"White Balance: Temp={wb_temp_k}K, Tint={'Magenta' if wb_tint < 0 else 'Green' if wb_tint > 0 else 'Neutral'}")
                    except Exception as e:
                        self.log_message(f"Failed to set WB via separate options: {e}")
                
                # Method 2: Combined white balance value (packed format)
                elif hasattr(amcam, 'AMCAM_OPTION_WHITEBALANCE'):
                    try:
                        # Pack temp (high byte) and tint (low byte)
                        wb_packed = (wb_temp_value << 8) | wb_tint_value
                        detector.hcam.put_Option(amcam.AMCAM_OPTION_WHITEBALANCE, wb_packed)
                        self.log_message(f"White Balance: Temp={wb_temp_k}K, Tint={'Magenta' if wb_tint < 0 else 'Green' if wb_tint > 0 else 'Neutral'}")
                    except Exception as e:
                        self.log_message(f"Failed to set WB via packed option: {e}")
                
                # Method 3: Try put_WhiteBalance if available
                elif hasattr(detector.hcam, 'put_WhiteBalance'):
                    try:
                        # Some cameras may accept (temp, tint) tuple or combined value
                        detector.hcam.put_WhiteBalance(wb_temp_value, wb_tint_value)
                        self.log_message(f"White Balance: Temp={wb_temp_k}K, Tint={'Magenta' if wb_tint < 0 else 'Green' if wb_tint > 0 else 'Neutral'}")
                    except TypeError:
                        # Try single value
                        detector.hcam.put_WhiteBalance(wb_temp_value)
                        self.log_message(f"White Balance: Temp={wb_temp_k}K")
                    except Exception as e:
                        self.log_message(f"Failed to set WB: {e}")
                
            except Exception as e:
                self.log_message(f"Failed to update white balance: {e}")
        
        self._wb_update_job = None
    
    def update_exposure_display(self, value):
        """Update exposure label and apply to camera automatically"""
        exposure_ms = float(value)
        # Display with 3 decimal places for precision
        self.exposure_label.config(text=f"{exposure_ms:.3f}")
        # Convert milliseconds to microseconds for camera SDK
        with self._lock:
            self.exposure_time_us = int(exposure_ms * 1000)
        
        # Cancel any pending update
        if self._exposure_update_job is not None:
            self.root.after_cancel(self._exposure_update_job)
        
        # Schedule update after 300ms of no changes (debouncing)
        self._exposure_update_job = self.root.after(300, self._apply_exposure_immediately)
    
    def _apply_exposure_immediately(self):
        """Apply exposure setting to camera immediately"""
        with self._lock:
            detector = self.detector
            exposure_us = self.exposure_time_us
        
        if detector and hasattr(detector, 'hcam') and detector.hcam is not None:
            try:
                detector.update_exposure_gain(exposure_time_us=exposure_us, gain=None)
            except Exception as e:
                self.log_message(f"Failed to update exposure: {e}")
        self._exposure_update_job = None
    
    def update_gain_display(self, value):
        """Update gain label and apply to camera automatically"""
        # Map 0-500 slider to 100-5000 camera SDK units
        # Formula: camera_gain = 100 + (value * (5000 - 100) / 500)
        camera_gain = 100 + int(float(value) * (5000 - 100) / 500)
        self.gain_label.config(text=f"{camera_gain}")
        with self._lock:
            self.gain = camera_gain
        
        # Cancel any pending update
        if self._gain_update_job is not None:
            self.root.after_cancel(self._gain_update_job)
        
        # Schedule update after 300ms of no changes (debouncing)
        self._gain_update_job = self.root.after(300, self._apply_gain_immediately)
    
    def _apply_gain_immediately(self):
        """Apply gain setting to camera immediately"""
        with self._lock:
            detector = self.detector
            gain = self.gain
        
        if detector and hasattr(detector, 'hcam') and detector.hcam is not None:
            try:
                detector.update_exposure_gain(exposure_time_us=None, gain=gain)
            except Exception as e:
                self.log_message(f"Failed to update gain: {e}")
        self._gain_update_job = None
    
    def apply_roi_settings(self):
        """Apply ROI settings from GUI controls"""
        with self._lock:
            detector = self.detector
            
        if not detector or detector.width == 0:
            self.log_message("Cannot apply ROI: Camera/Video not running")
            return
        
        # Get values from sliders
        roi_x = self.roi_x_var.get()
        roi_y = self.roi_y_var.get()
        roi_width = self.roi_width_var.get()
        roi_height = self.roi_height_var.get()
        
        # Validate ROI stays within frame bounds
        if roi_x + roi_width > detector.width:
            roi_width = detector.width - roi_x
            self.roi_width_var.set(roi_width)
            self.roi_width_label.config(text=f"{roi_width}")
            self.log_message(f"Warning: Width adjusted to {roi_width} to fit frame")
        
        if roi_y + roi_height > detector.height:
            roi_height = detector.height - roi_y
            self.roi_height_var.set(roi_height)
            self.roi_height_label.config(text=f"{roi_height}")
            self.log_message(f"Warning: Height adjusted to {roi_height} to fit frame")
        
        # Update detector ROI (Zone 1)
        success = detector.update_roi(x=roi_x, y=roi_y, width=roi_width, height=roi_height)
        
        if success:
            self.log_message(f"Zone 1 ROI updated: x={roi_x}, y={roi_y}, w={roi_width}, h={roi_height}")
            self.roi_info_label.config(
                text=f"Zone 1: x={roi_x}, y={roi_y}, w={roi_width}, h={roi_height} "
                     f"(Frame: {detector.width}x{detector.height})"
            )
        else:
            self.log_message("Failed to update Zone 1 ROI")
    
    def apply_zone2_settings(self):
        """Apply Zone 2 (trigger zone) settings from GUI controls"""
        with self._lock:
            detector = self.detector
            
        if not detector or detector.width == 0:
            self.log_message("Cannot apply Zone 2: Camera/Video not running")
            return
        
        # Get values from sliders
        zone2_enabled = self.zone2_enabled_var.get()
        zone2_x = self.zone2_x_var.get()
        zone2_y = self.zone2_y_var.get()
        zone2_width = self.zone2_width_var.get()
        zone2_height = self.zone2_height_var.get()
        zone1_window = self.zone1_window_var.get()
        
        # Validate Zone 2 stays within frame bounds
        if zone2_x + zone2_width > detector.width:
            zone2_width = detector.width - zone2_x
            self.zone2_width_var.set(zone2_width)
            self.zone2_width_label.config(text=f"{zone2_width}")
            self.log_message(f"Warning: Zone 2 width adjusted to {zone2_width} to fit frame")
        
        if zone2_y + zone2_height > detector.height:
            zone2_height = detector.height - zone2_y
            self.zone2_height_var.set(zone2_height)
            self.zone2_height_label.config(text=f"{zone2_height}")
            self.log_message(f"Warning: Zone 2 height adjusted to {zone2_height} to fit frame")
        
        # Update detector Zone 2 ROI
        success = detector.update_zone2_roi(
            x=zone2_x, y=zone2_y, width=zone2_width, height=zone2_height, enabled=zone2_enabled
        )
        
        # Update Zone 1 decision window
        detector.update_zone1_decision_window(zone1_window)
        
        if success:
            status = "enabled" if zone2_enabled else "disabled"
            self.log_message(f"Zone 2 updated: x={zone2_x}, y={zone2_y}, w={zone2_width}, h={zone2_height} ({status})")
            self.zone2_info_label.config(
                text=f"Zone 2: x={zone2_x}, y={zone2_y}, w={zone2_width}, h={zone2_height} ({status})"
            )
        else:
            self.log_message("Failed to update Zone 2 settings")

    def update_roi_from_detector(self):
        """Update ROI sliders from detector's current values (both zones)"""
        with self._lock:
            detector = self.detector
            
        if detector and detector.width > 0:
            # Update slider ranges to match current frame dimensions
            self.roi_x_scale.configure(to=max(0, detector.width - 1))
            self.roi_y_scale.configure(to=max(0, detector.height - 1))
            self.roi_width_scale.configure(to=detector.width)
            self.roi_height_scale.configure(to=detector.height)
            
            # Zone 2 slider ranges
            self.zone2_x_scale.configure(to=max(0, detector.width - 1))
            self.zone2_y_scale.configure(to=max(0, detector.height - 1))
            self.zone2_width_scale.configure(to=detector.width)
            self.zone2_height_scale.configure(to=detector.height)
            
            # Update Zone 1 slider values from detector
            if hasattr(detector, 'roi_x') and detector.roi_width > 0:
                self.roi_x_var.set(detector.roi_x)
                self.roi_y_var.set(detector.roi_y)
                self.roi_width_var.set(detector.roi_width)
                self.roi_height_var.set(detector.roi_height)
                self.roi_x_label.config(text=f"{detector.roi_x}")
                self.roi_y_label.config(text=f"{detector.roi_y}")
                self.roi_width_label.config(text=f"{detector.roi_width}")
                self.roi_height_label.config(text=f"{detector.roi_height}")
            
            # Update Zone 2 slider values from detector
            if hasattr(detector, 'zone2_x') and detector.zone2_width > 0:
                self.zone2_enabled_var.set(detector.zone2_enabled)
                self.zone2_x_var.set(detector.zone2_x)
                self.zone2_y_var.set(detector.zone2_y)
                self.zone2_width_var.set(detector.zone2_width)
                self.zone2_height_var.set(detector.zone2_height)
                self.zone2_x_label.config(text=f"{detector.zone2_x}")
                self.zone2_y_label.config(text=f"{detector.zone2_y}")
                self.zone2_width_label.config(text=f"{detector.zone2_width}")
                self.zone2_height_label.config(text=f"{detector.zone2_height}")
            
            # Update Zone 1 decision window
            if hasattr(detector, 'zone1_decision_window_ms'):
                self.zone1_window_var.set(detector.zone1_decision_window_ms)
                self.zone1_window_label.config(text=f"{detector.zone1_decision_window_ms}")
            
            # Update exposure and gain sliders from detector
            if hasattr(detector, 'exposure_time_us'):
                # Convert from microseconds to milliseconds for display
                exposure_ms = detector.exposure_time_us / 1000.0
                self.exposure_var.set(exposure_ms)
                self.exposure_label.config(text=f"{exposure_ms:.3f}")
            if hasattr(detector, 'gain'):
                # Convert from camera SDK range (100-5000) to slider range (0-500)
                camera_gain = detector.gain
                slider_value = int((camera_gain - 100) * 500 / (5000 - 100))
                slider_value = max(0, min(500, slider_value))  # Clamp to slider range
                self.gain_var.set(slider_value)
                self.gain_label.config(text=f"{camera_gain}")
    
    # ==================== ZONE EDITING METHODS ====================
    
    def _bind_mouse_events(self):
        """Bind mouse events to video label for zone editing"""
        self.video_label.bind('<Button-1>', self._on_mouse_down)
        self.video_label.bind('<B1-Motion>', self._on_mouse_drag)
        self.video_label.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.video_label.bind('<Motion>', self._on_mouse_move)
    
    def toggle_zone_edit_mode(self):
        """Toggle zone editing mode on/off"""
        self.roi_edit_mode = self.zone_edit_var.get()
        if self.roi_edit_mode:
            self.log_message("Zone editing enabled - click and drag to move/resize zones")
            self.video_label.config(cursor="crosshair")
        else:
            self.log_message("Zone editing disabled")
            self.video_label.config(cursor="")
            self.active_zone = None
            self.active_zone_label.config(text="Active: None")
    
    def _draw_zone_handles(self, frame, x, y, w, h, color):
        """Draw resize handles on zone corners and edges.
        
        Handles are scaled based on frame size so they remain clickable after
        the frame is resized for display (960px width).
        """
        frame_h, frame_w = frame.shape[:2]
        # Scale handle size based on frame-to-display ratio
        # Frame is resized to 960px width for display, so handles need to be larger
        # in frame coords to appear the same size in display coords
        scale_factor = frame_w / self.display_width if frame_w > 0 else 1.0
        handle_size = int(12 * scale_factor)  # ~12px in display coords
        handle_size_edge = int(10 * scale_factor)  # ~10px in display coords
        border_thickness = max(1, int(2 * scale_factor))  # Scale border too
        
        # Corner handles (filled squares)
        corners = [
            (x, y),                    # NW
            (x + w, y),                # NE
            (x, y + h),                # SW
            (x + w, y + h),            # SE
        ]
        
        for cx, cy in corners:
            cv2.rectangle(frame, 
                         (cx - handle_size, cy - handle_size),
                         (cx + handle_size, cy + handle_size),
                         color, -1)  # Filled
            cv2.rectangle(frame, 
                         (cx - handle_size, cy - handle_size),
                         (cx + handle_size, cy + handle_size),
                         (255, 255, 255), border_thickness)  # White border
        
        # Edge handles (smaller rectangles at midpoints)
        edges = [
            (x + w // 2, y),           # N
            (x + w // 2, y + h),       # S
            (x, y + h // 2),           # W
            (x + w, y + h // 2),       # E
        ]
        
        for ex, ey in edges:
            cv2.rectangle(frame,
                         (ex - handle_size_edge, ey - handle_size_edge),
                         (ex + handle_size_edge, ey + handle_size_edge),
                         color, -1)
            cv2.rectangle(frame,
                         (ex - handle_size_edge, ey - handle_size_edge),
                         (ex + handle_size_edge, ey + handle_size_edge),
                         (255, 255, 255), border_thickness)
    
    def _display_to_frame_coords(self, display_x, display_y):
        """Convert display coordinates to actual frame coordinates"""
        with self._lock:
            detector = self.detector
        
        if not detector or detector.width == 0:
            return display_x, display_y
        
        # Calculate scale factor
        frame_w, frame_h = detector.width, detector.height
        display_h = int(frame_h * self.display_width / frame_w)
        
        scale_x = frame_w / self.display_width
        scale_y = frame_h / display_h
        
        # Account for flip if enabled
        flip_h = self.flip_horizontal_var.get()
        flip_v = self.flip_vertical_var.get()
        
        if flip_h:
            display_x = self.display_width - display_x
        if flip_v:
            display_x = display_x  # y flip doesn't affect x
            display_y = display_h - display_y
        
        frame_x = int(display_x * scale_x)
        frame_y = int(display_y * scale_y)
        
        return frame_x, frame_y
    
    def _frame_to_display_coords(self, frame_x, frame_y, frame_w, frame_h):
        """Convert frame coordinates to display coordinates"""
        display_h = int(frame_h * self.display_width / frame_w)
        
        scale_x = self.display_width / frame_w
        scale_y = display_h / frame_h
        
        display_x = int(frame_x * scale_x)
        display_y = int(frame_y * scale_y)
        
        # Account for flip if enabled
        flip_h = self.flip_horizontal_var.get()
        flip_v = self.flip_vertical_var.get()
        
        if flip_h:
            display_x = self.display_width - display_x
        if flip_v:
            display_y = display_h - display_y
        
        return display_x, display_y
    
    def _get_zone_at_point(self, frame_x, frame_y):
        """Determine which zone and which part (corner/edge/center) is at the given frame point.
        Returns (zone_name, handle_type) or (None, None) if no zone at point.
        handle_type can be: 'nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w', 'center'
        """
        with self._lock:
            detector = self.detector
        
        if not detector:
            return None, None
        
        # Scale handle margin based on frame-to-display ratio (matches _draw_zone_handles)
        frame_w = detector.width if detector.width > 0 else 1824
        scale_factor = frame_w / self.display_width
        handle_margin = int(20 * scale_factor)  # ~20px margin in display coords
        
        zones = []
        
        # Zone 1
        if hasattr(detector, 'roi_width') and detector.roi_width > 0:
            zones.append(('zone1', detector.roi_x, detector.roi_y, 
                         detector.roi_width, detector.roi_height))
        
        # Zone 2
        if hasattr(detector, 'zone2_width') and detector.zone2_width > 0 and detector.zone2_enabled:
            zones.append(('zone2', detector.zone2_x, detector.zone2_y,
                         detector.zone2_width, detector.zone2_height))
        
        for zone_name, zx, zy, zw, zh in zones:
            # Check corners first (higher priority)
            corners = [
                ('nw', zx, zy),
                ('ne', zx + zw, zy),
                ('sw', zx, zy + zh),
                ('se', zx + zw, zy + zh),
            ]
            
            for handle, hx, hy in corners:
                if abs(frame_x - hx) < handle_margin and abs(frame_y - hy) < handle_margin:
                    return zone_name, handle
            
            # Check edges
            edges = [
                ('n', zx + zw/2, zy, zw, handle_margin),  # top edge
                ('s', zx + zw/2, zy + zh, zw, handle_margin),  # bottom edge
                ('w', zx, zy + zh/2, handle_margin, zh),  # left edge
                ('e', zx + zw, zy + zh/2, handle_margin, zh),  # right edge
            ]
            
            for handle, hx, hy, hw, hh in edges:
                if handle in ('n', 's'):
                    if zx < frame_x < zx + zw and abs(frame_y - hy) < handle_margin:
                        return zone_name, handle
                else:  # 'w', 'e'
                    if zy < frame_y < zy + zh and abs(frame_x - hx) < handle_margin:
                        return zone_name, handle
            
            # Check center (inside zone)
            if zx < frame_x < zx + zw and zy < frame_y < zy + zh:
                return zone_name, 'center'
        
        return None, None
    
    def _on_mouse_move(self, event):
        """Update cursor based on position when in edit mode"""
        if not self.roi_edit_mode:
            return
        
        frame_x, frame_y = self._display_to_frame_coords(event.x, event.y)
        zone, handle = self._get_zone_at_point(frame_x, frame_y)
        
        if zone:
            cursor_map = {
                'nw': 'top_left_corner',
                'ne': 'top_right_corner',
                'sw': 'bottom_left_corner',
                'se': 'bottom_right_corner',
                'n': 'sb_v_double_arrow',
                's': 'sb_v_double_arrow',
                'e': 'sb_h_double_arrow',
                'w': 'sb_h_double_arrow',
                'center': 'fleur',  # move cursor
            }
            self.video_label.config(cursor=cursor_map.get(handle, 'crosshair'))
        else:
            self.video_label.config(cursor='crosshair')
    
    def _on_mouse_down(self, event):
        """Handle mouse button press - start drag operation"""
        if not self.roi_edit_mode:
            return
        
        frame_x, frame_y = self._display_to_frame_coords(event.x, event.y)
        zone, handle = self._get_zone_at_point(frame_x, frame_y)
        
        if zone and handle:
            self.active_zone = zone
            self.roi_drag_mode = handle
            self.roi_drag_start = (frame_x, frame_y)
            
            # Store initial zone position/size for relative movement
            with self._lock:
                detector = self.detector
            
            if detector:
                if zone == 'zone1':
                    self._drag_initial = (detector.roi_x, detector.roi_y, 
                                         detector.roi_width, detector.roi_height)
                else:
                    self._drag_initial = (detector.zone2_x, detector.zone2_y,
                                         detector.zone2_width, detector.zone2_height)
            
            zone_display = "Zone 1 (Classification)" if zone == 'zone1' else "Zone 2 (Trigger)"
            self.active_zone_label.config(text=f"Active: {zone_display}")
            self.log_message(f"Started {handle} drag on {zone_display}")
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag - update zone position/size"""
        if not self.roi_edit_mode or not self.roi_drag_start or not self.active_zone:
            return
        
        frame_x, frame_y = self._display_to_frame_coords(event.x, event.y)
        start_x, start_y = self.roi_drag_start
        init_x, init_y, init_w, init_h = self._drag_initial
        
        dx = frame_x - start_x
        dy = frame_y - start_y
        
        with self._lock:
            detector = self.detector
        
        if not detector:
            return
        
        # Calculate new position/size based on drag mode
        new_x, new_y, new_w, new_h = init_x, init_y, init_w, init_h
        
        if self.roi_drag_mode == 'center':
            # Move the entire zone
            new_x = init_x + dx
            new_y = init_y + dy
        elif self.roi_drag_mode == 'nw':
            # Resize from top-left corner
            new_x = init_x + dx
            new_y = init_y + dy
            new_w = init_w - dx
            new_h = init_h - dy
        elif self.roi_drag_mode == 'ne':
            # Resize from top-right corner
            new_y = init_y + dy
            new_w = init_w + dx
            new_h = init_h - dy
        elif self.roi_drag_mode == 'sw':
            # Resize from bottom-left corner
            new_x = init_x + dx
            new_w = init_w - dx
            new_h = init_h + dy
        elif self.roi_drag_mode == 'se':
            # Resize from bottom-right corner
            new_w = init_w + dx
            new_h = init_h + dy
        elif self.roi_drag_mode == 'n':
            # Resize from top edge
            new_y = init_y + dy
            new_h = init_h - dy
        elif self.roi_drag_mode == 's':
            # Resize from bottom edge
            new_h = init_h + dy
        elif self.roi_drag_mode == 'w':
            # Resize from left edge
            new_x = init_x + dx
            new_w = init_w - dx
        elif self.roi_drag_mode == 'e':
            # Resize from right edge
            new_w = init_w + dx
        
        # Ensure minimum size and bounds
        new_w = max(50, new_w)
        new_h = max(50, new_h)
        new_x = max(0, min(new_x, detector.width - new_w))
        new_y = max(0, min(new_y, detector.height - new_h))
        
        # Apply to detector (without resetting background - just update coords)
        if self.active_zone == 'zone1':
            detector.roi_x = int(new_x)
            detector.roi_y = int(new_y)
            detector.roi_width = int(new_w)
            detector.roi_height = int(new_h)
            detector.roi_start_y = detector.roi_y
            detector.roi_end_y = detector.roi_y + detector.roi_height
            # Update GUI sliders and labels
            self.roi_x_var.set(int(new_x))
            self.roi_y_var.set(int(new_y))
            self.roi_width_var.set(int(new_w))
            self.roi_height_var.set(int(new_h))
            self.roi_x_label.config(text=f"{int(new_x)}")
            self.roi_y_label.config(text=f"{int(new_y)}")
            self.roi_width_label.config(text=f"{int(new_w)}")
            self.roi_height_label.config(text=f"{int(new_h)}")
        else:
            detector.zone2_x = int(new_x)
            detector.zone2_y = int(new_y)
            detector.zone2_width = int(new_w)
            detector.zone2_height = int(new_h)
            detector.zone2_roi_start_y = detector.zone2_y
            detector.zone2_roi_end_y = detector.zone2_y + detector.zone2_height
            # Update GUI sliders and labels
            self.zone2_x_var.set(int(new_x))
            self.zone2_y_var.set(int(new_y))
            self.zone2_width_var.set(int(new_w))
            self.zone2_height_var.set(int(new_h))
            self.zone2_x_label.config(text=f"{int(new_x)}")
            self.zone2_y_label.config(text=f"{int(new_y)}")
            self.zone2_width_label.config(text=f"{int(new_w)}")
            self.zone2_height_label.config(text=f"{int(new_h)}")
    
    def _on_mouse_up(self, event):
        """Handle mouse button release - end drag operation"""
        if not self.roi_edit_mode:
            return
        
        if self.roi_drag_start and self.active_zone:
            # Log the final position
            with self._lock:
                detector = self.detector
            
            if detector:
                if self.active_zone == 'zone1':
                    self.log_message(f"Zone 1 updated: x={detector.roi_x}, y={detector.roi_y}, "
                                   f"w={detector.roi_width}, h={detector.roi_height}")
                    # Update the info label
                    self.roi_info_label.config(
                        text=f"Zone 1: x={detector.roi_x}, y={detector.roi_y}, "
                             f"w={detector.roi_width}, h={detector.roi_height}"
                    )
                else:
                    self.log_message(f"Zone 2 updated: x={detector.zone2_x}, y={detector.zone2_y}, "
                                   f"w={detector.zone2_width}, h={detector.zone2_height}")
                    # Update the info label
                    status = "enabled" if detector.zone2_enabled else "disabled"
                    self.zone2_info_label.config(
                        text=f"Zone 2: x={detector.zone2_x}, y={detector.zone2_y}, "
                             f"w={detector.zone2_width}, h={detector.zone2_height} ({status})"
                    )
        
        # Reset drag state
        self.roi_drag_start = None
        self.roi_drag_mode = None
            
    def start_camera(self):
        """Start camera detection in a separate thread"""
        with self._lock:
            if self.is_running:
                self.log_message("Already running. Please stop first.")
                return
            
            # Clean up any existing detector before starting new one
            if self.detector is not None:
                self.log_message("Cleaning up previous detector...")
                try:
                    if hasattr(self.detector, 'stop_event'):
                        self.detector.stop_event.set()
                    
                    # Wait for detector thread to finish if it exists
                    if hasattr(self, 'detector_thread') and self.detector_thread.is_alive():
                        self.log_message("Waiting for detector thread to finish...")
                        self.detector_thread.join(timeout=3.0)
                    
                    if hasattr(self.detector, 'hcam') and self.detector.hcam is not None:
                        try:
                            # Stop camera before closing
                            try:
                                self.detector.hcam.Stop()
                                time.sleep(0.5)  # Increased wait time for camera to fully stop
                            except Exception:
                                pass
                            self.detector.hcam.Close()
                            time.sleep(0.2)  # Additional wait after closing
                        except Exception as e:
                            self.log_message(f"Warning during camera cleanup: {e}")
                except Exception as e:
                    self.log_message(f"Warning during cleanup: {e}")
                self.detector = None
            
            self.is_running = True
        
        self.start_camera_btn.config(state=tk.DISABLED)
        self.load_video_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.detection_btn.config(state=tk.NORMAL)  # Enable detection toggle
        self.status_label.config(text="Status: Initializing camera...")
        self.log_message("Starting camera detection...")
        
        # Reset FPS tracking
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Run detector in separate thread (non-daemon for proper cleanup)
        self.detector_thread = threading.Thread(target=self._run_camera_detection, daemon=False)
        self.detector_thread.start()
        
        # Start frame update loop
        self.update_frame()
        
    def load_video(self):
        """Load and process video file"""
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if not video_path:
            return
        
        with self._lock:
            if self.is_running:
                return
            self.is_running = True
            
        self.start_camera_btn.config(state=tk.DISABLED)
        self.load_video_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Status: Processing video...")
        self.log_message(f"Loading video: {video_path}")
        
        # Reset FPS tracking
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Run detector in separate thread (non-daemon for proper cleanup)
        self.detector_thread = threading.Thread(
            target=self._run_video_detection, 
            args=(video_path,), 
            daemon=False
        )
        self.detector_thread.start()
        
        # Start frame update loop
        self.update_frame()
        
    def _run_camera_detection(self):
        """Run camera detection (called in separate thread)"""
        try:
            self.log_message("Creating detector...")
            detector = EmbryoDetector()
            # IMPORTANT: Start with detection OFF - user must click 'Start Detection'
            detector.detection_enabled = False
            # Set binning from GUI if available
            with self._lock:
                if hasattr(self, 'binning'):
                    detector.binning = self.binning
            # Store detector with lock
            with self._lock:
                self.detector = detector
            
            self.log_message("Detector created. Modifying process_frame...")
            
            # Modify detector to send frames to GUI
            original_process = detector.process_frame
            
            def gui_process_frame(frame):
                # Put frame in queue for display FIRST (before heavy processing)
                # This ensures display gets frames even if processing is slow
                try:
                    self.frame_queue.put_nowait(frame)  # Don't copy here - just pass reference
                except queue.Full:
                    # Queue full - drain it and add latest to minimize latency
                    try:
                        while True:
                            self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass  # Still full, skip this frame
                
                # Call original processing (this is heavy - background subtraction, detection, ML inference)
                # This blocks the camera callback, but it's necessary for detection
                # TODO: Consider making this async or moving to separate thread
                return original_process(frame)
            
            detector.process_frame = gui_process_frame
            
            # Update ROI sliders with detector's values after initialization
            self.root.after(100, self.update_roi_from_detector)
            
            self.log_message("Starting camera initialization...")
            self.root.after(0, lambda: self.status_label.config(text="Status: Initializing camera..."))
            
            # Add a timeout check - if initialization takes too long, report it
            def check_initialization_status():
                """Check if camera initialized after a delay"""
                with self._lock:
                    detector = self.detector
                if detector and hasattr(detector, 'hcam') and detector.hcam is not None:
                    self.root.after(0, lambda: self.status_label.config(text="Status: Camera initialized, starting..."))
                else:
                    # Still initializing, check again in 2 seconds
                    self.root.after(2000, check_initialization_status)
            
            # Start checking after 3 seconds
            self.root.after(3000, check_initialization_status)
            
            # Wrap the run call to catch initialization failures
            try:
                self.log_message("Calling detector.run()...")
                detector.run(video_path=None, show_window=False)
                self.log_message("detector.run() returned (this should not happen immediately)")
            except Exception as e:
                error_msg = f"Camera initialization or run failed: {str(e)}"
                self.log_message(error_msg)
                import traceback
                self.log_message(traceback.format_exc())
                self.root.after(0, lambda: self.status_label.config(text="Status: Initialization failed - see log"))
                self.root.after(0, self.stop_detection)
            
        except Exception as e:
            import traceback
            error_msg = f"Error in camera detection thread: {str(e)}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            print(f"EXCEPTION in _run_camera_detection: {e}")
            traceback.print_exc()
            # Schedule stop on main thread
            self.root.after(0, self.stop_detection)
            
    def _run_video_detection(self, video_path):
        """Run video detection (called in separate thread)"""
        try:
            detector = EmbryoDetector()
            # IMPORTANT: Start with detection OFF - user must click 'Start Detection'
            detector.detection_enabled = False
            # Store detector with lock
            with self._lock:
                self.detector = detector
            
            # Modify detector to send frames to GUI
            original_process = detector.process_frame
            
            def gui_process_frame(frame):
                # Put frame in queue for display (use reference, not copy, to avoid expensive copy operation)
                try:
                    self.frame_queue.put_nowait(frame)  # Don't copy here - just pass reference
                except queue.Full:
                    # Queue full - drain it and add latest
                    try:
                        while True:
                            self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass  # Still full, skip this frame
                return original_process(frame)
            
            detector.process_frame = gui_process_frame
            
            # Update ROI sliders with detector's values after initialization
            self.root.after(100, self.update_roi_from_detector)
            
            detector.run(video_path=video_path, show_window=False)
            
            self.log_message("Video processing complete.")
            # Schedule stop on main thread
            self.root.after(0, self.stop_detection)
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            # Schedule stop on main thread
            self.root.after(0, self.stop_detection)
            
    def update_frame(self):
        """Update video display (called periodically from main thread)"""
        with self._lock:
            is_running = self.is_running
            
        if not is_running:
            return
            
        # Get only the latest frame without draining (keep detector frame rate)
        frame = None
        try:
            # Non-blocking peek at queue
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        
        if frame is None:
            # No frame available, schedule next update
            with self._lock:
                if self.is_running:
                    self.root.after(10, self.update_frame)
            return
        
        try:
            # Get detector and display settings (no copy yet)
            with self._lock:
                detector = self.detector
                flip_h = self.flip_horizontal
                flip_v = self.flip_vertical
            
            # Only copy ONCE when drawing (avoid early copy)
            display_frame = frame.copy()
            frame_h, frame_w = display_frame.shape[:2]
            
            # Draw zones with different colors and handles when in edit mode
            if detector:
                # Zone 1 (Classification) - Green
                if hasattr(detector, 'roi_width') and detector.roi_width > 0 and detector.roi_height > 0:
                    roi_x = max(0, min(detector.roi_x, frame_w - 1))
                    roi_y = max(0, min(detector.roi_y, frame_h - 1))
                    roi_w = max(1, min(detector.roi_width, frame_w - roi_x))
                    roi_h = max(1, min(detector.roi_height, frame_h - roi_y))
                    
                    # Draw zone rectangle
                    zone1_color = (0, 255, 0)  # Green in BGR
                    thickness = 3 if (self.roi_edit_mode and self.active_zone == 'zone1') else 2
                    cv2.rectangle(display_frame, 
                                 (roi_x, roi_y), 
                                 (roi_x + roi_w, roi_y + roi_h), 
                                 zone1_color, thickness)
                    
                    # Draw label
                    cv2.putText(display_frame, "Zone 1 (ML)", (roi_x + 5, roi_y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone1_color, 2)
                    
                    # Draw handles when in edit mode
                    if self.roi_edit_mode:
                        self._draw_zone_handles(display_frame, roi_x, roi_y, roi_w, roi_h, zone1_color)
                
                # Zone 2 (Trigger) - Orange
                # Show Zone 2 if it has valid dimensions OR use GUI slider values as fallback
                zone2_enabled = getattr(detector, 'zone2_enabled', True)
                z2_x = getattr(detector, 'zone2_x', self.zone2_x_var.get())
                z2_y = getattr(detector, 'zone2_y', self.zone2_y_var.get())
                z2_w = getattr(detector, 'zone2_width', self.zone2_width_var.get())
                z2_h = getattr(detector, 'zone2_height', self.zone2_height_var.get())
                
                if zone2_enabled and z2_w > 0 and z2_h > 0:
                    z2_x = max(0, min(z2_x, frame_w - 1))
                    z2_y = max(0, min(z2_y, frame_h - 1))
                    z2_w = max(1, min(z2_w, frame_w - z2_x))
                    z2_h = max(1, min(z2_h, frame_h - z2_y))
                    
                    # Draw zone rectangle
                    zone2_color = (0, 165, 255)  # Orange in BGR
                    thickness = 3 if (self.roi_edit_mode and self.active_zone == 'zone2') else 2
                    cv2.rectangle(display_frame, 
                                 (z2_x, z2_y), 
                                 (z2_x + z2_w, z2_y + z2_h), 
                                 zone2_color, thickness)
                    
                    # Draw label
                    cv2.putText(display_frame, "Zone 2 (Trigger)", (z2_x + 5, z2_y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone2_color, 2)
                    
                    # Draw handles when in edit mode
                    if self.roi_edit_mode:
                        self._draw_zone_handles(display_frame, z2_x, z2_y, z2_w, z2_h, zone2_color)
            
            # Track and draw GUI FPS (light operation)
            current_time = time.time()
            self.gui_fps_frame_count += 1
            if current_time - self.gui_last_fps_time >= 1.0:
                self.gui_current_fps = self.gui_fps_frame_count / (current_time - self.gui_last_fps_time)
                self.gui_fps_frame_count = 0
                self.gui_last_fps_time = current_time
            
            # Build FPS text
            detector_fps_text = "Detector: --"
            if detector and hasattr(detector, 'current_fps'):
                detector_fps_text = f"Detector: {detector.current_fps:.1f}"
            gui_fps_text = f"GUI: {self.gui_current_fps:.1f}"
            
            # Convert to RGB for PIL (single conversion)
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display - use dynamic display size
            height, width = frame_rgb.shape[:2]
            if width > 0 and height > 0:
                # Calculate display size maintaining aspect ratio
                aspect_ratio = width / height
                target_width = self.display_width
                target_height = int(target_width / aspect_ratio)
                
                # If calculated height exceeds display_height, scale by height instead
                if target_height > self.display_height:
                    target_height = self.display_height
                    target_width = int(target_height * aspect_ratio)
                
                # Ensure minimum size
                target_width = max(320, target_width)
                target_height = max(240, target_height)
                
                frame_resized = cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                
                # Apply flip if needed
                if flip_h or flip_v:
                    flip_code = -1 if (flip_h and flip_v) else (1 if flip_h else 0)
                    frame_resized = cv2.flip(frame_resized, flip_code)
                
                # Draw FPS overlay AFTER flip so text stays readable (bottom-left corner)
                # Scale overlay size based on display size
                overlay_scale = target_width / 960.0
                font_scale = max(0.4, 0.7 * overlay_scale)
                box_width = int(200 * overlay_scale)
                box_height = int(50 * overlay_scale)
                
                cv2.rectangle(frame_resized, (8, target_height - box_height - 8), 
                             (box_width + 8, target_height - 8), (20, 20, 20), -1)
                cv2.rectangle(frame_resized, (8, target_height - box_height - 8), 
                             (box_width + 8, target_height - 8), (60, 60, 60), 1)
                cv2.putText(frame_resized, detector_fps_text, (12, target_height - int(box_height * 0.55)), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, gui_fps_text, (12, target_height - int(box_height * 0.15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 100), 1, cv2.LINE_AA)
                
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            
            # Update stats only every 500ms (reduce UI overhead)
            current_time = time.time()
            if current_time - self.last_stats_update >= 0.5:  # Increased from 0.1 to 0.5
                self.update_stats()
                self.last_stats_update = current_time
            
            self.status_label.config(text="● Running")
            
        except Exception as e:
            # Don't flood the log with display errors - only log occasionally
            if not hasattr(self, '_last_display_error_time') or (time.time() - self._last_display_error_time) > 5.0:
                self.log_message(f"Display error: {str(e)}")
                self._last_display_error_time = time.time()
            
        # Schedule next update with small delay to prevent overwhelming the GUI
        with self._lock:
            if self.is_running:
                self.root.after(16, self.update_frame)  # ~60fps max to reduce CPU load
            
    def stop_detection(self):
        """Stop detection (must be called from main thread)"""
        with self._lock:
            if not self.is_running:
                return
            self.is_running = False
            
        self.status_label.config(text="○ Stopping...")
        self.log_message("Stopping detection...")
        
        # Signal detector to stop
        with self._lock:
            detector = self.detector
        
        if detector:
            try:
                detector.stop_event.set()
            except AttributeError:
                pass  # Detector may not have stop_event yet
            
            # Ensure camera is closed
            if hasattr(detector, 'hcam') and detector.hcam is not None:
                try:
                    # Stop camera before closing
                    try:
                        detector.hcam.Stop()
                        time.sleep(0.5)  # Increased wait time for camera to fully stop
                    except Exception:
                        pass  # May not be running
                    
                    detector.hcam.Close()
                    time.sleep(0.2)  # Additional wait after closing to ensure resource is released
                    self.log_message("Camera closed during stop")
                except Exception as e:
                    self.log_message(f"Warning: Error closing camera during stop: {e}")
        
        # Wait for detector thread to finish (with timeout)
        if hasattr(self, 'detector_thread') and self.detector_thread.is_alive():
            self.detector_thread.join(timeout=5.0)  # Increased timeout to 5 seconds
            if self.detector_thread.is_alive():
                self.log_message("Warning: Detector thread did not stop gracefully - camera may still be in use")
        
        # Clean up detector reference
        with self._lock:
            self.detector = None
            self.detection_enabled = False
            
        self.start_camera_btn.config(state=tk.NORMAL)
        self.load_video_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.detection_btn.config(state=tk.DISABLED, text="Start Detection")
        self.detection_status_label.config(text="Detection: OFF", foreground="gray")
        self.status_label.config(text="● Stopped")
        
        # Clear video display
        self.video_label.configure(image='')
        
        # Clear frame queue
        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass

def main():
    root = tk.Tk()
    app = EmbryoDetectionGUI(root)
    
    # Handle window close event
    def on_closing():
        with app._lock:
            is_running = app.is_running
        if is_running:
            app.stop_detection()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()