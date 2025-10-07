"""
Pro Smooth Modern Camera App with Face, Hand & Finger, and Object Detection

Features added on top of previous version:
- Face detection (Haar Cascade)
- Hand detection + finger counting (MediaPipe Hands)
- Simple moving-object detection (background subtraction + contours)
- UI toggles to enable/disable each detector
- Visual overlays for detected faces, hands (landmarks), finger count, and objects

Dependencies:
- Python 3.8+
- opencv-python
- PyQt6
- numpy
- mediapipe

Install:
    pip install opencv-python PyQt6 numpy mediapipe

Run:
    python pro_camera_app.py

Notes:
- MediaPipe may require additional system dependencies on some platforms; follow its install guide if you hit issues.
- Object detection here is motion-based (background subtraction) to avoid large external model files. It detects moving blobs and draws boxes.

"""

import sys
import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque

from PyQt6 import QtCore, QtGui, QtWidgets

# Try to import MediaPipe for hand detection; handle gracefully if missing
import traceback
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception as e:
    print("[DEBUG] Failed to import mediapipe:", e)
    traceback.print_exc()
    HAS_MEDIAPIPE = False

# ---------------------------
# Configuration
# ---------------------------
CAMERA_INDEX = 0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
VIDEO_CODEC = 'mp4v'  # FourCC
OUTPUT_DIR = Path.cwd() / "captures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------
# Utility filters
# ---------------------------

def apply_sepia(frame: np.ndarray) -> np.ndarray:
    img = frame.astype(np.float32)
    tr = (img[:, :, 2] * 0.393) + (img[:, :, 1] * 0.769) + (img[:, :, 0] * 0.189)
    tg = (img[:, :, 2] * 0.349) + (img[:, :, 1] * 0.686) + (img[:, :, 0] * 0.168)
    tb = (img[:, :, 2] * 0.272) + (img[:, :, 1] * 0.534) + (img[:, :, 0] * 0.131)
    sepia = np.stack([np.clip(tb, 0, 255), np.clip(tg, 0, 255), np.clip(tr, 0, 255)], axis=-1)
    return sepia.astype(np.uint8)


def apply_edges(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_color

# ---------------------------
# Detection helpers
# ---------------------------

# Face detector - use Haar cascade included with OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MediaPipe hands setup
if HAS_MEDIAPIPE:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    # We'll instantiate the hands object later in the worker to run inside the capture thread

# Simple moving object detector using BackgroundSubtractorMOG2
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def detect_moving_objects(frame, min_area=500):
    # returns list of bounding boxes for moving contours
    mask = bg_subtractor.apply(frame)
    # clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # threshold & find contours
    _, thresh = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
    return boxes, mask


def count_fingers_mediapipe(hand_landmarks, image_width, image_height):
    """
    Count extended fingers using landmark coordinates from MediaPipe for one hand.
    Return integer finger count (0-5) and a simple list of which fingers are up.
    """
    # Landmark indices: thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20
    # We'll compare y-coordinates for fingers (except thumb) and for thumb compare x depending on handedness.
    tips_ids = [4, 8, 12, 16, 20]
    coords = {}
    for idx, lm in enumerate(hand_landmarks.landmark):
        coords[idx] = (int(lm.x * image_width), int(lm.y * image_height))
    fingers_up = []
    # For fingers except thumb: tip y < pip y -> finger is up
    for tip_id in [8, 12, 16, 20]:
        tip_y = hand_landmarks.landmark[tip_id].y
        pip_y = hand_landmarks.landmark[tip_id - 2].y
        fingers_up.append(tip_y < pip_y)
    # Thumb: compare tip x with ip x depending on hand orientation
    # We'll use a simple heuristic: if tip x > ip x => thumb open (for right hand in image coordinates)
    thumb_open = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
    fingers_up.insert(0, thumb_open)
    count = sum(1 for v in fingers_up if v)
    return count, fingers_up

# ---------------------------
# Camera thread (QThread)
# ---------------------------

class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    fps_updated = QtCore.pyqtSignal(float)

    def __init__(self, camera_index=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._running = False
        self.cap = None
        self.target_fps = 30

        # Detection toggles
        self.enable_face = True
        self.enable_hand = True
        self.enable_motion = True

        # mediapipe hands instance (created when thread runs)
        self.hands = None

    def run(self):
        # Initialize MediaPipe hands if available and requested
        if HAS_MEDIAPIPE and self.enable_hand:
            self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY)
        # Try to set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Read loop
        self._running = True
        last_time = time.time()
        frame_count = 0
        fps_timer = time.time()
        while self._running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame_count += 1
            now = time.time()

            # Prepare data for overlay detections
            overlay = frame.copy()
            detections = {
                'faces': [],
                'hands': [],
                'fingers': [],
                'objects': [],
                'motion_mask': None,
            }

            # Face detection
            if self.enable_face:
                try:
                    faces = detect_faces(frame)
                    detections['faces'] = faces
                except Exception:
                    detections['faces'] = []

            # Motion / moving objects
            if self.enable_motion:
                try:
                    boxes, mask = detect_moving_objects(frame)
                    detections['objects'] = boxes
                    detections['motion_mask'] = mask
                except Exception:
                    detections['objects'] = []
                    detections['motion_mask'] = None

            # Hand detection (MediaPipe)
            if HAS_MEDIAPIPE and self.enable_hand:
                try:
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(img_rgb)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Collect landmarks and compute finger count
                            h, w, _ = frame.shape
                            count, fingers_up = count_fingers_mediapipe(hand_landmarks, w, h)
                            detections['hands'].append(hand_landmarks)
                            detections['fingers'].append((count, fingers_up))
                except Exception:
                    pass

            # Attach detections as a dictionary to frame by drawing overlay here to keep thread simple
            # We'll draw in the main thread after smoothing to avoid duplicated transforms; so pass raw frame and detection dict via an attached attribute.
            # To keep things simple, we'll encode detections into a structured numpy array by drawing onto overlay; then emit overlay and an auxiliary frame as needed.

            # For now, package both frame and detections into a single dict-like container: we'll use a 2-tuple: (frame, detections)
            payload = (frame, detections)
            # emit tuple by encoding using numpy object array - PyQt signals allow arbitrary python object types but we declared np.ndarray signal; instead use send via a custom pyqtSignal? To keep it's simplest, we'll emit the raw BGR frame and store latest detections in a shared attribute.
            # Simpler approach: store detections on self.latest_detections and emit only frame. The main thread reads self.latest_detections.
            self.latest_detections = detections

            self.frame_ready.emit(frame)

            # update fps once per second
            if now - fps_timer >= 1.0:
                fps = frame_count / (now - fps_timer)
                self.fps_updated.emit(fps)
                fps_timer = now
                frame_count = 0
            # Gentle sleep to avoid starving CPU (approx target fps)
            time.sleep(max(0, 1.0 / self.target_fps - (time.time() - now)))

        # cleanup
        if self.cap is not None:
            self.cap.release()
        if self.hands is not None:
            self.hands.close()

    def stop(self):
        self._running = False
        self.wait(2000)

    def set_resolution(self, w, h):
        self.width = w
        self.height = h
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def set_target_fps(self, fps):
        self.target_fps = fps

# ---------------------------
# Main Window
# ---------------------------

class CameraApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Camera — Smooth Modern + Detection")
        self.setMinimumSize(1100, 700)

        # Central layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Left: preview
        self.preview_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.preview_label.setStyleSheet("background: #111; border-radius: 8px;")
        main_layout.addWidget(self.preview_label, 3)

        # Right: controls
        controls = QtWidgets.QWidget()
        controls.setFixedWidth(380)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(10)

        # Top controls row
        row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        controls_layout.addLayout(row)

        # Capture / Record
        self.btn_capture = QtWidgets.QPushButton("Capture Photo")
        self.btn_record = QtWidgets.QPushButton("Start Recording")
        self.btn_record.setCheckable(True)
        self.btn_capture.setEnabled(False)
        self.btn_record.setEnabled(False)
        controls_layout.addWidget(self.btn_capture)
        controls_layout.addWidget(self.btn_record)

        # Filters
        controls_layout.addWidget(QtWidgets.QLabel("Filter"))
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(["None", "Grayscale", "Sepia", "Edges"])
        controls_layout.addWidget(self.filter_combo)

        # Detection toggles
        controls_layout.addWidget(QtWidgets.QLabel("Detections"))
        self.face_check = QtWidgets.QCheckBox("Face Detection")
        self.hand_check = QtWidgets.QCheckBox("Hand & Finger Detection")
        self.motion_check = QtWidgets.QCheckBox("Motion/Object Detection")
        self.face_check.setChecked(True)
        self.hand_check.setChecked(HAS_MEDIAPIPE)
        self.motion_check.setChecked(True)
        controls_layout.addWidget(self.face_check)
        controls_layout.addWidget(self.hand_check)
        controls_layout.addWidget(self.motion_check)

        # Resolution
        controls_layout.addWidget(QtWidgets.QLabel("Resolution"))
        self.res_combo = QtWidgets.QComboBox()
        self.resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        self.res_combo.addItems([f"{w}x{h}" for (w, h) in self.resolutions])
        self.res_combo.setCurrentIndex(1)
        controls_layout.addWidget(self.res_combo)

        # Brightness & Contrast
        controls_layout.addWidget(QtWidgets.QLabel("Brightness"))
        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        controls_layout.addWidget(self.brightness_slider)

        controls_layout.addWidget(QtWidgets.QLabel("Contrast"))
        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        controls_layout.addWidget(self.contrast_slider)

        # FPS smoothing (controls worker target FPS)
        controls_layout.addWidget(QtWidgets.QLabel("Preview FPS Target"))
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(30)
        controls_layout.addWidget(self.fps_spin)

        # Smoothing alpha
        controls_layout.addWidget(QtWidgets.QLabel("Temporal Smoothing (alpha)"))
        self.smooth_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(1, 99)
        self.smooth_slider.setValue(60)
        controls_layout.addWidget(self.smooth_slider)

        # Status & histogram placeholder
        self.status = QtWidgets.QLabel("Status: Stopped")
        controls_layout.addWidget(self.status)
        controls_layout.addStretch(1)

        # Small footer
        footer = QtWidgets.QLabel("Pro Camera • OpenCV + PyQt6 + MediaPipe (optional)")
        footer.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(footer)

        main_layout.addWidget(controls)

        # Camera worker
        self.worker = CameraWorker(camera_index=CAMERA_INDEX, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.fps_updated.connect(self.on_fps_updated)

        # Frame smoothing buffer
        self.smoothing_alpha = 0.6  # exponential smoothing, controlled by slider
        self.smoothed_frame = None

        # Recording
        self.recording = False
        self.video_writer = None
        self.video_fps = 30

        # Latest detections from worker
        self.latest_detections = {}

        # Connect signals
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_capture.clicked.connect(self.capture_photo)
        self.btn_record.toggled.connect(self.toggle_record)
        self.res_combo.currentIndexChanged.connect(self.change_resolution)
        self.fps_spin.valueChanged.connect(self.set_target_fps)

        # Detection toggles
        self.face_check.stateChanged.connect(self.on_detection_toggled)
        self.hand_check.stateChanged.connect(self.on_detection_toggled)
        self.motion_check.stateChanged.connect(self.on_detection_toggled)

        # Sliders adjustments
        self.brightness_slider.valueChanged.connect(lambda v: None)
        self.contrast_slider.valueChanged.connect(lambda v: None)
        self.smooth_slider.valueChanged.connect(lambda v: None)

        # Keyboard shortcuts
        self.shortcut_capture = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        self.shortcut_capture.activated.connect(self.capture_photo)

        # Warn if MediaPipe missing but hand detection requested
        if not HAS_MEDIAPIPE:
            self.hand_check.setChecked(False)
            self.hand_check.setEnabled(False)
            QtWidgets.QMessageBox.warning(
                self,
                "MediaPipe not available",
                "mediapipe is not installed. Hand & finger detection will be disabled.\nInstall with: pip install mediapipe"
            )

    # ---------------------------
    # Camera control
    # ---------------------------
    def start_camera(self):
        self.worker.enable_face = self.face_check.isChecked()
        self.worker.enable_hand = self.hand_check.isChecked()
        self.worker.enable_motion = self.motion_check.isChecked()
        self.worker.set_resolution(*self.resolutions[self.res_combo.currentIndex()])
        self.worker.set_target_fps(self.fps_spin.value())
        # set smoothing alpha
        self.smoothing_alpha = self.smooth_slider.value() / 100.0
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_capture.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.status.setText("Status: Running")

    def stop_camera(self):
        self.worker.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.status.setText("Status: Stopped")
        # stop recording if active
        if self.recording:
            self.finish_recording()

    def change_resolution(self, idx):
        w, h = self.resolutions[idx]
        self.worker.set_resolution(w, h)

    def set_target_fps(self, v):
        self.worker.set_target_fps(v)

    def on_detection_toggled(self, _state):
        # push settings to worker if running
        try:
            self.worker.enable_face = self.face_check.isChecked()
            self.worker.enable_hand = self.hand_check.isChecked()
            self.worker.enable_motion = self.motion_check.isChecked()
        except Exception:
            pass

    # ---------------------------
    # Frame processing and UI
    # ---------------------------
    def on_fps_updated(self, fps):
        self.status.setText(f"Status: Running — Camera FPS: {fps:.1f}")

    def on_frame(self, frame: np.ndarray):
        # Apply brightness/contrast
        frame = self.apply_brightness_contrast(frame, self.brightness_slider.value(), self.contrast_slider.value())

        # Filter
        filt = self.filter_combo.currentText()
        if filt == 'Grayscale':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif filt == 'Sepia':
            frame = apply_sepia(frame)
        elif filt == 'Edges':
            frame = apply_edges(frame)

        # Smooth (exponential smoothing in float space)
        frame_float = frame.astype(np.float32)
        if self.smoothed_frame is None:
            self.smoothed_frame = frame_float
        else:
            alpha = self.smooth_slider.value() / 100.0
            self.smoothed_frame = (alpha * frame_float) + ((1 - alpha) * self.smoothed_frame)
        display_frame = np.clip(self.smoothed_frame, 0, 255).astype(np.uint8)

        # Pull latest detections from worker if available
        try:
            detections = getattr(self.worker, 'latest_detections', {}) or {}
        except Exception:
            detections = {}

        # Draw overlays for faces
        faces = detections.get('faces', [])
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (50, 200, 255), 2)
            cv2.putText(display_frame, 'Face', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2)

        # Draw motion objects
        objects = detections.get('objects', [])
        for (x, y, w, h) in objects:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(display_frame, 'Object', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        # Draw hand landmarks and finger counts (using MediaPipe drawing utilities if available)
        hands = detections.get('hands', [])
        fingers = detections.get('fingers', [])
        if HAS_MEDIAPIPE and hands:
            for idx, hand_landmarks in enumerate(hands):
                try:
                    mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                except Exception:
                    # fallback to drawing simple circles
                    for lm in hand_landmarks.landmark:
                        h, w, _ = display_frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display_frame, (cx, cy), 3, (255, 0, 255), -1)
        # Draw finger counts near first landmark of each hand
        # fingers is list of tuples (count, fingers_up)
        if fingers:
            # we need positions to place counts; approximate by using wrist landmark (0)
            for i, (count_info) in enumerate(fingers):
                count, fingers_up = count_info
                # find an approximate position: use hand landmarks if present
                try:
                    hand_landmarks = hands[i]
                    h, w, _ = display_frame.shape
                    wx, wy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                except Exception:
                    wx, wy = 30 + i * 80, 30
                cv2.putText(display_frame, f'Fingers: {count}', (wx - 10, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 80, 255), 2)

        # If recording, write frame
        if self.recording and self.video_writer is not None:
            self.video_writer.write(display_frame)

        # Convert to QImage and display
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(display_frame.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(image).scaled(self.preview_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(pix)

    def apply_brightness_contrast(self, img, brightness=0, contrast=0):
        # brightness: -100..100, contrast: -100..100
        if brightness == 0 and contrast == 0:
            return img
        beta = float(brightness)
        alpha = 1.0 + (contrast / 100.0)
        out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return out

    # ---------------------------
    # Capture / Record
    # ---------------------------
    def capture_photo(self):
        if self.smoothed_frame is None:
            return
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = OUTPUT_DIR / f"photo_{ts}.jpg"
        cv2.imwrite(str(path), np.clip(self.smoothed_frame, 0, 255).astype(np.uint8))
        QtWidgets.QMessageBox.information(self, "Captured", f"Saved photo to: {path}")

    def toggle_record(self, toggled: bool):
        if toggled:
            self.start_recording()
            self.btn_record.setText("Stop Recording")
        else:
            self.finish_recording()
            self.btn_record.setText("Start Recording")

    def start_recording(self):
        if self.smoothed_frame is None:
            QtWidgets.QMessageBox.warning(self, "Recording", "No frame yet — wait for preview to start")
            self.btn_record.setChecked(False)
            return
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = OUTPUT_DIR / f"video_{ts}.mp4"
        h, w, ch = np.clip(self.smoothed_frame, 0, 255).astype(np.uint8).shape
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        fps = max(1, self.fps_spin.value())
        self.video_writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not self.video_writer.isOpened():
            QtWidgets.QMessageBox.critical(self, "Recording", "Failed to open video writer")
            self.video_writer = None
            self.btn_record.setChecked(False)
            return
        self.recording = True
        self.status.setText(f"Recording → {path.name}")

    def finish_recording(self):
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            QtWidgets.QMessageBox.information(self, "Recording", "Saved recording")
        self.status.setText("Status: Running")

    # ---------------------------
    # Clean up
    # ---------------------------
    def closeEvent(self, event):
        # ensure worker stops and writer closes
        try:
            if self.worker is not None:
                self.worker.stop()
        except Exception:
            pass
        try:
            if self.video_writer is not None:
                self.video_writer.release()
        except Exception:
            pass
        super().closeEvent(event)


# ---------------------------
# Run
# ---------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = CameraApp()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()