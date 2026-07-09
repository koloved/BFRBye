import os
import csv
import cv2
import mediapipe as mp
import winsound
import time
import threading
from datetime import datetime
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections, FaceLandmarksConnections

# Path to model files
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_HAND_MODEL = os.path.join(_MODELS_DIR, "hand_landmarker.task")
_FACE_LANDMARK_MODEL = os.path.join(_MODELS_DIR, "face_landmarker.task")

# Fingertip landmark indices in the MediaPipe hand model (0-20)
_FINGERTIP_INDICES = {4, 8, 12, 16, 20}

# Mouth (lips) landmark indices from FaceLandmarksConnections
_LIPS_CONNECTIONS = FaceLandmarksConnections.FACE_LANDMARKS_LIPS
_MOUTH_INDICES = sorted(set(
    c.start for c in _LIPS_CONNECTIONS
) | set(
    c.end for c in _LIPS_CONNECTIONS
))

# ── state machine ──────────────────────────────────────────────
_IDLE = 0
_ACTIVE = 1
_COOLDOWN = 2


class HandTracker:
    def __init__(self, config):
        self.config = config

        # ----- Hand Landmarker -----
        hand_base = python.BaseOptions(model_asset_path=_HAND_MODEL)
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # ----- Face Landmarker -----
        face_base = python.BaseOptions(model_asset_path=_FACE_LANDMARK_MODEL)
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        # Drawing references
        self.mp_hand_connections = HandLandmarksConnections.HAND_CONNECTIONS

        # Tunable parameters (from config)
        proc = config.get("processing", {})
        self.processing_interval = proc.get("interval", 1)
        self.mouth_padding = proc.get("mouth_padding", 0.5)
        self.trigger_frames = proc.get("trigger_frames", 5)
        self.min_duration = proc.get("min_duration", 0.5)
        self.cooldown = proc.get("cooldown", 0.5)
        self._cam_w = proc.get("camera_width", 640)
        self._cam_h = proc.get("camera_height", 480)

        self.webcam = cv2.VideoCapture(0)
        self._ensure_webcam()
        self.counter = 0  # episode counter

        # State machine
        self._state = _IDLE
        self._consecutive = 0
        self._episode_start = 0.0  # when ACTIVE was entered
        self._cooldown_until = 0.0  # when COOLDOWN expires

        # Beep thread control
        self._beep_active = False

        # Episode log file (same dir as config)
        self._episode_log = Path("episodes.csv")
        if not self._episode_log.exists():
            with open(self._episode_log, "w", newline="") as f:
                csv.writer(f).writerow(["start_time", "end_time", "duration_s"])

    # ── webcam ──────────────────────────────────────────────────

    def _ensure_webcam(self):
        if not self.webcam.isOpened():
            self.webcam.open(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self._cam_w)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cam_h)

    # ── inference ──────────────────────────────────────────────

    def _process(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        return (
            self.face_landmarker.detect(mp_image),
            self.hand_landmarker.detect(mp_image),
        )

    # ── beep ────────────────────────────────────────────────────

    def _beep_loop(self):
        """Daemon thread: short beeps while self._beep_active is True."""
        while self._beep_active:
            winsound.Beep(1500, 120)
            time.sleep(0.04)

    def _start_beep(self):
        if self._beep_active:
            return
        self._beep_active = True
        t = threading.Thread(target=self._beep_loop, daemon=True)
        t.start()

    def _stop_beep(self):
        self._beep_active = False

    # ── episode log ─────────────────────────────────────────────

    def _save_episode(self, start, end):
        dur = round(end - start, 1)
        with open(self._episode_log, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.fromtimestamp(start).isoformat(),
                datetime.fromtimestamp(end).isoformat(),
                f"{dur}s",
            ])
        print(f"Episode: start={datetime.fromtimestamp(start).isoformat()}, duration={dur}s")

    # ── state machine ───────────────────────────────────────────

    def _update_state(self, picked, now):
        """Tick the state machine once per processed frame. Returns (beep_on, just_logged)."""
        beep_on = False
        just_logged = False

        if self._state == _IDLE:
            if picked:
                self._consecutive += 1
                if self._consecutive >= self.trigger_frames:
                    # Enter ACTIVE
                    self._state = _ACTIVE
                    self._episode_start = now
                    self._consecutive = 0
                    self._start_beep()
                    beep_on = True
            else:
                self._consecutive = 0

        elif self._state == _ACTIVE:
            elapsed = now - self._episode_start
            beep_on = True  # keep beeping

            if not picked and elapsed >= self.min_duration:
                # Episode complete
                self._stop_beep()
                self.counter += 1
                self._save_episode(self._episode_start, now)
                just_logged = True
                self._state = _COOLDOWN
                self._cooldown_until = now + self.cooldown
                beep_on = False

        elif self._state == _COOLDOWN:
            if now >= self._cooldown_until:
                self._state = _IDLE
                self._consecutive = 0

        return beep_on, just_logged

    # ── HUD ─────────────────────────────────────────────────────

    def _draw_hud(self, img_bgr, fps=None):
        now = time.time()
        state_labels = {_IDLE: "IDLE", _ACTIVE: "ACTIVE", _COOLDOWN: "COOLDOWN"}
        label = state_labels.get(self._state, "?")

        lines = [
            f"Interval: {self.processing_interval}  ([ ])   "
            f"State: {label}",
            f"Padding:  {self.mouth_padding:.1f}  (- =)     "
            f"Consec: {self._consecutive}/{self.trigger_frames}",
            f"Episodes: {self.counter}",
        ]
        if self._state == _ACTIVE:
            elapsed = now - self._episode_start
            lines.insert(0, f">>> BEEP ({elapsed:.1f}s)")
        elif self._state == _COOLDOWN:
            left = max(0, self._cooldown_until - now)
            lines.insert(0, f"--- cooldown {left:.1f}s ---")

        if fps is not None:
            lines.insert(0, f"FPS: {fps}")
        for i, line in enumerate(lines):
            cv2.putText(img_bgr, line, (12, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2,
                        cv2.LINE_AA)

    # ── run (background) ────────────────────────────────────────

    def run(self):
        proc = self.config.get("processing", {})
        self._cam_w = proc.get("camera_width", 640)
        self._cam_h = proc.get("camera_height", 480)
        self._ensure_webcam()

        while self.webcam.isOpened():
            ret, img_bgr = self.webcam.read()
            if not ret:
                break

            face_result, hand_result = self._process(img_bgr)
            img_bgr = cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                    cv2.COLOR_RGB2BGR)

            picked = False
            if hand_result.hand_landmarks and face_result.face_landmarks:
                picked, _ = self.detect_fingertips_near_mouth(
                    img_bgr, face_result, hand_result, draw=False
                )

            now = time.time()
            self._update_state(picked, now)
            time.sleep(0.005)  # tiny breath to avoid busy-wait spinning

        time.sleep(1)
        self._stop_beep()
        self.webcam.release()
        cv2.destroyAllWindows()

    # ── run_preview ─────────────────────────────────────────────

    def run_preview(self):
        proc = self.config.get("processing", {})
        self.processing_interval = proc.get("interval", 1)
        self.mouth_padding = proc.get("mouth_padding", 0.5)
        self._cam_w = proc.get("camera_width", 640)
        self._cam_h = proc.get("camera_height", 480)

        self._ensure_webcam()
        window_name = "BFRBye — Overlay Preview ([ ] interval, - = padding, R reset, Q quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        frame_count = 0
        last_overlay = None
        fps_timer = time.time()
        fps_counter = 0
        fps_display = 0

        while self.webcam.isOpened():
            ret, raw_bgr = self.webcam.read()
            if not ret:
                break

            frame_count += 1
            fps_counter += 1
            now = time.time()

            if fps_counter >= 30:
                elapsed = now - fps_timer
                fps_display = int(fps_counter / elapsed) if elapsed > 0 else 0
                fps_counter = 0
                fps_timer = now

            do_process = (frame_count % self.processing_interval == 0)

            if do_process:
                face_result, hand_result = self._process(raw_bgr)
                display = cv2.cvtColor(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB),
                                       cv2.COLOR_RGB2BGR)

                picked = False
                if hand_result.hand_landmarks and face_result.face_landmarks:
                    picked, display = self.detect_fingertips_near_mouth(
                        display, face_result, hand_result, draw=True
                    )

                beep_on, just_logged = self._update_state(picked, now)

                if beep_on:
                    cv2.putText(display, "!!! BEEP !!!", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                if just_logged:
                    cv2.putText(display, "LOGGED", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                last_overlay = display.copy()
            else:
                # Skipped frame — state machine not ticked, but consecutive resets
                # (we don't know if hand is still there)
                if self._state == _IDLE:
                    self._consecutive = 0
                display = last_overlay.copy() if last_overlay is not None else raw_bgr.copy()

            self._draw_hud(display, fps_display)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(']') or key == ord('}'):
                self.processing_interval = min(self.processing_interval + 1, 20)
            elif key == ord('[') or key == ord('{'):
                self.processing_interval = max(self.processing_interval - 1, 1)
            elif key == ord('=') or key == ord('+'):
                self.mouth_padding = min(round(self.mouth_padding + 0.1, 1), 3.0)
            elif key == ord('-') or key == ord('_'):
                self.mouth_padding = max(round(self.mouth_padding - 0.1, 1), 0.0)
            elif key == ord('r') or key == ord('R'):
                self.processing_interval = self.config.get("processing", {}).get("interval", 1)
                self.mouth_padding = self.config.get("processing", {}).get("mouth_padding", 0.5)

        self._stop_beep()
        self.webcam.release()
        cv2.destroyWindow(window_name)

    # ── detection ──────────────────────────────────────────────

    def detect_fingertips_near_mouth(self, img_bgr, face_result, hand_result, draw=True):
        h, w, _ = img_bgr.shape
        picked = 0

        face_landmarks_list = face_result.face_landmarks
        hand_landmarks_list = hand_result.hand_landmarks

        if not face_landmarks_list or not hand_landmarks_list:
            return False, img_bgr

        face_landmarks = face_landmarks_list[0]

        # --- Mouth landmarks ---
        mouth_pts = []
        for idx in _MOUTH_INDICES:
            lm = face_landmarks[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            mouth_pts.append((lm.x, lm.y))
            if draw:
                cv2.circle(img_bgr, (px, py), 3, (0, 255, 0), -1)

        if not mouth_pts:
            return False, img_bgr

        # --- Mouth bounding box ---
        mouth_x = [p[0] for p in mouth_pts]
        mouth_y = [p[1] for p in mouth_pts]
        mx_min, mx_max = min(mouth_x), max(mouth_x)
        my_min, my_max = min(mouth_y), max(mouth_y)

        pad_x = (mx_max - mx_min) * self.mouth_padding
        pad_y = (my_max - my_min) * self.mouth_padding
        mx_min -= pad_x
        mx_max += pad_x
        my_min -= pad_y
        my_max += pad_y

        if draw:
            box_x1 = int(mx_min * w)
            box_y1 = int(my_min * h)
            box_x2 = int(mx_max * w)
            box_y2 = int(my_max * h)
            cv2.rectangle(img_bgr, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)

        # --- Draw hand landmarks ---
        if draw:
            for hand_landmarks in hand_landmarks_list:
                drawing_utils.draw_landmarks(img_bgr, hand_landmarks,
                                              self.mp_hand_connections)
                for i, lm in enumerate(hand_landmarks):
                    if i in _FINGERTIP_INDICES:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img_bgr, (px, py), 8, (0, 0, 255), 2)

        # --- Check fingertips near mouth ---
        for hand_landmarks in hand_landmarks_list:
            for i, lm in enumerate(hand_landmarks):
                if i not in _FINGERTIP_INDICES:
                    continue
                if (mx_min <= lm.x <= mx_max) and (my_min <= lm.y <= my_max):
                    picked += 1
                    break

        return picked > 0, img_bgr
