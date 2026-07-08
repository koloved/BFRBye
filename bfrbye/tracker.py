import os
import cv2
import mediapipe as mp
import winsound
import time
import threading
from datetime import datetime

from bfrbye.dialog import show_input_dialog
from bfrbye.storage import save_response

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

# How much to inflate the mouth bounding box (fraction of its size)
_MOUTH_PADDING = 0.5


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

        # ----- Face Landmarker (replaces Face Detector) -----
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

        self.webcam = cv2.VideoCapture(0)
        self.counter = 0

    def run(self):
        """Background tracking mode — no window, just triggers on detection."""
        while self.webcam.isOpened():
            ret, img_bgr = self.webcam.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            face_result = self.face_landmarker.detect(mp_image)
            hand_result = self.hand_landmarker.detect(mp_image)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if hand_result.hand_landmarks and face_result.face_landmarks:
                picked, img_bgr = self.detect_fingertips_near_mouth(
                    img_bgr, face_result, hand_result
                )
                if picked:
                    winsound.Beep(1500, 1000)
                    response = []
                    thread = threading.Thread(target=show_input_dialog, args=(response,))
                    thread.start()
                    thread.join()

                    if len(response):
                        save_response(response[0], self.config)

        time.sleep(2)
        self.webcam.release()
        cv2.destroyAllWindows()

    def run_preview(self):
        """Preview mode — shows camera feed with overlays in an OpenCV window."""
        window_name = "BFRBye — Overlay Preview (press Q to close)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        while self.webcam.isOpened():
            ret, img_bgr = self.webcam.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            face_result = self.face_landmarker.detect(mp_image)
            hand_result = self.hand_landmarker.detect(mp_image)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if hand_result.hand_landmarks and face_result.face_landmarks:
                picked, img_bgr = self.detect_fingertips_near_mouth(
                    img_bgr, face_result, hand_result
                )
                if picked:
                    # Show a brief visual flash instead of dialog in preview mode
                    cv2.putText(img_bgr, "TRIGGER!", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow(window_name, img_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.webcam.release()
        cv2.destroyWindow(window_name)

    def detect_fingertips_near_mouth(self, img_bgr, face_result, hand_result):
        """Check if any fingertip is near the mouth area using Face Landmarker."""
        h, w, _ = img_bgr.shape
        picked = 0

        # Get face landmarks for the first face
        face_landmarks_list = face_result.face_landmarks
        hand_landmarks_list = hand_result.hand_landmarks

        if not face_landmarks_list or not hand_landmarks_list:
            return False, img_bgr

        face_landmarks = face_landmarks_list[0]  # first (and only) face

        # --- Highlight mouth landmarks ---
        mouth_pts = []
        for idx in _MOUTH_INDICES:
            lm = face_landmarks[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            mouth_pts.append((lm.x, lm.y))
            cv2.circle(img_bgr, (px, py), 3, (0, 255, 0), -1)

        if not mouth_pts:
            return False, img_bgr

        # --- Compute mouth bounding box (normalized) ---
        mouth_x = [p[0] for p in mouth_pts]
        mouth_y = [p[1] for p in mouth_pts]
        mx_min, mx_max = min(mouth_x), max(mouth_x)
        my_min, my_max = min(mouth_y), max(mouth_y)

        # Add padding
        pad_x = (mx_max - mx_min) * _MOUTH_PADDING
        pad_y = (my_max - my_min) * _MOUTH_PADDING
        mx_min -= pad_x
        mx_max += pad_x
        my_min -= pad_y
        my_max += pad_y

        # Draw mouth bounding box
        box_x1 = int(mx_min * w)
        box_y1 = int(my_min * h)
        box_x2 = int(mx_max * w)
        box_y2 = int(my_max * h)
        cv2.rectangle(img_bgr, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)

        # --- Draw hand landmarks ---
        for hand_landmarks in hand_landmarks_list:
            drawing_utils.draw_landmarks(
                img_bgr,
                hand_landmarks,
                self.mp_hand_connections,
            )
            # Highlight fingertips
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
                    self.counter += 1
                    print(self.counter)
                    break  # one hand = one trigger per frame

        return picked > 0, img_bgr
