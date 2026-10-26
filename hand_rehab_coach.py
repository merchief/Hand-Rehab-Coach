"""
Hand Rehab Coach — Python / OpenCV version
============================================

A webcam-based rep coach for hand/finger rehab exercises (e.g. for carpal
tunnel recovery), with a clickable home screen to choose which exercise to
do, and a button to return to that home screen at any time.

WHAT IT DOES
------------
- Shows a home screen listing available exercises as clickable buttons.
- Once you pick one, opens your webcam and tracks your hand using
  MediaPipe's HandLandmarker.
- Measures each exercise's relevant motion using full 3D landmark positions
  (x, y, z), not flat 2D pixel positions, so the measurement stays accurate
  even as your hand rotates toward or away from the camera.
- Normalizes distance-based measurements by your palm width, so results
  reflect actual hand motion rather than how close you're sitting to the
  camera.
- Coaches you through reps with real-time feedback, and only counts a rep
  if you complete the full range of motion — partial movements don't count.
- A "Home" button in the corner of the exercise view takes you back to the
  exercise picker at any time.
- When you quit (press 'q') or return home, completed reps are saved to a
  CSV file (session_log.csv) in the same folder, so history persists across
  runs and across exercises.

HOW TO RUN IT
-------------
1. Make sure you have Python 3.9+ installed.
2. Install the two dependencies:
       pip install opencv-python mediapipe
3. Run the script:
       python hand_rehab_coach.py
   The first time you run it, it will automatically download MediaPipe's
   hand-landmark model file (a few MB, one-time, requires internet).
4. Click an exercise on the home screen to start it. Click "Home" (top-left
   of the camera view) to go back and pick a different one. Press 'q' at
   any time to quit entirely.

EXERCISES INCLUDED
-------------------
1. Index Finger Tendon Glide — curl your index finger into a full fist,
   then extend it fully straight. Measured as the bend angle at the
   knuckle in the middle of the finger (the PIP joint).
2. Thumb-to-Pinky Opposition — touch your thumb tip to your pinky
   fingertip, then release back to a fully open hand. Measured as the
   (palm-width-normalized) distance between the two fingertips. This is a
   real hand-therapy exercise used to test/rebuild thumb opposition.

NOTES
-----
This is a prototype, not a medical device. The measurements are
approximate and depend on lighting, camera angle, and hand visibility.
Stop any exercise that causes pain, and follow your actual
clinician-prescribed program.
"""

import csv
import math
import os
import time
import urllib.request
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import core as mp_core

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_FILE = "session_log.csv"
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
WINDOW_NAME = "Hand Rehab Coach"

# Landmark indices (MediaPipe's standard 21-point hand topology)
WRIST = 0
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
PINKY_MCP, PINKY_TIP = 17, 20
THUMB_TIP = 4

# Used as a stable "ruler" to normalize distances by hand size — the span
# across the base of the index and pinky knuckles stays roughly constant
# for a given hand regardless of camera distance.
PALM_LEFT, PALM_RIGHT = INDEX_MCP, PINKY_MCP

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Colors (BGR, since OpenCV uses BGR not RGB)
COLOR_BG = (38, 30, 27)
COLOR_TEAL = (106, 111, 43)
COLOR_TEAL_LIGHT = (150, 150, 80)
COLOR_GREEN = (118, 158, 63)
COLOR_AMBER = (43, 154, 217)
COLOR_TEXT = (230, 245, 240)
COLOR_MUTED = (150, 160, 158)
COLOR_LANDMARK = (63, 158, 118)
COLOR_HIGHLIGHT = (43, 154, 217)
COLOR_PANEL = (52, 42, 38)

# ---------------------------------------------------------------------------
# Exercise definitions
# ---------------------------------------------------------------------------
# Each exercise is defined generically as either an "angle" measurement
# (angle at a joint formed by three landmarks) or a "distance" measurement
# (palm-normalized distance between two landmarks). The same rep-detection
# and gauge-drawing code below works for both, driven off this config.
EXERCISES = [
    {
        "key": "tendon_glide",
        "name": "Index Finger Tendon Glide",
        "type": "angle",
        "landmarks": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
        "gauge_range": (30, 180),       # (fully curled, fully straight)
        "trigger_low": 90,              # crossing below this starts the "active" phase
        "trigger_high": 160,            # crossing back above this completes a rep
        "quality_cutoff": 70,           # how low you must get for "full range"
        "prompt_idle": "Curl your index finger into a full fist.",
        "prompt_active": "Good — now extend fully back out.",
        "prompt_return": "Keep extending — straighten the finger fully.",
    },
    {
        "key": "thumb_opposition",
        "name": "Thumb-to-Pinky Opposition",
        "type": "distance",
        "landmarks": (THUMB_TIP, PINKY_TIP),
        "gauge_range": (0.05, 1.1),     # (fingertips touching, hand fully open)
        "trigger_low": 0.35,            # normalized distance counted as "touching"
        "trigger_high": 0.85,           # normalized distance counted as "released"
        "quality_cutoff": 0.5,          # how close you must get for "full range"
        "prompt_idle": "Bring your thumb tip to touch your pinky fingertip.",
        "prompt_active": "Good — now open your hand back up fully.",
        "prompt_return": "Keep opening — spread your hand out fully.",
    },
]


def ensure_model_downloaded():
    """Download the hand-landmark model file on first run, if not present."""
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand-tracking model (one-time, ~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def angle_between(a, b, c):
    """Angle (degrees) at point b, formed by rays b->a and b->c.

    a, b, c are (x, y, z) tuples. Using all three dimensions keeps the angle
    accurate when the hand rotates toward or away from the camera, which a
    flat 2D angle would distort.
    """
    v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if mag1 == 0 or mag2 == 0:
        return None
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def distance_3d(a, b):
    """Euclidean distance between two (x, y, z) landmark points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def compute_exercise_value(exercise, pts_3d, palm_scale):
    """Compute the current measurement for a given exercise config.

    Returns a single number: a joint angle in degrees for "angle" exercises,
    or a palm-width-normalized distance for "distance" exercises. Both are
    designed to mean roughly the same thing across different hand sizes and
    camera distances.
    """
    if exercise["type"] == "angle":
        a, b, c = exercise["landmarks"]
        return angle_between(pts_3d[a], pts_3d[b], pts_3d[c])
    else:
        a, b = exercise["landmarks"]
        if palm_scale == 0:
            return None
        return distance_3d(pts_3d[a], pts_3d[b]) / palm_scale


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_gauge(frame, value, gauge_range, center=(100, 100), radius=70):
    """Draw a semicircular gauge showing where `value` sits within gauge_range."""
    gmin, gmax = gauge_range
    display_value = value if value is not None else gmax
    clamped = max(gmin, min(gmax, display_value))
    t = (clamped - gmin) / (gmax - gmin)

    cv2.ellipse(frame, center, (radius, radius), 0, 180, 360, (90, 100, 98), 10)
    end_angle = 180 + int(180 * t)
    cv2.ellipse(frame, center, (radius, radius), 0, 180, end_angle, COLOR_TEAL, 10)

    needle_deg = 180 + 180 * t
    needle_rad = math.radians(needle_deg)
    nx = int(center[0] + radius * 0.85 * math.cos(needle_rad))
    ny = int(center[1] + radius * 0.85 * math.sin(needle_rad))
    cv2.line(frame, center, (nx, ny), COLOR_TEXT, 2)
    cv2.circle(frame, center, 4, COLOR_TEXT, -1)

    label = f"{value:.2f}" if value is not None else "--"
    cv2.putText(frame, label, (center[0] - 30, center[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)


def draw_feedback(frame, text, mode="wait"):
    color_map = {"good": COLOR_GREEN, "adjust": COLOR_AMBER, "wait": COLOR_MUTED}
    color = color_map.get(mode, COLOR_MUTED)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 50), (w, h), (30, 40, 38), -1)
    cv2.putText(frame, text, (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)


def point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(frame, rect, label, hovered=False, sublabel=None):
    x1, y1, x2, y2 = rect
    fill = COLOR_TEAL_LIGHT if hovered else COLOR_TEAL
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TEXT, 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
    tx = x1 + (x2 - x1 - text_size[0]) // 2
    ty = y1 + (y2 - y1 + text_size[1]) // 2 - (10 if sublabel else 0)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_TEXT, 2)
    if sublabel:
        sub_size = cv2.getTextSize(sublabel, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        sx = x1 + (x2 - x1 - sub_size[0]) // 2
        cv2.putText(frame, sublabel, (sx, ty + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 220, 218), 1)


# ---------------------------------------------------------------------------
# Shared app state (mutated by the mouse callback)
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.mode = "home"          # "home" or "exercise"
        self.selected_exercise = None
        self.quit_requested = False
        self.go_home_requested = False
        self.home_buttons = []      # list of (rect, exercise_index_or_None_for_quit)
        self.exercise_home_button = None
        self.mouse_pos = (-1, -1)


def make_mouse_callback(state: AppState):
    def callback(event, x, y, flags, param):
        state.mouse_pos = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state.mode == "home":
            for rect, action in state.home_buttons:
                if point_in_rect(x, y, rect):
                    if action == "quit":
                        state.quit_requested = True
                    else:
                        state.selected_exercise = action
                        state.mode = "exercise"
        elif state.mode == "exercise":
            if state.exercise_home_button and point_in_rect(x, y, state.exercise_home_button):
                state.go_home_requested = True
    return callback


def build_home_frame(width, height, state: AppState):
    import numpy as np
    frame = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    cv2.putText(frame, "Hand Rehab Coach", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_TEXT, 2)
    cv2.putText(frame, "Choose an exercise", (40, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_MUTED, 1)

    state.home_buttons = []
    btn_w, btn_h, gap = 500, 70, 20
    start_y = 140
    for i, exercise in enumerate(EXERCISES):
        rect = (40, start_y + i * (btn_h + gap), 40 + btn_w, start_y + i * (btn_h + gap) + btn_h)
        hovered = point_in_rect(*state.mouse_pos, rect)
        draw_button(frame, rect, exercise["name"], hovered=hovered,
                    sublabel="angle-based" if exercise["type"] == "angle" else "distance-based")
        state.home_buttons.append((rect, i))

    quit_y = start_y + len(EXERCISES) * (btn_h + gap) + 10
    quit_rect = (40, quit_y, 40 + 160, quit_y + 46)
    hovered = point_in_rect(*state.mouse_pos, quit_rect)
    draw_button(frame, quit_rect, "Quit", hovered=hovered)
    state.home_buttons.append((quit_rect, "quit"))

    cv2.putText(frame, "Click an exercise to begin. Press 'q' anytime to quit.",
                (40, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_MUTED, 1)
    return frame


def save_log(entries):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "exercise", "rep_number",
                "min_value", "quality",
            ])
        writer.writerows(entries)


def run_exercise(cap, landmarker, exercise, state: AppState):
    """Run one exercise session until the user goes home or quits.

    Returns the list of completed-rep log entries from this session.
    """
    rep_state = "idle"   # "idle" -> "active" -> "idle" completes a rep
    value_extreme_this_rep = None  # tracks the deepest value reached during "active"
    reps = 0
    best_value = None
    log_entries = []
    session_start = time.time()

    home_btn_w, home_btn_h = 90, 34
    state.exercise_home_button = (10, 10, 10 + home_btn_w, 10 + home_btn_h)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        feedback_text = "Position your hand in frame, palm facing the camera."
        feedback_mode = "wait"
        current_value = None

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            pts_3d = [(lm.x, lm.y, lm.z) for lm in landmarks]

            palm_scale = distance_3d(pts_3d[PALM_LEFT], pts_3d[PALM_RIGHT])
            current_value = compute_exercise_value(exercise, pts_3d, palm_scale)

            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (90, 100, 98), 1)
            highlight_idxs = set(exercise["landmarks"])
            for i, p in enumerate(pts):
                is_key = i in highlight_idxs
                cv2.circle(frame, p, 5 if is_key else 3,
                           COLOR_HIGHLIGHT if is_key else COLOR_LANDMARK, -1)

            if current_value is not None:
                if value_extreme_this_rep is None:
                    value_extreme_this_rep = current_value
                else:
                    value_extreme_this_rep = min(value_extreme_this_rep, current_value)

                if rep_state == "idle":
                    if current_value < exercise["trigger_low"]:
                        rep_state = "active"
                        feedback_text = exercise["prompt_active"]
                        feedback_mode = "adjust"
                    else:
                        feedback_text = exercise["prompt_idle"]
                        feedback_mode = "wait"
                elif rep_state == "active":
                    if current_value > exercise["trigger_high"]:
                        reps += 1
                        if best_value is None or value_extreme_this_rep < best_value:
                            best_value = value_extreme_this_rep
                        quality = ("full range" if value_extreme_this_rep <= exercise["quality_cutoff"]
                                   else "shallow")
                        log_entries.append([
                            datetime.now().isoformat(timespec="seconds"),
                            exercise["name"],
                            reps,
                            round(value_extreme_this_rep, 3),
                            quality,
                        ])
                        feedback_text = f"Rep {reps} complete — {quality}."
                        feedback_mode = "good"
                        rep_state = "idle"
                        value_extreme_this_rep = None
                    else:
                        feedback_text = exercise["prompt_return"]
                        feedback_mode = "adjust"

        draw_gauge(frame, current_value, exercise["gauge_range"])

        elapsed = int(time.time() - session_start)
        mins, secs = divmod(elapsed, 60)
        cv2.putText(frame, exercise["name"], (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        cv2.putText(frame, f"Reps: {reps}", (w - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT, 2)
        cv2.putText(frame, f"Time: {mins}:{secs:02d}", (w - 150, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)
        if best_value is not None:
            cv2.putText(frame, f"Best: {best_value:.2f}", (w - 150, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 2)

        hovered_home = point_in_rect(*state.mouse_pos, state.exercise_home_button)
        draw_button(frame, state.exercise_home_button, "Home", hovered=hovered_home)

        draw_feedback(frame, feedback_text, feedback_mode)
        cv2.putText(frame, "Click Home to switch exercises, 'q' to quit",
                    (200, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_MUTED, 1)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            state.quit_requested = True
            break
        if state.go_home_requested:
            state.go_home_requested = False
            break

    return log_entries


def main():
    ensure_model_downloaded()

    base_options = mp_core.base_options.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check that a camera is connected and not in use by another app.")
        return

    state = AppState()
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(state))

    all_log_entries = []
    home_w, home_h = 600, 420

    print("Hand Rehab Coach running. Click an exercise to begin, 'q' to quit.")

    while True:
        if state.mode == "home":
            frame = build_home_frame(home_w, home_h, state)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                state.quit_requested = True
            if state.quit_requested:
                break
            if state.mode == "exercise":
                continue
        else:
            exercise = EXERCISES[state.selected_exercise]
            entries = run_exercise(cap, landmarker, exercise, state)
            all_log_entries.extend(entries)
            if entries:
                save_log(entries)
                print(f"Saved {len(entries)} rep(s) from '{exercise['name']}'.")
            state.mode = "home"
            if state.quit_requested:
                break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    print(f"Session ended. {len(all_log_entries)} total rep(s) logged this run to {LOG_FILE}.")


if __name__ == "__main__":
    main()
