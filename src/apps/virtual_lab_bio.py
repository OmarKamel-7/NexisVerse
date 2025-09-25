import cv2
import numpy as np
import mediapipe as mp
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def run_biology_lab():
    """
    Biology Lab – Interactive Heart Simulation
    - One-hand pinch (index+thumb) = drag/move heart
    - Two-hand pinch (both hands) = pinch distance controls zoom
    - Two-finger pinch (index+middle) = rotation toggle
    """

    # ---------------- Select Heart Image ----------------
    Tk().withdraw()  # hide root window
    heart_path = askopenfilename(title="Select Heart Image", 
                                 filetypes=[("PNG Images", "*.png"), ("All Files", "*.*")])
    if not heart_path:
        print("⚠️ No heart image selected. Exiting.")
        return

    heart_img = cv2.imread(heart_path, cv2.IMREAD_UNCHANGED)
    if heart_img is None:
        print(f"⚠️ Could not load image: {heart_path}")
        return

    print(f"🖐️ Loaded heart image: {heart_path}")
    print("🖐️ Biology Lab (Heart with Pinch Control)... (press Q to exit)")

    # ---------------- Mediapipe Hands Setup ----------------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # ---------------- Camera Setup ----------------
    cap = cv2.VideoCapture(0)
    width, height = 960, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ---------------- Heart State ----------------
    zoom = 1.0
    offset_x, offset_y = 0, 0
    beat_phase = 0
    last_drag = None
    rotating = False
    rotation_angle = 0

    # ---------------- Helper Functions ----------------
    def is_pinch(landmarks, idx1, idx2, threshold=40):
        """Return True if distance between two landmarks is below threshold"""
        x1, y1 = landmarks[idx1]
        x2, y2 = landmarks[idx2]
        return np.hypot(x2 - x1, y2 - y1) < threshold

    def draw_heart(frame, zoom, offset_x, offset_y, beat_phase, rotation_angle):
        """Draw a beating and rotating heart overlay on the frame"""
        # Simulate heartbeat
        scale = 1.0 + 0.1 * np.sin(beat_phase)

        # Resize heart
        new_w = int(200 * zoom * scale)
        new_h = int(200 * zoom * scale)
        heart_resized = cv2.resize(heart_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Split RGBA
        if heart_resized.shape[2] == 4:
            b, g, r, a = cv2.split(heart_resized)
            heart_rgb = cv2.merge((b, g, r))
            mask = a
        else:
            heart_rgb = heart_resized
            mask = cv2.cvtColor(heart_rgb, cv2.COLOR_BGR2GRAY)

        # Rotation
        if rotation_angle != 0:
            M = cv2.getRotationMatrix2D((new_w//2, new_h//2), rotation_angle, 1)
            heart_rgb = cv2.warpAffine(heart_rgb, M, (new_w, new_h), borderMode=cv2.BORDER_TRANSPARENT)
            mask = cv2.warpAffine(mask, M, (new_w, new_h), borderMode=cv2.BORDER_TRANSPARENT)

        # Heart position
        x = frame.shape[1]//2 - new_w//2 + offset_x
        y = frame.shape[0]//2 - new_h//2 + offset_y

        # Clip bounds
        x = max(0, min(x, frame.shape[1]-new_w))
        y = max(0, min(y, frame.shape[0]-new_h))

        # ROI
        roi = frame[y:y+new_h, x:x+new_w]

        # Overlay using mask
        mask_resized = cv2.resize(mask, (new_w, new_h))
        mask_inv = cv2.bitwise_not(mask_resized)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(heart_rgb, heart_rgb, mask=mask_resized)
        combined = cv2.add(bg, fg)
        frame[y:y+new_h, x:x+new_w] = combined

    # ---------------- Main Loop ----------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        beat_phase += 0.1

        pinch_points = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [(int(lm.x * width), int(lm.y * height))
                             for lm in hand_landmarks.landmark]

                # One-hand pinch (index+thumb)
                if is_pinch(landmarks, 4, 8):
                    pinch_points.append(landmarks[8])

                # Two-finger pinch (index+middle) for rotation
                rotating = is_pinch(landmarks, 8, 12)

        # Drag with one hand pinch
        if len(pinch_points) == 1:
            if last_drag is None:
                last_drag = pinch_points[0]
            else:
                dx = pinch_points[0][0] - last_drag[0]
                dy = pinch_points[0][1] - last_drag[1]
                offset_x += dx
                offset_y += dy
                last_drag = pinch_points[0]
        else:
            last_drag = None

        # Zoom with two-hand pinch
        if len(pinch_points) == 2:
            (x1,y1), (x2,y2) = pinch_points
            dist = np.hypot(x2-x1, y2-y1)
            zoom = np.clip(dist / 300, 0.5, 3.0)

        # Rotation
        if rotating:
            rotation_angle = (rotation_angle + 2) % 360

        # Draw heart
        draw_heart(frame, zoom, offset_x, offset_y, beat_phase, rotation_angle)

        # Info text
        cv2.putText(frame, "One-hand pinch = Move | Two-hand pinch = Zoom | Two-finger pinch = Rotate",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Biology Lab - Heart Simulation", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Biology Lab closed.")
