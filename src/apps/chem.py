# physical_chemistry_ui_v4.py
# Requirements: opencv-python, mediapipe, pyserial
# Run: python physical_chemistry_ui_v4.py

import cv2
import mediapipe as mp
import time
import math
import serial
import numpy as np
import os

# ---------------- CONFIG ----------------
PICO_PORT = "COM7"
BAUD_RATE = 115200
PINCH_THRESHOLD = 40
WINDOW_NAME = "Physical Chemistry Lab"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
VIDEO_PATH = r"C:\Users\amora\OneDrive\Desktop\New folder (2)\NexisVerse-main\src\apps\assets\Water Electrolysis Mechanism.mp4"

# ---------------- SERIAL CONNECTION ----------------
try:
    pico = serial.Serial(PICO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Pico on {PICO_PORT}")
except Exception as e:
    pico = None
    print(f"Pico not connected: {e}")

def send_command_to_pico(cmd: str):
    """Send a command string to the Pico over serial."""
    if pico and pico.is_open:
        try:
            pico.write((cmd + "\n").encode())
        except Exception as e:
            print(f"Serial error: {e}")
    else:
        print(f"[SIMULATION] Would send command: {cmd}")

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5,
                                max_num_hands=1)

# ---------------- UI HELPERS ----------------
def draw_button(frame, rect, label, bg_color, text_color=(255, 255, 255)):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cx = x1 + (x2 - x1) // 2 - tw // 2
    cy = y1 + (y2 - y1) // 2 + th // 2
    cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

def draw_text_box(frame, text_lines, pos, box_size, bg_color=(20,20,30), text_color=(220,220,220)):
    x, y = pos
    w, h = box_size
    cv2.rectangle(frame, (x, y), (x+w, y+h), bg_color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 180, 180), 2)
    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (x+10, y+30 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

def draw_3d_beaker(frame, center, radius_x=80, radius_y=120):
    cx, cy = center
    left, right = cx - radius_x, cx + radius_x
    top, bottom = cy - radius_y//2, cy + radius_y//2
    cv2.rectangle(frame, (left, top), (right, bottom), (200, 210, 220), -1)
    liquid_h = int(bottom - (bottom-top)*0.35)
    cv2.ellipse(frame, (cx, liquid_h), (radius_x-8, 16), 0, 0, 360, (180, 200, 255), -1)
    cv2.ellipse(frame, (cx, cy-radius_y//2), (radius_x, 20), 0, 0, 360, (140, 140, 160), 2)
    cv2.rectangle(frame, (left, top), (right, bottom), (140, 140, 160), 2)
    cv2.putText(frame, "Beaker", (cx-40, bottom+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
    return (left, top, right, bottom, liquid_h)

def draw_3d_sodium_block(frame, pos, size=60):
    x, y = pos
    s = size
    pts = np.array([[x-s, y-s//2],[x, y-s-s//6],[x+s, y-s//2],[x, y+s//3]], np.int32)
    cv2.drawContours(frame, [pts], -1, (160,80,60), -1)
    front = np.array([[x-s, y-s//2],[x-s, y+s//2],[x, y+s//3],[x, y-s//2]], np.int32)
    right = np.array([[x+s, y-s//2],[x+s, y+s//2],[x, y+s//3],[x, y-s//2]], np.int32)
    cv2.drawContours(frame, [front], -1, (140,60,40), -1)
    cv2.drawContours(frame, [right], -1, (120,50,30), -1)
    cv2.polylines(frame, [pts], True, (80,40,30), 2)
    cv2.putText(frame, "Na", (x-12, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,200), 2)

# ---------------- APP STATES ----------------
STATE_MAIN = "MAIN"
STATE_ELECTROLYSIS = "ELECTROLYSIS"
STATE_SODIUM = "SODIUM"

# ---------------- MAIN APP ----------------
def run_physical_chemistry_ui():
    state = STATE_MAIN
    electrolysis_on = False
    sodium_hold_on = False
    sodium_pos = [200, 180]
    dragging = False
    last_pin_time = 0

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cv2.namedWindow(WINDOW_NAME)

    # Load electrolysis video
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    electrolysis_cap = cv2.VideoCapture(VIDEO_PATH)

    # UI rects
    btn_main_e = (60, 120, 420, 220)
    btn_main_s = (60, 260, 420, 360)
    back_btn = (FRAME_WIDTH-160, FRAME_HEIGHT-80, FRAME_WIDTH-40, FRAME_HEIGHT-20)
    switch_rect = (FRAME_WIDTH-240, 140, FRAME_WIDTH-80, 200)

    def toggle_electrolysis(on: bool):
        nonlocal electrolysis_on
        electrolysis_on = on
        send_command_to_pico("ELECTROLYSIS_START" if on else "ELECTROLYSIS_STOP")

    def toggle_sodium(on: bool):
        nonlocal sodium_hold_on
        sodium_hold_on = on
        send_command_to_pico("SODIUM_WATER_START" if on else "SODIUM_WATER_STOP")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)
        ix = iy = None
        pinch = False
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            ix = int(hand.landmark[8].x * w)
            iy = int(hand.landmark[8].y * h)
            tx = int(hand.landmark[4].x * w)
            ty = int(hand.landmark[4].y * h)
            pinch = math.hypot(ix-tx, iy-ty) < PINCH_THRESHOLD
            cv2.circle(frame, (ix, iy), 10, (0,255,0), -1)

        # Top bar
        cv2.rectangle(frame, (0,0), (w,72), (18,24,32), -1)
        cv2.putText(frame, "Physical Chemistry — NexisVerse", (18,48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,240,255), 2)

        # ---------------- STATE MAIN ----------------
        if state == STATE_MAIN:
            draw_button(frame, btn_main_e, "Electrolysis", (200,30,30))
            draw_button(frame, btn_main_s, "Sodium + Water", (200,30,30))
            if ix and iy and pinch:
                x1, y1, x2, y2 = btn_main_e
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    state = STATE_ELECTROLYSIS
                    time.sleep(0.25)
                x1, y1, x2, y2 = btn_main_s
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    state = STATE_SODIUM
                    sodium_pos = [200, 180]
                    toggle_sodium(False)
                    time.sleep(0.25)

        # ---------------- STATE ELECTROLYSIS ----------------
        elif state == STATE_ELECTROLYSIS:
            # Video
            ret_vid, video_frame = electrolysis_cap.read()
            if not ret_vid:
                electrolysis_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_vid, video_frame = electrolysis_cap.read()
            vh, vw, _ = video_frame.shape
            new_h = 400
            new_w = int(vw * (new_h / vh))
            video_frame = cv2.resize(video_frame, (new_w, new_h))
            x_offset, y_offset = 50, 100
            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = video_frame

            # Info box
            draw_text_box(frame, [
                "Water Electrolysis",
                "Reaction: 2NaCl(aq)+2H2O(l)→2NaOH(aq)+H2(g)+Cl2(g)",
                "Electrolysis of brine. Anode produces Cl2, cathode H2 & NaOH.",
                "Safety: goggles, gloves, ventilate gases."
            ], (x_offset+new_w+20, 120), (FRAME_WIDTH-(x_offset+new_w+40), 140))

            # Electrolysis switch
            sx1, sy1, sx2, sy2 = switch_rect
            bg_color = (0,180,0) if electrolysis_on else (120,120,120)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), bg_color, -1)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (200,200,200), 2)
            knob_x = sx2-18 if electrolysis_on else sx1+18
            cv2.circle(frame, (knob_x, (sy1+sy2)//2), 18, (255,255,255), -1)
            cv2.putText(frame, f"Electrolysis: {'ON' if electrolysis_on else 'OFF'}", 
                        (sx1, sy2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2)

            draw_button(frame, back_btn, "Back", (60,60,90))

            if ix and iy and pinch:
                if sx1 <= ix <= sx2 and sy1 <= iy <= sy2 and time.time()-last_pin_time > 0.35:
                    toggle_electrolysis(not electrolysis_on)
                    last_pin_time = time.time()
                bx1, by1, bx2, by2 = back_btn
                if bx1 <= ix <= bx2 and by1 <= iy <= by2 and time.time()-last_pin_time > 0.35:
                    toggle_electrolysis(False)
                    state = STATE_MAIN
                    last_pin_time = time.time()

        # ---------------- STATE SODIUM ----------------
        elif state == STATE_SODIUM:
            beaker_center = (w//2+140, h//2-20)
            left, top, right, bottom, liquid_h = draw_3d_beaker(frame, beaker_center, 120, 180)
            draw_3d_sodium_block(frame, sodium_pos, 50)
            draw_text_box(frame, ["Pinch & drag the Na block into the beaker",
                                  "Relay stays ON while inside."], (60, 120), (500, 60))
            draw_button(frame, back_btn, "Back", (60,60,90))

            if ix and iy:
                if pinch and not dragging and math.hypot(ix-sodium_pos[0], iy-sodium_pos[1]) < 60:
                    dragging = True
                if dragging:
                    sodium_pos[0], sodium_pos[1] = ix, iy
                if not pinch and dragging:
                    dragging = False
                    beaker_cx, beaker_cy = beaker_center
                    if (sodium_pos[0]-beaker_cx)**2 + (sodium_pos[1]-liquid_h)**2 < 80**2:
                        toggle_sodium(True)
                    else:
                        toggle_sodium(False)

                bx1, by1, bx2, by2 = back_btn
                if pinch and bx1 <= ix <= bx2 and by1 <= iy <= by2:
                    toggle_sodium(False)
                    state = STATE_MAIN
                    time.sleep(0.3)

            cv2.putText(frame, f"Sodium HOLD: {'ON' if sodium_hold_on else 'OFF'}", (60, h-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,0) if sodium_hold_on else (160,160,160), 2)

        # Footer
        cv2.putText(frame, "Pinch to interact | ESC to quit", (18, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,190), 1)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            toggle_electrolysis(False)
            toggle_sodium(False)
            break

    cap.release()
    electrolysis_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_physical_chemistry_ui()
