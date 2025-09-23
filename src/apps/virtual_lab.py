# apps/virtual_lab.py
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from apps.common import draw_text_centered, draw_icon_placeholder

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# We'll simulate a 3D organ viewer by overlaying a rotating image (circle) and allow gesture-driven scale/rotate.
HOVER_DURATION = 1.0

def run_virtual_lab():
    cap = cv2.VideoCapture(0)
    angle = 0.0
    scale = 1.0
    last_time = time.time()
    hover_idx = -1
    hover_start = None
    mode = "Select"  # or "Rotate" or "Scale" or "Back"
    items = ["Organs (3D)", "Sine Wave", "Back"]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # Draw UI
        draw_text_centered(frame, "Virtual Lab", (w//2, 40), fontsize=1.4, color=(200,200,20))
        # draw item buttons left
        box_w = 260
        box_h = 70
        start_x = 30
        start_y = 100
        boxes = []
        for i, it in enumerate(items):
            y = start_y + i*(box_h+18)
            boxes.append((start_x, y, start_x+box_w, y+box_h))
            color=(200,200,200)
            if i==hover_idx: color=(0,200,255)
            cv2.rectangle(frame, (start_x,y), (start_x+box_w,y+box_h), color, 2)
            draw_text_centered(frame, it, ((start_x+start_x+box_w)//2, y+box_h//2), fontsize=0.9, color=color)

        fingertip = None
        if res.multi_hand_landmarks:
            # use the first hand's index finger for selection and use differences between hands for scale/rotate
            for hms in res.multi_hand_landmarks:
                x = int(hms.landmark[8].x * w)
                y = int(hms.landmark[8].y * h)
                fingertip = (x,y)
                cv2.circle(frame, fingertip, 12, (0,0,255), -1)
                for i,b in enumerate(boxes):
                    if b[0] < x < b[2] and b[1] < y < b[3]:
                        hover_idx = i
                        break

        # hover logic to open sections
        progress = 0
        now = time.time()
        if hover_idx != -1:
            if hover_idx == hover_idx: # trivial to keep stable
                if hover_start is None:
                    hover_start = now
                elapsed = now - hover_start
                progress = min(1, elapsed / HOVER_DURATION)
                if elapsed >= HOVER_DURATION:
                    # select item
                    chosen = items[hover_idx]
                    if chosen == "Organs (3D)":
                        show_organs_viewer(frame, cap)
                    elif chosen == "Sine Wave":
                        show_sine_wave_simulator(frame, cap)
                    elif chosen == "Back":
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    hover_start = None
                    hover_idx = -1
            else:
                hover_start = now
        else:
            hover_start = None

        # main AR overlay: a rotating "organ" circle in center for 3D feel
        angle += 20 * (time.time() - last_time)
        last_time = time.time()
        cx, cy = w//2 + 150, h//2
        rad = int(120*scale)
        # draw a "rotating" organ with gradient and simple veins
        overlay = frame.copy()
        cv2.circle(overlay, (cx,cy), rad, (80,10,10), -1)
        # rotate some lines
        for i in range(8):
            a = math.radians(angle + i*45)
            x2 = int(cx + rad*0.8*math.cos(a))
            y2 = int(cy + rad*0.8*math.sin(a))
            cv2.line(overlay, (cx,cy), (x2,y2), (120,20,20), 2)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        draw_text_centered(frame, "Try selecting 'Organs (3D)' or 'Sine Wave' from left", (w//2, h-40), fontsize=0.6)

        cv2.imshow("Virtual Lab", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def show_organs_viewer(parent_frame, cap):
    # immersive viewer: keep camera active and allow hover-back
    mpf = mp.solutions.hands
    handss = mpf.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    angle = 0.0
    scale = 1.0
    hover_back = False
    hover_start = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = handss.process(rgb)
        # draw a rotating organ in center
        cx,cy = w//2, h//2
        rad = int(140*scale)
        overlay = frame.copy()
        angle += 18
        cv2.circle(overlay, (cx,cy), rad, (70, 15, 20), -1)
        for i in range(12):
            a = math.radians(angle + i*30)
            x2 = int(cx + rad*0.75*math.cos(a))
            y2 = int(cy + rad*0.75*math.sin(a))
            cv2.line(overlay, (cx,cy), (x2,y2), (120,30,30), 3)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        draw_text_centered(frame, "Organs Viewer - Hover on top-left Back to return", (w//2, 40), fontsize=1.0)
        # draw back box
        bx,by,bw,bh = 20,20,140,60
        cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (200,200,200), 2)
        draw_text_centered(frame, "Back", (bx + bw//2, by + bh//2), fontsize=0.8)
        fingertip=None
        if res.multi_hand_landmarks:
            for hms in res.multi_hand_landmarks:
                x = int(hms.landmark[8].x * w)
                y = int(hms.landmark[8].y * h)
                fingertip=(x,y)
                cv2.circle(frame, (x,y), 12, (0,0,255), -1)
                if bx < x < bx+bw and by < y < by+bh:
                    if hover_start is None:
                        hover_start = time.time()
                    if time.time() - hover_start > 1.0:
                        handss.close()
                        return
                else:
                    hover_start = None
        cv2.imshow("Organs Viewer", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    handss.close()

def show_sine_wave_simulator(parent_frame, cap):
    # show an animated sine wave overlay; user can change freq/amp by moving hand up/down while hovering on controls
    mpf = mp.solutions.hands
    handss = mpf.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    freq = 1.0
    amp = 1.0
    hover_control = None
    hover_start = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = handss.process(rgb)
        # draw ui
        draw_text_centered(frame, "Sine Wave Simulator - Hover controls: Freq (left), Amp (right), Back (top-left)", (w//2, 30), fontsize=0.7)
        # controls
        fx,fy,fw,fh = 40, 100, 180, 70
        ax,ay,aw,ah = w-220, 100, 180, 70
        bx,by,bw,bh = 20,20,140,60  # back
        cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (200,200,200), 2)
        draw_text_centered(frame, f"Freq {freq:.2f}", (fx+fw//2, fy+fh//2), fontsize=0.7)
        cv2.rectangle(frame, (ax,ay), (ax+aw,ay+ah), (200,200,200), 2)
        draw_text_centered(frame, f"Amp {amp:.2f}", (ax+aw//2, ay+ah//2), fontsize=0.7)
        cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (200,200,200), 2)
        draw_text_centered(frame, "Back", (bx+bw//2, by+bh//2), fontsize=0.7)
        fingertip=None
        if res.multi_hand_landmarks:
            for hms in res.multi_hand_landmarks:
                x=int(hms.landmark[8].x*w)
                y=int(hms.landmark[8].y*h)
                fingertip=(x,y)
                cv2.circle(frame, (x,y), 12, (0,0,255), -1)
                # controls hover
                if fx < x < fx+fw and fy < y < fy+fh:
                    hover_control = "freq"
                elif ax < x < ax+aw and ay < y < ay+ah:
                    hover_control = "amp"
                elif bx < x < bx+bw and by < y < by+bh:
                    hover_control = "back"
                else:
                    hover_control = None
                if hover_control:
                    if hover_start is None:
                        hover_start = time.time()
                    else:
                        if time.time() - hover_start > 0.8:
                            if hover_control == "back":
                                handss.close()
                                return
                            elif hover_control == "freq":
                                # map y to freq 0.5..5.0
                                freq = max(0.5, 5.0 - (y / h) * 4.5)
                            elif hover_control == "amp":
                                amp = max(0.2, 3.0 - (y / h) * 2.8)
                            hover_start = None
                else:
                    hover_start = None
        # draw sine on an overlay
        overlay = frame.copy()
        pts = []
        for px in range(0, w, 4):
            xnorm = (px / w) * 2 * math.pi * freq
            yval = int(h//2 + amp * 80 * math.sin(xnorm + time.time()*2*math.pi*0.5))
            pts.append((px, yval))
        for i in range(len(pts)-1):
            cv2.line(overlay, pts[i], pts[i+1], (50,200,50), 2)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        cv2.imshow("Sine Wave", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    handss.close()
