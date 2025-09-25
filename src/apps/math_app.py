# apps/math_app.py
import cv2
import mediapipe as mp
import random
import time
import math
import numpy as np
from apps.common import draw_text_centered, draw_icon_placeholder

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

HOVER_DURATION = 1.0

def make_question():
    # random arithmetic or geometry (simple)
    if random.random() < 0.5:
        # arithmetic
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        op = random.choice(["+", "-", "*"])
        expr = f"{a} {op} {b}"
        answer = eval(expr)
        prompt = f"{expr} = ?"
        choices = [answer, answer + random.randint(1,5), answer - random.randint(1,5), answer + random.randint(6,12)]
    else:
        # geometry: area of rectangle or circle radius
        if random.random() < 0.6:
            w = random.randint(2, 12)
            h = random.randint(2, 12)
            prompt = f"Area rect {w}x{h}?"
            answer = w*h
        else:
            r = random.randint(1,6)
            prompt = f"Area circle r={r}? (pi~3.14)"
            answer = round(3.14 * r * r, 1)
        choices = [answer]
        while len(choices) < 4:
            val = round(answer + random.uniform(-10,10), 1)
            if val not in choices:
                choices.append(val)
    random.shuffle(choices)
    return {"prompt": prompt, "choices": choices, "answer": answer}

def run_math_app():
    cap = cv2.VideoCapture(0)
    question = make_question()
    score = 0
    hover_idx = -1
    hover_start = None
    chosen_idx = -1
    last_q_time = time.time()
    wtext = "Math App - Hover a choice to answer. Back to return."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # Draw UI
        draw_text_centered(frame, wtext, (w//2, 30), fontsize=0.7)
        draw_text_centered(frame, question["prompt"], (w//2, 90), fontsize=1.2, color=(220,220,20))

        # draw choices boxes
        boxes = []
        box_w = 400
        box_h = 70
        start_x = (w - box_w) // 2
        start_y = 140
        for i, c in enumerate(question["choices"]):
            y = start_y + i*(box_h+18)
            boxes.append((start_x, y, start_x + box_w, y + box_h))
            color = (200,200,200)
            if i == hover_idx:
                color = (0,200,255)
            cv2.rectangle(frame, (start_x, y), (start_x+box_w, y+box_h), color, 2)
            draw_text_centered(frame, str(c), ((start_x+start_x+box_w)//2, y+box_h//2), fontsize=0.9, color=color)

        fingertip = None
        if res.multi_hand_landmarks:
            for hms in res.multi_hand_landmarks:
                x = int(hms.landmark[8].x * w)
                y = int(hms.landmark[8].y * h)
                fingertip = (x,y)
                cv2.circle(frame, fingertip, 14, (0,0,255), -1)

                for i, b in enumerate(boxes):
                    if b[0] < x < b[2] and b[1] < y < b[3]:
                        hover_idx = i
                        break

        # hover logic
        progress = 0
        now = time.time()
        if hover_idx != -1:
            if hover_idx == chosen_idx:
                # continuing hover on same choice
                elapsed = now - hover_start if hover_start else 0
                progress = min(1, elapsed / HOVER_DURATION)
                if elapsed >= HOVER_DURATION:
                    # commit choice
                    val = question["choices"][hover_idx]
                    if abs(val - question["answer"]) < 1e-6:
                        score += 1
                    # next question
                    question = make_question()
                    chosen_idx = -1
                    hover_idx = -1
                    hover_start = None
            else:
                chosen_idx = hover_idx
                hover_start = now
        else:
            chosen_idx = -1
            hover_start = None

        if fingertip and progress > 0:
            cx,cy = fingertip
            r = 26
            cv2.circle(frame, (cx,cy), r, (120,120,120), 2)
            cv2.ellipse(frame, (cx,cy), (r,r), -90, 0, int(360*progress), (0,200,255), 5)

        # score display
        draw_text_centered(frame, f"Score: {score}", (w-120, 30), fontsize=0.7, align_left=True)

        cv2.imshow("Math App", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
