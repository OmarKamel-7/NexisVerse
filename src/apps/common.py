# apps/common.py
import cv2
import numpy as np

def draw_text_centered(frame, text, center_xy, fontsize=1.0, color=(255,255,255), align_left=False):
    x,y = center_xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = fontsize
    thickness = 2 if fontsize >= 0.9 else 1
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    if align_left:
        org = (int(x), int(y))
    else:
        org = (int(x - size[0]//2), int(y + size[1]//2))
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)

def draw_icon_placeholder(frame, top_left, size, label=""):
    x,y = top_left
    x,y = int(x), int(y)
    cv2.rectangle(frame, (x, y), (x+size, y+size), (60,60,70), -1)
    cv2.rectangle(frame, (x, y), (x+size, y+size), (120,120,120), 1)
    if label:
        cv2.putText(frame, label, (x+6, y+size//2+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
