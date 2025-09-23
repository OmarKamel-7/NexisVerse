# menu.py
import cv2
import mediapipe as mp
import time
import math
from apps.common import draw_text_centered, draw_icon_placeholder

# -------- MENU CONFIG (matches your answers) ----------
menus = {
    "Main Menu": ["Math App", "Virtual Lab", "Study Session", "Music", "Notebook", "Exit"],
    "Math App": ["Launch", "Back"],
    "Virtual Lab": ["Launch", "Back"],
    "Study Session": ["Launch", "Back"],
    "Music": ["Launch", "Back"],
    "Notebook": ["Open", "Back"]
}

# hover selection settings
HOVER_DURATION = 1.0  # seconds to select (you said fast)
ICON_SIZE = 48

# Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# persistent state
current_menu = "Main Menu"
menu_selected = -1
hover_start_time = None

# TASKBAR / Pomodoro state (global to menu)
pomodoro_running = False
pomodoro_start = None
pomodoro_seconds = 25 * 60  # default 25 minutes
pomodoro_remaining = pomodoro_seconds

# helper: create button boxes depending on layout mix
def compute_button_layout(frame, items):
    h, w, _ = frame.shape
    # We'll design a mixed layout: top title, central vertical list, and a horizontal radial-ish bar for quick items
    center_w = int(w * 0.6)
    btn_h = 70
    start_x = (w - center_w) // 2
    start_y = 140

    boxes = []
    for i, item in enumerate(items):
        y = start_y + i * (btn_h + 22)
        boxes.append((start_x, y, start_x + center_w, y + btn_h))
    return boxes

def draw_taskbar(frame):
    global pomodoro_remaining, pomodoro_running, pomodoro_start
    h, w, _ = frame.shape
    bar_h = 80
    y0 = h - bar_h
    cv2.rectangle(frame, (0, y0), (w, h), (20, 20, 25), -1)  # dark bar

    # clock
    now = time.localtime()
    timestr = time.strftime("%Y-%m-%d  %H:%M:%S", now)
    draw_text_centered(frame, timestr, (10, y0 + 26), fontsize=0.6, color=(220,220,220), align_left=True)

    # Pomodoro display (center of taskbar)
    px = w // 2
    py = y0 + 40
    # update remaining
    if pomodoro_running and pomodoro_start:
        elapsed = int(time.time() - pomodoro_start)
        pomodoro_remaining = max(0, pomodoro_seconds - elapsed)
        if pomodoro_remaining == 0:
            pomodoro_running = False
            pomodoro_start = None

    mins = pomodoro_remaining // 60
    secs = pomodoro_remaining % 60
    pstr = f"Pomodoro: {mins:02d}:{secs:02d} {'▶' if pomodoro_running else '◻'}"
    draw_text_centered(frame, pstr, (px - 80, py), fontsize=0.8, align_left=True)

    # Notebook icon (right)
    nb_x = w - 180
    draw_icon_placeholder(frame, (nb_x, y0 + 10), ICON_SIZE, label="Note")

    # Return bounding boxes for interactive elements on the taskbar
    # Pomodoro area
    pomobox = (px - 100, y0 + 8, px + 160, y0 + bar_h - 8)
    notebook_box = (nb_x, y0 + 10, nb_x + ICON_SIZE + 10, y0 + 10 + ICON_SIZE + 10)
    return {"pomodoro": pomobox, "notebook": notebook_box}

def run_menu():
    global current_menu, menu_selected, hover_start_time
    global pomodoro_running, pomodoro_start, pomodoro_remaining

    cap = cv2.VideoCapture(0)
    last_frame_time = time.time()
    selected_result = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        items = menus[current_menu]
        button_boxes = compute_button_layout(frame, items)

        # draw menu title
        draw_text_centered(frame, current_menu, (w // 2, 60), fontsize=1.6, color=(200, 200, 20))

        # draw buttons
        for i, box in enumerate(button_boxes):
            x1,y1,x2,y2 = box
            color = (200,200,200)
            thickness = 2
            if i == menu_selected:
                color = (0, 200, 255)
                thickness = 4
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
            draw_text_centered(frame, items[i], ((x1+x2)//2, (y1+y2)//2), fontsize=1.0, color=color)

            # placeholder icon left of text
            draw_icon_placeholder(frame, (x1+12, y1+10), 48, label=items[i][:2])

        # draw taskbar and get interactive areas
        taskareas = draw_taskbar(frame)

        hover_index = -1
        hovered_task = None
        fingertip = None

        if result.multi_hand_landmarks:
            # allow multi-hand; if any fingertip is over a control we use it
            for handLms in result.multi_hand_landmarks:
                x = int(handLms.landmark[8].x * w)
                y = int(handLms.landmark[8].y * h)
                fingertip = (x, y)
                cv2.circle(frame, fingertip, 14, (0,0,255), -1)
                # check menu buttons
                for i, box in enumerate(button_boxes):
                    if box[0] < x < box[2] and box[1] < y < box[3]:
                        hover_index = i
                        break
                # check taskbar interactions (pomodoro/notebook)
                if hover_index == -1:
                    for name, tb in taskareas.items():
                        if tb[0] < x < tb[2] and tb[1] < y < tb[3]:
                            hovered_task = name
                            break

        # handle hover selection or taskbar hover
        progress = 0
        now = time.time()
        if hovered_task:
            # taskbar has precedence: immediate hover triggers actions
            if hover_start_time and menu_selected == -99 and hovered_task == menu_selected_task:
                elapsed = now - hover_start_time
                progress = min(1, elapsed / HOVER_DURATION)
                if elapsed >= HOVER_DURATION:
                    # execute task
                    if hovered_task == "pomodoro":
                        if pomodoro_running:
                            # pause
                            pomodoro_running = False
                            pomodoro_start = None
                        else:
                            pomodoro_running = True
                            pomodoro_start = time.time() - (pomodoro_seconds - pomodoro_remaining)
                    elif hovered_task == "notebook":
                        # open notebook file externally
                        import os, subprocess, sys
                        notebook_path = "data/notebook.txt"
                        if sys.platform.startswith("win"):
                            os.startfile(os.path.abspath(notebook_path))
                        elif sys.platform.startswith("darwin"):
                            subprocess.Popen(["open", notebook_path])
                        else:
                            subprocess.Popen(["xdg-open", notebook_path])
                    hover_start_time = None
                    menu_selected = -1
                    menu_selected_task = None
            else:
                # set special menu_selected code for taskbar hover
                menu_selected = -99
                menu_selected_task = hovered_task
                hover_start_time = now
        elif hover_index != -1:
            if hover_index == menu_selected:
                elapsed = now - hover_start_time if hover_start_time else 0
                progress = min(1, elapsed / HOVER_DURATION)
                if elapsed >= HOVER_DURATION:
                    chosen = items[hover_index]
                    # behavior when chosen
                    if chosen == "Back":
                        current_menu = "Main Menu"
                    elif chosen == "Exit":
                        cap.release()
                        cv2.destroyAllWindows()
                        return "Exit"
                    elif chosen == "Launch" or chosen == "Open":
                        # launching a named app: the app name is the parent menu title
                        app_name = current_menu
                        cap.release()
                        cv2.destroyAllWindows()
                        return app_name
                    elif chosen in menus:
                        current_menu = chosen
                    hover_start_time = None
                    menu_selected = -1
            else:
                menu_selected = hover_index
                hover_start_time = now
        else:
            menu_selected = -1
            hover_start_time = None

        # show progress ring at fingertip
        if fingertip and progress > 0:
            cx, cy = fingertip
            radius = 26
            end_angle = int(360 * progress)
            cv2.circle(frame, (cx, cy), radius, (120,120,120), 2)
            cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, end_angle, (0,200,255), 5)

        # small FPS
        tdiff = time.time() - last_frame_time
        fps = 1.0/tdiff if tdiff>0 else 0
        draw_text_centered(frame, f"FPS:{int(fps)}", (10,10), fontsize=0.5, align_left=True)
        last_frame_time = time.time()

        cv2.imshow("HIZMOS AR Menu", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
