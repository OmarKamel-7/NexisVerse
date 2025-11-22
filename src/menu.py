# menu.py – Vision Pro Style Hand-Gesture Menu System (Clean Sidebar Version)
import cv2
import mediapipe as mp
import time, math, sys, os, subprocess
from apps.chem import run_physical_chemistry_ui

# ---------- IMPORT APP MODULES ----------
from apps.common import draw_text_centered
from apps.virtual_lab_bio import run_biology_lab
from apps.virtual_lab_chemistry import run_chemistry_lab
from apps.math_app import run_math_app

# ---------- MENU CONFIGURATION ----------
MENUS = {
    "Main Menu": ["Math App", "Virtual Lab", "Study Session", "Music", "Notebook", "Exit"],
    "Math App": ["Algebra", "Geometry", "Calculator", "Back"],
    "Virtual Lab": ["Physics Lab", "Chemistry Lab", "Biology Lab", "chem","Back"],
    "Study Session": ["Start Timer", "Review Notes", "Back"],
    "Music": ["Play", "Library", "Back"],
    "Notebook": ["Open Notes", "New Note", "Back"],
}

PINCH_THRESHOLD = 40   # Distance threshold between thumb & index for pinch
BUTTON_HEIGHT = 70
BUTTON_SPACING = 20
SIDEBAR_WIDTH = 380

# ---------- MEDIAPIPE HANDS ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
)

# ---------- GLOBAL STATE ----------
current_menu = "Main Menu"
hovered_button = -1


# ---------- HELPER FUNCTIONS ----------
def compute_button_layout(items: list):
    """Return bounding boxes for sidebar menu buttons."""
    boxes = []
    start_x = 40
    start_y = 140
    for i, _ in enumerate(items):
        y = start_y + i * (BUTTON_HEIGHT + BUTTON_SPACING)
        boxes.append((start_x, y, start_x + SIDEBAR_WIDTH, y + BUTTON_HEIGHT))
    return boxes


def draw_sidebar(frame, items, selected_index):
    """Draw styled sidebar buttons and return bounding boxes."""
    button_boxes = compute_button_layout(items)

    # Sidebar background
    cv2.rectangle(frame, (20, 100), (SIDEBAR_WIDTH + 60, 900), (25, 25, 40), -1)
    cv2.rectangle(frame, (20, 100), (SIDEBAR_WIDTH + 60, 900), (80, 150, 200), 2)

    for i, (x1, y1, x2, y2) in enumerate(button_boxes):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Hover animation
        if i == selected_index:
            bg_color = (60, 200, 255)
            border_color = (255, 255, 255)
        else:
            bg_color = (50, 60, 90)
            border_color = (150, 180, 220)

        # Shadow
        cv2.rectangle(frame, (x1+4, y1+4), (x2+4, y2+4), (0, 0, 0), -1)

        # Button
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

        # Text
        draw_text_centered(frame, items[i], (cx, cy), fontsize=0.9, color=(255, 255, 255))

    return button_boxes


def handle_selection(choice: str):
    """Execute action when a menu option is selected."""
    global current_menu

    if choice == "Back":
        current_menu = "Main Menu"
        return None

    if choice == "Exit":
        return "Exit"

    if choice in MENUS:
        current_menu = choice
        return None

    if current_menu == "Virtual Lab":
        if choice == "Biology Lab":
            run_biology_lab()
        elif choice == "Chemistry Lab":
            run_chemistry_lab()

    if current_menu == "Notebook":
        if choice == "Open Notes":
            path = "data/notebook.txt"
            abs_path = os.path.abspath(path)
            if sys.platform.startswith("win"):
                os.startfile(abs_path)
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", abs_path])
            else:
                subprocess.Popen(["xdg-open", abs_path])
                
    if choice == "Algebra":
        run_math_app()
        return None
    
    
    if choice == "chem":
        run_physical_chemistry_ui()
        return None
    
    current_menu = choice
    
    
    return None
                

    return None



# ---------- MAIN MENU LOOP (Faster Startup / Lighter) ----------
def run_menu():
    """Start the AR hand-gesture controlled sidebar menu system (optimized for lower RAM)."""
    global current_menu, hovered_button

    # Lower resolution to reduce memory and GPU load
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_click_time = 0
    DEBOUNCE_DELAY = 0.6  # seconds

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror
            h, w, _ = frame.shape

            # Mediapipe hand processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Draw sidebar
            items = MENUS[current_menu]
            button_boxes = draw_sidebar(frame, items, hovered_button)

            # Draw menu title
            draw_text_centered(frame, current_menu, (SIDEBAR_WIDTH // 2 + 40, 80),
                               fontsize=1.2, color=(0, 255, 200))

            # Gesture detection
            hover_index, pinch_detected = -1, False
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                ix, iy = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)
                tx, ty = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)

                # Draw fingertip markers
                cv2.circle(frame, (ix, iy), 10, (0, 255, 255), -1)
                cv2.circle(frame, (tx, ty), 8, (255, 200, 0), -1)

                # Detect pinch
                dist = math.hypot(ix - tx, iy - ty)
                pinch_detected = dist < PINCH_THRESHOLD

                # Check hover over button
                for i, (x1, y1, x2, y2) in enumerate(button_boxes):
                    if x1 < ix < x2 and y1 < iy < y2:
                        hover_index = i
                        break

            # Handle selection with debounce
            if pinch_detected and hover_index != -1:
                now = time.time()
                if now - last_click_time > DEBOUNCE_DELAY:
                    choice = items[hover_index]
                    action = handle_selection(choice)
                    if action == "Exit":
                        return "Exit"
                    last_click_time = now

            hovered_button = hover_index

            # Display frame
            cv2.imshow("NexisVerse AR Menu", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return None
