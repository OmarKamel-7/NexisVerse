import cv2
import mediapipe as mp
import time

# === Mediapipe Hands ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# === New Realistic App Menu Structure ===
menus = {
    "Main Menu": ["Camera", "Music", "Files", "Settings", "Exit"],
    "Camera": ["Take Photo", "Record Video", "Back"],
    "Music": ["Playlists", "Now Playing", "Back"],
    "Files": ["Documents", "Downloads", "Back"],
    "Settings": ["Wi-Fi", "Bluetooth", "Display", "Back"]
}

current_menu = "Main Menu"
menu_selected = -1
hover_start_time = None
hover_duration = 1  # seconds to select (faster)

# === Draw Menu with Rectangles ===
def draw_menu(frame, hover_index):
    h, w, _ = frame.shape
    items = menus[current_menu]

    button_w = int(w * 0.6)   # 60% screen width
    button_h = 70
    start_x = (w - button_w) // 2
    start_y = 120

    button_boxes = []

    for i, item in enumerate(items):
        y = start_y + i * (button_h + 25)
        box = (start_x, y, start_x + button_w, y + button_h)
        button_boxes.append(box)

        color = (230, 230, 230)
        thickness = 2

        if i == hover_index:
            color = (0, 200, 255)  # highlight color
            thickness = 4

        # Button rectangle
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)

        # Button text (centered)
        text_size = cv2.getTextSize(item, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = box[0] + (button_w - text_size[0]) // 2
        text_y = box[1] + (button_h + text_size[1]) // 2
        cv2.putText(frame, item, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return button_boxes

# === Main Loop ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hover_index = -1
    fingertip = None

    # Draw menu and get button boxes
    button_boxes = draw_menu(frame, menu_selected)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # Index fingertip
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)
            fingertip = (x, y)

            cv2.circle(frame, fingertip, 18, (0, 0, 255), cv2.FILLED)

            # Check if fingertip is inside a button
            for i, box in enumerate(button_boxes):
                if box[0] < x < box[2] and box[1] < y < box[3]:
                    hover_index = i
                    break

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    progress = 0
    if hover_index != -1:
        if hover_index == menu_selected:
            elapsed = time.time() - hover_start_time
            progress = min(1, elapsed / hover_duration)
            if elapsed >= hover_duration:
                chosen = menus[current_menu][hover_index]
                print(f"✅ Selected: {chosen}")

                if chosen == "Back":
                    current_menu = "Main Menu"
                elif chosen in menus:  # go into submenu
                    current_menu = chosen
                elif chosen == "Exit":
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                hover_start_time = None
                menu_selected = -1
        else:
            menu_selected = hover_index
            hover_start_time = time.time()
    else:
        menu_selected = -1
        hover_start_time = None

    # Progress ring
    if fingertip and progress > 0:
        radius = 30
        end_angle = int(360 * progress)
        cv2.circle(frame, fingertip, radius, (120, 120, 120), 2)
        cv2.ellipse(frame, fingertip, (radius, radius), -90, 0, end_angle, (0, 200, 255), 5)

    # Menu title
    cv2.putText(frame, f"{current_menu}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    cv2.imshow("VR Menu", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
