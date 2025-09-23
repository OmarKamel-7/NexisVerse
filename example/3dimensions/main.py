import cv2
import mediapipe as mp
import pyvista as pv
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# === Ask user to select 3D brain file ===
Tk().withdraw()  # Hide Tkinter root window
file_path = askopenfilename(title="Select Brain Mesh File",
                            filetypes=[("3D Models", "*.glb *.gltf *.obj *.stl")])
if not file_path:
    print("No file selected. Exiting.")
    exit()

# === Load 3D Brain ===
brain = pv.read(file_path)
brain.scale([0.5, 0.5, 0.5])

plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
actor = plotter.add_mesh(brain, smooth_shading=True, color="lightblue")
plotter.show(auto_close=False)

# === Mediapipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Webcam ===
cap = cv2.VideoCapture(0)

# Interaction states
last_x, last_y = None, None
zoom_factor = 1.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hover_effect = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Index fingertip
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            # Thumb tip
            tx = int(handLms.landmark[4].x * w)
            ty = int(handLms.landmark[4].y * h)

            # Distance = pinch (for zoom)
            distance = np.hypot(tx - x, ty - y)
            zoom_factor = 1 + (distance / 300.0)

            # Rotation based on fingertip movement
            if last_x is not None:
                dx = x - last_x
                dy = y - last_y

                plotter.camera.azimuth_angle(dx * 0.2)
                plotter.camera.elevation_angle(dy * 0.2)

            last_x, last_y = x, y

            # Hover detection (if hand near center)
            if abs(x - w//2) < 100 and abs(y - h//2) < 100:
                hover_effect = True

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Apply zoom
    plotter.camera.zoom(zoom_factor)

    # Change color if hovering
    if hover_effect:
        actor.prop.color = (1, 0.5, 0.5)  # reddish glow
    else:
        actor.prop.color = (0.6, 0.8, 1.0)  # default blue

    # Render brain overlay
    brain_img = plotter.screenshot(return_img=True)
    brain_img = cv2.resize(brain_img, (w, h))

    blended = cv2.addWeighted(frame, 0.5, brain_img, 0.5, 0)

    cv2.putText(blended, "Finger move = rotate | Pinch = zoom | Hover = glow",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Floating Brain AR", blended)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
