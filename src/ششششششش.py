import cv2
import mediapipe as mp
import numpy as np
import trimesh
import pyrender

# Load 3D model (.glb file)
mesh = trimesh.load("your_model.glb")
scene = pyrender.Scene()
mesh_node = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_node)

# Create renderer (offscreen so we can overlay later)
renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

# Camera
cap = cv2.VideoCapture(0)

# Mediapipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Detect hands
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # --- Render 3D model ---
    color, _ = renderer.render(scene)

    # Convert RGBA → BGR
    if color.shape[2] == 4:
        color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)

    # Resize model render to camera size
    color_resized = cv2.resize(color, (frame.shape[1], frame.shape[0]))

    # Ensure both are 3-channel BGR
    if frame.shape[2] != 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if color_resized.shape[2] != 3:
        color_resized = cv2.cvtColor(color_resized, cv2.COLOR_GRAY2BGR)

    # Blend camera + model
    blended = cv2.addWeighted(frame, 0.7, color_resized, 0.3, 0)

    cv2.imshow("AR Model", blended)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
