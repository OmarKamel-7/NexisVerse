# apps/virtual_lab_bio.py
import cv2
import mediapipe as mp
import math, os, time
import trimesh, pyrender
import numpy as np
from apps.common import draw_text_centered

# ---------------- MENU CONFIG ----------------
BIO_MENUS = {
    "Biology Lab": ["Organs", "Sample 2", "Back"],
    "Organs": ["Kidney", "Heart", "Brain", "Back"]
}

PINCH_THRESHOLD = 40
BUTTON_HEIGHT = 70
BUTTON_SPACING = 20
SIDEBAR_WIDTH = 380
DEBOUNCE_TIME = 1.0   # seconds

# Thumbnail images
THUMBNAILS = {
    "Kidney": "apps/assets/kidney.jpg",
    "Heart": "apps/assets/heart.jpg",
    "Brain": "apps/assets/brain.jpg"
}

# 3D model file paths
MODELS = {
    "Kidney": r"C:\Users\amora\OneDrive\Desktop\New folder (2)\NexisVerse-main\src\apps\assets\stylizedhumanheart.glb",
    "Heart":  r"C:\Users\amora\OneDrive\Desktop\New folder (2)\NexisVerse-main\src\apps\assets\scalp_anatomy.glb",
    "Brain":  r"C:\Users\amora\OneDrive\Desktop\New folder (2)\NexisVerse-main\src\apps\assets\source.glb"
}

# ---------------- MEDIAPIPE HANDS ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=2
)

# ---------------- STATE ----------------
current_menu = "Biology Lab"
hovered_button = -1
last_click_time = 0


# ---------------- HELPERS ----------------
def compute_button_layout(items):
    """Calculate positions for sidebar buttons."""
    boxes = []
    start_x = 40
    start_y = 140
    for i, _ in enumerate(items):
        y = start_y + i * (BUTTON_HEIGHT + BUTTON_SPACING)
        boxes.append((start_x, y, start_x + SIDEBAR_WIDTH, y + BUTTON_HEIGHT))
    return boxes


def draw_sidebar(frame, items, selected_index):
    """Draw sidebar menu with buttons and thumbnails."""
    button_boxes = compute_button_layout(items)

    cv2.rectangle(frame, (20, 100), (SIDEBAR_WIDTH + 60, 900), (25, 25, 40), -1)
    cv2.rectangle(frame, (20, 100), (SIDEBAR_WIDTH + 60, 900), (80, 150, 200), 2)

    for i, (x1, y1, x2, y2) in enumerate(button_boxes):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if i == selected_index:
            bg_color = (60, 200, 255)
            border_color = (255, 255, 255)
        else:
            bg_color = (50, 60, 90)
            border_color = (150, 180, 220)

        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

        draw_text_centered(frame, items[i], (cx, cy), fontsize=0.9, color=(255, 255, 255))

        if items[i] in THUMBNAILS and os.path.exists(THUMBNAILS[items[i]]):
            thumb = cv2.imread(THUMBNAILS[items[i]])
            if thumb is not None:
                thumb = cv2.resize(thumb, (60, 60))
                frame[y1+5:y1+65, x2-70:x2-10] = thumb

    return button_boxes


# ---------------- VR MODEL VIEWER ----------------
def run_3d_model(model_path):
    """Interactive 3D viewer with pinch controls."""
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}")
        return

    mesh = trimesh.load(model_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    scene = pyrender.Scene()
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    node = scene.add(mesh_node)

    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    cam_pose = np.array([
        [1,0,0,0],
        [0,1,0,-0.1],
        [0,0,1,1.5],
        [0,0,0,1]
    ])
    scene.add(camera, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)

    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

    # model state
    model_translation = np.array([0.0, 0.0, 0.0])
    model_rotation = 0.0
    model_scale = 1.0
    last_drag, last_free_hand = None, None

    cap = cv2.VideoCapture(0)
    w, h = 960, 720
    cap.set(3, w)
    cap.set(4, h)

    def is_pinch(landmarks, i1, i2, thresh=40):
        x1,y1 = landmarks[i1]
        x2,y2 = landmarks[i2]
        return np.hypot(x2-x1, y2-y1) < thresh

    def update_model():
        T = np.eye(4); T[:3,3] = model_translation
        R = np.array([
            [np.cos(model_rotation),0,np.sin(model_rotation),0],
            [0,1,0,0],
            [-np.sin(model_rotation),0,np.cos(model_rotation),0],
            [0,0,0,1]
        ])
        S = np.eye(4)*model_scale; S[3,3]=1
        node.matrix = T @ R @ S

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        pinch_points, pinch_flags, landmarks_list = [], [], []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(int(lm.x*w), int(lm.y*h)) for lm in hand_landmarks.landmark]
                landmarks_list.append(landmarks)
                pinching = is_pinch(landmarks, 4, 8)
                pinch_flags.append(pinching)
                if pinching:
                    pinch_points.append(landmarks[8])
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # drag
        if len(pinch_points)==1:
            if last_drag is None: last_drag=pinch_points[0]
            else:
                dx=(pinch_points[0][0]-last_drag[0])/300.0
                dy=(pinch_points[0][1]-last_drag[1])/300.0
                model_translation[0]+=dx; model_translation[1]-=dy
                last_drag=pinch_points[0]
        else: last_drag=None

        # zoom
        if len(pinch_points)==2:
            (x1,y1),(x2,y2)=pinch_points
            dist=np.hypot(x2-x1,y2-y1)
            model_scale=np.clip(dist/200,0.5,3.0)

        # rotate with free hand
        if len(pinch_flags)==2:
            if pinch_flags[0] and not pinch_flags[1]:
                free_hand=landmarks_list[1][0]
            elif pinch_flags[1] and not pinch_flags[0]:
                free_hand=landmarks_list[0][0]
            else: free_hand=None
            if free_hand is not None:
                if last_free_hand is None: last_free_hand=free_hand
                else:
                    dx=free_hand[0]-last_free_hand[0]
                    model_rotation+=dx/200.0
                    last_free_hand=free_hand
            else: last_free_hand=None
        else: last_free_hand=None

        update_model()
        cv2.imshow("Hand Control", frame)
        if cv2.waitKey(10)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- MAIN LOOP ----------------
def run_biology_lab():
    global current_menu, hovered_button, last_click_time

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        items = BIO_MENUS[current_menu]
        boxes = draw_sidebar(frame, items, hovered_button)
        draw_text_centered(frame, current_menu, (SIDEBAR_WIDTH//2+40, 80),
                           fontsize=1.2, color=(0,255,200))

        hover_index, pinch_detected = -1, False
        if result.multi_hand_landmarks:
            hand=result.multi_hand_landmarks[0]
            ix,iy=int(hand.landmark[8].x*w),int(hand.landmark[8].y*h)
            tx,ty=int(hand.landmark[4].x*w),int(hand.landmark[4].y*h)
            cv2.circle(frame,(ix,iy),10,(0,255,255),-1)
            dist=math.hypot(ix-tx,iy-ty)
            pinch_detected=dist<PINCH_THRESHOLD
            for i,(x1,y1,x2,y2) in enumerate(boxes):
                if x1<ix<x2 and y1<iy<y2: hover_index=i; break

        if pinch_detected and hover_index!=-1:
            now=time.time()
            if now-last_click_time>DEBOUNCE_TIME:
                choice=items[hover_index]
                if choice=="Back":
                    if current_menu=="Biology Lab":
                        cap.release(); cv2.destroyAllWindows(); return
                    else: current_menu="Biology Lab"
                elif choice in BIO_MENUS:
                    current_menu=choice
                elif choice in MODELS:
                    cap.release(); cv2.destroyAllWindows()
                    run_3d_model(MODELS[choice])
                    cap=cv2.VideoCapture(0)
                    cap.set(3,1280); cap.set(4,720)
                last_click_time=now

        hovered_button=hover_index
        cv2.imshow("Biology Lab", frame)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release()
    cv2.destroyAllWindows()
