# apps/virtual_lab.py
import cv2, time, math
import mediapipe as mp
from apps.common import draw_text_centered

CAM_W, CAM_H = 960, 540

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.55,
                       min_tracking_confidence=0.55,
                       max_num_hands=2)

TOOLS = ["Organs", "Sine", "Chemistry"]

class VirtualLabMenu:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.current_screen = "menu"
        self.current_tool = 0   # index in TOOLS
        self.running = True

    def draw_side_panel(self, frame):
        h, w, _ = frame.shape
        x0 = 12
        y0 = 20
        cv2.rectangle(frame, (x0, y0), (x0+180, y0+240), (30,30,36), -1)
        draw_text_centered(frame, "Virtual Lab", (x0+90, y0+20), fontsize=0.8)
        for i, tool in enumerate(TOOLS):
            ty = y0 + 60 + i*60
            color = (0,200,180) if i == self.current_tool else (200,200,200)
            cv2.rectangle(frame, (x0+10, ty-20), (x0+170, ty+20), (50,50,60), -1)
            cv2.rectangle(frame, (x0+10, ty-20), (x0+170, ty+20), color, 2)
            draw_text_centered(frame, tool, (x0+90, ty), fontsize=0.6, color=color)

    def handle_gestures(self, res, w, h):
        """Detect swipe or pinch gestures for navigation"""
        if not res.multi_hand_landmarks:
            return
        first = res.multi_hand_landmarks[0]
        ix = int(first.landmark[8].x * w)
        iy = int(first.landmark[8].y * h)
        tx = int(first.landmark[4].x * w)
        ty = int(first.landmark[4].y * h)

        # simple pinch detect
        d = math.hypot(ix - tx, iy - ty)
        if d < 40 and self.current_screen == "menu":
            # Enter tool
            self.current_screen = TOOLS[self.current_tool].lower()

    def run(self):
        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAM_W, CAM_H))
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            self.draw_side_panel(frame)
            draw_text_centered(frame, f"Screen: {self.current_screen}", (w//2, 30), fontsize=0.7)

            # Navigation handling
            self.handle_gestures(res, w, h)

            # Tool rendering
            if self.current_screen == "menu":
                draw_text_centered(frame, "Pinch to enter selected tool", (w//2, h-30), fontsize=0.6)
            elif self.current_screen == "organs":
                draw_text_centered(frame, "Organs Viewer (Back = ESC)", (w//2, 60), fontsize=0.7)
                # TODO: call a light draw_organs_preview(frame)
            elif self.current_screen == "sine":
                draw_text_centered(frame, "Sine Simulator (Back = ESC)", (w//2, 60), fontsize=0.7)
            elif self.current_screen == "chemistry":
                draw_text_centered(frame, "Chemistry Lab (Back = ESC)", (w//2, 60), fontsize=0.7)

            cv2.imshow("Virtual Lab", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC → back or quit
                if self.current_screen == "menu":
                    self.running = False
                else:
                    self.current_screen = "menu"

        self.cap.release()
        cv2.destroyAllWindows()

def run_virtual_lab():
    app = VirtualLabMenu()
    app.run()
