# apps/study_session.py
import cv2
import mediapipe as mp
import time
import math
from apps.common import draw_text_centered

# We'll use MediaPipe Face Mesh to detect eye openness proportion.
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Eye landmark indices (from Mediapipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # left eye indices approx
RIGHT_EYE = [263, 387, 385, 362, 380, 373] # right eye indices approx

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    # compute a simple openness ratio using vertical/horizontal distances
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
    # horizontal
    left = pts[0]
    right = pts[3]
    horiz = math.hypot(right[0]-left[0], right[1]-left[1]) + 1e-6
    # vertical average of two vertical pairs
    v1 = math.hypot(pts[1][0]-pts[5][0], pts[1][1]-pts[5][1])
    v2 = math.hypot(pts[2][0]-pts[4][0], pts[2][1]-pts[4][1])
    vert = (v1+v2)/2.0 + 1e-6
    return vert / horiz

def run_study_session():
    cap = cv2.VideoCapture(0)
    start_time = None
    eyes_open_seconds = 0.0
    running = False
    last_frame = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape
        now = time.time()
        dt = now - last_frame
        last_frame = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        eyes_open = False
        if res.multi_face_landmarks:
            for faceLms in res.multi_face_landmarks:
                # compute left and right EAR
                ear_l = eye_aspect_ratio(faceLms.landmark, LEFT_EYE, w, h)
                ear_r = eye_aspect_ratio(faceLms.landmark, RIGHT_EYE, w, h)
                ear = (ear_l + ear_r) / 2.0
                # threshold: when ratio > 0.25 we consider open (empirical)
                if ear > 0.25:
                    eyes_open = True
                # draw small overlay
                draw_text_centered(frame, f"EAR:{ear:.2f}", (w-120, 30), fontsize=0.6, align_left=True)
                break

        # session control: hover top-left back to exit; hover bottom-center to start/pause
        # we'll use a simple cv2 UI for start/pause/back using finger hover
        # For selection we re-use a simple hover time
        # draw controls
        cv2.rectangle(frame, (20,20), (160,80), (200,200,200), 2)
        draw_text_centered(frame, "Back", (100,50), fontsize=0.7)
        cv2.rectangle(frame, (w//2-150, h-100), (w//2+150, h-20), (200,200,200), 2)
        draw_text_centered(frame, "Start/Pause (hover)", (w//2, h-60), fontsize=0.8)

        fingertip = None
        # reuse hand detection (small overhead): detect fingertip with MediaPipe Hands for hover interactions
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = hands.process(rgb_small)
        hover_target = None
        if hand_res.multi_hand_landmarks:
            for hms in hand_res.multi_hand_landmarks:
                x=int(hms.landmark[8].x * w)
                y=int(hms.landmark[8].y * h)
                fingertip=(x,y)
                cv2.circle(frame, (x,y), 12, (0,0,255), -1)
                # back
                if 20 < x < 160 and 20 < y < 80:
                    hover_target = "back"
                # start/pause
                if w//2-150 < x < w//2+150 and h-100 < y < h-20:
                    hover_target = "toggle"
                break
        hands.close()

        # hover handling with simple persistent state
        # store hover start timestamp in closure
        if not hasattr(run_study_session, "hover_start"):
            run_study_session.hover_start = None
            run_study_session.hover_target = None

        if hover_target:
            if run_study_session.hover_target == hover_target and run_study_session.hover_start:
                if time.time() - run_study_session.hover_start > 0.9:
                    if hover_target == "back":
                        hands.close()
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif hover_target == "toggle":
                        if not running:
                            running = True
                            start_time = time.time()
                        else:
                            running = False
                            if start_time:
                                eyes_open_seconds += time.time() - start_time
                            start_time = None
                    run_study_session.hover_start = None
                    run_study_session.hover_target = None
            else:
                run_study_session.hover_target = hover_target
                run_study_session.hover_start = time.time()
        else:
            run_study_session.hover_start = None
            run_study_session.hover_target = None

        # counting eyes-open time only while session running and eyes are open
        if running:
            if eyes_open:
                # increment by dt
                eyes_open_seconds += dt

        # draw status
        draw_text_centered(frame, f"Session running: {'Yes' if running else 'No'}", (w//2, 80), fontsize=0.7)
        draw_text_centered(frame, f"Eyes-open study time: {int(eyes_open_seconds)}s", (w//2, 110), fontsize=0.9, color=(220,220,20))

        cv2.imshow("Study Session", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
