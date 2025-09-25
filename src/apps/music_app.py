# apps/music_app.py
import webbrowser
import cv2
import mediapipe as mp
from apps.common import draw_text_centered

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

HOVER_DURATION = 1.0

def run_music_app():
    # Ask user for a link (you said you'll provide one later) — use input prompt
    link = input("Paste your music link (e.g., Spotify/URL). Press Enter to open default sample: ").strip()
    if not link:
        # sample url (no audio handling by this app — it opens browser)
        link = "https://open.spotify.com/"
    # open in browser
    webbrowser.open(link, new=2)

    # show overlay until user hovers Back
    cap = cv2.VideoCapture(0)
    hover_start = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h,w,_ = frame.shape
        draw_text_centered(frame, "Music App (browser opened). Hover Back to return.", (w//2, 40), fontsize=0.9)
        # back button
        bx,by,bw,bh = 20,20,140,60
        cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (200,200,200), 2)
        draw_text_centered(frame, "Back", (bx+bw//2, by+bh//2), fontsize=0.8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        fingertip = None
        hover = False
        if res.multi_hand_landmarks:
            for hms in res.multi_hand_landmarks:
                x=int(hms.landmark[8].x*w)
                y=int(hms.landmark[8].y*h)
                fingertip=(x,y)
                cv2.circle(frame, (x,y), 12, (0,0,255), -1)
                if bx < x < bx+bw and by < y < by+bh:
                    hover = True
                    break
        if hover:
            if hover_start is None:
                hover_start = time.time()
            elif time.time() - hover_start > HOVER_DURATION:
                break
        else:
            hover_start = None

        cv2.imshow("Music App", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
