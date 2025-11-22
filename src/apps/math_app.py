# apps/math_app.py
"""
Interactive Math App for NexisVerse
- internal menu with 3 modes:
  1) Math Game (pinch selection + 1s circular timer to confirm)
  2) Calculator (on-screen keypad, pinch to press, supports expressions)
  3) Equation Plotter (pick examples or enter custom, plots y = f(x))

Requirements: OpenCV, mediapipe, numpy, matplotlib

Notes:
- Input is done via hand pinch (thumb-tip to index-tip). A "press" is a pinch held for HOVER_DURATION.
- For safety, expression evaluation uses a restricted namespace and `ast` checks.
"""

import cv2
import mediapipe as mp
import time
import math
import random
import numpy as np
import ast
import operator as op
import matplotlib.pyplot as plt
import io

# mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)

# UI constants
HOVER_DURATION = 1.0  # seconds to confirm selection
PINCH_THRESHOLD = 40  # pixels

# ---------- Safe expression evaluation ----------
# Allow basic math functions
SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
SAFE_NAMES.update({
    'abs': abs, 'min': min, 'max': max, 'round': round,
    'np': np
})

# Allowed AST nodes
ALLOWED_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.UAdd, ast.USub,
    ast.Call, ast.Name, ast.Tuple, ast.List, ast.Constant
}


def is_safe_expr(expr: str) -> bool:
    try:
        node = ast.parse(expr, mode='eval')
    except Exception:
        return False

    for n in ast.walk(node):
        if not isinstance(n, tuple(ALLOWED_NODES)):
            return False
        # Disallow calls to attributes (like os.system)
        if isinstance(n, ast.Call):
            if not isinstance(n.func, (ast.Name,)):
                return False
    return True


def safe_eval(expr: str, extra_env=None):
    """Evaluate math expression safely."""
    if not is_safe_expr(expr):
        raise ValueError("Expression contains unsafe nodes")
    env = dict(SAFE_NAMES)
    if extra_env:
        env.update(extra_env)
    return eval(compile(ast.parse(expr, mode='eval'), '<string>', 'eval'), {'__builtins__': {}}, env)


# ---------- Math Game Mode ----------

def make_question():
    if random.random() < 0.5:
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        op = random.choice(['+', '-', '*'])
        expr = f"{a} {op} {b}"
        answer = eval(expr)
        prompt = f"{expr} = ?"
        choices = [answer, answer + random.randint(1,5), answer - random.randint(1,5), answer + random.randint(6,12)]
    else:
        if random.random() < 0.6:
            w = random.randint(2, 12)
            h = random.randint(2, 12)
            prompt = f"Area rect {w}x{h}?"
            answer = w*h
        else:
            r = random.randint(1,6)
            prompt = f"Area circle r={r}? (pi~3.14)"
            answer = round(3.14 * r * r, 1)
        choices = [answer]
        while len(choices) < 4:
            val = round(answer + random.uniform(-10,10), 1)
            if val not in choices:
                choices.append(val)
    random.shuffle(choices)
    return {'prompt': prompt, 'choices': choices, 'answer': answer}


def run_math_game(cap):
    question = make_question()
    score = 0
    chosen_idx = -1
    hover_start = None
    hover_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # UI
        cv2.rectangle(frame, (40,40), (w-40,120), (10,10,30), -1)
        cv2.putText(frame, 'Math Game - Hover & Pinch 1s to choose (ESC to back)', (60,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,220,255), 2)
        cv2.putText(frame, f'Score: {score}', (w-180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,220,255), 2)

        # Render question and choices
        cv2.putText(frame, question['prompt'], (w//2 - 200, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220,220,20), 2)
        boxes = []
        box_w = 500
        box_h = 90
        start_x = (w - box_w) // 2
        start_y = 220
        for i, c in enumerate(question['choices']):
            y = start_y + i*(box_h+20)
            boxes.append((start_x, y, start_x+box_w, y+box_h))
            color = (180,180,180)
            if i == hover_idx:
                color = (0,200,255)
            cv2.rectangle(frame, (start_x,y), (start_x+box_w, y+box_h), color, 3)
            cv2.putText(frame, str(c), (start_x+40, y+box_h//2+12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        fingertip = None
        if res.multi_hand_landmarks:
            for hms in res.multi_hand_landmarks:
                x = int(hms.landmark[8].x * w)
                y = int(hms.landmark[8].y * h)
                fingertip = (x,y)
                cv2.circle(frame, fingertip, 12, (0,0,255), -1)
                tx = int(hms.landmark[4].x * w)
                ty = int(hms.landmark[4].y * h)
                dist = math.hypot(x-tx, y-ty)
                is_pinch = dist < PINCH_THRESHOLD

                # detect hover over boxes
                found = -1
                for i, b in enumerate(boxes):
                    if b[0] < x < b[2] and b[1] < y < b[3]:
                        found = i
                        break

                if found != -1:
                    if hover_idx == found and is_pinch:
                        if hover_start is None:
                            hover_start = time.time()
                        elapsed = time.time() - hover_start
                        progress = min(1.0, elapsed / HOVER_DURATION)
                        # draw circular loading around fingertip
                        cv2.circle(frame, fingertip, 30, (120,120,120), 3)
                        cv2.ellipse(frame, fingertip, (30,30), -90, 0, int(360*progress), (0,200,255), 6)
                        if elapsed >= HOVER_DURATION:
                            val = question['choices'][found]
                            if abs(val - question['answer']) < 1e-6:
                                score += 1
                            # next question
                            question = make_question()
                            hover_start = None
                            hover_idx = -1
                            chosen_idx = -1
                    else:
                        hover_idx = found
                        hover_start = None
                else:
                    hover_idx = -1
                    hover_start = None

        # draw fingertip progress even if no pinch but hovering with finger
        if fingertip and hover_start:
            cx,cy = fingertip
            elapsed = time.time() - hover_start
            progress = min(1.0, elapsed / HOVER_DURATION)
            cv2.circle(frame, (cx,cy), 30, (120,120,120), 3)
            cv2.ellipse(frame, (cx,cy), (30,30), -90, 0, int(360*progress), (0,200,255), 6)

        cv2.imshow('Math Game', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow('Math Game')
            break


# ---------- Calculator Mode (on-screen keypad) ----------

KEYS = [
    ['7','8','9','/'],
    ['4','5','6','*'],
    ['1','2','3','-'],
    ['0','.','^','+'],
    ['(',')','C','=']
]


def draw_keypad(frame, keys, selected=None):
    h, w, _ = frame.shape
    keypad_w = 600
    keypad_h = 500
    start_x = (w - keypad_w)//2
    start_y = 150
    box_w = keypad_w // 4
    box_h = keypad_h // 5
    boxes = []
    for r, row in enumerate(keys):
        for c, label in enumerate(row):
            x1 = start_x + c*box_w
            y1 = start_y + r*box_h
            x2 = x1 + box_w
            y2 = y1 + box_h
            boxes.append((x1,y1,x2,y2,label))
            color = (60,60,80)
            border = (150,150,170)
            if selected == (r,c):
                color = (0,160,220)
                border = (255,255,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, -1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), border, 2)
            cv2.putText(frame, label, (x1+box_w//3, y1+box_h//2+12), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 3)
    return boxes


def run_calculator(cap):
    expr = ''
    selected = None
    hover_start = None
    hover_cell = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        cv2.putText(frame, 'Calculator - Pinch to press keys (ESC to back)', (40,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,220,255), 2)
        cv2.putText(frame, expr, (60,120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220,220,20), 2)

        boxes = draw_keypad(frame, KEYS, selected)

        fingertip = None
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)
            fingertip = (x,y)
            cv2.circle(frame, fingertip, 10, (0,0,255), -1)
            tx = int(hand.landmark[4].x * w)
            ty = int(hand.landmark[4].y * h)
            dist = math.hypot(x-tx, y-ty)
            is_pinch = dist < PINCH_THRESHOLD

            found = None
            for i, (x1,y1,x2,y2,label) in enumerate(boxes):
                if x1 < x < x2 and y1 < y < y2:
                    r = i // 4
                    c = i % 4
                    found = (r,c,label)
                    break

            if found:
                sel = (found[0], found[1])
                selected = sel
                if is_pinch:
                    if hover_cell == sel:
                        if hover_start is None:
                            hover_start = time.time()
                        elapsed = time.time() - hover_start
                        progress = min(1.0, elapsed / HOVER_DURATION)
                        cv2.circle(frame, fingertip, 28, (120,120,120), 3)
                        cv2.ellipse(frame, (fingertip[0], fingertip[1]), (28,28), -90, 0, int(360*progress), (0,200,255), 6)
                        if elapsed >= HOVER_DURATION:
                            label = found[2]
                            if label == 'C':
                                expr = ''
                            elif label == '=':
                                try:
                                    # replace '^' with **
                                    safe = expr.replace('^','**')
                                    resv = safe_eval(safe)
                                    expr = str(resv)
                                except Exception as e:
                                    expr = 'ERR'
                            else:
                                expr += label
                            hover_start = None
                            hover_cell = None
                            selected = None
                    else:
                        hover_cell = sel
                        hover_start = None
                else:
                    hover_cell = None
                    hover_start = None
            else:
                selected = None
                hover_cell = None
                hover_start = None

        cv2.imshow('Calculator', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow('Calculator')
            break


# ---------- Equation Plotter Mode ----------

EXAMPLES = [
    'sin(x)',
    'cos(x) * x',
    'x**2 - 3*x + 2',
    'np.exp(-x**2)',
    'np.sin(2*x) + 0.5*np.cos(x)'
]


def plot_expression(expr: str, width=640, height=480):
    # produce a matplotlib figure as an OpenCV image
    x = np.linspace(-10, 10, 1000)
    safe = expr.replace('^','**')
    try:
        y = safe_eval(safe, {'x': x})
        y = np.array(y)
    except Exception as e:
        return None, str(e)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, y)
    ax.set_title(f'y = {expr}')
    ax.grid(True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, height))
    return img, None


def run_plotter(cap):
    selected_idx = 0
    hover_start = None
    hover_idx = -1
    custom_expr = ''
    show_plot = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        cv2.putText(frame, 'Equation Plotter - Pinch to choose example or "Custom" (ESC to back)', (40,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,220,255), 2)

        # list examples
        start_x = 60
        start_y = 130
        box_w = 560
        box_h = 60
        items = EXAMPLES + ['CUSTOM']
        boxes = []
        for i, it in enumerate(items):
            y = start_y + i*(box_h+12)
            boxes.append((start_x,y,start_x+box_w,y+box_h))
            color = (70,70,90)
            if i == hover_idx:
                color = (0,180,220)
            cv2.rectangle(frame, (start_x,y), (start_x+box_w,y+box_h), color, -1)
            cv2.rectangle(frame, (start_x,y), (start_x+box_w,y+box_h), (200,200,220), 2)
            cv2.putText(frame, it, (start_x+16, y+box_h//2+8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        fingertip = None
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)
            fingertip = (x,y)
            cv2.circle(frame, fingertip, 10, (0,0,255), -1)
            tx = int(hand.landmark[4].x * w)
            ty = int(hand.landmark[4].y * h)
            dist = math.hypot(x-tx, y-ty)
            is_pinch = dist < PINCH_THRESHOLD

            found = -1
            for i, b in enumerate(boxes):
                if b[0] < x < b[2] and b[1] < y < b[3]:
                    found = i
                    break

            if found != -1:
                if hover_idx == found and is_pinch:
                    if hover_start is None:
                        hover_start = time.time()
                    elapsed = time.time() - hover_start
                    progress = min(1.0, elapsed / HOVER_DURATION)
                    cv2.circle(frame, fingertip, 26, (120,120,120), 3)
                    cv2.ellipse(frame, fingertip, (26,26), -90, 0, int(360*progress), (0,200,255), 6)
                    if elapsed >= HOVER_DURATION:
                        if found < len(EXAMPLES):
                            expr = EXAMPLES[found]
                            img, err = plot_expression(expr)
                            if img is None:
                                show_plot = f'ERR: {err}'
                            else:
                                show_plot = img
                        else:
                            # CUSTOM selected — toggle to input mode (quick keypad reuse)
                            custom_expr = ''
                            # open a simple text-entry overlay: we'll let user use calculator keypad mapping
                            show_plot = 'INPUT'
                        hover_start = None
                        hover_idx = -1
                else:
                    hover_idx = found
                    hover_start = None
            else:
                hover_idx = -1
                hover_start = None

        # render plot preview on right side
        if isinstance(show_plot, np.ndarray):
            ph, pw, _ = show_plot.shape
            # place it on the right
            fx = w - pw - 40
            fy = 120
            frame[fy:fy+ph, fx:fx+pw] = show_plot
        elif isinstance(show_plot, str) and show_plot.startswith('ERR'):
            cv2.putText(frame, show_plot, (w-420, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        elif show_plot == 'INPUT':
            cv2.putText(frame, 'Enter custom expression with Calculator, then press = to plot', (60, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,20), 2)

        cv2.imshow('Equation Plotter', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow('Equation Plotter')
            break




# ---------- Top-level Math App Menu ----------

def run_math_app():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    menu_items = ['Math Game', 'Calculator', 'Equation Plotter', 'Back']
    hover_idx = -1
    last_selection_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # Draw sidebar menu
        sx = 30
        sy = 80
        bw = 320
        bh = 70
        for i, it in enumerate(menu_items):
            y = sy + i*(bh + 16)
            color = (50,60,80)
            if i == hover_idx:
                color = (0,180,220)
            cv2.rectangle(frame, (sx, y), (sx+bw, y+bh), color, -1)
            cv2.rectangle(frame, (sx, y), (sx+bw, y+bh), (200,200,220), 2)
            cv2.putText(frame, it, (sx+24, y+bh//2+12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        fingertip = None
        pinch = False
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)
            fingertip = (x,y)
            cv2.circle(frame, fingertip, 10, (0,0,255), -1)
            tx = int(hand.landmark[4].x * w)
            ty = int(hand.landmark[4].y * h)
            dist = math.hypot(x-tx, y-ty)
            pinch = dist < PINCH_THRESHOLD

            found = -1
            for i in range(len(menu_items)):
                yi = sy + i*(bh+16)
                if sx < x < sx+bw and yi < y < yi+bh:
                    found = i
                    break

            if found != -1:
                if hover_idx == found and pinch:
                    if last_selection_time == 0:
                        last_selection_time = time.time()
                    elapsed = time.time() - last_selection_time
                    progress = min(1.0, elapsed / HOVER_DURATION)
                    cv2.circle(frame, fingertip, 26, (120,120,120), 3)
                    cv2.ellipse(frame, fingertip, (26,26), -90, 0, int(360*progress), (0,200,255), 6)
                    if elapsed >= HOVER_DURATION:
                        choice = menu_items[found]
                        if choice == 'Math Game':
                            run_math_game(cap)
                        elif choice == 'Calculator':
                            run_calculator(cap)
                        elif choice == 'Equation Plotter':
                            run_plotter(cap)
                        elif choice == 'Back':
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        last_selection_time = 0
                else:
                    hover_idx = found
                    last_selection_time = 0
            else:
                hover_idx = -1
                last_selection_time = 0

        cv2.imshow('NexisVerse - Math App', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
